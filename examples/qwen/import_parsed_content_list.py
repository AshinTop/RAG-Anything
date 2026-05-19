#!/usr/bin/env python
"""Rebuild a Qwen policy RAG index from existing MinerU content_list JSON."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from qwen_policy_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORKING_DIR,
    build_qwen_policy_rag,
    describe_storage_backends,
    flush_runtime_artifact_records,
    insert_retrieval_only_chunks,
    normalize_storage,
    restore_path_from_save,
    set_runtime_import_job_id,
    sync_tree_to_save,
)
from import_policy_files import (
    assert_text_insert_succeeded,
    normalize_content_list_for_text_index,
)
from raganything.utils import insert_text_content
from policy_structure import build_structured_chunks, write_structured_chunks_sidecar


class TableHTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.current_cell is not None:
            text = normalize_ws("".join(self.current_cell))
            if self.current_row is not None:
                self.current_row.append(text)
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if any(cell for cell in self.current_row):
                self.rows.append(self.current_row)
            self.current_row = None

    def as_text(self) -> str:
        return "\n".join(" | ".join(row) for row in self.rows)


def normalize_ws(text: str) -> str:
    text = str(text or "").replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def inline_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return normalize_ws(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [inline_text(item) for item in value]
        return normalize_ws("".join(part for part in parts if part))
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return normalize_ws(value["content"])
        if isinstance(value.get("text"), str):
            return normalize_ws(value["text"])
        if isinstance(value.get("latex"), str):
            return normalize_ws(value["latex"])
    return ""


def html_table_to_text(html: str) -> str:
    parser = TableHTMLToText()
    parser.feed(html)
    text = parser.as_text()
    if text:
        return text
    return normalize_ws(re.sub(r"<[^>]+>", " ", html))


def mineru_block_to_text(block: dict[str, Any], page_idx: int, block_index: int) -> dict | None:
    block_type = str(block.get("type", "unknown"))
    content = block.get("content")
    parts: list[str] = []

    if isinstance(content, dict):
        if block_type == "title":
            parts.append(inline_text(content.get("title_content")))
        elif block_type == "paragraph":
            parts.append(inline_text(content.get("paragraph_content")))
        elif block_type == "list":
            for item in content.get("list_items") or []:
                item_text = inline_text(item.get("item_content") if isinstance(item, dict) else item)
                if item_text:
                    parts.append(item_text)
        elif block_type == "table":
            caption = inline_text(content.get("table_caption"))
            if caption:
                parts.append(caption)
            html = content.get("html")
            if html:
                parts.append(html_table_to_text(str(html)))
            footnote = inline_text(content.get("table_footnote"))
            if footnote:
                parts.append(footnote)
        elif block_type in {"interline_equation", "equation_interline", "equation"}:
            for key in ("math_content", "latex", "text", "content"):
                value = content.get(key)
                if isinstance(value, str):
                    parts.append(value)
                    break
        else:
            for key in (
                "title_content",
                "paragraph_content",
                "text",
                "latex",
                "html",
            ):
                value = content.get(key)
                if key == "html" and value:
                    parts.append(html_table_to_text(str(value)))
                else:
                    parts.append(inline_text(value))
    else:
        parts.append(inline_text(content))

    text = "\n".join(part for part in (normalize_ws(part) for part in parts) if part)
    if not text:
        return None

    if block_type in {"table", "list", "interline_equation", "equation_interline", "equation"}:
        label_map = {
            "table": "表格",
            "list": "列表",
            "interline_equation": "公式",
            "equation_interline": "公式",
            "equation": "公式",
        }
        label = label_map.get(block_type, block_type)
        text = f"[{label}，第 {page_idx + 1} 页，块 {block_index + 1}]\n{text}"

    return {"type": "text", "text": text, "page_idx": page_idx}


def content_list_to_text_items(data: Any) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError("content_list JSON 顶层必须是 list")

    if all(isinstance(page, list) for page in data):
        text_items: list[dict] = []
        skipped: dict[str, int] = {}
        for page_idx, page in enumerate(data):
            for block_index, block in enumerate(page):
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type", "unknown"))
                text_item = mineru_block_to_text(block, page_idx, block_index)
                if text_item is None:
                    skipped[block_type] = skipped.get(block_type, 0) + 1
                    continue
                text_items.append(text_item)

        print(f"解析复用模式: MinerU v2 页数 {len(data)}，入库文本块 {len(text_items)} 个")
        if skipped:
            print(f"解析复用模式: 已跳过无文本块 {skipped}")
        return text_items

    if all(isinstance(item, dict) for item in data):
        return normalize_content_list_for_text_index(data)

    raise ValueError("不支持的 content_list JSON 结构")


def make_parsed_doc_id(content_list_path: Path, text_items: list[dict]) -> str:
    digest = hashlib.md5()
    digest.update(str(content_list_path.resolve()).encode("utf-8"))
    for item in text_items:
        digest.update(str(item.get("page_idx", "")).encode("utf-8"))
        digest.update(str(item.get("text", "")).encode("utf-8"))
    return f"qwen-parsed-{digest.hexdigest()}"


def infer_source_reference(content_list_path: Path, source_file: str | None) -> str:
    if source_file:
        return source_file

    name = content_list_path.name
    name = re.sub(r"_content_list(?:_v\d+)?\.json$", "", name)
    return name or str(content_list_path)


def resolve_content_list_paths(paths: list[str], recursive: bool) -> list[Path]:
    result: list[Path] = []
    for raw_path in paths:
        path = restore_path_from_save(Path(raw_path).expanduser())
        if path.is_file():
            result.append(path)
            continue
        if path.is_dir():
            iterator = path.rglob("*content_list_v2.json") if recursive else path.glob("*content_list_v2.json")
            found = list(iterator)
            if not found:
                iterator = path.rglob("*content_list*.json") if recursive else path.glob("*content_list*.json")
                found = list(iterator)
            result.extend(item for item in found if item.is_file())
            continue
        raise FileNotFoundError(f"content_list 文件或目录不存在: {path}")

    return sorted(dict.fromkeys(result))


async def import_parsed_content_lists(args: argparse.Namespace) -> None:
    args.import_job_id = set_runtime_import_job_id(args.import_job_id)
    restore_path_from_save(args.working_dir)
    files = resolve_content_list_paths(args.content_lists, recursive=args.recursive)
    if not files:
        raise RuntimeError("没有找到 content_list JSON 文件。")
    if args.source_file and len(files) > 1:
        raise RuntimeError("--source-file 只能在导入单个 content_list 时使用。")
    if args.doc_id and len(files) > 1:
        raise RuntimeError("--doc-id 只能在导入单个 content_list 时使用。")

    storage = normalize_storage(args.storage)
    print("Qwen 已解析 content_list 重建索引")
    print(f"工作目录: {Path(args.working_dir).resolve()}")
    print(f"QA 策略: {'structured_retrieval_first' if args.retrieval_only else 'default_lightrag'}")
    print(f"存储后端: {describe_storage_backends(storage)}")
    print(f"import_job_id: {args.import_job_id}")
    print(f"待导入 content_list 数: {len(files)}")

    if args.dry_run:
        for index, content_list_path in enumerate(files, start=1):
            print(f"\n[{index}/{len(files)}] 检查解析结果: {content_list_path}", flush=True)
            data = json.loads(content_list_path.read_text(encoding="utf-8"))
            text_items = content_list_to_text_items(data)
            doc_id = args.doc_id or make_parsed_doc_id(content_list_path, text_items)
            source_ref = infer_source_reference(content_list_path, args.source_file)
            print(f"dry-run: doc_id={doc_id}")
            print(f"dry-run: source_ref={source_ref}")
            if args.chunk_mode == "structured":
                structured_chunks = build_structured_chunks(
                    text_items, doc_id=doc_id, file_name=Path(source_ref).name
                )
                print(f"dry-run: 结构化 chunks={len(structured_chunks)}")
            else:
                text_content = "\n\n".join(item["text"] for item in text_items if item.get("text"))
                print(f"dry-run: 文本字符数={len(text_content)}")
        print("\ndry-run 完成，未写入索引。")
        return

    rag = build_qwen_policy_rag(
        working_dir=args.working_dir,
        storage=storage,
        parser="mineru",
        parse_method="auto",
        output_dir=args.output_dir,
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=True,
        display_content_stats=False,
    )

    try:
        init_result = await rag._ensure_lightrag_initialized()
        if not init_result.get("success"):
            raise RuntimeError(init_result.get("error", "LightRAG 初始化失败。"))

        for index, content_list_path in enumerate(files, start=1):
            print(f"\n[{index}/{len(files)}] 开始导入解析结果: {content_list_path}", flush=True)
            data = json.loads(content_list_path.read_text(encoding="utf-8"))
            text_items = content_list_to_text_items(data)
            if not text_items:
                raise RuntimeError(f"content_list 没有可入库文本: {content_list_path}")

            doc_id = args.doc_id or make_parsed_doc_id(content_list_path, text_items)
            source_ref = infer_source_reference(content_list_path, args.source_file)

            print(f"解析复用模式: doc_id={doc_id}")
            print(f"解析复用模式: source_ref={source_ref}")
            if args.chunk_mode == "structured":
                structured_chunks = build_structured_chunks(
                    text_items, doc_id=doc_id, file_name=Path(source_ref).name
                )
                if not structured_chunks:
                    raise RuntimeError(f"结构化切分没有生成 chunk: {content_list_path}")
                sidecar = write_structured_chunks_sidecar(
                    structured_chunks, args.working_dir, doc_id
                )
                print(
                    f"结构化切分: chunks={len(structured_chunks)}, sidecar={sidecar}"
                )
                structured_texts = [
                    chunk.to_index_text() for chunk in structured_chunks
                ]
                if args.retrieval_only:
                    await insert_retrieval_only_chunks(
                        rag,
                        doc_id=doc_id,
                        source_ref=source_ref,
                        chunk_texts=structured_texts,
                        chunk_ids=[chunk.chunk_id for chunk in structured_chunks],
                        full_text="\n\n".join(structured_texts),
                    )
                    await assert_text_insert_succeeded(rag, doc_id)
                else:
                    await insert_text_content(
                        rag.lightrag,
                        input=structured_texts,
                        file_paths=[source_ref for _ in structured_chunks],
                        ids=[chunk.chunk_id for chunk in structured_chunks],
                    )
                    for chunk in structured_chunks:
                        await assert_text_insert_succeeded(rag, chunk.chunk_id)
            else:
                text_content = "\n\n".join(item["text"] for item in text_items if item.get("text"))
                await insert_text_content(
                    rag.lightrag,
                    input=text_content,
                    file_paths=source_ref,
                    ids=doc_id,
                )
                await assert_text_insert_succeeded(rag, doc_id)
            print(f"[{index}/{len(files)}] 导入完成: {content_list_path.name}", flush=True)
    finally:
        await rag.finalize_storages()
        sync_tree_to_save(args.working_dir)
        sync_tree_to_save(args.output_dir)
        await flush_runtime_artifact_records()

    print("\n全部解析结果导入完成。该脚本不会重新运行 MinerU，也不执行问答。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="复用 MinerU content_list_v2.json，直接重建 Qwen 中文政策 RAG 索引。"
    )
    parser.add_argument(
        "content_lists",
        nargs="+",
        help="content_list_v2.json 文件或包含该文件的目录。",
    )
    parser.add_argument(
        "--working-dir",
        default=DEFAULT_WORKING_DIR,
        help="RAG 工作目录；postgres 模式用于缓存/运行文件。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="保留给 RAG-Anything 配置的解析输出目录；本脚本不会重新解析 PDF。",
    )
    parser.add_argument(
        "--storage",
        default=None,
        choices=["local", "postgres", "pg", "postgresql", "postgres-age", "pg-age"],
        help=(
            "索引存储后端。postgres=PostgreSQL KV/向量/状态 + NetworkX 图；"
            "postgres-age=全 PostgreSQL，需安装 Apache AGE。默认读取 QWEN_STORAGE。"
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="输入目录时递归查找 *content_list_v2.json。",
    )
    parser.add_argument(
        "--source-file",
        help="原始文件路径或引用名；只适用于单个 content_list。",
    )
    parser.add_argument("--doc-id", help="可选固定 doc_id；只建议单文件导入时使用。")
    parser.add_argument(
        "--import-job-id",
        default=None,
        help="导入任务 UUID；默认自动生成。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析 content_list 并打印统计，不写入索引。",
    )
    parser.add_argument(
        "--chunk-mode",
        default="auto",
        choices=["auto", "structured"],
        help=(
            "切分方式。auto=沿用 LightRAG 自动 chunk；"
            "structured=按章/节/条/表格预切分并注入 metadata。"
        ),
    )
    parser.add_argument(
        "--retrieval-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "结构化 chunk 时默认启用检索优先入库：仅写入文本/向量/doc_status，"
            "不阻塞在实体关系抽取。关闭后回到 LightRAG 默认抽取流程。"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(import_parsed_content_lists(parse_args()))
