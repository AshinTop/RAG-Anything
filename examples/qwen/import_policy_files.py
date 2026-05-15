#!/usr/bin/env python
"""Import Chinese policy files into a RAG-Anything index with Qwen."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path

from qwen_policy_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORKING_DIR,
    build_qwen_policy_rag,
    describe_storage_backends,
    insert_retrieval_only_chunks,
    normalize_storage,
    resolve_input_paths,
)

from raganything.utils import insert_text_content
from policy_structure import build_structured_chunks, write_structured_chunks_sidecar


def detect_cuda_status() -> tuple[bool | None, str | None]:
    """Best-effort CUDA availability probe for user-facing import hints."""
    try:
        import torch

        available = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if available else None
        return available, device_name
    except Exception:
        return None, None


def content_item_to_text(item: dict, index: int) -> dict | None:
    content_type = item.get("type", "text")
    page_idx = item.get("page_idx")

    if content_type == "text":
        text = str(item.get("text", "") or "").strip()
        if not text:
            return None
        return {"type": "text", "text": text, "page_idx": page_idx}

    if content_type == "page_number":
        return None

    text_fields = [
        "text",
        "table_body",
        "table_data",
        "latex",
        "equation",
        "content",
        "list",
    ]
    parts: list[str] = []
    for field in text_fields:
        value = item.get(field)
        if value in (None, ""):
            continue
        if isinstance(value, list):
            value = "\n".join(str(v) for v in value if str(v).strip())
        else:
            value = str(value)
        value = value.strip()
        if value:
            parts.append(value)

    if not parts:
        return None

    label_map = {
        "table": "表格",
        "equation": "公式",
        "list": "列表",
    }
    label = label_map.get(str(content_type), str(content_type))
    page_text = f"第 {int(page_idx) + 1} 页" if isinstance(page_idx, int) else "未知页"
    text = f"[{label}，{page_text}，块 {index + 1}]\n" + "\n\n".join(parts)
    return {"type": "text", "text": text, "page_idx": page_idx}


def normalize_content_list_for_text_index(content_list: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    skipped: dict[str, int] = {}
    converted: dict[str, int] = {}

    for index, item in enumerate(content_list):
        if not isinstance(item, dict):
            continue
        content_type = str(item.get("type", "text"))
        text_item = content_item_to_text(item, index)
        if text_item is None:
            skipped[content_type] = skipped.get(content_type, 0) + 1
            continue
        normalized.append(text_item)
        if content_type != "text":
            converted[content_type] = converted.get(content_type, 0) + 1

    print(f"文本索引模式: 入库文本块 {len(normalized)} 个")
    if converted:
        print(f"文本索引模式: 已转成文本的结构化块 {converted}")
    if skipped:
        print(f"文本索引模式: 已跳过无正文/页码块 {skipped}")

    return normalized


def make_text_doc_id(file_path: Path, content_list: list[dict]) -> str:
    digest = hashlib.md5()
    digest.update(str(file_path.resolve()).encode("utf-8"))
    for item in content_list:
        digest.update(str(item.get("page_idx", "")).encode("utf-8"))
        digest.update(str(item.get("text", "")).encode("utf-8"))
    return f"qwen-text-{digest.hexdigest()}"


async def assert_text_insert_succeeded(rag, doc_id: str) -> None:
    status = await rag.lightrag.doc_status.get_by_id(doc_id)
    if not status:
        raise RuntimeError(f"文本索引写入失败：未找到 doc_status 记录 {doc_id}")

    status_value = str(status.get("status", "")).lower()
    chunks_count = int(status.get("chunks_count") or 0)
    error_msg = status.get("error_msg") or ""
    if "processed" not in status_value or chunks_count <= 0:
        raise RuntimeError(
            "文本索引写入失败：embedding 或索引处理未成功。\n"
            f"doc_id={doc_id}\n"
            f"status={status.get('status')}\n"
            f"chunks_count={chunks_count}\n"
            f"error_msg={error_msg}\n"
            "请先确认本地 Qwen 服务的 /v1/embeddings 可用，"
            "且 .env 中 QWEN_EMBEDDING_MODEL、QWEN_EMBEDDING_DIM 与服务实际模型一致。"
        )


async def import_files(args: argparse.Namespace) -> None:
    files = resolve_input_paths(args.files, recursive=args.recursive)
    if not files:
        raise RuntimeError("没有找到可导入的文件。")

    rag = build_qwen_policy_rag(
        working_dir=args.working_dir,
        storage=args.storage,
        parser=args.parser,
        parse_method=args.parse_method,
        output_dir=args.output_dir,
        enable_image_processing=args.enable_images,
        enable_table_processing=not args.disable_tables,
        enable_equation_processing=not args.disable_equations,
        display_content_stats=not args.no_stats,
    )
    storage = normalize_storage(args.storage)

    print("Qwen 中文政策文件导入")
    if storage == "postgres":
        print(f"工作目录: {Path(args.working_dir).resolve()} (仅用于缓存/运行文件)")
    else:
        print(f"本地索引目录: {Path(args.working_dir).resolve()}")
    print(f"解析输出目录: {Path(args.output_dir).resolve()}")
    print(f"解析器: {args.parser}, 解析模式: {args.parse_method}")
    print(f"MinerU 设备: {args.mineru_device or 'default'}")
    cuda_available, cuda_device_name = detect_cuda_status()
    if args.parser == "mineru":
        requested_device = (args.mineru_device or "").strip().lower()
        gpu_requested = requested_device.startswith("cuda")
        if gpu_requested:
            if cuda_available is False:
                print(
                    "提示: 当前命令显式指定了 `--mineru-device cuda`，但本机未检测到可用 CUDA。"
                    " MinerU 可能无法使用 GPU。"
                )
            elif cuda_available and cuda_device_name:
                print(f"提示: 已请求 GPU 解析，检测到 CUDA 设备: {cuda_device_name}")
        else:
            if cuda_available and cuda_device_name:
                print(
                    "提示: 当前命令没有显式使用 GPU。检测到可用 CUDA 设备: "
                    f"{cuda_device_name}。扫描件建议传 `--mineru-device cuda`。"
                )
            elif cuda_available is False:
                print("提示: 当前未检测到可用 CUDA，MinerU 将不会使用 GPU。")
            else:
                print(
                    "提示: 当前未显式指定 GPU，且无法确认 CUDA 状态。"
                    " 如需强制走 GPU，可传 `--mineru-device cuda`。"
                )
    print(f"索引模式: {args.index_mode}")
    print(f"QA 策略: {'structured_retrieval_first' if args.retrieval_only else 'default_lightrag'}")
    print(f"存储后端: {describe_storage_backends(storage)}")
    print(f"待导入文件数: {len(files)}")

    try:
        for index, file_path in enumerate(files, start=1):
            print(f"\n[{index}/{len(files)}] 开始导入: {file_path}", flush=True)
            doc_id = None
            if args.doc_id_prefix:
                doc_id = f"{args.doc_id_prefix}-{index}"
            parser_kwargs = {
                "lang": args.ocr_lang,
                "timeout": args.mineru_timeout,
            }
            if args.mineru_device:
                parser_kwargs["device"] = args.mineru_device
            if args.model_source:
                parser_kwargs["source"] = args.model_source
                parser_kwargs["env"] = {
                    **os.environ,
                    "MINERU_MODEL_SOURCE": args.model_source,
                }

            if args.index_mode == "full":
                await rag.process_document_complete(
                    file_path=str(file_path),
                    output_dir=args.output_dir,
                    parse_method=args.parse_method,
                    display_stats=not args.no_stats,
                    doc_id=doc_id,
                    **parser_kwargs,
                )
            else:
                init_result = await rag._ensure_lightrag_initialized()
                if not init_result.get("success"):
                    raise RuntimeError(init_result.get("error", "LightRAG 初始化失败。"))

                content_list, parsed_doc_id = await rag.parse_document(
                    file_path=str(file_path),
                    output_dir=args.output_dir,
                    parse_method=args.parse_method,
                    display_stats=not args.no_stats,
                    **parser_kwargs,
                )
                text_content_list = normalize_content_list_for_text_index(content_list)
                if not text_content_list:
                    raise RuntimeError(f"解析成功但没有可入库的文本内容: {file_path}")

                text_doc_id = doc_id or make_text_doc_id(file_path, text_content_list)
                print(f"文本索引模式: doc_id={text_doc_id}")
                if args.chunk_mode == "structured":
                    structured_chunks = build_structured_chunks(
                        text_content_list,
                        doc_id=text_doc_id,
                        file_name=file_path.name,
                    )
                    if not structured_chunks:
                        raise RuntimeError(f"结构化切分没有生成 chunk: {file_path}")
                    sidecar = write_structured_chunks_sidecar(
                        structured_chunks, args.working_dir, text_doc_id
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
                            doc_id=text_doc_id,
                            source_ref=str(file_path),
                            chunk_texts=structured_texts,
                            chunk_ids=[chunk.chunk_id for chunk in structured_chunks],
                            full_text="\n\n".join(structured_texts),
                        )
                        await assert_text_insert_succeeded(rag, text_doc_id)
                    else:
                        await insert_text_content(
                            rag.lightrag,
                            input=structured_texts,
                            file_paths=[
                                rag._get_file_reference(str(file_path))
                                for _ in structured_chunks
                            ],
                            ids=[chunk.chunk_id for chunk in structured_chunks],
                        )
                        for chunk in structured_chunks:
                            await assert_text_insert_succeeded(rag, chunk.chunk_id)
                else:
                    text_content = "\n\n".join(
                        item["text"] for item in text_content_list if item.get("text")
                    )
                    await insert_text_content(
                        rag.lightrag,
                        input=text_content,
                        file_paths=rag._get_file_reference(str(file_path)),
                        ids=text_doc_id,
                    )
                    await assert_text_insert_succeeded(rag, text_doc_id)
            print(f"[{index}/{len(files)}] 导入完成: {file_path.name}", flush=True)
    finally:
        await rag.finalize_storages()

    print("\n全部文件导入完成。该脚本只创建索引，不执行问答。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用本地 Qwen3 8B 为中文政策文件构建 RAG-Anything 索引。"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="要导入的文件或目录；目录会按 --recursive 设置扫描。",
    )
    parser.add_argument(
        "--working-dir",
        default=DEFAULT_WORKING_DIR,
        help="RAG 工作目录；local 模式会写入本地索引，postgres 模式用于缓存/运行文件。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="MinerU/解析器输出目录。",
    )
    parser.add_argument(
        "--storage",
        default=None,
        choices=["local", "postgres", "pg", "postgresql", "postgres-age", "pg-age"],
        help=(
            "索引存储后端。local=本地文件；postgres=PostgreSQL KV/向量/状态 + "
            "NetworkX 图；postgres-age=全 PostgreSQL，需安装 Apache AGE。"
        ),
    )
    parser.add_argument(
        "--parser",
        default="mineru",
        choices=["mineru", "docling", "paddleocr"],
        help="RAG-Anything 文档解析器。",
    )
    parser.add_argument(
        "--parse-method",
        default="auto",
        choices=["auto", "ocr", "txt"],
        help="解析模式。",
    )
    parser.add_argument(
        "--ocr-lang",
        default="ch",
        help="OCR 语言；中文政策默认 ch。",
    )
    parser.add_argument(
        "--model-source",
        default=None,
        choices=["huggingface", "modelscope", "local"],
        help=(
            "MinerU 模型来源。国内网络建议传 modelscope；"
            "本地已配置 MinerU 模型目录时传 local。"
        ),
    )
    parser.add_argument(
        "--mineru-timeout",
        type=int,
        default=None,
        help="MinerU 单个文件解析超时时间（秒）；默认不限制。",
    )
    parser.add_argument(
        "--mineru-device",
        default=os.getenv("MINERU_DEVICE") or os.getenv("QWEN_MINERU_DEVICE"),
        help="MinerU 推理设备，例如 cpu、cuda、cuda:0、mps。扫描件建议显式传 cuda。",
    )
    parser.add_argument(
        "--index-mode",
        default="text",
        choices=["text", "full"],
        help=(
            "索引模式。text=把表格/列表/公式等可文本化内容转成普通文本入库，"
            "避免调用多模态描述；full=启用 RAG-Anything 原生多模态处理。"
        ),
    )
    parser.add_argument(
        "--chunk-mode",
        default="auto",
        choices=["auto", "structured"],
        help=(
            "text 索引下的切分方式。auto=沿用 LightRAG 自动 chunk；"
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
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="当输入为目录时递归扫描支持的文件。",
    )
    parser.add_argument(
        "--doc-id-prefix",
        help="可选固定 doc_id 前缀；传入后会生成 <prefix>-1、<prefix>-2。",
    )
    parser.add_argument(
        "--enable-images",
        action="store_true",
        help="启用图片多模态处理；本地 Qwen3 8B 通常不是视觉模型，默认关闭。",
    )
    parser.add_argument(
        "--disable-tables",
        action="store_true",
        help="关闭表格处理；中文政策默认开启。",
    )
    parser.add_argument(
        "--disable-equations",
        action="store_true",
        help="关闭公式处理；默认开启。",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="不显示内容统计。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(import_files(parse_args()))
