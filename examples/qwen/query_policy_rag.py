#!/usr/bin/env python
"""Query an existing Qwen Chinese policy RAG index."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from qwen_policy_common import (
    ANSWER_SYSTEM_PROMPT,
    STRICT_ANSWER_SYSTEM_PROMPT,
    build_chunk_focused_context,
    build_document_overview_answer,
    build_document_overview_context,
    build_extractive_clause_answer,
    build_extractive_policy_answer,
    build_extractive_table_answer,
    build_protected_table_answer_prefix,
    build_strict_answer_prompt,
    build_qwen_policy_rag,
    clean_answer_text,
    flush_runtime_artifact_records,
    load_index_chunks,
    merge_protected_table_answer,
    normalize_storage,
    resolve_path_from_save,
    resolve_working_dir,
    restore_path_from_save,
    set_postgres_workspace,
    sync_file_to_save,
    sync_tree_to_save,
)
from lightrag import QueryParam


DEFAULT_QA_OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "output" / "qwen_policy_qa")


def local_index_has_chunks(working_dir: Path) -> bool:
    text_chunks_path = resolve_path_from_save(working_dir / "kv_store_text_chunks.json")
    if text_chunks_path.exists():
        try:
            with text_chunks_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and len(data) > 0:
                return True
        except Exception:
            pass

    vector_chunks_path = resolve_path_from_save(working_dir / "vdb_chunks.json")
    if vector_chunks_path.exists():
        try:
            with vector_chunks_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return bool(data.get("data") or data.get("_data"))
        except Exception:
            pass

    return False


def safe_filename_part(text: str, max_length: int = 40) -> str:
    text = re.sub(r"\s+", "", text or "")
    text = re.sub(r'[\\/:*?"<>|`]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return (text[:max_length] or "question")


def write_qa_output(
    *,
    output_dir: str,
    args: argparse.Namespace,
    question_index: int,
    question: str,
    raw_prompt: str,
    focused_context: str,
    answer: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_path / f"{timestamp}_q{question_index}_{safe_filename_part(question)}.txt"
    args_dict = {
        key: value
        for key, value in vars(args).items()
        if key != "system_prompt"
    }
    content = [
        "Qwen Policy RAG QA Result",
        "=" * 80,
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"command: {' '.join(sys.argv)}",
        "",
        "Execution Arguments",
        "-" * 80,
        json.dumps(args_dict, ensure_ascii=False, indent=2),
        "",
        "Question",
        "-" * 80,
        question,
        "",
        "Raw Prompt",
        "-" * 80,
        raw_prompt or "",
        "",
        "Focused Context",
        "-" * 80,
        focused_context or "",
        "",
        "Answer",
        "-" * 80,
        answer or "",
        "",
    ]
    file_path.write_text("\n".join(content), encoding="utf-8")
    sync_file_to_save(file_path)
    return file_path


async def query(args: argparse.Namespace) -> None:
    storage = normalize_storage(args.storage)
    if storage in {"postgres", "postgres-age"}:
        set_postgres_workspace(args.workspace)
    args.working_dir = resolve_working_dir(args.working_dir, storage)
    restore_path_from_save(args.working_dir)
    working_dir = Path(args.working_dir)
    if storage == "local" and not working_dir.exists():
        raise FileNotFoundError(f"索引目录不存在，请先运行导入脚本: {working_dir}")
    if storage == "local" and not local_index_has_chunks(working_dir):
        raise RuntimeError(
            f"索引目录没有可检索的文本块: {working_dir}\n"
            "请先用 import_policy_files.py 重新导入到一个新的空目录，"
            "例如 --working-dir ../rag_storage/qwen_policy_text_v2。"
        )

    rag = build_qwen_policy_rag(
        working_dir=args.working_dir,
        storage=storage,
        parser=args.parser,
        parse_method="auto",
        enable_image_processing=args.vlm,
        enable_table_processing=True,
        enable_equation_processing=True,
        display_content_stats=False,
    )

    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success"):
        raise RuntimeError(init_result.get("error", "LightRAG 初始化失败。"))

    questions: list[str] = []
    if args.question:
        questions.append(args.question)
    if args.questions:
        questions.extend(args.questions)
    if not questions:
        raise RuntimeError("请传入一个问题，或用 -q 传入多个问题。")

    try:
        index_chunks: list[dict] = []
        for index, question in enumerate(questions, start=1):
            print(f"\n[{index}] Q: {question}\n", flush=True)
            raw_prompt = ""
            focused_context = ""
            if args.strict_answer:
                raw_prompt = await rag.lightrag.aquery(
                    question,
                    param=QueryParam(
                        mode=args.mode,
                        top_k=args.top_k,
                        chunk_top_k=args.chunk_top_k,
                        enable_rerank=args.enable_rerank,
                        only_need_prompt=True,
                    ),
                )
                if args.dump_prompt:
                    print("RAW PROMPT:\n")
                    print(raw_prompt)
                    print("\n--- END RAW PROMPT ---\n")
                focused_context = build_chunk_focused_context(
                    index_chunks,
                    question,
                    raw_prompt=raw_prompt,
                )
                overview_context = build_document_overview_context(question, raw_prompt)
                if overview_context:
                    if not index_chunks:
                        index_chunks = await load_index_chunks(args.working_dir, storage)
                    overview_context = build_document_overview_context(
                        question,
                        raw_prompt,
                        index_chunks=index_chunks,
                    )
                if overview_context:
                    focused_context = overview_context
                if not focused_context:
                    index_chunks = await load_index_chunks(args.working_dir, storage)
                    focused_context = build_chunk_focused_context(index_chunks, question)
                if args.dump_prompt and focused_context:
                    print("FOCUSED CONTEXT:\n")
                    print(focused_context)
                    print("\n--- END FOCUSED CONTEXT ---\n")
                answer = await rag.llm_model_func(
                    build_strict_answer_prompt(raw_prompt, question, focused_context),
                    system_prompt=args.system_prompt,
                    temperature=0,
                )
                protected_table_prefix = build_protected_table_answer_prefix(
                    question, focused_context
                )
                if protected_table_prefix:
                    answer = build_extractive_table_answer(
                        question, focused_context, protected_table_prefix
                    ) or merge_protected_table_answer(str(answer), protected_table_prefix)
                else:
                    answer = build_document_overview_answer(
                        question, focused_context
                    ) or build_extractive_policy_answer(
                        question, focused_context
                    ) or build_extractive_clause_answer(
                        question, focused_context
                    ) or clean_answer_text(str(answer))
            else:
                answer = await rag.aquery(
                    question,
                    mode=args.mode,
                    system_prompt=args.system_prompt,
                    top_k=args.top_k,
                    chunk_top_k=args.chunk_top_k,
                    enable_rerank=args.enable_rerank,
                    vlm_enhanced=args.vlm,
                )
                answer = clean_answer_text(str(answer))
            print("A:\n")
            print(answer)
            if args.save_output:
                output_file = write_qa_output(
                    output_dir=args.qa_output_dir,
                    args=args,
                    question_index=index,
                    question=question,
                    raw_prompt=raw_prompt,
                    focused_context=focused_context,
                    answer=str(answer),
                )
                print(f"\nQA 结果已保存: {output_file}")
    finally:
        await rag.finalize_storages()
        sync_tree_to_save(args.working_dir)
        sync_tree_to_save(args.qa_output_dir)
        await flush_runtime_artifact_records()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对已导入的 Qwen 中文政策索引执行问答验证。"
    )
    parser.add_argument("question", nargs="?", help="要提问的问题。")
    parser.add_argument(
        "-q",
        "--question-list",
        action="append",
        dest="questions",
        help="要提问的问题，可重复传入。",
    )
    parser.add_argument(
        "--working-dir",
        default=None,
        help=(
            "RAG 工作目录。local 模式默认读取本地索引目录；postgres 模式可省略，"
            "省略时按数据库 workspace 派生为 rag_storage/<workspace>，仅用于运行文件。"
        ),
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help=(
            "PostgreSQL workspace/知识库名；等价于设置 POSTGRES_WORKSPACE。"
            "仅 postgres/postgres-age 模式生效。"
        ),
    )
    parser.add_argument(
        "--storage",
        default="postgres",
        choices=["local", "postgres", "pg", "postgresql", "postgres-age", "pg-age"],
        help=(
            "索引存储后端，默认 postgres。local=本地文件；postgres=PostgreSQL KV/向量/状态 + "
            "NetworkX 图；postgres-age=全 PostgreSQL，需安装 Apache AGE。"
        ),
    )
    parser.add_argument(
        "--parser",
        default="mineru",
        choices=["mineru", "docling", "paddleocr"],
        help="初始化 RAG-Anything 时使用的解析器名称。",
    )
    parser.add_argument(
        "--mode",
        default="naive",
        choices=["hybrid", "local", "global", "naive", "mix", "bypass"],
        help="LightRAG 查询模式；中文政策原文问答默认使用 naive，避免图谱检索带偏。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="最多检索多少个实体/关系。",
    )
    parser.add_argument(
        "--chunk-top-k",
        type=int,
        default=20,
        help="最多检索多少个文本块。",
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="启用 rerank，前提是已配置 rerank 模型。",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="启用 VLM 图片增强；本地 Qwen3 8B 默认不建议开启。",
    )
    parser.add_argument(
        "--system-prompt",
        default=STRICT_ANSWER_SYSTEM_PROMPT,
        help="问答阶段 system prompt。",
    )
    parser.add_argument(
        "--strict-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="先获取 LightRAG 原始检索提示，再用抽取式 prompt 生成答案。",
    )
    parser.add_argument(
        "--dump-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认打印 LightRAG 生成的原始检索提示；传 --no-dump-prompt 可关闭。",
    )
    parser.add_argument(
        "--save-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认每个 QA 保存一个本地 text 文件；传 --no-save-output 可关闭。",
    )
    parser.add_argument(
        "--qa-output-dir",
        default=DEFAULT_QA_OUTPUT_DIR,
        help="QA 结果保存目录，默认 output/qwen_policy_qa。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(query(parse_args()))
