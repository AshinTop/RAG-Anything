#!/usr/bin/env python
"""Query an existing Qwen Chinese policy RAG index."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from qwen_policy_common import (
    ANSWER_SYSTEM_PROMPT,
    DEFAULT_WORKING_DIR,
    build_qwen_policy_rag,
    normalize_storage,
)


def local_index_has_chunks(working_dir: Path) -> bool:
    text_chunks_path = working_dir / "kv_store_text_chunks.json"
    if text_chunks_path.exists():
        try:
            with text_chunks_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and len(data) > 0:
                return True
        except Exception:
            pass

    vector_chunks_path = working_dir / "vdb_chunks.json"
    if vector_chunks_path.exists():
        try:
            with vector_chunks_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return bool(data.get("data") or data.get("_data"))
        except Exception:
            pass

    return False


async def query(args: argparse.Namespace) -> None:
    working_dir = Path(args.working_dir)
    storage = normalize_storage(args.storage)
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
        for index, question in enumerate(questions, start=1):
            print(f"\n[{index}] Q: {question}\n", flush=True)
            answer = await rag.aquery(
                question,
                mode=args.mode,
                system_prompt=args.system_prompt,
                top_k=args.top_k,
                chunk_top_k=args.chunk_top_k,
                enable_rerank=args.enable_rerank,
                vlm_enhanced=args.vlm,
            )
            print("A:\n")
            print(answer)
    finally:
        await rag.finalize_storages()


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
        default=DEFAULT_WORKING_DIR,
        help="RAG 工作目录；local 模式从这里读取索引，postgres 模式用于运行文件。",
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
        help="初始化 RAG-Anything 时使用的解析器名称。",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "local", "global", "naive", "mix", "bypass"],
        help="LightRAG 查询模式。",
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
        default=ANSWER_SYSTEM_PROMPT,
        help="问答阶段 system prompt。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(query(parse_args()))
