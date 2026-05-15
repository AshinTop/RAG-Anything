#!/usr/bin/env python
"""
Ask questions against an existing RAGAnything/LightRAG storage directory.

This script does not parse documents and does not insert content_list data. It only
initializes LightRAG from an existing working directory and runs queries.

Examples:
    python -X utf8 examples/query_existing_rag.py "这份文件主要规定了什么？"
    python -X utf8 examples/query_existing_rag.py -q "适用于哪些工程？" -q "污水处理厂用地如何规定？"
"""

import argparse
import asyncio
import os
import sys
from functools import partial
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

load_dotenv(dotenv_path=".env", override=False)


def build_rag(args: argparse.Namespace) -> RAGAnything:
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")
    if not api_key:
        raise RuntimeError("Set LLM_BINDING_API_KEY in .env or the environment first.")

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))

    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    vision_model_func = None
    if args.vlm:
        vision_model = os.getenv("VISION_MODEL", "gpt-4o")

        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return openai_complete_if_cache(
                    vision_model,
                    "",
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
        ),
    )

    config = RAGAnythingConfig(
        working_dir=args.working_dir,
        enable_image_processing=args.vlm,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    lightrag_kwargs = {}
    if args.use_env_storage:
        storage_env_map = {
            "LIGHTRAG_KV_STORAGE": "kv_storage",
            "LIGHTRAG_VECTOR_STORAGE": "vector_storage",
            "LIGHTRAG_GRAPH_STORAGE": "graph_storage",
            "LIGHTRAG_DOC_STATUS_STORAGE": "doc_status_storage",
        }
        for env_name, kwarg_name in storage_env_map.items():
            value = os.getenv(env_name)
            if value:
                lightrag_kwargs[kwarg_name] = value

    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs=lightrag_kwargs,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query an existing RAGAnything storage directory without re-indexing."
    )
    parser.add_argument("question", nargs="?", help="Question to ask.")
    parser.add_argument(
        "-q",
        "--question-list",
        action="append",
        dest="questions",
        help="Question to ask. Can be used multiple times.",
    )
    parser.add_argument(
        "--working-dir",
        default="./rag_storage",
        help="Existing RAGAnything/LightRAG storage directory.",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "local", "global", "naive", "mix", "bypass"],
        help="LightRAG query mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Maximum number of entities/relations to retrieve.",
    )
    parser.add_argument(
        "--chunk-top-k",
        type=int,
        default=20,
        help="Maximum number of chunks to include.",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Enable VLM image enhancement for retrieved image paths. Slower.",
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Enable rerank if a rerank model is configured.",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "请始终使用自然、正式的简体中文回答。资料中的专有名词、表名、指标名、条文名应尽量保留原文中文。"
            "如果检索上下文或知识图谱中出现英文实体、英文关系或英文概括，请先理解其含义，再翻译为贴合本文档语境的中文表达后输出。"
            "不要在中文句子中夹杂英文短语，例如应写“城市给水和污水处理领域”，不要写“urban water supply 和 wastewater management 的领域”。"
            "只有原文确实是英文、标准编号、模型名、API 名或没有合适中文表达时，才保留英文。"
        ),
        help="System prompt used for answering.",
    )
    parser.add_argument(
        "--use-env-storage",
        action="store_true",
        help="Use LIGHTRAG_* storage environment variables, e.g. PostgreSQL.",
    )
    args = parser.parse_args()

    questions = []
    if args.question:
        questions.append(args.question)
    if args.questions:
        questions.extend(args.questions)
    if not questions:
        parser.error("Provide a question argument or at least one -q/--question-list.")

    working_dir = Path(args.working_dir)
    if not working_dir.exists():
        raise FileNotFoundError(f"Storage directory not found: {working_dir}")

    rag = build_rag(args)
    init_result = await rag._ensure_lightrag_initialized()
    if not init_result.get("success"):
        raise RuntimeError(init_result.get("error", "LightRAG initialization failed."))

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


if __name__ == "__main__":
    asyncio.run(main())
