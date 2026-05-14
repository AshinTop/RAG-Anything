#!/usr/bin/env python
"""
Query a PDF that has already been parsed to a MinerU/RAGAnything content_list.

Example:
    python examples/query_parsed_pdf.py "污水处理工程项目的建设用地指标有哪些？"
"""

import argparse
import asyncio
import json
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


def find_default_content_list() -> Path:
    output_dir = Path("output/raganything_parser_test")
    matches = sorted(output_dir.rglob("*_content_list.json"))
    if not matches:
        raise FileNotFoundError(
            "No *_content_list.json found under output/raganything_parser_test. "
            "Pass --content-list explicitly."
        )
    return matches[0]


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insert an already parsed PDF content_list and ask a question."
    )
    parser.add_argument("question", help="Question to ask the indexed PDF")
    parser.add_argument(
        "--content-list",
        default=None,
        help="Path to MinerU/RAGAnything *_content_list.json",
    )
    parser.add_argument(
        "--working-dir",
        default="./rag_storage_pdf_query",
        help="LightRAG storage directory for this parsed PDF",
    )
    parser.add_argument(
        "--file-path",
        default="《城市生活垃圾处理和给水与污水处理工程项目建设用地指标》.pdf",
        help="Reference file name used in citations",
    )
    parser.add_argument(
        "--doc-id",
        default="city-waste-water-land-index",
        help="Stable document ID; keep the same value to avoid re-indexing as a new document",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "local", "global", "naive"],
        help="LightRAG query mode",
    )
    args = parser.parse_args()

    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")
    if not api_key:
        raise RuntimeError("Set LLM_BINDING_API_KEY in .env or the environment first.")

    content_list_path = Path(args.content_list) if args.content_list else find_default_content_list()
    with content_list_path.open("r", encoding="utf-8") as f:
        content_list = json.load(f)

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vision_model = os.getenv("VISION_MODEL", "gpt-4o")
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
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        display_content_stats=True,
    )

    lightrag_kwargs = {}
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

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs=lightrag_kwargs,
    )

    await rag.insert_content_list(
        content_list=content_list,
        file_path=args.file_path,
        doc_id=args.doc_id,
        display_stats=True,
    )

    result = await rag.aquery(args.question, mode=args.mode)
    print("\nAnswer:\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
