#!/usr/bin/env python
"""
PDF QA demo for an already parsed MinerU/RAGAnything content_list.

Before running:
    1. Parse the PDF first and generate *_content_list.json.
    2. Set LLM_BINDING_API_KEY in .env.

Run:
    python -X utf8 examples/pdf_qa_demo.py

Ask custom questions:
    python -X utf8 examples/pdf_qa_demo.py -q "本文件适用于哪些工程项目？" -q "污水处理工程用地指标怎么规定？"

Fast smoke test:
    python -X utf8 examples/pdf_qa_demo.py --max-blocks 40 -q "这份文件主要规定了什么？"
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


DEFAULT_QUESTIONS = [
    "这份文件主要规定了什么内容？",
]


def find_content_list() -> Path:
    """Find the newest parsed content_list from the previous PDF parsing test."""
    output_dir = Path("output/raganything_parser_test")
    matches = sorted(
        output_dir.rglob("*_content_list.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            "没有找到解析结果 *_content_list.json。请先运行 PDF 解析，或使用 --content-list 指定文件。"
        )
    return matches[0]


def load_content_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        content_list = json.load(f)
    if not isinstance(content_list, list):
        raise ValueError(f"content_list 格式不正确: {path}")
    return content_list


def prepare_content_list(
    content_list: list[dict],
    *,
    include_multimodal: bool,
    max_blocks: int | None,
) -> list[dict]:
    if not include_multimodal:
        content_list = [
            item for item in content_list if isinstance(item, dict) and item.get("type") == "text"
        ]
    if max_blocks is not None:
        content_list = content_list[:max_blocks]
    return content_list


def build_rag(working_dir: str, *, include_multimodal: bool) -> RAGAnything:
    api_key = os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST")
    if not api_key:
        raise RuntimeError("请先在 .env 中配置 LLM_BINDING_API_KEY。")

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
        working_dir=working_dir,
        enable_image_processing=include_multimodal,
        enable_table_processing=include_multimodal,
        enable_equation_processing=include_multimodal,
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

    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs=lightrag_kwargs,
    )


async def run_demo(args: argparse.Namespace) -> None:
    content_list_path = Path(args.content_list) if args.content_list else find_content_list()
    content_list = load_content_list(content_list_path)
    content_list = prepare_content_list(
        content_list,
        include_multimodal=args.full,
        max_blocks=args.max_blocks,
    )
    questions = args.question or DEFAULT_QUESTIONS

    print(f"使用解析结果: {content_list_path}", flush=True)
    print(f"索引内容块数量: {len(content_list)}", flush=True)
    print(f"工作目录: {args.working_dir}", flush=True)
    print(
        f"模式: {'完整多模态' if args.full else '快速文本测试'}",
        flush=True,
    )

    rag = build_rag(args.working_dir, include_multimodal=args.full)

    print("\n正在插入解析内容到 RAG 索引；第一次会调用 embedding/LLM，可能需要几分钟...", flush=True)
    await rag.insert_content_list(
        content_list=content_list,
        file_path=args.file_path,
        doc_id=args.doc_id,
        display_stats=True,
    )

    print("\n开始提问:", flush=True)
    for index, question in enumerate(questions, start=1):
        print(f"\n[{index}] Q: {question}", flush=True)
        answer = await rag.aquery(question, mode=args.mode)
        print(f"A: {answer}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test QA over a parsed PDF.")
    parser.add_argument(
        "--content-list",
        help="已解析出的 *_content_list.json 路径；不填则自动从 output/raganything_parser_test 查找",
    )
    parser.add_argument(
        "--working-dir",
        default="./rag_storage_pdf_demo_fast",
        help="RAG 索引存储目录",
    )
    parser.add_argument(
        "--file-path",
        default="《城市生活垃圾处理和给水与污水处理工程项目建设用地指标》.pdf",
        help="引用来源文件名",
    )
    parser.add_argument(
        "--doc-id",
        default="city-waste-water-land-index-demo-fast",
        help="固定文档 ID；重复运行同一 demo 时保持不变",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "local", "global", "naive"],
        help="检索模式",
    )
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        help="自定义测试问题；可重复传多个 -q",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=60,
        help="最多索引多少个内容块；测试时建议 40-80，完整测试可传 0",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="启用完整多模态处理；默认只用文本块做快速问答测试",
    )
    args = parser.parse_args()
    if args.max_blocks == 0:
        args.max_blocks = None
    return args


if __name__ == "__main__":
    asyncio.run(run_demo(parse_args()))
