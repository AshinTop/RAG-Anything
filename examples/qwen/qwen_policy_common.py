#!/usr/bin/env python
"""Shared Qwen + Chinese policy helpers for RAG-Anything examples."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv

RAGANYTHING_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = RAGANYTHING_ROOT.parent

if str(RAGANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(RAGANYTHING_ROOT))

load_dotenv(dotenv_path=RAGANYTHING_ROOT / ".env", override=False)

from lightrag.llm.openai import openai_complete_if_cache
from lightrag.prompt import PROMPTS as LIGHTRAG_PROMPTS
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig, set_prompt_language


DEFAULT_WORKING_DIR = str(RAGANYTHING_ROOT / "rag_storage" / "qwen_policy_text")
DEFAULT_OUTPUT_DIR = str(RAGANYTHING_ROOT / "output" / "qwen_policy_parser")

_GPU_UNAVAILABLE_REPORTED = False

LOCAL_STORAGE_BACKENDS = {
    "kv_storage": "JsonKVStorage",
    "vector_storage": "NanoVectorDBStorage",
    "graph_storage": "NetworkXStorage",
    "doc_status_storage": "JsonDocStatusStorage",
}

POSTGRES_STORAGE_BACKENDS = {
    "kv_storage": "PGKVStorage",
    "vector_storage": "PGVectorStorage",
    # PGGraphStorage depends on Apache AGE. NetworkX keeps graph import usable on
    # plain PostgreSQL installations while KV/vector/doc status live in Postgres.
    "graph_storage": "NetworkXStorage",
    "doc_status_storage": "PGDocStatusStorage",
}

POSTGRES_AGE_STORAGE_BACKENDS = {
    "kv_storage": "PGKVStorage",
    "vector_storage": "PGVectorStorage",
    "graph_storage": "PGGraphStorage",
    "doc_status_storage": "PGDocStatusStorage",
}

POLICY_ENTITY_TYPES = [
    "政策文件",
    "发布机构",
    "行政区域",
    "适用对象",
    "工程项目",
    "建设指标",
    "用地指标",
    "条款",
    "数值标准",
    "时间",
    "程序要求",
    "概念",
    "其他",
]

QWEN_POLICY_GUARD = (
    "你正在处理中文政策、标准、规范类文档。请使用简体中文完成任务，"
    "保持条款名、指标名、机构名和数值单位的原文表达。"
    "不要输出思考过程、<think> 标签、解释性前言或与任务格式无关的内容。"
)

ANSWER_SYSTEM_PROMPT = (
    "/no_think\n"
    "你是中文政策文件问答助手。必须只依据本轮检索上下文中的政策原文回答；"
    "知识图谱实体和关系只能作为定位线索，不能作为最终答案依据，不能覆盖政策原文。"
    "不得使用模型自身知识补充、推断、替换或改写政策结论。"
    "先识别用户问题中的核心对象、章节名、工程类型、指标名或条款主题，然后只使用"
    "与该核心对象精确匹配或直接相邻的原文内容作答。"
    "如果检索上下文中存在与问题核心对象同名的章节标题、条款或表格，必须优先使用"
    "这些内容，不要使用其他相似主题、其他文件或外部标准。"
    "如果原文 chunk 中出现明确条款、表格或指标，必须逐条展开相关内容，"
    "不要只给一句概括。涉及适用范围、指标、数值、单位、发布机构、文件名称时，"
    "必须保留原文中的工程名称、条件、数值和单位。"
    "当问题询问“如何规定”“建设用地如何规定”“指标如何规定”时，答案至少应覆盖"
    "原文中可检索到的分类/分级、用地组成、面积或容量控制、比例要求、上下限说明、"
    "计算或适用条件；如果某项在原文中没有出现，可省略，不得编造。"
    "禁止引用检索上下文中没有逐字出现的文件名、标准号、法规名称、机构名称或数值，"
    "例如不得凭常识补充其他国家标准、地方标准、规划标准或主管部门建议。"
    "禁止输出占位符或模板内容，例如“第X条”“页码 X”“XX平方米”“具体数值需另查”。"
    "禁止把相似但不同的主题混为一谈；必须优先使用与用户问题核心对象、章节、条款"
    "或表格最匹配的检索内容。"
    "不得判断“项目不适用”“不在适用范围内”或给出否定结论，除非检索上下文"
    "明确写出了该否定结论。"
    "如果政策原文和知识图谱信息冲突，以政策原文为准；如果原文没有足够依据，"
    "请回答“资料中未检索到相关依据”，不要编造。"
    "回答必须使用以下结构，不要增加其他标题：\n"
    "结论：\n"
    "用 2-5 句话直接回答问题，覆盖检索原文中与问题相关的主要条款和表格结论；"
    "如果资料不足，明确说明未检索到相关依据。\n\n"
    "依据：\n"
    "按条目说明依据来自哪些原文条款、表格或指标。每条依据应包含条款号或表格名，"
    "并保留原文中的关键规定、规模分类、面积指标、比例、上下限和单位。"
    "如果同一问题涉及多个连续条款，应尽量列全，不要遗漏关键条款。\n\n"
    "补充说明：\n"
    "只说明检索原文中明确存在的适用限制、计算条件或口径。"
    "如果原文已经足以回答且没有额外限制，写“无”。"
    "只有在确实未检索到依据时，才给出获取资料的建议。\n\n"
    "参考来源：\n"
    "列出使用到的来源。如果检索内容明确包含页码，格式为“1. 页码 N - 《文件名》”；"
    "如果没有明确页码，格式为“1. 《文件名》”，不要写“页码 X”或“页码未标明”。"
    "同一文件重复引用时可以合并为一条。"
)

STRICT_ANSWER_SYSTEM_PROMPT = (
    "/no_think\n"
    "你是中文政策原文抽取式问答助手。用户会提供带 metadata 的问题相关原文片段。"
    "你的任务不是自由问答，而是从这些片段中抽取原文依据并组织答案。\n"
    "必须遵守：\n"
    "1. 只使用检索提示中逐字出现的信息；禁止使用任何外部知识、常识、标准号、文件名、"
    "章节号、页码、数值或建议。\n"
    "2. 先在原文片段中定位与用户问题核心对象最匹配的标题、条款或表格；如果存在，"
    "只围绕这些原文回答。\n"
    "3. 不得输出检索提示中没有逐字出现的文件名、标准名称、条款号、页码、面积、距离、"
    "比例或下限。禁止输出“第X条”“页码 X”“XX平方米”等占位符。\n"
    "4. 如果问题问“如何规定”，应尽量覆盖原文中的分类/分级、用地组成、容量或面积控制、"
    "比例要求、上下限说明、适用条件。只要原文出现了这些内容，就不要省略。\n"
    "对于连续条款，不要只列前两条；必须继续检查后续条款，直到该小节结束或进入下一节。\n"
    "如果片段含相同“章节路径”的多条内容或表格，必须综合这些同章节内容回答，不要只使用第一条。\n"
    "5. 答案必须使用以下结构：\n"
    "结论：\n"
    "依据：\n"
    "补充说明：\n"
    "参考来源：\n"
    "6. 没有明确页码时，参考来源只写“《文件名》”；不要写页码 X 或页码未标明。"
)


def build_strict_answer_prompt(raw_prompt: str, question: str, focused_context: str | None = None) -> str:
    focused_context = focused_context or extract_focused_context(raw_prompt, question)
    return f"""/no_think
请根据下面的“问题相关原文片段”回答用户问题。

用户问题：
{question}

问题相关原文片段：
```
{focused_context}
```

只能使用上方原文片段。不要使用外部知识，不要引用上方片段没有出现的文件名、标准号、条款号、页码、数值或单位。
"""


async def load_index_chunks(working_dir: str, storage: str) -> list[dict[str, Any]]:
    if storage in {"postgres", "postgres-age"}:
        return await load_postgres_chunks()
    return load_local_chunks(Path(working_dir))


async def load_postgres_chunks() -> list[dict[str, Any]]:
    try:
        import asyncpg
    except ImportError:
        return []

    workspace = (
        os.getenv("PG_WORKSPACE")
        or os.getenv("POSTGRES_WORKSPACE")
        or os.getenv("QWEN_POSTGRES_WORKSPACE")
        or "default"
    )
    connection = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST") or os.getenv("PGHOST") or "localhost",
        port=int(os.getenv("POSTGRES_PORT") or os.getenv("PGPORT") or "5432"),
        user=os.getenv("POSTGRES_USER") or os.getenv("PGUSER") or "postgres",
        password=os.getenv("POSTGRES_PASSWORD") or os.getenv("PGPASSWORD"),
        database=os.getenv("POSTGRES_DATABASE") or os.getenv("PGDATABASE") or "postgres",
    )
    try:
        rows = await connection.fetch(
            """
            SELECT id, workspace, full_doc_id, chunk_order_index, content, file_path
            FROM LIGHTRAG_VDB_CHUNKS
            WHERE workspace = $1
            ORDER BY full_doc_id, chunk_order_index
            """,
            workspace,
        )
        return [dict(row) for row in rows]
    finally:
        await connection.close()


def load_local_chunks(working_dir: Path) -> list[dict[str, Any]]:
    chunks_path = working_dir / "kv_store_text_chunks.json"
    if not chunks_path.exists():
        return []
    with chunks_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return []

    chunks: list[dict[str, Any]] = []
    for index, (chunk_id, value) in enumerate(data.items()):
        if not isinstance(value, dict):
            continue
        chunks.append(
            {
                "id": chunk_id,
                "workspace": "local",
                "full_doc_id": value.get("full_doc_id") or value.get("doc_id") or "",
                "chunk_order_index": value.get("chunk_order_index", index),
                "content": value.get("content") or value.get("text") or "",
                "file_path": value.get("file_path") or value.get("source") or "",
            }
        )
    return sorted(chunks, key=lambda item: (item.get("full_doc_id") or "", int(item.get("chunk_order_index") or 0)))


async def insert_retrieval_only_chunks(
    rag: RAGAnything,
    *,
    doc_id: str,
    source_ref: str,
    chunk_texts: list[str],
    chunk_ids: list[str],
    full_text: str | None = None,
) -> None:
    """Insert chunks for retrieval QA without running KG extraction.

    This writes full docs, text chunks, vectors, and doc_status so the policy QA
    path can rely on structured retrieval and citations even when small local
    models are not stable enough for entity/relation extraction.
    """
    if rag.lightrag is None:
        raise RuntimeError("LightRAG is not initialized")
    if len(chunk_texts) != len(chunk_ids):
        raise ValueError("chunk_texts and chunk_ids must have the same length")
    if not chunk_texts:
        raise ValueError("chunk_texts must not be empty")

    file_ref = rag._get_file_reference(source_ref)
    timestamp = datetime.now(timezone.utc)
    tokenizer = rag.lightrag.tokenizer

    prepared_chunks: dict[str, dict[str, Any]] = {}
    normalized_texts: list[str] = []
    for index, (chunk_id, chunk_text) in enumerate(zip(chunk_ids, chunk_texts)):
        normalized_text = str(chunk_text or "").strip()
        if not normalized_text:
            continue
        normalized_texts.append(normalized_text)
        prepared_chunks[chunk_id] = {
            "content": normalized_text,
            "tokens": len(tokenizer.encode(normalized_text)),
            "full_doc_id": doc_id,
            "chunk_order_index": index,
            "file_path": file_ref,
            "llm_cache_list": [],
        }

    if not prepared_chunks:
        raise ValueError("No non-empty chunk text available for insertion")

    full_doc_text = full_text or "\n\n".join(normalized_texts)
    full_doc_record = {
        doc_id: {
            "content": full_doc_text,
            "file_path": file_ref,
        }
    }
    doc_status_record = {
        doc_id: {
            "status": "processed",
            "chunks_count": len(prepared_chunks),
            "chunks_list": list(prepared_chunks.keys()),
            "content_summary": full_doc_text[:240],
            "content_length": len(full_doc_text),
            "created_at": timestamp,
            "updated_at": timestamp,
            "file_path": file_ref,
            "metadata": {
                "ingest_strategy": "retrieval_only",
                "qa_strategy": "structured_retrieval_first",
                "kg_extraction": "skipped",
            },
        }
    }

    await rag.lightrag.full_docs.upsert(full_doc_record)
    await rag.lightrag.full_docs.index_done_callback()

    await rag.lightrag.text_chunks.upsert(prepared_chunks)
    await rag.lightrag.text_chunks.index_done_callback()

    await rag.lightrag.chunks_vdb.upsert(prepared_chunks)
    await rag.lightrag.chunks_vdb.index_done_callback()

    await rag.lightrag.doc_status.upsert(doc_status_record)
    await rag.lightrag.doc_status.index_done_callback()

    stored_full_doc = await rag.lightrag.full_docs.get_by_id(doc_id)
    if not stored_full_doc:
        raise RuntimeError(f"retrieval-only full_docs 写入失败: {doc_id}")

    stored_doc_status = await rag.lightrag.doc_status.get_by_id(doc_id)
    if not stored_doc_status:
        raise RuntimeError(f"retrieval-only doc_status 写入失败: {doc_id}")

    stored_chunk_vectors = await rag.lightrag.chunks_vdb.get_by_ids(
        list(prepared_chunks.keys())
    )
    if not stored_chunk_vectors:
        raise RuntimeError(
            f"retrieval-only vdb_chunks 写入失败: doc_id={doc_id}, chunks={len(prepared_chunks)}"
        )


def build_chunk_focused_context(
    chunks: list[dict[str, Any]], question: str, *, max_chunks: int = 8
) -> str:
    if not chunks:
        return ""

    terms = build_focus_terms(question)
    high_priority_terms = terms

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        content = str(chunk.get("content") or "")
        score = score_focus_snippet(content, terms, high_priority_terms)
        scored.append((score, index, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        selected_indexes = [index for _, index, _ in scored[:max_chunks]]
    else:
        _, top_index, top_chunk = scored[0]
        selected_indexes = [top_index]
        top_doc_id = top_chunk.get("full_doc_id")
        top_meta = parse_structured_chunk_metadata(str(top_chunk.get("content") or ""))
        top_section_path = top_meta.get("章节路径")

        # Include only close adjacent chunks that still look related. This keeps
        # cross-chunk clauses/tables while avoiding drift into the next section.
        for offset in [1, -1]:
            neighbor_index = top_index + offset
            if 0 <= neighbor_index < len(chunks):
                neighbor = chunks[neighbor_index]
                neighbor_content = str(neighbor.get("content") or "")
                if (
                    neighbor.get("full_doc_id") == top_doc_id
                    and is_related_neighbor(neighbor_content, high_priority_terms)
                ):
                    selected_indexes.append(neighbor_index)
            if len(selected_indexes) >= max_chunks:
                break

        if top_section_path:
            for neighbor_index, neighbor in enumerate(chunks):
                if len(selected_indexes) >= max_chunks:
                    break
                if neighbor_index in selected_indexes:
                    continue
                if neighbor.get("full_doc_id") != top_doc_id:
                    continue
                neighbor_meta = parse_structured_chunk_metadata(
                    str(neighbor.get("content") or "")
                )
                if neighbor_meta.get("章节路径") != top_section_path:
                    continue
                selected_indexes.append(neighbor_index)

    selected_indexes = sorted(dict.fromkeys(selected_indexes))
    parts: list[str] = []
    for ref_index, chunk_index in enumerate(selected_indexes, start=1):
        chunk = chunks[chunk_index]
        file_path = Path(str(chunk.get("file_path") or "")).name or "来源未标明"
        order = chunk.get("chunk_order_index")
        parts.append(
            f"[参考内容{ref_index}]\n"
            f"chunk_id: {chunk.get('id')}\n"
            f"chunk_order_index: {order}\n"
            f"文件: 《{file_path}》\n"
            f"原文:\n{str(chunk.get('content') or '').strip()}"
        )
    return "\n\n---\n\n".join(parts)


def is_related_neighbor(content: str, high_priority_terms: list[str]) -> bool:
    return any(term and term in content for term in high_priority_terms)


def parse_structured_chunk_metadata(content: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in str(content or "").splitlines():
        stripped = line.strip()
        if stripped == "正文：":
            break
        if "：" not in stripped:
            continue
        key, value = stripped.split("：", 1)
        if key in {
            "文件",
            "chunk_id",
            "chunk_order_index",
            "内容类型",
            "页码",
            "章节路径",
            "条款",
        }:
            metadata[key] = value.strip()
    return metadata


def extract_focused_context(raw_prompt: str, question: str, window: int = 1800) -> str:
    """Extract query-focused snippets so small local LLMs do not drift to nearby topics."""
    if not raw_prompt:
        return ""

    terms = build_focus_terms(question)
    generic_terms = {"建设用地", "用地面积", "用地指标"}
    high_priority_terms = [term for term in terms if term not in generic_terms]
    candidates: list[tuple[int, int, int, str]] = []

    for term in terms:
        if not term:
            continue
        for match in re.finditer(re.escape(term), raw_prompt):
            start = max(0, match.start() - window)
            end = min(len(raw_prompt), match.end() + window)
            snippet = raw_prompt[start:end].strip()
            score = score_focus_snippet(snippet, terms, high_priority_terms)
            candidates.append((score, start, end, snippet))

    if not candidates:
        return raw_prompt

    candidates.sort(key=lambda item: item[0], reverse=True)
    snippets: list[str] = []
    used_ranges: list[tuple[int, int]] = []

    for score, start, end, snippet in candidates:
        if score <= 0:
            continue
        if any(not (end < old_start or start > old_end) for old_start, old_end in used_ranges):
            continue
        used_ranges.append((start, end))
        snippets.append(snippet)
        if len(snippets) >= 3:
            break

    return "\n\n---\n\n".join(snippets) if snippets else raw_prompt


def score_focus_snippet(
    snippet: str, terms: list[str], high_priority_terms: list[str]
) -> int:
    score = 0
    for term in high_priority_terms:
        if term in snippet:
            score += 100 + len(term) * 5
    for term in terms:
        if term in snippet:
            score += snippet.count(term) * max(1, len(term))

    if re.search(r"第[一二三四五六七八九十百零〇\d]+条", snippet):
        score += 40
    if "表格" in snippet or "<table" in snippet:
        score += 30
    if "工程项目" in snippet:
        score += 20

    # Avoid snippets that only match broad words such as "建设用地".
    if not any(term in snippet for term in high_priority_terms):
        score -= 200
    return score


def build_focus_terms(question: str) -> list[str]:
    text = re.sub(r"[？?，,。；;：:\s]+", "", question or "")

    terms: list[str] = []
    if len(text) >= 4:
        terms.append(text)

    # Generate long n-grams directly from the question instead of maintaining
    # domain-specific stop-word lists.
    for size in range(min(12, len(text)), 3, -1):
        for index in range(0, max(0, len(text) - size + 1)):
            gram = text[index : index + size]
            if gram not in terms:
                terms.append(gram)
        if len(terms) >= 10:
            break

    deduped: list[str] = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {sec:.1f}s"
    if minutes:
        return f"{minutes}m {sec:.1f}s"
    return f"{sec:.1f}s"


def log_runtime(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[TIME {timestamp}] {message}", flush=True)


def print_gpu_snapshot(label: str) -> None:
    global _GPU_UNAVAILABLE_REPORTED

    if not env_bool("QWEN_SHOW_GPU_STATS", True):
        return

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        if not _GPU_UNAVAILABLE_REPORTED:
            print("[GPU] nvidia-smi 不可用，跳过 GPU 使用情况输出。", flush=True)
            _GPU_UNAVAILABLE_REPORTED = True
        return

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=5,
            check=False,
        )
    except Exception as exc:
        if not _GPU_UNAVAILABLE_REPORTED:
            print(f"[GPU] 读取 GPU 状态失败: {exc}", flush=True)
            _GPU_UNAVAILABLE_REPORTED = True
        return

    if result.returncode != 0 or not result.stdout.strip():
        if not _GPU_UNAVAILABLE_REPORTED:
            detail = (result.stderr or result.stdout or "").strip()
            print(f"[GPU] nvidia-smi 无可用输出: {detail}", flush=True)
            _GPU_UNAVAILABLE_REPORTED = True
        return

    for raw_line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 7:
            print(f"[GPU {label}] {raw_line}", flush=True)
            continue
        index, name, util, mem_used, mem_total, temp, power = parts[:7]
        print(
            f"[GPU {label}] GPU{index} {name}: util={util}% "
            f"mem={mem_used}/{mem_total} MiB temp={temp}C power={power}W",
            flush=True,
        )


def start_stage(label: str, *, gpu: bool = False) -> float:
    log_runtime(f"{label} 开始")
    if gpu:
        print_gpu_snapshot(f"{label} 开始")
    return time.perf_counter()


def end_stage(label: str, start_time: float, *, gpu: bool = False) -> None:
    log_runtime(f"{label} 完成，用时 {format_duration(time.perf_counter() - start_time)}")
    if gpu:
        print_gpu_snapshot(f"{label} 完成")


def strip_qwen_thinking(text: str) -> str:
    """Remove thinking tags returned by local reasoning-style Qwen servers."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    # Some Qwen3-compatible servers omit the opening tag but still return
    # reasoning followed by a closing tag.
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def repair_lightrag_extraction_output(text: str) -> str:
    """Repair narrow LightRAG extraction format slips from local small models."""
    if not isinstance(text, str) or "relation" not in text.lower():
        return text

    tuple_delimiters = ["<|#|>"]
    repaired_lines: list[str] = []
    changed = False

    for line in text.splitlines():
        stripped = line.strip()
        repaired = line
        for delimiter in tuple_delimiters:
            if delimiter not in stripped:
                continue
            parts = [part.strip() for part in stripped.split(delimiter)]
            if not parts or parts[0].lower() not in {"relation", "relationship"}:
                continue

            parts[0] = "relation"
            if len(parts) == 4:
                source, target, value = parts[1], parts[2], parts[3]
                if len(value) <= 12 and not re.search(r"[，。；,.、]", value):
                    keyword = value or "相关"
                    description = f"{source}与{target}存在{keyword}关系。"
                else:
                    keyword = "相关"
                    description = value or f"{source}与{target}存在政策相关关系。"
                repaired = delimiter.join(["relation", source, target, keyword, description])
                changed = True
                break

            if len(parts) > 5:
                repaired = delimiter.join(parts[:4] + ["；".join(parts[4:])])
                changed = True
                break

        repaired_lines.append(repaired)

    return "\n".join(repaired_lines) if changed else text


def sanitize_embedding_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or "空文本"


def fallback_embedding_vector(text: str, dim: int) -> list[float]:
    """Return a deterministic non-zero vector when local embedding returns NaN."""
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    repeats = (dim + len(digest) - 1) // len(digest)
    raw = (digest * repeats)[:dim]
    vector = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    vector = (vector - 127.5) / 127.5
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        vector = np.ones(dim, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32).tolist()


def apply_chinese_policy_prompts() -> None:
    """Switch RAG-Anything and LightRAG indexing prompts to Chinese policy prompts."""
    set_prompt_language("zh")

    LIGHTRAG_PROMPTS["entity_extraction_system_prompt"] = """---角色---
你是一名中文政策知识图谱专家，负责从输入文本中抽取适合检索问答的实体和关系。

---抽取要求---
1. 实体抽取：
   - 只抽取文本中明确出现、对政策理解有价值的实体。
   - 实体名称应使用原文中文名称，机构、文件名、条款名、指标名、工程名称和数值单位不要随意改写。
   - 实体类型必须从以下类型中选择：{entity_types}。如果都不适用，使用“其他”。
   - 实体描述必须只依据输入文本，简洁说明该实体在政策中的身份、属性、职责、适用条件或指标含义。
   - 如果输入文本含有“文件、章节路径、条款、页码、正文”等元数据，实体描述必须保留相关条款号、章节路径或页码。

2. 关系抽取：
   - 抽取实体之间明确存在的政策关系，例如“发布/批准/适用于/规定/包含/对应/要求/限定/计算依据/取值为”。
   - 关系描述必须说明关系依据，尽量保留原文中的条件、范围、数值和单位。
   - 关系只能来自当前输入文本中的明确表述，不允许跨条款、跨章节或凭常识推断。
   - 关系描述必须包含能回溯的 evidence；如果输入文本含条款号或页码，应写入关系描述。

3. 输出格式：
   - 每个实体一行，字段用 `{tuple_delimiter}` 分隔，格式必须为：
     entity{tuple_delimiter}<实体名称>{tuple_delimiter}<实体类型>{tuple_delimiter}<实体描述>
   - 每个关系一行，字段用 `{tuple_delimiter}` 分隔，格式必须为：
     relation{tuple_delimiter}<源实体>{tuple_delimiter}<目标实体>{tuple_delimiter}<关系关键词>{tuple_delimiter}<关系描述>
   - 关系行总共只能有 5 个字段，第一字段必须是 `relation`，不要输出关系强度、评分或额外字段。
   - 关系行不能省略“关系关键词”。如果无法确定关键词，请用“相关”作为第 4 个字段，并把完整依据写在第 5 个字段。
   - 不要输出解释、Markdown、编号或代码块。
   - 所有实体、关键词和描述均使用{language}，但原文中的标准编号、英文缩写、模型名、API 名可保留原样。
"""

    LIGHTRAG_PROMPTS["entity_extraction_user_prompt"] = """---任务---
请从下面“待处理文本”中抽取中文政策知识图谱的实体和关系。待处理文本可能包含结构化 metadata，例如文件、章节路径、条款、页码和正文。

---要求---
1. 严格遵守系统提示中的实体、关系字段顺序和分隔符。
2. 只输出实体和关系列表，不要添加开场白、总结或解释。
3. 抽取完成后，最后一行必须输出 `{completion_delimiter}`。
4. 输出语言必须为{language}。
5. 实体或关系描述必须绑定当前文本中的 evidence；优先保留“条款：...”“章节路径：...”“页码：...”。
6. 不要抽取当前文本没有明确说明的适用范围、指标、数值或关系。

---待处理文本---
<实体类型>
[{entity_types}]

<输入文本>
```
{input_text}
```

<输出>
"""

    LIGHTRAG_PROMPTS["entity_continue_extraction_user_prompt"] = """---任务---
请基于上一轮抽取结果，继续检查下面输入文本中是否还有遗漏或格式错误的中文政策实体和关系。

---要求---
1. 不要重复输出上一轮已经正确抽取的实体或关系。
2. 如发现遗漏、截断、字段缺失或格式错误，请按系统提示的格式补充或修正。
3. 每个实体一行，格式为：
   entity{tuple_delimiter}<实体名称>{tuple_delimiter}<实体类型>{tuple_delimiter}<实体描述>
4. 每个关系一行，格式为：
   relation{tuple_delimiter}<源实体>{tuple_delimiter}<目标实体>{tuple_delimiter}<关系关键词>{tuple_delimiter}<关系描述>
5. 最后一行必须输出 `{completion_delimiter}`。
6. 关系行总共只能有 5 个字段，第一字段必须是 `relation`，不要输出关系强度、评分或额外字段。
7. 不允许输出 4 字段关系行；缺少关键词时使用“相关”作为第 4 个字段。
8. 描述必须包含当前输入中的 evidence；优先保留“条款：...”“章节路径：...”“页码：...”。
9. 不要跨条款或跨章节推断关系。
10. 输出语言必须为{language}，不要添加解释、Markdown 或代码块。

---输入文本---
```
{input_text}
```

<输出>
"""

    LIGHTRAG_PROMPTS["entity_extraction_examples"] = [
        """示例文本：
《城市生活垃圾处理和给水与污水处理工程项目建设用地指标》由建设部、国土资源部批准发布，适用于城市生活垃圾处理工程、给水工程和污水处理工程项目。

示例输出：
entity{tuple_delimiter}《城市生活垃圾处理和给水与污水处理工程项目建设用地指标》{tuple_delimiter}政策文件{tuple_delimiter}该文件是由建设部、国土资源部批准发布的工程项目建设用地指标文件。
entity{tuple_delimiter}建设部{tuple_delimiter}发布机构{tuple_delimiter}建设部是批准发布该建设用地指标的机构之一。
entity{tuple_delimiter}国土资源部{tuple_delimiter}发布机构{tuple_delimiter}国土资源部是批准发布该建设用地指标的机构之一。
entity{tuple_delimiter}城市生活垃圾处理工程{tuple_delimiter}工程项目{tuple_delimiter}城市生活垃圾处理工程是该建设用地指标的适用对象之一。
relation{tuple_delimiter}建设部{tuple_delimiter}《城市生活垃圾处理和给水与污水处理工程项目建设用地指标》{tuple_delimiter}批准发布{tuple_delimiter}建设部批准发布该建设用地指标文件。
relation{tuple_delimiter}《城市生活垃圾处理和给水与污水处理工程项目建设用地指标》{tuple_delimiter}城市生活垃圾处理工程{tuple_delimiter}适用于{tuple_delimiter}该文件适用于城市生活垃圾处理工程项目。
{completion_delimiter}"""
    ]

    LIGHTRAG_PROMPTS["summarize_entity_descriptions"] = """---角色---
你是一名中文政策知识整理专家。

---任务---
请把给定实体或关系的多条描述合并成一段准确、连贯的中文摘要。

---要求---
1. 只依据“描述列表”中的信息，不补充外部知识。
2. 摘要应客观、第三人称表达，并在开头明确提及 `{description_name}`。
3. 保留政策文件名、条款名、指标名、数值、单位和机构名称。
4. 如果多条描述存在差异，请合并可兼容信息；无法兼容时说明存在不同表述。
5. 输出为纯文本，不要添加标题、Markdown 或解释。
6. 输出语言为{language}，建议长度约 {summary_length} 字。

---描述类型---
{description_type}

---实体或关系名称---
{description_name}

---描述列表---
{description_list}

<输出>
"""

    LIGHTRAG_PROMPTS["keywords_extraction"] = """---角色---
你是一名中文政策问答检索关键词抽取专家。

---目标---
从用户问题中抽取用于 RAG 检索的高层关键词和低层关键词。

---要求---
1. 只输出合法 JSON 对象，不要输出 Markdown、代码块或解释。
2. high_level_keywords 表示政策主题、业务领域、问题意图或指标类别。
3. low_level_keywords 表示具体文件名、机构、条款、工程类型、指标名、数值、区域或对象。
4. 所有关键词必须来自用户问题或其直接同义表达，输出语言为{language}。

---用户问题---
{query}

---输出格式---
{{"high_level_keywords": ["关键词1", "关键词2"], "low_level_keywords": ["关键词1", "关键词2"]}}
"""


def get_qwen_settings() -> dict[str, Any]:
    base_url = os.getenv("QWEN_BASE_URL") or os.getenv(
        "LLM_BINDING_HOST", "http://localhost:8000/v1"
    )
    api_key = os.getenv("QWEN_API_KEY") or os.getenv(
        "LLM_BINDING_API_KEY", "local-qwen"
    )
    embedding_base_url = (
        os.getenv("QWEN_EMBEDDING_BASE_URL")
        or os.getenv("EMBEDDING_BINDING_HOST")
        or base_url
    )
    embedding_api_key = (
        os.getenv("QWEN_EMBEDDING_API_KEY")
        or os.getenv("EMBEDDING_BINDING_API_KEY")
        or api_key
    )
    return {
        "base_url": base_url,
        "api_key": api_key,
        "llm_model": os.getenv("QWEN_LLM_MODEL")
        or os.getenv("LLM_MODEL", "Qwen3-8B"),
        "embedding_model": os.getenv("QWEN_EMBEDDING_MODEL")
        or os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"),
        "embedding_base_url": embedding_base_url,
        "embedding_api_key": embedding_api_key,
        "embedding_dim": int(os.getenv("QWEN_EMBEDDING_DIM") or os.getenv("EMBEDDING_DIM", "1024")),
        "embedding_max_tokens": int(
            os.getenv("QWEN_EMBEDDING_MAX_TOKENS")
            or os.getenv("EMBEDDING_MAX_TOKENS", "8192")
        ),
        "embedding_timeout": int(
            os.getenv("QWEN_EMBEDDING_TIMEOUT")
            or os.getenv("EMBEDDING_TIMEOUT", "180")
        ),
        "timeout": int(os.getenv("QWEN_TIMEOUT", "180")),
        "temperature": float(os.getenv("QWEN_TEMPERATURE", "0")),
    }


def normalize_storage(storage: str | None) -> str:
    value = (storage or os.getenv("QWEN_STORAGE") or "local").strip().lower()
    aliases = {
        "file": "local",
        "files": "local",
        "json": "local",
        "pg": "postgres",
        "postgresql": "postgres",
        "pg-age": "postgres-age",
        "postgresql-age": "postgres-age",
    }
    value = aliases.get(value, value)
    if value not in {"local", "postgres", "postgres-age"}:
        raise ValueError(
            f"Unsupported storage: {storage}. Use local, postgres, or postgres-age."
        )
    return value


def storage_backend_names(storage: str | None = None) -> dict[str, str]:
    storage = normalize_storage(storage)
    if storage == "postgres-age":
        backends = POSTGRES_AGE_STORAGE_BACKENDS.copy()
    elif storage == "postgres":
        backends = POSTGRES_STORAGE_BACKENDS.copy()
    else:
        backends = LOCAL_STORAGE_BACKENDS.copy()

    env_overrides = {
        "kv_storage": os.getenv("QWEN_KV_STORAGE") or os.getenv("LIGHTRAG_KV_STORAGE"),
        "vector_storage": os.getenv("QWEN_VECTOR_STORAGE")
        or os.getenv("LIGHTRAG_VECTOR_STORAGE"),
        "graph_storage": os.getenv("QWEN_GRAPH_STORAGE")
        or os.getenv("LIGHTRAG_GRAPH_STORAGE"),
        "doc_status_storage": os.getenv("QWEN_DOC_STATUS_STORAGE")
        or os.getenv("LIGHTRAG_DOC_STATUS_STORAGE"),
    }
    for key, value in env_overrides.items():
        if value:
            backends[key] = value
    return backends


def describe_storage_backends(storage: str | None = None) -> str:
    backends = storage_backend_names(storage)
    return (
        f"{backends['kv_storage']} + {backends['vector_storage']} + "
        f"{backends['graph_storage']} + {backends['doc_status_storage']}"
    )


def make_lightrag_kwargs(
    settings: dict[str, Any], storage: str | None = None
) -> dict[str, Any]:
    backends = storage_backend_names(storage)
    llm_timeout = int(
        os.getenv("QWEN_DEFAULT_LLM_TIMEOUT", str(settings["timeout"]))
    )
    return {
        **backends,
        "llm_model_name": settings["llm_model"],
        "tiktoken_model_name": os.getenv("QWEN_TIKTOKEN_MODEL", "gpt-4o-mini"),
        "chunk_token_size": int(os.getenv("QWEN_CHUNK_TOKEN_SIZE", "900")),
        "chunk_overlap_token_size": int(os.getenv("QWEN_CHUNK_OVERLAP_TOKEN_SIZE", "120")),
        "max_extract_input_tokens": int(os.getenv("QWEN_MAX_EXTRACT_INPUT_TOKENS", "12000")),
        "summary_max_tokens": int(os.getenv("QWEN_SUMMARY_MAX_TOKENS", "900")),
        "summary_context_size": int(os.getenv("QWEN_SUMMARY_CONTEXT_SIZE", "8000")),
        "summary_length_recommended": int(os.getenv("QWEN_SUMMARY_LENGTH", "450")),
        "llm_model_max_async": int(os.getenv("QWEN_LLM_MAX_ASYNC", "1")),
        "default_llm_timeout": llm_timeout,
        "llm_model_kwargs": {
            "timeout": llm_timeout,
        },
        "embedding_batch_num": int(os.getenv("QWEN_EMBEDDING_BATCH_NUM", "4")),
        "embedding_func_max_async": int(os.getenv("QWEN_EMBEDDING_MAX_ASYNC", "1")),
        "default_embedding_timeout": int(settings["embedding_timeout"]),
        "max_parallel_insert": int(os.getenv("QWEN_MAX_PARALLEL_INSERT", "1")),
        "entity_extract_max_gleaning": int(os.getenv("QWEN_ENTITY_MAX_GLEANING", "1")),
        "enable_llm_cache": env_bool("QWEN_ENABLE_LLM_CACHE", True),
        "enable_llm_cache_for_entity_extract": env_bool(
            "QWEN_ENABLE_ENTITY_LLM_CACHE", True
        ),
        "addon_params": {
            "language": "Simplified Chinese",
            "entity_types": POLICY_ENTITY_TYPES,
        },
    }


def make_llm_func(settings: dict[str, Any]):
    async def qwen_llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        guarded_system_prompt = QWEN_POLICY_GUARD
        if system_prompt:
            guarded_system_prompt = f"{QWEN_POLICY_GUARD}\n\n{system_prompt}"

        # LightRAG passes keyword_extraction=True for query keyword extraction.
        # The OpenAI helper turns that into beta.chat.completions.parse() with a
        # Pydantic response_format, which many local OpenAI-compatible servers
        # return as 503/unsupported. Our prompt already demands JSON, so use a
        # normal chat completion instead.
        kwargs.pop("keyword_extraction", None)
        kwargs.pop("response_format", None)
        kwargs.setdefault("temperature", settings["temperature"])
        kwargs.setdefault("timeout", settings["timeout"])
        kwargs.setdefault("enable_cot", False)

        response = await openai_complete_if_cache(
            settings["llm_model"],
            prompt,
            system_prompt=guarded_system_prompt,
            history_messages=history_messages or [],
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            **kwargs,
        )
        response = strip_qwen_thinking(response)
        return repair_lightrag_extraction_output(response)

    return qwen_llm_model_func


def make_embedding_func(settings: dict[str, Any]) -> EmbeddingFunc:
    async def create_embeddings(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
        params: dict[str, Any] = {
            "model": settings["embedding_model"],
            "input": texts,
        }
        if env_bool("QWEN_EMBEDDING_SEND_DIMENSIONS", False):
            params["dimensions"] = settings["embedding_dim"]
        response = await client.embeddings.create(**params)
        return [item.embedding for item in response.data]

    def valid_vector(vector: list[float]) -> bool:
        if len(vector) != settings["embedding_dim"]:
            return False
        array = np.asarray(vector, dtype=np.float32)
        return bool(np.all(np.isfinite(array)) and np.linalg.norm(array) > 0)

    async def qwen_embedding_func(texts: list[str]) -> np.ndarray:
        clean_texts = [sanitize_embedding_text(text) for text in texts]
        client = AsyncOpenAI(
            api_key=settings["embedding_api_key"],
            base_url=settings["embedding_base_url"],
            timeout=settings["timeout"],
        )
        try:
            try:
                vectors = await create_embeddings(client, clean_texts)
                if len(vectors) == len(clean_texts) and all(
                    valid_vector(vector) for vector in vectors
                ):
                    return np.array(vectors, dtype=np.float32)
            except Exception as batch_exc:
                if "unsupported value: NaN" not in str(batch_exc):
                    raise

            vectors = []
            for text in clean_texts:
                try:
                    single_vectors = await create_embeddings(client, [text])
                    vector = single_vectors[0]
                    if not valid_vector(vector):
                        raise ValueError("embedding returned non-finite or zero vector")
                except Exception:
                    vector = fallback_embedding_vector(text, settings["embedding_dim"])
                vectors.append(vector)
            return np.array(vectors, dtype=np.float32)
        finally:
            await client.close()

    return EmbeddingFunc(
        embedding_dim=settings["embedding_dim"],
        max_token_size=settings["embedding_max_tokens"],
        func=qwen_embedding_func,
    )


def build_qwen_policy_rag(
    *,
    working_dir: str,
    storage: str | None = None,
    parser: str = "mineru",
    parse_method: str = "auto",
    output_dir: str | None = None,
    enable_image_processing: bool = False,
    enable_table_processing: bool = True,
    enable_equation_processing: bool = True,
    display_content_stats: bool = True,
) -> RAGAnything:
    apply_chinese_policy_prompts()
    settings = get_qwen_settings()

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        parse_method=parse_method,
        parser_output_dir=output_dir or DEFAULT_OUTPUT_DIR,
        enable_image_processing=enable_image_processing,
        enable_table_processing=enable_table_processing,
        enable_equation_processing=enable_equation_processing,
        display_content_stats=display_content_stats,
        context_window=int(os.getenv("QWEN_CONTEXT_WINDOW", "2")),
        context_mode=os.getenv("QWEN_CONTEXT_MODE", "page"),
        max_context_tokens=int(os.getenv("QWEN_MAX_CONTEXT_TOKENS", "2400")),
        include_headers=True,
        include_captions=True,
        use_full_path=False,
    )

    return RAGAnything(
        config=config,
        llm_model_func=make_llm_func(settings),
        embedding_func=make_embedding_func(settings),
        lightrag_kwargs=make_lightrag_kwargs(settings, storage=storage),
    )


def resolve_input_paths(paths: list[str], recursive: bool) -> list[Path]:
    supported = {
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".txt",
        ".md",
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
    }
    result: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            result.extend(
                item for item in iterator if item.is_file() and item.suffix.lower() in supported
            )
        elif path.is_file():
            result.append(path)
        else:
            fallback = find_unique_file_by_name(path.name)
            if fallback is not None:
                print(f"未找到指定路径 {path}，已按文件名匹配到: {fallback}")
                result.append(fallback)
                continue
            candidates = find_file_candidates(path.name)
            hint = ""
            if candidates:
                preview = "\n".join(f"  - {candidate}" for candidate in candidates[:10])
                hint = f"\n在 data 目录中找到相近文件，请使用完整路径：\n{preview}"
            raise FileNotFoundError(f"文件不存在: {path}{hint}")
    return sorted(dict.fromkeys(result))


def find_file_candidates(file_name: str) -> list[Path]:
    """Find files with a similar name under common project data directories."""
    if not file_name:
        return []

    search_roots = [WORKSPACE_ROOT / "data", RAGANYTHING_ROOT / "data"]
    candidates: list[Path] = []
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix.lower()

    for root in search_roots:
        if not root.exists():
            continue
        for item in root.rglob("*"):
            if not item.is_file():
                continue
            if item.name == file_name:
                candidates.append(item)
            elif stem and stem in item.stem and (not suffix or item.suffix.lower() == suffix):
                candidates.append(item)

    return sorted(dict.fromkeys(candidates))


def find_unique_file_by_name(file_name: str) -> Path | None:
    candidates = [item for item in find_file_candidates(file_name) if item.name == file_name]
    if len(candidates) == 1:
        return candidates[0]
    return None
