#!/usr/bin/env python
"""Shared Qwen + Chinese policy helpers for RAG-Anything examples."""

from __future__ import annotations

import asyncio
import html
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
    "如果同一分类同时出现库容、容量、面积、处理能力等不同口径或单位，必须分开描述；"
    "不得用“或”“/”把万m³、t/d、万m³/d、hm²、m²等不同单位合并成一个指标。"
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
    "如果问题问“包含哪些”“有哪些”“名录”“清单”，且原文片段中有表格，必须按表格逐行抽取。"
    "数量较多时可以只列名称或关键列，但不得写“因篇幅限制仅展示部分”“其余略”等截断语。"
    "除非用户明确要求示例，否则不能只给部分示例。\n"
    "如果原文片段包含“|”分隔的表格行，回答涉及表格时必须原样使用表格中的行、列和值；"
    "不得重排表头、不得把空值或破折号改成数值、不得换算单位、不得补充表格中不存在的数值。"
    "需要整理成 Markdown 表格时，只能复制原表格单元格内容。\n"
    "如果同一分类同时有容量单位和处理能力单位，必须按原文分开列出；"
    "不得写成“≥1200万m³或t/d”“万m³或t/d”这类混合单位表达。\n"
    "5. 答案必须使用以下结构：\n"
    "结论：\n"
    "依据：\n"
    "补充说明：\n"
    "参考来源：\n"
    "6. 没有明确页码时，参考来源只写“《文件名》”；不要写页码 X 或页码未标明。"
)


def build_strict_answer_prompt(raw_prompt: str, question: str, focused_context: str | None = None) -> str:
    focused_context = focused_context or extract_focused_context(raw_prompt, question)
    protected_tables = build_protected_table_context(question, focused_context)
    protected_instruction = ""
    if protected_tables:
        protected_instruction = f"""

不可改写表格：
```
{protected_tables}
```
上方“不可改写表格”是答案中必须原样使用的表格依据。你仍然必须按照“结论、依据、补充说明、参考来源”的结构完整回答问题。
回答涉及该表格时，只能原样复制这些表格行和值；同时要结合同一问题相关片段中的非表格条款说明控制原则、适用条件和来源。不得新增或改写任何数字、百分比、标准号、条款编号或单位。"""
    return f"""/no_think
请根据下面的“问题相关原文片段”回答用户问题。

用户问题：
{question}

问题相关原文片段：
```
{focused_context}
```
{protected_instruction}

只能使用上方原文片段。不要使用外部知识，不要引用上方片段没有出现的文件名、标准号、条款号、页码、数值或单位。
如果用户询问“包含哪些”“有哪些”“名录”“清单”，必须尽量完整列出原文片段中对应表格的所有条目；不得只展示部分示例，不得使用“因篇幅限制”“其余略”等截断表述。
如果问题相关原文片段中含有“|”分隔表格，回答中涉及该表格时必须原样复制表格行和值；不得自行转置、重排、换算、补齐空值或改写任何数值。
如果同一分类同时有容量单位和处理能力单位，必须分开描述，不得用“或”“/”合并不同单位；例如不得写成“≥1200万m³或t/d”。
"""


def build_protected_table_answer_prefix(question: str, focused_context: str) -> str:
    protected_tables = build_protected_table_context(question, focused_context)
    if not protected_tables:
        return ""
    return protected_tables


def merge_protected_table_answer(answer: str, protected_table: str) -> str:
    """Preserve exact source table rows without discarding the prose answer."""
    cleaned = clean_answer_text(answer)
    if not protected_table:
        return cleaned

    protected_table = normalize_policy_text_for_answer(protected_table).strip()
    if not protected_table:
        return cleaned

    without_tables = strip_markdown_tables(cleaned)
    if protected_table in without_tables:
        return without_tables

    table_block = f"原文表格：\n{protected_table}"
    if "依据：" in without_tables:
        return without_tables.replace("依据：", f"依据：\n{table_block}\n", 1).strip()
    if "补充说明：" in without_tables:
        return without_tables.replace("补充说明：", f"{table_block}\n\n补充说明：", 1).strip()
    if without_tables:
        return f"{without_tables}\n\n{table_block}".strip()
    return protected_table


def build_extractive_table_answer(
    question: str, focused_context: str, protected_table: str
) -> str:
    """Build a grounded answer for table-centric policy questions.

    This avoids asking a small LLM to paraphrase dense tables and clause numbers.
    It uses only the currently retrieved focused context, not question-specific
    rules or external knowledge.
    """
    raw_protected_table = str(protected_table or "").strip()
    protected_table = normalize_policy_text_for_answer(raw_protected_table).strip()
    if not protected_table:
        return ""

    chunks = parse_focused_context_chunks(focused_context)
    if not chunks:
        return ""

    table_chunk = next(
        (chunk for chunk in chunks if raw_protected_table in chunk.get("body", "")),
        None,
    )
    table_section = table_chunk.get("section_path") if table_chunk else ""
    table_doc = table_chunk.get("file") if table_chunk else ""
    terms = build_focus_terms(question)
    long_terms = [term for term in terms if len(term) >= 5]
    object_terms = [
        term
        for term in long_terms
        if not any(marker in term for marker in ["如何", "规定", "控制", "面积", "建设用地"])
    ]

    table_refs = set(re.findall(r"表\s*\d+", protected_table))
    area_control_markers = [
        "建设用地面积",
        "用地面积",
        "控制面积",
        "用地控制面积",
        "不应超过",
        "不得超过",
        "应根据",
        "取上限",
        "取下限",
        "内插法",
        "插入法",
    ]
    article_candidates: list[tuple[int, dict[str, str]]] = []
    for chunk in chunks:
        if chunk.get("content_type") != "article":
            continue
        body = chunk.get("body", "")
        if table_doc and chunk.get("file") != table_doc:
            continue
        same_section = table_section and chunk.get("section_path") == table_section
        object_hit = any(term in body for term in object_terms)
        fallback_hit = not object_terms and any(term in body for term in long_terms)
        if not same_section or not (object_hit or fallback_hit):
            continue

        table_ref_hits = sum(1 for ref in table_refs if ref and ref in body)
        marker_hits = sum(1 for marker in area_control_markers if marker in body)
        body_score = table_ref_hits * 120 + marker_hits * 35
        body_score += score_focus_snippet(body, terms, terms) // 10
        if table_ref_hits or marker_hits:
            article_candidates.append((body_score, chunk))

    article_candidates.sort(
        key=lambda item: (-item[0], safe_int(item[1].get("chunk_order_index")))
    )
    if not article_candidates:
        for chunk in chunks:
            if chunk.get("content_type") != "article":
                continue
            body = chunk.get("body", "")
            if table_doc and chunk.get("file") != table_doc:
                continue
            same_section = table_section and chunk.get("section_path") == table_section
            object_hit = any(term in body for term in object_terms)
            fallback_hit = not object_terms and any(term in body for term in long_terms)
            if same_section and (object_hit or fallback_hit):
                article_candidates.append((score_focus_snippet(body, terms, terms), chunk))
    article_chunks = [chunk for _, chunk in article_candidates[:6]]
    article_chunks.sort(key=lambda item: safe_int(item.get("chunk_order_index")))

    conclusion = build_extractive_conclusion(
        article_chunks,
        table_title=extract_table_title(protected_table),
        max_sentences=4,
    )
    lines: list[str] = [
        "结论：",
        conclusion
        or "检索原文中有相关控制表格，具体面积、单位和上下限以“原文表格”列示内容为准。",
        "",
        "依据：",
    ]

    seen_bodies: set[str] = set()
    for chunk in article_chunks:
        body = normalize_answer_line(chunk.get("body", ""))
        if not body or body in seen_bodies:
            continue
        seen_bodies.add(body)
        clause = chunk.get("clause")
        page = chunk.get("page")
        prefix_parts = []
        if clause:
            prefix_parts.append(clause)
        if page:
            prefix_parts.append(f"页码 {page}")
        prefix = f"{'，'.join(prefix_parts)}：" if prefix_parts else ""
        lines.append(f"{len(seen_bodies)}. {prefix}{body}")

    if not seen_bodies:
        lines.append("1. 资料中未检索到可单独抽取的非表格条款；请以原文表格为准。")

    lines.extend(["", "原文表格：", protected_table, "", "补充说明："])
    lines.append("以上内容均为检索片段中的原文条款或表格，未补充外部标准或未检索到的数值。")
    lines.extend(["", "参考来源："])
    reference_chunks = article_chunks + ([table_chunk] if table_chunk else [])
    lines.extend(build_reference_source_lines(reference_chunks))
    return "\n".join(lines).strip()


def build_extractive_clause_answer(question: str, focused_context: str) -> str:
    """Build a clause-by-clause answer for normative "how is it regulated" queries."""
    if not is_clause_extractive_question(question):
        return ""

    chunks = parse_focused_context_chunks(focused_context)
    article_chunks = [chunk for chunk in chunks if chunk.get("content_type") == "article"]
    if not article_chunks:
        return ""

    terms = build_focus_terms(question)
    scored = [
        (score_focus_snippet(chunk.get("body", ""), terms, terms), index, chunk)
        for index, chunk in enumerate(article_chunks)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        return ""

    top_chunk = scored[0][2]
    top_section = top_chunk.get("section_path")
    top_file = top_chunk.get("file")
    selected = [
        chunk
        for chunk in article_chunks
        if chunk.get("file") == top_file
        and (not top_section or chunk.get("section_path") == top_section)
    ]
    selected.sort(key=lambda item: safe_int(item.get("chunk_order_index")))
    if not selected:
        return ""

    conclusion = build_extractive_conclusion(selected, max_sentences=6)
    lines: list[str] = [
        "结论：",
        conclusion
        or "检索原文列出了相关条款，具体分类、组成、容量、比例和计算要求如下。",
        "",
        "依据：",
    ]

    seen_bodies: set[str] = set()
    for chunk in selected:
        body = normalize_answer_line(chunk.get("body", ""))
        if not body or body in seen_bodies:
            continue
        seen_bodies.add(body)
        clause = chunk.get("clause")
        page = chunk.get("page")
        prefix_parts = []
        if clause:
            prefix_parts.append(clause)
        if page:
            prefix_parts.append(f"页码 {page}")
        prefix = f"{'，'.join(prefix_parts)}：" if prefix_parts else ""
        lines.append(f"{len(seen_bodies)}. {prefix}{body}")

    lines.extend(["", "补充说明："])
    lines.append("以上内容均为检索片段中的原文条款或表格，未补充外部标准或未检索到的数值。")
    lines.extend(["", "参考来源："])
    lines.extend(build_reference_source_lines(selected))
    return "\n".join(lines).strip()


def is_clause_extractive_question(question: str) -> bool:
    question = str(question or "")
    return any(
        marker in question
        for marker in [
            "如何规定",
            "怎么规定",
            "怎样规定",
            "如何控制",
            "怎么控制",
            "怎样控制",
        ]
    )


def parse_focused_context_chunks(focused_context: str) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for block in re.split(r"\n\s*---\s*\n", str(focused_context or "")):
        if "原文:" not in block:
            continue
        content = block.split("原文:", 1)[1].strip()
        meta = parse_structured_chunk_metadata(content)
        body = content.split("正文：", 1)[1].strip() if "正文：" in content else content
        body = body.split("表格作答要求：", 1)[0].strip()
        chunks.append(
            {
                "file": clean_source_title(meta.get("文件", "")),
                "page": meta.get("页码", ""),
                "section_path": meta.get("章节路径", ""),
                "clause": meta.get("条款", ""),
                "content_type": meta.get("内容类型", ""),
                "chunk_order_index": meta.get("chunk_order_index", ""),
                "body": body,
            }
        )
    return chunks


def build_reference_source_lines(chunks: list[dict[str, str]]) -> list[str]:
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for chunk in chunks:
        file_name = clean_source_title(chunk.get("file", ""))
        if not file_name:
            continue
        page = chunk.get("page", "").strip()
        if file_name not in grouped:
            grouped[file_name] = []
            order.append(file_name)
        if page and page not in grouped[file_name]:
            grouped[file_name].append(page)

    if not order:
        return ["1. 来源未标明"]

    lines: list[str] = []
    for index, file_name in enumerate(order[:5], start=1):
        pages = sort_page_values(grouped[file_name])
        if pages:
            lines.append(f"{index}. 页码 {'、'.join(pages)} - 《{file_name}》")
        else:
            lines.append(f"{index}. 《{file_name}》")
    return lines


def clean_source_title(title: str) -> str:
    title = str(title or "").strip()
    title = title.strip("《》")
    return title.strip()


def safe_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return 10**9


def normalize_answer_line(text: str) -> str:
    text = normalize_policy_text_for_answer(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sort_page_values(pages: list[str]) -> list[str]:
    def page_key(page: str) -> tuple[int, str]:
        number = re.search(r"\d+", page)
        return (int(number.group()) if number else 10**9, page)

    return sorted(pages, key=page_key)


def normalize_policy_text_for_answer(text: str) -> str:
    """Clean parser math artifacts for QA display without changing values."""
    text = str(text or "")
    replacements = {
        "\\leq": "≤",
        "\\geq": "≥",
        "\\times": "×",
        "\\%": "%",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    text = re.sub(r"([A-Za-z]+)_\{([^{}]+)\}", r"\1\2", text)
    text = re.sub(r"([A-Za-z]+)_([A-Za-z0-9]+)", r"\1\2", text)
    text = re.sub(r"hm\^\{?2\}?", "hm²", text)
    text = re.sub(r"m\^\{?3\}?", "m³", text)
    text = re.sub(r"m\^\{?2\}?", "m²", text)
    text = text.replace("\\", "")
    text = re.sub(r"\s+([,，;；。])", r"\1", text)
    text = re.sub(r"([（(])\s+", r"\1", text)
    text = re.sub(r"\s+([）)])", r"\1", text)
    return text.strip()


def extract_table_title(table_text: str) -> str:
    for line in str(table_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("表"):
            return stripped
    return ""


def build_extractive_conclusion(
    chunks: list[dict[str, str]], *, table_title: str = "", max_sentences: int = 4
) -> str:
    sentences: list[str] = []
    priority_markers = [
        "不应超过",
        "不得超过",
        "应根据",
        "按照",
        "由",
        "应满足",
        "取上限",
        "取下限",
        "内插法",
        "插入法",
    ]
    for chunk in chunks:
        body = normalize_answer_line(chunk.get("body", ""))
        if not body:
            continue
        body = re.sub(r"^第[一二三四五六七八九十百零〇\d]+条\s*", "", body).strip()
        pieces = split_policy_sentences(body)
        relevant = [piece for piece in pieces if any(marker in piece for marker in priority_markers)]
        if not relevant and pieces:
            relevant = pieces[:1]
        for piece in relevant:
            piece = piece.strip("；;。 ")
            if piece and piece not in sentences:
                sentences.append(piece)
            if len(sentences) >= max_sentences:
                break
        if len(sentences) >= max_sentences:
            break

    if table_title:
        table_sentence = f"具体控制面积以{table_title}为准"
        if table_sentence not in sentences:
            sentences.insert(0, table_sentence)

    if not sentences:
        return ""
    return "；".join(sentences[:max_sentences]) + "。"


def split_policy_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。；;])\s*", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def build_protected_table_context(question: str, focused_context: str) -> str:
    blocks = extract_pipe_table_blocks(focused_context)
    if not blocks:
        return ""
    terms = build_focus_terms(question)
    scored = [
        (score_focus_snippet(block, terms, terms), index, block)
        for index, block in enumerate(blocks)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] <= 0:
        return ""
    return scored[0][2]


def extract_pipe_table_blocks(text: str) -> list[str]:
    lines = str(text or "").splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if "|" not in lines[index]:
            index += 1
            continue

        start = index
        for cursor in range(index - 1, -1, -1):
            stripped = lines[cursor].strip()
            if not stripped or stripped.startswith("---") or stripped.startswith("[参考内容"):
                break
            if stripped == "正文：":
                start = cursor + 1
                break
            start = cursor

        end = index + 1
        while end < len(lines):
            stripped = lines[end].strip()
            if stripped.startswith("---") or stripped.startswith("[参考内容"):
                break
            if stripped.startswith("表格作答要求："):
                break
            if not stripped and end > index:
                break
            end += 1

        block = "\n".join(line.rstrip() for line in lines[start:end]).strip()
        if block and block not in blocks:
            blocks.append(block)
        index = max(end, index + 1)
    return blocks


def strip_markdown_tables(text: str) -> str:
    """Remove model-generated pipe tables when a protected source table is present."""
    lines = str(text or "").splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        is_table_line = "|" in stripped and stripped.count("|") >= 2
        is_separator = bool(re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", stripped))
        if is_table_line or is_separator:
            skipping = True
            continue
        if skipping and not stripped:
            skipping = False
            continue
        skipping = False
        kept.append(line)
    return "\n".join(kept).strip()


def clean_answer_text(text: str) -> str:
    """Remove common model output noise without changing factual content."""
    cleaned_lines: list[str] = []
    noise_lines = {"出手", "好的", "以下是答案", "根据提供的内容"}
    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if stripped in noise_lines:
            continue
        stripped = re.sub(r"^\*+([^*]+)\*+[:：]?", r"\1：", stripped)
        stripped = re.sub(r"\*\*", "", stripped)
        stripped = re.sub(r"(?<!\*)\*(?!\*)", "", stripped)
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_protected_table_explanation(text: str, protected_table: str) -> str:
    """Keep LLM explanation conservative when the table itself is program-copied."""
    allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?(?:~\d+(?:\.\d+)?)?%?", protected_table))
    allowed_numbers.update(re.findall(r"第[一二三四五六七八九十百零〇\d]+条", protected_table))

    kept: list[str] = []
    for raw_line in strip_markdown_tables(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("---"):
            continue
        line = re.sub(r"^\s*[-*]\s*", "", line)
        line = re.sub(r"^\s*\d+[\.、]\s*", "", line)

        numbers = set(re.findall(r"\d+(?:\.\d+)?(?:~\d+(?:\.\d+)?)?%?", line))
        clauses = set(re.findall(r"第[一二三四五六七八九十百零〇\d]+条", line))
        if any(value not in allowed_numbers for value in numbers | clauses):
            continue

        # Drop lines that look like fresh normative claims with numbers; the
        # copied table already carries exact values.
        if numbers and any(word in line for word in ["不得超过", "不应超过", "增加", "比例", "为"]):
            if line not in protected_table:
                continue

        if line and line not in kept:
            kept.append(line)
        if len(kept) >= 6:
            break

    return "\n".join(f"- {line}" for line in kept)


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
    chunks: list[dict[str, Any]],
    question: str,
    *,
    max_chunks: int = 8,
    raw_prompt: str | None = None,
) -> str:
    raw_chunks = extract_raw_prompt_document_chunks(raw_prompt or "")
    if raw_chunks:
        chunks = raw_chunks

    if not chunks:
        return ""

    terms = build_focus_terms(question)
    high_priority_terms = terms

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        content = str(chunk.get("content") or "")
        score = score_focus_snippet(focus_scoring_text(content), terms, high_priority_terms)
        scored.append((score, index, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        selected_indexes = [index for _, index, _ in scored[:max_chunks]]
    else:
        _, top_index, top_chunk = scored[0]
        selected_indexes = [top_index]
        top_doc_id = top_chunk.get("full_doc_id") or top_chunk.get("reference_id")
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
                    (neighbor.get("full_doc_id") or neighbor.get("reference_id")) == top_doc_id
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
                if (neighbor.get("full_doc_id") or neighbor.get("reference_id")) != top_doc_id:
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
        content = format_focused_chunk_content(str(chunk.get("content") or ""))
        parts.append(
            f"[参考内容{ref_index}]\n"
            f"chunk_id: {chunk.get('id')}\n"
            f"chunk_order_index: {order}\n"
            f"文件: 《{file_path}》\n"
            f"原文:\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def format_focused_chunk_content(content: str) -> str:
    content = str(content or "").strip()
    if "<table" not in content.lower():
        return annotate_pipe_table_context(content)

    rows = html_table_rows_to_text(content)
    if not rows:
        return content

    prefix = content
    table_start = re.search(r"<table\b", content, flags=re.IGNORECASE)
    if table_start:
        prefix = content[: table_start.start()].strip()
    return annotate_pipe_table_context(prefix + "\n表格转写：\n" + "\n".join(rows))


def annotate_pipe_table_context(content: str) -> str:
    """Add a generic instruction near pipe-delimited tables to reduce value drift."""
    if "表格作答要求：" in content:
        return content
    pipe_line_count = sum(1 for line in content.splitlines() if "|" in line)
    if pipe_line_count < 2:
        return content
    return (
        content
        + "\n\n表格作答要求：上方包含“|”分隔表格行；回答涉及该表格时必须逐行原样引用"
        "表格中的单元格和值，不得自行重排表头、换算单位、补齐空值或改写数值。"
    )


def html_table_rows_to_text(content: str) -> list[str]:
    rows: list[str] = []
    for row_match in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", content, flags=re.IGNORECASE | re.DOTALL):
        row_html = row_match.group(1)
        cells: list[str] = []
        for cell_match in re.finditer(
            r"<t[dh]\b[^>]*>(.*?)</t[dh]>",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            cell = re.sub(r"<[^>]+>", "", cell_match.group(1))
            cell = html.unescape(cell)
            cell = re.sub(r"\s+", " ", cell).strip()
            cells.append(cell)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def extract_raw_prompt_document_chunks(raw_prompt: str) -> list[dict[str, Any]]:
    """Parse LightRAG's only_need_prompt output and keep retrieval-scoped chunks.

    The strict answer stage must not re-rank the whole database when LightRAG has
    already retrieved a relevant set. Reusing the raw prompt chunks prevents a
    correct hit from being replaced by unrelated old documents during local
    focusing.
    """
    if not raw_prompt:
        return []

    chunks: list[dict[str, Any]] = []
    for line in raw_prompt.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        meta = parse_structured_chunk_metadata(content)
        chunks.append(
            {
                "id": meta.get("chunk_id") or f"raw-{len(chunks)}",
                "reference_id": str(item.get("reference_id") or ""),
                "full_doc_id": meta.get("文件hash") or str(item.get("reference_id") or ""),
                "chunk_order_index": meta.get("chunk_order_index") or len(chunks),
                "content": content,
                "file_path": meta.get("相对路径") or meta.get("文件") or "",
            }
        )
    return chunks


def is_related_neighbor(content: str, high_priority_terms: list[str]) -> bool:
    scoring_text = focus_scoring_text(content)
    return any(term and term in scoring_text for term in high_priority_terms)


def focus_scoring_text(content: str) -> str:
    content = str(content or "")
    meta = parse_structured_chunk_metadata(content)
    body = content.split("正文：", 1)[1] if "正文：" in content else content
    parts = [
        meta.get("章节路径", ""),
        meta.get("条款", ""),
        body,
    ]
    return "\n".join(part for part in parts if part)


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
            "来源集合",
            "相对路径",
            "目录路径",
            "目录标签",
            "业务分类",
            "文件hash",
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
    high_priority_terms = terms
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
    first_line = str(snippet or "").splitlines()[0] if str(snippet or "").splitlines() else ""
    for term in high_priority_terms:
        if term in snippet:
            score += 100 + len(term) * 5
        if term in first_line:
            score += 250 + len(term) * 10
    for term in terms:
        if term in snippet:
            score += snippet.count(term) * max(1, len(term))

    if re.search(r"第[一二三四五六七八九十百零〇\d]+条", snippet):
        score += 40
    if "表格" in snippet or "<table" in snippet:
        score += 30
    if not any(term in snippet for term in high_priority_terms):
        score -= 200
    return score


def build_focus_terms(question: str) -> list[str]:
    question = question or ""
    text = re.sub(r"[？?，,。；;：:\s（）()【】\[\]《》“”\"'、]+", "", question)

    terms: list[str] = []
    normalized = re.sub(r"[？?，,。；;：:\s（）()【】\[\]《》“”\"'、]+", " ", question)
    phrase_parts = re.split(
        r"(?:的|是什么|什么是|有哪些|包含哪些|主要讲|主要内容|如何|怎么|怎样|规定|控制)+",
        normalized,
    )
    for part in phrase_parts:
        part = re.sub(r"\s+", "", part)
        if len(part) >= 3 and part not in terms:
            terms.append(part)
        for size in range(min(10, len(part)), 4, -1):
            for index in range(0, max(0, len(part) - size + 1)):
                gram = part[index : index + size]
                if gram not in terms:
                    terms.append(gram)
            if len(terms) >= 40:
                break

    if len(text) >= 4:
        terms.append(text)

    # Generate long n-grams directly from the question instead of maintaining
    # domain-specific stop-word lists.
    for size in range(min(12, len(text)), 4, -1):
        for index in range(0, max(0, len(text) - size + 1)):
            gram = text[index : index + size]
            if gram not in terms:
                terms.append(gram)
        if len(terms) >= 80:
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
《某专项管理办法》由甲部门、乙部门联合发布，适用于本行政区域内相关项目的申报、审批和监督管理工作。

示例输出：
entity{tuple_delimiter}《某专项管理办法》{tuple_delimiter}政策文件{tuple_delimiter}该文件由甲部门、乙部门联合发布，规定相关项目的申报、审批和监督管理要求。
entity{tuple_delimiter}甲部门{tuple_delimiter}发布机构{tuple_delimiter}甲部门是发布《某专项管理办法》的机构之一。
entity{tuple_delimiter}乙部门{tuple_delimiter}发布机构{tuple_delimiter}乙部门是发布《某专项管理办法》的机构之一。
entity{tuple_delimiter}相关项目{tuple_delimiter}适用对象{tuple_delimiter}相关项目是《某专项管理办法》明确适用的对象。
relation{tuple_delimiter}甲部门{tuple_delimiter}《某专项管理办法》{tuple_delimiter}发布{tuple_delimiter}甲部门联合发布《某专项管理办法》。
relation{tuple_delimiter}《某专项管理办法》{tuple_delimiter}相关项目{tuple_delimiter}适用于{tuple_delimiter}《某专项管理办法》适用于本行政区域内相关项目的申报、审批和监督管理工作。
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
        allow_fallback = env_bool("QWEN_EMBEDDING_ALLOW_FALLBACK", False)
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
                if not allow_fallback:
                    raise RuntimeError(
                        "Embedding batch returned NaN/invalid values. "
                        "Fix the embedding service or set QWEN_EMBEDDING_ALLOW_FALLBACK=true "
                        "only for non-evaluation experiments."
                    ) from batch_exc

            vectors = []
            for text in clean_texts:
                try:
                    single_vectors = await create_embeddings(client, [text])
                    vector = single_vectors[0]
                    if not valid_vector(vector):
                        raise ValueError("embedding returned non-finite or zero vector")
                except Exception as exc:
                    if not allow_fallback:
                        raise RuntimeError(
                            "Embedding returned invalid vector. "
                            "Fix the embedding service or set QWEN_EMBEDDING_ALLOW_FALLBACK=true "
                            "only for non-evaluation experiments."
                        ) from exc
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
