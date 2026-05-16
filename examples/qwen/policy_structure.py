#!/usr/bin/env python
"""Structure Chinese policy text into article/table chunks with metadata."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_policy_common import sync_file_to_save


HEADING_NUM = r"[一二三四五六七八九十百千万零〇\d]+"
CHAPTER_RE = re.compile(rf"^(第{HEADING_NUM}章)\s*(.+)$")
SECTION_RE = re.compile(rf"^(第{HEADING_NUM}节)\s*(.+)$")
ARTICLE_RE = re.compile(rf"^(第{HEADING_NUM}条)\s*(.*)$")
TABLE_RE = re.compile(r"^\[表格[，,]\s*第\s*(\d+)\s*页[，,]\s*块\s*(\d+)\]")
PAGE_RE = re.compile(r"第\s*(\d+)\s*页")
DEFAULT_MAX_CHUNK_CHARS = 3500


@dataclass
class StructuredChunk:
    doc_id: str
    chunk_id: str
    chunk_order_index: int
    file_name: str
    page_idx: int | None
    chapter_title: str | None
    section_title: str | None
    article_no: str | None
    content_type: str
    content: str
    source_block_range: list[int]
    source_metadata: dict[str, Any] | None = None

    @property
    def section_path(self) -> str:
        return " / ".join(
            part for part in [self.chapter_title, self.section_title] if part
        )

    def to_index_text(self) -> str:
        metadata_lines = [
            f"文件：{self.file_name}",
            f"chunk_id：{self.chunk_id}",
            f"chunk_order_index：{self.chunk_order_index}",
            f"内容类型：{self.content_type}",
        ]
        if self.source_metadata:
            source_collection = self.source_metadata.get("source_collection")
            relative_path = self.source_metadata.get("relative_path")
            directory_path = self.source_metadata.get("directory_path")
            directory_tags = self.source_metadata.get("directory_tags")
            business_category = self.source_metadata.get("business_category")
            file_hash = self.source_metadata.get("file_hash")
            if source_collection:
                metadata_lines.append(f"来源集合：{source_collection}")
            if relative_path:
                metadata_lines.append(f"相对路径：{relative_path}")
            if directory_path:
                metadata_lines.append(f"目录路径：{directory_path}")
            if directory_tags:
                if isinstance(directory_tags, list):
                    directory_tags = " > ".join(str(tag) for tag in directory_tags)
                metadata_lines.append(f"目录标签：{directory_tags}")
            if business_category:
                metadata_lines.append(f"业务分类：{business_category}")
            if file_hash:
                metadata_lines.append(f"文件hash：{file_hash}")
        if self.page_idx is not None:
            metadata_lines.append(f"页码：{self.page_idx + 1}")
        if self.section_path:
            metadata_lines.append(f"章节路径：{self.section_path}")
        if self.article_no:
            metadata_lines.append(f"条款：{self.article_no}")
        return "\n".join(metadata_lines) + "\n正文：\n" + self.content.strip()

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["section_path"] = self.section_path
        data["index_text"] = self.to_index_text()
        return data


def normalize_ws(text: str) -> str:
    text = str(text or "").replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    text = normalize_ws(text)
    if not text:
        return []
    return [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]


def page_from_text(text: str, fallback: int | None) -> int | None:
    table_match = TABLE_RE.search(text)
    if table_match:
        return int(table_match.group(1)) - 1
    page_match = PAGE_RE.search(text)
    if page_match:
        return int(page_match.group(1)) - 1
    return fallback


def max_structured_chunk_chars() -> int:
    try:
        return max(500, int(os.getenv("QWEN_STRUCTURED_CHUNK_MAX_CHARS", str(DEFAULT_MAX_CHUNK_CHARS))))
    except Exception:
        return DEFAULT_MAX_CHUNK_CHARS


def split_long_text(text: str, max_chars: int) -> list[str]:
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return [text] if text else []

    segments: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in split_paragraphs(text):
        if len(paragraph) > max_chars:
            if current:
                segments.append(normalize_ws("\n\n".join(current)))
                current = []
                current_len = 0
            for start in range(0, len(paragraph), max_chars):
                part = paragraph[start : start + max_chars].strip()
                if part:
                    segments.append(part)
            continue

        extra_len = len(paragraph) + (2 if current else 0)
        if current and current_len + extra_len > max_chars:
            segments.append(normalize_ws("\n\n".join(current)))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += extra_len

    if current:
        segments.append(normalize_ws("\n\n".join(current)))
    return [segment for segment in segments if segment]


def build_structured_chunks(
    text_items: list[dict[str, Any]],
    *,
    doc_id: str,
    file_name: str,
    source_metadata: dict[str, Any] | None = None,
) -> list[StructuredChunk]:
    chunks: list[StructuredChunk] = []
    chapter_title: str | None = None
    section_title: str | None = None
    current_article_no: str | None = None
    current_parts: list[str] = []
    current_page_idx: int | None = None
    current_start_block: int | None = None
    max_chunk_chars = max_structured_chunk_chars()

    def append_chunk(
        *,
        page_idx: int | None,
        article_no: str | None,
        content_type: str,
        content: str,
        source_block_range: list[int],
    ) -> None:
        for segment in split_long_text(content, max_chunk_chars):
            order = len(chunks)
            chunks.append(
                StructuredChunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}-chunk-{order:04d}",
                    chunk_order_index=order,
                    file_name=file_name,
                    page_idx=page_idx,
                    chapter_title=chapter_title,
                    section_title=section_title,
                    article_no=article_no,
                    content_type=content_type,
                    content=segment,
                    source_block_range=source_block_range,
                    source_metadata=source_metadata,
                )
            )

    def flush_article(end_block: int) -> None:
        nonlocal current_article_no, current_parts, current_page_idx, current_start_block
        content = normalize_ws("\n\n".join(current_parts))
        if not content:
            current_article_no = None
            current_parts = []
            current_page_idx = None
            current_start_block = None
            return
        append_chunk(
            page_idx=current_page_idx,
            article_no=current_article_no,
            content_type="article" if current_article_no else "section_text",
            content=content,
            source_block_range=[current_start_block or end_block, end_block],
        )
        current_article_no = None
        current_parts = []
        current_page_idx = None
        current_start_block = None

    def add_table(paragraph: str, page_idx: int | None, block_index: int) -> None:
        append_chunk(
            page_idx=page_idx,
            article_no=current_article_no,
            content_type="table",
            content=paragraph,
            source_block_range=[block_index, block_index],
        )

    for block_index, item in enumerate(text_items):
        if not isinstance(item, dict):
            continue
        item_page_idx = item.get("page_idx")
        if not isinstance(item_page_idx, int):
            item_page_idx = None
        for paragraph in split_paragraphs(str(item.get("text") or "")):
            paragraph_page_idx = page_from_text(paragraph, item_page_idx)
            chapter_match = CHAPTER_RE.match(paragraph)
            if chapter_match:
                flush_article(block_index)
                chapter_title = normalize_ws(paragraph)
                section_title = None
                continue

            section_match = SECTION_RE.match(paragraph)
            if section_match:
                flush_article(block_index)
                section_title = normalize_ws(paragraph)
                continue

            article_match = ARTICLE_RE.match(paragraph)
            if article_match:
                flush_article(block_index)
                current_article_no = article_match.group(1)
                current_parts = [paragraph]
                current_page_idx = paragraph_page_idx
                current_start_block = block_index
                continue

            if TABLE_RE.match(paragraph):
                flush_article(block_index)
                add_table(paragraph, paragraph_page_idx, block_index)
                continue

            if current_parts:
                current_parts.append(paragraph)
                if current_page_idx is None:
                    current_page_idx = paragraph_page_idx
            else:
                current_article_no = None
                current_parts = [paragraph]
                current_page_idx = paragraph_page_idx
                current_start_block = block_index

    flush_article(len(text_items) - 1)
    return chunks


def write_structured_chunks_sidecar(
    chunks: list[StructuredChunk], output_dir: str | Path, doc_id: str
) -> Path:
    output_path = Path(output_dir) / "structured_chunks" / f"{doc_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([chunk.to_json() for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sync_file_to_save(output_path)
    return output_path
