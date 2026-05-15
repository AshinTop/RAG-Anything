#!/usr/bin/env python
"""Shared Qwen + Chinese policy helpers for RAG-Anything examples."""

from __future__ import annotations

import os
import re
import sys
import hashlib
import shutil
import subprocess
import time
from datetime import datetime
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
    "请始终使用自然、正式的简体中文回答。回答必须基于检索到的政策原文、"
    "知识图谱实体和关系，不要编造未在资料中出现的政策要求。"
    "涉及条款、适用范围、指标、数值、单位、发布机构和文件名称时，"
    "优先保留原文中文表述；无法从资料中确认时，请明确说明资料中未检索到。"
)


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

2. 关系抽取：
   - 抽取实体之间明确存在的政策关系，例如“发布/批准/适用于/规定/包含/对应/要求/限定/计算依据/取值为”。
   - 关系描述必须说明关系依据，尽量保留原文中的条件、范围、数值和单位。

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
请从下面“待处理文本”中抽取中文政策知识图谱的实体和关系。

---要求---
1. 严格遵守系统提示中的实体、关系字段顺序和分隔符。
2. 只输出实体和关系列表，不要添加开场白、总结或解释。
3. 抽取完成后，最后一行必须输出 `{completion_delimiter}`。
4. 输出语言必须为{language}。

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
8. 输出语言必须为{language}，不要添加解释、Markdown 或代码块。

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
        "embedding_batch_num": int(os.getenv("QWEN_EMBEDDING_BATCH_NUM", "4")),
        "embedding_func_max_async": int(os.getenv("QWEN_EMBEDDING_MAX_ASYNC", "1")),
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
