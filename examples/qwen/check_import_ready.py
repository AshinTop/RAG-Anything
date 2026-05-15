#!/usr/bin/env python
"""Preflight checks for the local Qwen policy import pipeline."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

RAGANYTHING_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = RAGANYTHING_ROOT.parent


def status_line(ok: bool, title: str, detail: str = "") -> None:
    mark = "OK " if ok else "BAD"
    print(f"[{mark}] {title}")
    if detail:
        print(f"     {detail}")


def warn_line(title: str, detail: str = "") -> None:
    print(f"[WARN] {title}")
    if detail:
        print(f"       {detail}")


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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
        value = "local"
    return value


def check_modules(model_source: str | None, storage: str) -> int:
    failures = 0
    required = {
        "openai": "pip install openai",
        "dotenv": "pip install python-dotenv",
        "numpy": "pip install numpy",
        "raganything": "在 RAG-Anything 目录执行: pip install -e .",
        "lightrag": "pip install lightrag-hku",
        "mineru": "pip install -U \"mineru[core]\"",
    }
    if model_source == "modelscope":
        required["modelscope"] = "pip install modelscope"
    if storage == "postgres":
        required["asyncpg"] = "pip install asyncpg"
        required["pgvector"] = "pip install pgvector"

    for module, install_hint in required.items():
        ok = module_exists(module)
        status_line(ok, f"Python 模块: {module}", "" if ok else install_hint)
        failures += 0 if ok else 1
    return failures


def check_postgres_env(args: argparse.Namespace) -> tuple[int, dict[str, str]]:
    required_keys = ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"]
    optional_defaults = {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_WORKSPACE": "qwen_policy",
        "POSTGRES_MAX_CONNECTIONS": "12",
        "POSTGRES_VECTOR_INDEX_TYPE": "HNSW",
    }

    env = {key: os.getenv(key, "") for key in required_keys}
    env.update({key: os.getenv(key, default) for key, default in optional_defaults.items()})

    failures = 0
    for key in required_keys:
        value = env[key]
        display = "***" if key == "POSTGRES_PASSWORD" and value else value
        status_line(bool(value), f"PostgreSQL 环境变量: {key}", display or "请在 .env 中配置")
        failures += 0 if value else 1

    for key in optional_defaults:
        status_line(True, f"PostgreSQL 环境变量: {key}", env[key])

    try:
        int(env["POSTGRES_PORT"])
    except Exception:
        status_line(False, "POSTGRES_PORT 必须是整数", env["POSTGRES_PORT"])
        failures += 1

    return failures, env


async def _check_postgres_connection(args: argparse.Namespace, env: dict[str, str]) -> int:
    try:
        import asyncpg
    except Exception as exc:
        status_line(False, "asyncpg 可用性", str(exc))
        return 1

    conn = None
    try:
        conn = await asyncpg.connect(
            host=env["POSTGRES_HOST"],
            port=int(env["POSTGRES_PORT"]),
            user=env["POSTGRES_USER"],
            password=env["POSTGRES_PASSWORD"],
            database=env["POSTGRES_DATABASE"],
            timeout=args.timeout,
        )
        version = await conn.fetchval("select version()")
        status_line(True, "PostgreSQL 连接", str(version).split(",", 1)[0])

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        vector_version = await conn.fetchval(
            "select extversion from pg_extension where extname = 'vector'"
        )
        status_line(
            bool(vector_version),
            "PostgreSQL pgvector 扩展",
            f"version={vector_version}" if vector_version else "未启用 vector 扩展",
        )
        return 0 if vector_version else 1
    except Exception as exc:
        status_line(
            False,
            "PostgreSQL 连接/pgvector 检查",
            f"{exc}\n     处理: 安装并启动 PostgreSQL，创建数据库，并执行 CREATE EXTENSION IF NOT EXISTS vector;",
        )
        return 1
    finally:
        if conn is not None:
            await conn.close()


def check_postgres(args: argparse.Namespace) -> int:
    failures, env = check_postgres_env(args)
    if failures:
        return failures
    graph_storage = os.getenv("QWEN_GRAPH_STORAGE") or os.getenv(
        "LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"
    )
    if normalize_storage(args.storage) == "postgres-age" or graph_storage == "PGGraphStorage":
        warn_line(
            "当前图谱存储需要 Apache AGE",
            "PGGraphStorage 会调用 create_graph()；普通 PostgreSQL/pgvector 不包含该函数。未安装 AGE 时请使用 QWEN_GRAPH_STORAGE=NetworkXStorage。",
        )
    return asyncio.run(_check_postgres_connection(args, env))


def check_mineru_command() -> int:
    mineru_path = shutil.which("mineru")
    ok = mineru_path is not None
    status_line(
        ok,
        "MinerU 命令行",
        mineru_path or "安装: pip install -U \"mineru[core]\"，然后重新打开终端",
    )
    return 0 if ok else 1


def check_paths(args: argparse.Namespace) -> int:
    failures = 0
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = (Path.cwd() / file_path).resolve()
        ok = file_path.exists()
        status_line(ok, "待导入文件", str(file_path))
        failures += 0 if ok else 1

    for label, path_value in [
        ("索引目录父目录", args.working_dir),
        ("解析输出目录父目录", args.output_dir),
    ]:
        path = Path(path_value)
        parent = path if path.exists() and path.is_dir() else path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            probe = parent / ".qwen_preflight_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            status_line(True, label, str(parent.resolve()))
        except Exception as exc:
            status_line(False, label, f"{parent} 不可写: {exc}")
            failures += 1

    working_dir = Path(args.working_dir)
    if normalize_storage(args.storage) == "local" and working_dir.exists():
        chunks_file = working_dir / "kv_store_text_chunks.json"
        vdb_file = working_dir / "vdb_chunks.json"
        has_chunks = False
        for file_path in [chunks_file, vdb_file]:
            if not file_path.exists():
                continue
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and (data.get("data") or len(data) > 0):
                    has_chunks = True
            except Exception:
                pass
        if has_chunks:
            warn_line("索引目录已有数据", f"如需重建，建议换一个新的 --working-dir: {working_dir}")
        else:
            warn_line("索引目录存在但没有 chunks", f"可能是上次失败残留，建议换新目录: {working_dir}_v2")

    return failures


def check_env(args: argparse.Namespace) -> tuple[int, dict[str, str]]:
    env_path = RAGANYTHING_ROOT / ".env"
    load_dotenv(env_path, override=False)
    status_line(env_path.exists(), ".env 文件", str(env_path))

    base_url = os.getenv("QWEN_BASE_URL") or os.getenv("LLM_BINDING_HOST", "")
    api_key = os.getenv("QWEN_API_KEY") or os.getenv(
        "LLM_BINDING_API_KEY", "local-qwen"
    )
    llm_model = os.getenv("QWEN_LLM_MODEL") or os.getenv("LLM_MODEL", "")
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
    embedding_model = os.getenv("QWEN_EMBEDDING_MODEL") or os.getenv(
        "EMBEDDING_MODEL", ""
    )
    embedding_dim = os.getenv("QWEN_EMBEDDING_DIM") or os.getenv("EMBEDDING_DIM", "")

    values = {
        "QWEN_BASE_URL / LLM_BINDING_HOST": base_url,
        "QWEN_API_KEY / LLM_BINDING_API_KEY": api_key,
        "QWEN_LLM_MODEL / LLM_MODEL": llm_model,
        "QWEN_EMBEDDING_BASE_URL / EMBEDDING_BINDING_HOST / LLM_BINDING_HOST": embedding_base_url,
        "QWEN_EMBEDDING_API_KEY / EMBEDDING_BINDING_API_KEY / LLM_BINDING_API_KEY": embedding_api_key,
        "QWEN_EMBEDDING_MODEL / EMBEDDING_MODEL": embedding_model,
        "QWEN_EMBEDDING_DIM / EMBEDDING_DIM": embedding_dim,
    }

    failures = 0
    for key, value in values.items():
        ok = bool(value)
        display = value
        if "API_KEY" in key and value:
            display = value[:6] + "..." if len(value) > 6 else "***"
        status_line(ok, f"环境变量: {key}", display or "请在 RAG-Anything/.env 中配置")
        failures += 0 if ok else 1

    try:
        int(embedding_dim)
    except Exception:
        status_line(False, "QWEN_EMBEDDING_DIM / EMBEDDING_DIM 必须是整数", embedding_dim)
        failures += 1

    resolved = {
        "QWEN_BASE_URL": base_url,
        "QWEN_API_KEY": api_key,
        "QWEN_LLM_MODEL": llm_model,
        "QWEN_EMBEDDING_BASE_URL": embedding_base_url,
        "QWEN_EMBEDDING_API_KEY": embedding_api_key,
        "QWEN_EMBEDDING_MODEL": embedding_model,
        "QWEN_EMBEDDING_DIM": embedding_dim,
    }
    return failures, resolved


def check_openai_services(args: argparse.Namespace, env: dict[str, str]) -> int:
    try:
        from openai import OpenAI
    except Exception as exc:
        status_line(False, "OpenAI SDK 可用性", str(exc))
        return 1

    failures = 0

    llm_client = OpenAI(base_url=env["QWEN_BASE_URL"], api_key=env["QWEN_API_KEY"])
    try:
        models = [model.id for model in llm_client.models.list().data]
        status_line(True, "Chat 服务 /v1/models", ", ".join(models[:8]) or "空列表")
        if env["QWEN_LLM_MODEL"] not in models:
            warn_line(
                "QWEN_LLM_MODEL 不在 /v1/models 返回列表中",
                f"当前配置: {env['QWEN_LLM_MODEL']}",
            )
    except Exception as exc:
        status_line(False, "Chat 服务 /v1/models", str(exc))
        failures += 1

    try:
        response = llm_client.chat.completions.create(
            model=env["QWEN_LLM_MODEL"],
            messages=[{"role": "user", "content": "只回答 OK"}],
            temperature=0,
            timeout=args.timeout,
        )
        content = response.choices[0].message.content or ""
        status_line(True, "Chat 服务 /v1/chat/completions", content[:80])
    except Exception as exc:
        status_line(
            False,
            "Chat 服务 /v1/chat/completions",
            f"{exc}\n     处理: 确认 QWEN_BASE_URL、QWEN_LLM_MODEL，并确保聊天模型服务已启动。",
        )
        failures += 1

    embed_client = OpenAI(
        base_url=env["QWEN_EMBEDDING_BASE_URL"],
        api_key=env["QWEN_EMBEDDING_API_KEY"],
    )
    try:
        response = embed_client.embeddings.create(
            model=env["QWEN_EMBEDDING_MODEL"],
            input=["污水处理工程建设用地指标"],
            timeout=args.timeout,
        )
        dim = len(response.data[0].embedding)
        expected_dim = int(env["QWEN_EMBEDDING_DIM"])
        ok = dim == expected_dim
        status_line(
            ok,
            "Embedding 服务 /v1/embeddings",
            f"实际维度={dim}, QWEN_EMBEDDING_DIM={expected_dim}",
        )
        if not ok:
            failures += 1
    except Exception as exc:
        status_line(
            False,
            "Embedding 服务 /v1/embeddings",
            f"{exc}\n     处理: 启动 embedding 模型服务，或修正 QWEN_EMBEDDING_BASE_URL / QWEN_EMBEDDING_MODEL。",
        )
        failures += 1

    return failures


def print_install_help() -> None:
    print("\n常用安装/启动参考:")
    print("1. Python 依赖:")
    print("   pip install openai python-dotenv numpy")
    print("   cd RAG-Anything")
    print("   pip install -e .")
    print("   pip install -U \"mineru[core]\"")
    print("   pip install modelscope")
    print("\n2. vLLM 单独启动 embedding 服务示例:")
    print(
        "   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-Embedding-0.6B --task embed --host 127.0.0.1 --port 8001"
    )
    print("   .env:")
    print("   QWEN_EMBEDDING_BASE_URL=http://localhost:8001/v1")
    print("   QWEN_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B")
    print("   QWEN_EMBEDDING_DIM=1024")
    print("\n3. 如果使用 Ollama embedding:")
    print("   ollama pull embeddinggemma")
    print("   QWEN_EMBEDDING_BASE_URL=http://localhost:11434/v1")
    print("   QWEN_EMBEDDING_API_KEY=ollama")
    print("   QWEN_EMBEDDING_MODEL=embeddinggemma")
    print("   QWEN_EMBEDDING_DIM=768")
    print("\n4. PostgreSQL 索引存储依赖:")
    print("   pip install asyncpg pgvector")
    print("   CREATE DATABASE rag_anything;")
    print("   \\c rag_anything")
    print("   CREATE EXTENSION IF NOT EXISTS vector;")
    print("   .env:")
    print("   QWEN_STORAGE=postgres")
    print("   POSTGRES_HOST=localhost")
    print("   POSTGRES_PORT=5432")
    print("   POSTGRES_USER=postgres")
    print("   POSTGRES_PASSWORD=你的密码")
    print("   POSTGRES_DATABASE=rag_anything")
    print("   POSTGRES_WORKSPACE=qwen_policy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 Qwen 中文政策导入前的本地环境、模型服务和索引目录。"
    )
    parser.add_argument("--file", help="可选：要导入的文件路径，用于检查文件是否存在。")
    parser.add_argument(
        "--working-dir",
        default=str(RAGANYTHING_ROOT / "rag_storage" / "qwen_policy_text"),
        help="计划使用的 RAG 工作目录；local 模式会写入索引，postgres 模式用于缓存/运行文件。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RAGANYTHING_ROOT / "output" / "qwen_policy_parser"),
        help="计划写入的解析输出目录。",
    )
    parser.add_argument(
        "--storage",
        default=None,
        choices=["local", "postgres", "pg", "postgresql", "postgres-age", "pg-age"],
        help=(
            "索引存储后端。postgres=PostgreSQL KV/向量/状态 + NetworkX 图；"
            "postgres-age=全 PostgreSQL，需安装 Apache AGE。默认读取 QWEN_STORAGE。"
        ),
    )
    parser.add_argument(
        "--model-source",
        choices=["huggingface", "modelscope", "local"],
        default="modelscope",
        help="MinerU 模型来源，用于判断是否需要 modelscope 包。",
    )
    parser.add_argument("--timeout", type=int, default=30, help="模型接口测试超时秒数。")
    parser.add_argument(
        "--skip-service",
        action="store_true",
        help="只检查本地 Python/路径/MinerU，不请求模型服务。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    storage = normalize_storage(args.storage)
    print("Qwen 导入预检")
    print("=" * 60)
    print(f"索引存储后端: {storage}")

    failures = 0
    failures += check_modules(args.model_source, storage)
    failures += check_mineru_command()
    failures += check_paths(args)
    env_failures, env = check_env(args)
    failures += env_failures
    if storage == "postgres":
        failures += check_postgres(args)

    if not args.skip_service and env_failures == 0:
        failures += check_openai_services(args, env)

    print("=" * 60)
    if failures:
        print(f"预检未通过：{failures} 项需要处理。")
        print_install_help()
        return 1

    print("预检通过：可以开始执行 import_policy_files.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
