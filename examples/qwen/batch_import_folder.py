#!/usr/bin/env python
"""Batch import policy files with scan, deduplication and resumable execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from qwen_policy_common import DEFAULT_OUTPUT_DIR, DEFAULT_WORKING_DIR


DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".wps",
    ".txt",
}
GIS_OR_SYSTEM_EXTENSIONS = {
    ".gdbtable",
    ".gdbtablx",
    ".gdbindexes",
    ".atx",
    ".spx",
    ".freelist",
    ".lock",
    ".ini",
}
OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".wps"}
PRESENTATION_EXTENSIONS = {".ppt", ".pptx"}
DEFAULT_CORE_DIRS = [
    "002相关政策/高标准农",
    "002相关政策/风电光伏",
    "002相关政策/00相关用地标准",
    "002相关政策/用地报批",
]


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            return data
    except Exception:
        pass
    return {"files": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rel_path(path: Path) -> str:
    return path.as_posix()


def get_pdf_page_count(path: Path) -> int | None:
    if path.suffix.lower() != ".pdf":
        return None
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def classify_file(path: Path, root: Path, args: argparse.Namespace) -> dict:
    suffix = path.suffix.lower()
    size_bytes = path.stat().st_size
    relative_path = normalize_rel_path(path.relative_to(root))
    directory_path = normalize_rel_path(Path(relative_path).parent)
    directory_tags = [] if directory_path == "." else list(Path(directory_path).parts)
    business_category = directory_tags[-1] if directory_tags else root.name
    size_mb = size_bytes / 1024 / 1024
    page_count = get_pdf_page_count(path)
    is_large_by_size = size_mb >= args.large_file_mb
    is_large_by_pages = (
        page_count is not None
        and args.large_file_pages is not None
        and page_count >= args.large_file_pages
    )
    is_large_file = is_large_by_size or is_large_by_pages

    item = {
        "file": str(path.resolve()),
        "relative_path": relative_path,
        "directory_path": "" if directory_path == "." else directory_path,
        "directory_tags": directory_tags,
        "business_category": business_category,
        "source_root": str(root.resolve()),
        "source_collection": root.name,
        "raw_extension": suffix,
        "normalized_extension": suffix,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "page_count": page_count,
        "is_large_file": is_large_file,
        "is_large_by_size": is_large_by_size,
        "is_large_by_pages": is_large_by_pages,
        "needs_office_convert": suffix in OFFICE_EXTENSIONS,
        "is_presentation": suffix in PRESENTATION_EXTENSIONS,
        "is_gis_asset": suffix in GIS_OR_SYSTEM_EXTENSIONS,
        "import_strategy": "skip",
        "skip_reason": None,
        "eligible": False,
    }

    if not suffix:
        item["skip_reason"] = "empty_extension"
    elif suffix in GIS_OR_SYSTEM_EXTENSIONS:
        item["skip_reason"] = "gis_or_system_asset"
    elif suffix not in DOCUMENT_EXTENSIONS:
        item["skip_reason"] = "unsupported_extension"
    elif item["is_large_file"] and args.large_file_policy == "defer":
        item["import_strategy"] = "parse_large_file"
        item["skip_reason"] = "large_file_deferred"
    elif not item["is_large_file"] and args.large_file_policy == "only":
        item["import_strategy"] = "parse_normal"
        item["skip_reason"] = "not_large_file"
    else:
        item["eligible"] = True
        item["import_strategy"] = choose_import_strategy(suffix, item["is_large_file"])

    if item["eligible"] or args.hash_skipped:
        item["file_hash"] = file_sha256(path)
    else:
        item["file_hash"] = None
    item["doc_id_prefix"] = f"qwen-{(item['file_hash'] or hashlib.md5(str(path).encode()).hexdigest())[:16]}"
    return item


def choose_import_strategy(suffix: str, is_large_file: bool) -> str:
    if suffix == ".txt":
        return "direct_text"
    if suffix in OFFICE_EXTENSIONS:
        return "convert_then_parse"
    if is_large_file:
        return "parse_large_file"
    return "parse_normal"


def scan_folder(root: Path, args: argparse.Namespace) -> list[dict]:
    iterator = root.rglob("*") if args.recursive else root.glob("*")
    items = []
    for path in iterator:
        if not path.is_file():
            continue
        items.append(classify_file(path, root, args))
    return sorted(items, key=lambda item: item["relative_path"])


def apply_hash_dedup(items: list[dict]) -> None:
    first_by_hash: dict[str, dict] = {}
    for item in items:
        file_hash = item.get("file_hash")
        if not item["eligible"] or not file_hash:
            continue
        first = first_by_hash.get(file_hash)
        if first is None:
            first_by_hash[file_hash] = item
            item["duplicate_of"] = None
            item["source_paths"] = [item["relative_path"]]
            continue

        first.setdefault("source_paths", [first["relative_path"]]).append(item["relative_path"])
        item["eligible"] = False
        item["duplicate_of"] = first["relative_path"]
        item["skip_reason"] = "duplicate_file_hash"
        item["import_strategy"] = "duplicate"


def priority_score(item: dict, args: argparse.Namespace) -> tuple[int, int, int, int, str]:
    relative_path = item["relative_path"]
    core_rank = 1
    if args.prioritize_core_dirs:
        normalized = relative_path.replace("\\", "/")
        for index, core_dir in enumerate(args.core_dir):
            if core_dir.replace("\\", "/") in normalized:
                core_rank = 0
                break
        else:
            index = len(args.core_dir)
    else:
        index = 0

    strategy_rank = {
        "direct_text": 0,
        "parse_normal": 1,
        "convert_then_parse": 2,
        "parse_large_file": 3,
    }.get(item["import_strategy"], 9)
    return core_rank, index, strategy_rank, item["size_bytes"], relative_path


def summarize_items(items: list[dict]) -> dict:
    by_extension: dict[str, dict] = {}
    by_directory: dict[str, dict] = {}
    summary = {
        "total_files": len(items),
        "total_size_mb": round(sum(item["size_bytes"] for item in items) / 1024 / 1024, 2),
        "eligible_files": sum(1 for item in items if item["eligible"]),
        "skipped_files": sum(1 for item in items if not item["eligible"]),
        "duplicate_files": sum(1 for item in items if item.get("skip_reason") == "duplicate_file_hash"),
        "large_files": sum(1 for item in items if item["is_large_file"]),
        "large_by_size_files": sum(1 for item in items if item.get("is_large_by_size")),
        "large_by_pages_files": sum(1 for item in items if item.get("is_large_by_pages")),
        "office_files": sum(1 for item in items if item["needs_office_convert"]),
        "gis_assets": sum(1 for item in items if item["is_gis_asset"]),
        "by_extension": by_extension,
        "by_directory": by_directory,
    }
    for item in items:
        ext = item["normalized_extension"] or "<none>"
        ext_row = by_extension.setdefault(ext, {"count": 0, "size_mb": 0.0})
        ext_row["count"] += 1
        ext_row["size_mb"] = round(ext_row["size_mb"] + item["size_bytes"] / 1024 / 1024, 2)

        directory = item["directory_path"] or "."
        dir_row = by_directory.setdefault(directory, {"count": 0, "eligible": 0, "size_mb": 0.0})
        dir_row["count"] += 1
        dir_row["eligible"] += 1 if item["eligible"] else 0
        dir_row["size_mb"] = round(dir_row["size_mb"] + item["size_bytes"] / 1024 / 1024, 2)
    return summary


def write_report(path: Path, manifest: dict) -> None:
    summary = manifest.get("scan_summary", {})
    import_summary = manifest.get("import_summary", {})
    files = manifest.get("files", {})
    records = list(files.values())
    failed = [
        record for record in records
        if record.get("status") in {"failed", "parse_failed", "embedding_failed"}
    ]
    skipped = [record for record in records if record.get("status") == "skipped"]
    largest = sorted(records, key=lambda record: record.get("size_bytes") or 0, reverse=True)[:10]
    skip_reasons: dict[str, int] = {}
    for record in skipped:
        reason = record.get("skip_reason") or "unknown"
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    lines = [
        "# 批量导入报告",
        "",
        f"- 更新时间: {manifest.get('updated_at')}",
        f"- 导入目录: `{manifest.get('folder')}`",
        f"- 工作目录: `{manifest.get('working_dir')}`",
        f"- 文件总数: {summary.get('total_files', 0)}",
        f"- 可入库文件数: {summary.get('eligible_files', 0)}",
        f"- 跳过文件数: {summary.get('skipped_files', 0)}",
        f"- 重复文件数: {summary.get('duplicate_files', 0)}",
        f"- 大文件数: {summary.get('large_files', 0)}",
        f"- 成功导入: {import_summary.get('success', 0)}",
        f"- 本次跳过: {import_summary.get('skipped', 0)}",
        f"- 失败: {import_summary.get('failed', 0)}",
        "",
        "## 文件类型统计",
        "",
        "| 扩展名 | 数量 | 大小 MB |",
        "| --- | ---: | ---: |",
    ]
    for ext, row in sorted(summary.get("by_extension", {}).items()):
        lines.append(f"| `{ext}` | {row['count']} | {row['size_mb']} |")

    lines.extend([
        "",
        "## 目录统计 Top 30",
        "",
        "| 目录 | 文件数 | 可入库 | 大小 MB |",
        "| --- | ---: | ---: | ---: |",
    ])
    by_directory = summary.get("by_directory", {})
    top_dirs = sorted(by_directory.items(), key=lambda item: item[1]["size_mb"], reverse=True)[:30]
    for directory, row in top_dirs:
        lines.append(f"| `{directory}` | {row['count']} | {row['eligible']} | {row['size_mb']} |")

    lines.extend([
        "",
        "## 最大文件 Top 10",
        "",
        "| 文件 | 大小 MB | 策略 |",
        "| --- | ---: | --- |",
    ])
    if largest:
        for record in largest:
            lines.append(
                f"| `{record.get('relative_path') or record.get('file')}` | "
                f"{record.get('size_mb', 0)} | `{record.get('import_strategy')}` |"
            )
    else:
        lines.append("| 无 | 0 | - |")

    lines.extend([
        "",
        "## 跳过原因统计",
        "",
        "| 原因 | 数量 |",
        "| --- | ---: |",
    ])
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("| 无 | 0 |")

    lines.extend(["", "## 失败文件", ""])
    if failed:
        for record in failed[:50]:
            lines.append(
                f"- `{record.get('relative_path') or record.get('file')}`: "
                f"{record.get('stage')} returncode={record.get('returncode')}"
            )
    else:
        lines.append("- 无")

    lines.extend(["", "## 跳过文件 Top 50", ""])
    if skipped:
        for record in skipped[:50]:
            lines.append(
                f"- `{record.get('relative_path') or record.get('file')}`: {record.get('skip_reason')}"
            )
    else:
        lines.append("- 无")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_metadata_for_import(file_item: dict) -> dict:
    keys = [
        "source_root",
        "source_collection",
        "relative_path",
        "directory_path",
        "directory_tags",
        "business_category",
        "raw_extension",
        "normalized_extension",
        "file_hash",
        "size_bytes",
        "size_mb",
        "page_count",
        "import_strategy",
        "is_large_file",
        "is_large_by_size",
        "is_large_by_pages",
        "needs_office_convert",
        "is_presentation",
        "source_paths",
    ]
    return {key: file_item[key] for key in keys if key in file_item}


def build_import_command(args: argparse.Namespace, file_item: dict) -> list[str]:
    script_path = Path(__file__).with_name("import_policy_files.py")
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(script_path),
        file_item["file"],
        "--storage",
        args.storage,
        "--working-dir",
        args.working_dir,
        "--output-dir",
        args.output_dir,
        "--parser",
        args.parser,
        "--parse-method",
        args.parse_method,
        "--ocr-lang",
        args.ocr_lang,
        "--index-mode",
        args.index_mode,
        "--chunk-mode",
        args.chunk_mode,
    ]

    if args.model_source:
        command.extend(["--model-source", args.model_source])
    if args.mineru_timeout is not None:
        command.extend(["--mineru-timeout", str(args.mineru_timeout)])
    if args.mineru_device:
        command.extend(["--mineru-device", args.mineru_device])
    if not args.mineru_resilient_pages:
        command.append("--no-mineru-resilient-pages")
    command.extend(["--mineru-page-window", str(args.mineru_page_window)])
    command.extend(["--mineru-min-page-window", str(args.mineru_min_page_window)])
    command.extend([
        "--mineru-segment-large-pages",
        str(args.mineru_segment_large_pages),
    ])
    if not args.no_stable_doc_id:
        command.extend(["--doc-id-prefix", file_item["doc_id_prefix"]])
    command.extend([
        "--source-metadata-json",
        json.dumps(source_metadata_for_import(file_item), ensure_ascii=False),
    ])
    if not args.retrieval_only:
        command.append("--no-retrieval-only")
    if args.enable_images:
        command.append("--enable-images")
    if args.disable_tables:
        command.append("--disable-tables")
    if args.disable_equations:
        command.append("--disable-equations")
    if args.no_stats:
        command.append("--no-stats")

    return command


def run_batch(args: argparse.Namespace) -> None:
    if args.large_file_pages is not None and args.large_file_pages <= 0:
        args.large_file_pages = None

    root = Path(args.folder).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"批量导入目录不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"批量导入路径不是目录: {root}")

    manifest_path = Path(args.manifest) if args.manifest else Path(args.working_dir) / "batch_import_manifest.json"
    report_path = Path(args.report) if args.report else Path(args.working_dir) / "batch_import_report.md"
    manifest = load_manifest(manifest_path)

    print("扫描并分类文件...", flush=True)
    items = scan_folder(root, args)
    apply_hash_dedup(items)
    if not items:
        raise RuntimeError(f"目录中没有找到可导入文件: {root}")

    eligible_items = [item for item in items if item["eligible"]]
    eligible_items = sorted(eligible_items, key=lambda item: priority_score(item, args))
    if args.limit:
        eligible_items = eligible_items[: args.limit]

    existing_records = manifest.get("files", {})
    file_records = {}
    for item in items:
        previous = existing_records.get(item["file"], {})
        status = previous.get("status")
        if not item["eligible"] and status != "success":
            status = "skipped"
        file_records[item["file"]] = {
            **previous,
            **item,
            "status": status or "pending",
            "stage": previous.get("stage") or ("skipped" if not item["eligible"] else "queued_parse"),
        }

    manifest.update(
        {
            "folder": str(root),
            "recursive": args.recursive,
            "large_file_mb": args.large_file_mb,
            "large_file_policy": args.large_file_policy,
            "storage": args.storage,
            "working_dir": args.working_dir,
            "output_dir": args.output_dir,
            "chunk_mode": args.chunk_mode,
            "scan_summary": summarize_items(items),
            "files": file_records,
            "updated_at": iso_now(),
        }
    )
    save_manifest(manifest_path, manifest)
    write_report(report_path, manifest)

    print("Qwen 中文政策文件批量导入")
    print(f"目录: {root}")
    print(f"扫描文件数: {len(items)}")
    print(f"可导入文件数: {len([item for item in items if item['eligible']])}")
    print(f"本次计划导入: {len(eligible_items)}")
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    print(f"chunk_mode: {args.chunk_mode}")
    print(f"storage: {args.storage}")
    print(f"working_dir: {args.working_dir}")
    page_threshold = args.large_file_pages if args.large_file_pages is not None else "off"
    print(
        f"large_file_policy: {args.large_file_policy} "
        f"({args.large_file_mb} MB, {page_threshold} pages)"
    )

    if args.scan_only:
        print("已完成扫描预览，未执行导入。")
        return

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for index, file_item in enumerate(eligible_items, start=1):
        key = file_item["file"]
        record = manifest["files"].get(key, {})
        if record.get("status") == "success" and not args.force:
            skipped_count += 1
            print(f"\n[{index}/{len(eligible_items)}] 跳过已成功文件: {Path(key).name}")
            continue

        retry_count = int(record.get("retry_count") or 0)
        if record.get("status") == "failed" and retry_count >= args.retry_limit and not args.force:
            skipped_count += 1
            manifest["files"][key].update({"status": "skipped", "skip_reason": "retry_limit_reached"})
            print(f"\n[{index}/{len(eligible_items)}] 跳过超过重试次数文件: {Path(key).name}")
            continue

        command = build_import_command(args, file_item)
        started = time.monotonic()
        manifest["files"][key] = {
            **record,
            **file_item,
            "status": "running",
            "stage": "parsing_embedding",
            "started_at": iso_now(),
            "retry_count": retry_count + (1 if record.get("status") == "failed" else 0),
            "command": command,
        }
        save_manifest(manifest_path, manifest)

        print(
        f"\n[{index}/{len(eligible_items)}] 开始导入: "
            f"{file_item['relative_path']} "
            f"({file_item['size_mb']} MB, "
            f"{file_item.get('page_count') or '-'} pages, "
            f"{file_item['import_strategy']})",
            flush=True,
        )
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2])
        duration_ms = int((time.monotonic() - started) * 1000)

        if result.returncode == 0:
            success_count += 1
            manifest["files"][key].update(
                {
                    "status": "success",
                    "stage": "completed",
                    "finished_at": iso_now(),
                    "duration_ms": duration_ms,
                    "returncode": result.returncode,
                }
            )
            print(f"[{index}/{len(eligible_items)}] 导入成功: {Path(key).name}", flush=True)
        else:
            failed_count += 1
            manifest["files"][key].update(
                {
                    "status": "failed",
                    "stage": "failed",
                    "finished_at": iso_now(),
                    "duration_ms": duration_ms,
                    "returncode": result.returncode,
                }
            )
            print(
                f"[{index}/{len(eligible_items)}] 导入失败: {Path(key).name}, returncode={result.returncode}",
                flush=True,
            )
            save_manifest(manifest_path, manifest)
            write_report(report_path, manifest)
            if not args.continue_on_error:
                raise RuntimeError(f"批量导入中断，失败文件: {key}")

        manifest["import_summary"] = {
            "success": success_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "planned": len(eligible_items),
        }
        manifest["updated_at"] = iso_now()
        save_manifest(manifest_path, manifest)
        write_report(report_path, manifest)

    manifest["import_summary"] = {
        "success": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "planned": len(eligible_items),
    }
    manifest["updated_at"] = iso_now()
    save_manifest(manifest_path, manifest)
    write_report(report_path, manifest)

    print("\n批量导入完成")
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {failed_count}")
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按文件夹批量导入中文政策文件；逐文件执行，支持失败后续跑。"
    )
    parser.add_argument("folder", help="要批量导入的文件夹。")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录。")
    parser.add_argument("--scan-only", action="store_true", help="只生成扫描清单和报告，不执行导入。")
    parser.add_argument(
        "--storage",
        default=os.getenv("QWEN_STORAGE", "postgres"),
        choices=["local", "postgres", "pg", "postgresql", "postgres-age", "pg-age"],
        help="索引存储后端。",
    )
    parser.add_argument(
        "--working-dir",
        default=DEFAULT_WORKING_DIR,
        help="RAG 工作目录；postgres 模式用于缓存/运行文件。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="MinerU/解析器输出目录。",
    )
    parser.add_argument(
        "--parser",
        default="mineru",
        choices=["mineru", "docling", "paddleocr"],
        help="RAG-Anything 文档解析器。",
    )
    parser.add_argument(
        "--parse-method",
        default="auto",
        choices=["auto", "ocr", "txt"],
        help="解析模式。",
    )
    parser.add_argument("--ocr-lang", default="ch", help="OCR 语言。")
    parser.add_argument(
        "--model-source",
        default=None,
        choices=["huggingface", "modelscope", "local"],
        help="MinerU 模型来源。",
    )
    parser.add_argument("--mineru-timeout", type=int, default=None)
    parser.add_argument(
        "--mineru-device",
        default=os.getenv("MINERU_DEVICE") or os.getenv("QWEN_MINERU_DEVICE"),
        help="MinerU 推理设备，例如 cpu、cuda、cuda:0。扫描件建议显式传 cuda。",
    )
    parser.add_argument(
        "--mineru-resilient-pages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MinerU 解析失败时启用分段重试并跳过局部失败页段。",
    )
    parser.add_argument(
        "--mineru-page-window",
        type=int,
        default=int(os.getenv("QWEN_MINERU_PAGE_WINDOW", "100")),
        help="MinerU 分段容错解析的初始页段大小。",
    )
    parser.add_argument(
        "--mineru-min-page-window",
        type=int,
        default=int(os.getenv("QWEN_MINERU_MIN_PAGE_WINDOW", "5")),
        help="MinerU 分段容错解析的最小页段大小。",
    )
    parser.add_argument(
        "--mineru-segment-large-pages",
        type=int,
        default=int(os.getenv("QWEN_MINERU_SEGMENT_LARGE_PAGES", "300")),
        help="PDF 页数达到该阈值时直接分段解析；传 0 表示整本失败后再分段。",
    )
    parser.add_argument(
        "--index-mode",
        default="text",
        choices=["text", "full"],
        help="索引模式。",
    )
    parser.add_argument(
        "--chunk-mode",
        default="structured",
        choices=["auto", "structured"],
        help="切分方式；批量政策导入默认 structured。",
    )
    parser.add_argument("--enable-images", action="store_true")
    parser.add_argument(
        "--retrieval-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="结构化 chunk 默认只写入检索索引；传 --no-retrieval-only 可启用实体关系抽取。",
    )
    parser.add_argument("--disable-tables", action="store_true")
    parser.add_argument("--disable-equations", action="store_true")
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", help="单文件失败后继续。")
    parser.add_argument("--force", action="store_true", help="重新导入已成功文件。")
    parser.add_argument("--limit", type=int, default=None, help="只导入前 N 个文件，用于压测。")
    parser.add_argument("--retry-limit", type=int, default=2, help="失败文件自动重跑次数上限。")
    parser.add_argument(
        "--large-file-mb",
        type=float,
        default=20,
        help="大文件阈值，默认 20MB。",
    )
    parser.add_argument(
        "--large-file-pages",
        type=int,
        default=300,
        help="PDF 大文档页数阈值，默认 300 页；传 0 可关闭页数判断。",
    )
    parser.add_argument(
        "--large-file-policy",
        choices=["import", "defer", "only"],
        default="import",
        help="大文件策略：import=正常导入，defer=本批跳过，only=只导入大文件。",
    )
    parser.add_argument(
        "--prioritize-core-dirs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="优先导入 data-import-analysis.md 建议的核心政策目录。",
    )
    parser.add_argument(
        "--core-dir",
        action="append",
        default=DEFAULT_CORE_DIRS.copy(),
        help="核心目录片段，可重复传入。",
    )
    parser.add_argument("--hash-skipped", action="store_true", help="也为跳过文件计算 hash。")
    parser.add_argument(
        "--no-stable-doc-id",
        action="store_true",
        help="不使用基于文件 hash 的稳定 doc_id 前缀。",
    )
    parser.add_argument("--manifest", default=None, help="导入进度 manifest 路径。")
    parser.add_argument("--report", default=None, help="导入报告 markdown 路径。")
    return parser.parse_args()


if __name__ == "__main__":
    run_batch(parse_args())
