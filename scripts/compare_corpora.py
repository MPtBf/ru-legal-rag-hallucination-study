"""Validate the project corpus and compare it with a RuLawRetrieval corpus file.

This script is deliberately offline: obtain the source files separately and pass
their paths explicitly. It does not download, modify, or replace any corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL = PROJECT_ROOT / "data" / "corpus" / "tk_rf_articles.jsonl"
REQUIRED_FIELDS = {"_id", "title", "text", "code"}


class InputError(Exception):
    """An input file is malformed or does not match the expected schema."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise InputError(f"{path}:{line_number}: blank JSONL line")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise InputError(f"{path}:{line_number}: each row must be a JSON object")
                missing = REQUIRED_FIELDS - row.keys()
                if missing:
                    raise InputError(
                        f"{path}:{line_number}: missing fields: {', '.join(sorted(missing))}"
                    )
                for field in REQUIRED_FIELDS:
                    value = row[field]
                    if not isinstance(value, str) or not value.strip():
                        raise InputError(
                            f"{path}:{line_number}: {field} must be a non-empty string"
                        )
                rows.append(row)
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    return rows


def read_sources(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read sources metadata {path}: {exc}") from exc
    if not isinstance(value, list):
        raise InputError(f"{path}: expected a JSON array")
    matches = [row for row in value if isinstance(row, dict) and row.get("code") == code]
    if len(matches) != 1:
        raise InputError(f"{path}: expected exactly one source entry for code={code!r}, found {len(matches)}")
    return matches[0]


def duplicates(rows: list[dict[str, Any]], field: str) -> list[str]:
    counts = Counter(row[field] for row in rows)
    return sorted(value for value, count in counts.items() if count > 1)


def compare(local: list[dict[str, Any]], source: list[dict[str, Any]], code: str) -> dict[str, Any]:
    local = [row for row in local if row["code"] == code]
    source = [row for row in source if row["code"] == code]
    local_by_id = {row["_id"]: row for row in local}
    source_by_id = {row["_id"]: row for row in source}
    local_ids, source_ids = set(local_by_id), set(source_by_id)

    changed = []
    for article_id in sorted(local_ids & source_ids):
        fields = [
            field for field in ("title", "text")
            if local_by_id[article_id][field] != source_by_id[article_id][field]
        ]
        if fields:
            changed.append({"_id": article_id, "different_fields": fields})

    return {
        "code": code,
        "local_count": len(local),
        "source_count": len(source),
        "local_duplicate_ids": duplicates(local, "_id"),
        "source_duplicate_ids": duplicates(source, "_id"),
        "only_in_local": sorted(local_ids - source_ids),
        "only_in_source": sorted(source_ids - local_ids),
        "content_differences": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-corpus", type=Path, help="RuLawRetrieval corpus/corpus.jsonl")
    parser.add_argument("--sources", type=Path, help="RuLawRetrieval corpus/sources.json")
    parser.add_argument("--local-corpus", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--code", default="tk", help="document code to compare (default: tk)")
    parser.add_argument("--revision", help="source repository revision used to obtain the source files")
    parser.add_argument("--report", type=Path, help="optional path for a JSON report")
    args = parser.parse_args()

    if bool(args.source_corpus) != bool(args.sources):
        parser.error("--source-corpus and --sources must be supplied together")

    try:
        local_rows = read_jsonl(args.local_corpus)
        local_duplicates = {
            field: duplicates([row for row in local_rows if row["code"] == args.code], field)
            for field in ("_id", "title")
        }
        report: dict[str, Any] = {
            "local": {
                "path": str(args.local_corpus),
                "sha256": sha256(args.local_corpus),
                "row_count": len(local_rows),
                "code_counts": dict(sorted(Counter(row["code"] for row in local_rows).items())),
                "duplicate_ids_for_code": local_duplicates["_id"],
                "duplicate_titles_for_code": local_duplicates["title"],
            },
            "source_revision": args.revision,
        }
        failed = bool(local_duplicates["_id"] or local_duplicates["title"])

        if args.source_corpus:
            source_rows = read_jsonl(args.source_corpus)
            source_entry = read_sources(args.sources, args.code)
            report["source"] = {
                "corpus_path": str(args.source_corpus),
                "corpus_sha256": sha256(args.source_corpus),
                "sources_path": str(args.sources),
                "sources_sha256": sha256(args.sources),
                "metadata_entry": source_entry,
            }
            report["comparison"] = compare(local_rows, source_rows, args.code)
            result = report["comparison"]
            failed = failed or any(
                result[key]
                for key in (
                    "local_duplicate_ids", "source_duplicate_ids", "only_in_local",
                    "only_in_source", "content_differences",
                )
            )
        else:
            report["comparison"] = None

        output = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        print(output)
        if args.report:
            args.report.write_text(output + "\n", encoding="utf-8")
        return 1 if failed else 0
    except InputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
