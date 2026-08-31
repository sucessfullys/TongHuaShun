#!/usr/bin/env python3
"""Convert GEditBench-v2 subtask ELO HTML to the same Markdown report format."""

from __future__ import annotations

import argparse
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


class ELOHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[tuple[str, list[list[str]]]] = []
        self._heading = ""
        self._in_heading = False
        self._heading_parts: list[str] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._table: list[list[str]] = []
        self._row: list[str] = []
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"h1", "h2", "h3"}:
            self._in_heading = True
            self._heading_parts = []
        elif tag == "table":
            self._in_table = True
            self._table = []
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._row = []
        elif self._in_row and tag in {"th", "td"}:
            self._in_cell = True
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2", "h3"} and self._in_heading:
            heading = clean_text(" ".join(self._heading_parts))
            if heading:
                self._heading = heading
            self._in_heading = False
        elif tag in {"th", "td"} and self._in_cell:
            self._row.append(clean_text(" ".join(self._cell_parts)))
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if any(self._row):
                self._table.append(self._row)
            self._in_row = False
        elif tag == "table" and self._in_table:
            if self._table:
                self.tables.append((self._heading, self._table))
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)
        elif self._in_heading:
            self._heading_parts.append(data)


def clean_text(value: str) -> str:
    return " ".join(unescape(value).replace("\xa0", " ").split())


def parse_int(value: str) -> int:
    text = clean_text(value).replace(",", "")
    if text in {"", "-", "nan", "None"}:
        return 0
    return int(float(text))


def parse_tables(html_path: Path) -> list[tuple[str, list[dict[str, str]]]]:
    parser = ELOHTMLParser()
    parser.feed(html_path.read_text(encoding="utf-8", errors="ignore"))

    tables: list[tuple[str, list[dict[str, str]]]] = []
    for heading, rows in parser.tables:
        if len(rows) < 2:
            continue
        headers = rows[0]
        records = []
        for row in rows[1:]:
            padded = row + [""] * max(0, len(headers) - len(row))
            records.append(dict(zip(headers, padded)))
        tables.append((heading, records))
    return tables


def build_summary_rows(html_path: Path, model: str) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for subtask, records in parse_tables(html_path):
        model_rows = [row for row in records if row.get("Model") == model]
        if not model_rows:
            continue

        model_row = model_rows[0]
        best_row = min(records, key=lambda row: parse_int(row.get("Rank", "999999")))
        overall = parse_int(model_row.get("Overall ELO", "0"))
        best_overall = parse_int(best_row.get("Overall ELO", "0"))

        summary_rows.append(
            {
                "Subtask": subtask,
                "Samples": parse_int(model_row.get("Samples", "0")),
                "VC": parse_int(model_row.get("VC ELO", "0")),
                "VQ": parse_int(model_row.get("VQ ELO", "0")),
                "IF": parse_int(model_row.get("IF ELO", "0")),
                "Overall": overall,
                "Rank": parse_int(model_row.get("Rank", "0")),
                "Best Model": best_row.get("Model", ""),
                "Best Overall": best_overall,
                "Gap": overall - best_overall,
            }
        )

    if not summary_rows:
        raise ValueError(f"Model {model!r} was not found in {html_path}")
    return summary_rows


def mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def format_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "Subtask",
        "Samples",
        "VC",
        "VQ",
        "IF",
        "Overall",
        "Rank",
        "Best Model",
        "Best Overall",
        "Gap",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    return "\n".join(lines)


def write_markdown(rows: list[dict[str, object]], model: str, output_path: Path) -> None:
    rows = sorted(rows, key=lambda row: str(row["Subtask"]))
    rank_counts = Counter(int(row["Rank"]) for row in rows)
    best_tasks = [str(row["Subtask"]) for row in rows if int(row["Rank"]) == 1]
    top3_tasks = [str(row["Subtask"]) for row in rows if int(row["Rank"]) <= 3]

    lines = [
        f"# GEditBench-v2 Subtask ELO Report: {model}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Subtasks | {len(rows)} |",
        f"| Mean VC ELO | {mean(rows, 'VC'):.1f} |",
        f"| Mean VQ ELO | {mean(rows, 'VQ'):.1f} |",
        f"| Mean IF ELO | {mean(rows, 'IF'):.1f} |",
        f"| Mean Overall ELO | {mean(rows, 'Overall'):.1f} |",
        f"| Rank-1 Subtasks | {len(best_tasks)} |",
        f"| Top-3 Subtasks | {len(top3_tasks)} |",
        "",
        "## Rank Distribution",
        "",
        "| Rank | Count |",
        "| --- | --- |",
    ]
    for rank in sorted(rank_counts):
        lines.append(f"| {rank} | {rank_counts[rank]} |")

    lines.extend(
        [
            "",
            "## Best-Performing Subtasks",
            "",
            ", ".join(best_tasks) if best_tasks else "None",
            "",
            "## Subtask Table",
            "",
            format_table(rows),
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    html_path = args.html.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else html_path.with_suffix(".md")

    rows = build_summary_rows(html_path, args.model)
    write_markdown(rows, args.model, output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
