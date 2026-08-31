#!/usr/bin/env python3
"""Convert GEditBench-v2 subtask ELO HTML to Markdown.

The output structure can follow an existing Markdown template, but every
model-specific title and every data table is generated from --model and --html.
"""

from __future__ import annotations

import argparse
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


DEFAULT_TEMPLATE_NAME = "tmp_elo_subtask_flux2_klein_9b_table.md"


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


def model_display_name(model: str) -> str:
    names = {
        "FLUX2_klein_9b": "FLUX2 Klein 9B",
        "FireRed_Image_Edit": "FireRed Image Edit",
        "FireRed_Image_Edit_1p1": "FireRed Image Edit 1.1",
        "Qwen_Image_Edit_2511": "Qwen Image Edit 2511",
        "LongCat_Image_Edit": "LongCat Image Edit",
    }
    return names.get(model, model.replace("_", " "))


def rank_display_name(model: str) -> str:
    if model.startswith("FLUX"):
        return "FLUX"
    if model.startswith("FireRed"):
        return "FireRed"
    if model.startswith("Qwen"):
        return "Qwen"
    if model.startswith("LongCat"):
        return "LongCat"
    return model_display_name(model)


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


def make_summary_section(rows: list[dict[str, object]]) -> list[str]:
    best_tasks = [str(row["Subtask"]) for row in rows if int(row["Rank"]) == 1]
    top3_tasks = [str(row["Subtask"]) for row in rows if int(row["Rank"]) <= 3]
    return [
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
    ]


def make_rank_section(rows: list[dict[str, object]]) -> list[str]:
    rank_counts = Counter(int(row["Rank"]) for row in rows)
    lines = ["", "| Rank | Count |", "| --- | --- |"]
    for rank in sorted(rank_counts):
        lines.append(f"| {rank} | {rank_counts[rank]} |")
    lines.append("")
    return lines


def make_best_tasks_section(rows: list[dict[str, object]]) -> list[str]:
    best_tasks = [str(row["Subtask"]) for row in rows if int(row["Rank"]) == 1]
    return ["", ", ".join(best_tasks) if best_tasks else "None", ""]


def make_subtask_table(rows: list[dict[str, object]]) -> list[str]:
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
    aligns = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---", "---:", "---:"]
    lines = [
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in sorted(rows, key=lambda item: str(item["Subtask"])):
        values = []
        for header in headers:
            value = row[header]
            if header in {"Subtask", "Best Model"}:
                value = f"`{value}`"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def make_sorted_rank_table(rows: list[dict[str, object]]) -> list[str]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            int(row["Rank"]),
            -int(row["Overall"]),
            str(row["Subtask"]),
        ),
    )
    lines = [
        "",
        "| Rank | Subtask | Overall | VC | VQ | IF | Gap to Best |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Rank"]),
                    f"`{row['Subtask']}`",
                    str(row["Overall"]),
                    str(row["VC"]),
                    str(row["VQ"]),
                    str(row["IF"]),
                    str(row["Gap"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def replace_title(lines: list[str], model: str) -> list[str]:
    if lines and lines[0].startswith("# "):
        lines[0] = f"# GEditBench-v2 Subtask ELO Report: {model}"
    return lines


def replace_section(lines: list[str], heading: str, body: list[str]) -> list[str]:
    try:
        start = lines.index(heading)
    except ValueError:
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend([heading, *body])
        return lines

    end = start + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    return lines[: start + 1] + body + lines[end:]


def replace_matching_section(
    lines: list[str], old_prefix: str, old_suffix: str, new_heading: str, body: list[str]
) -> list[str]:
    start = None
    for index, line in enumerate(lines):
        if line.startswith(old_prefix) and line.endswith(old_suffix):
            start = index
            break
    if start is None:
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend([new_heading, *body])
        return lines

    end = start + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    return lines[:start] + [new_heading, *body] + lines[end:]


def fallback_markdown(rows: list[dict[str, object]], model: str) -> str:
    subtask_heading = f"## {model_display_name(model)} By Subtask"
    sorted_heading = f"## Sorted By {rank_display_name(model)} Rank"
    lines = [
        f"# GEditBench-v2 Subtask ELO Report: {model}",
        "",
        "## Summary",
        *make_summary_section(rows),
        "## Rank Distribution",
        *make_rank_section(rows),
        "## Best-Performing Subtasks",
        *make_best_tasks_section(rows),
        subtask_heading,
        *make_subtask_table(rows),
        sorted_heading,
        *make_sorted_rank_table(rows),
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(rows: list[dict[str, object]], model: str, template_path: Path | None) -> str:
    if template_path is None or not template_path.is_file():
        return fallback_markdown(rows, model)

    lines = template_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = replace_title(lines, model)
    lines = replace_section(lines, "## Summary", make_summary_section(rows))
    lines = replace_section(lines, "## Rank Distribution", make_rank_section(rows))
    lines = replace_section(lines, "## Best-Performing Subtasks", make_best_tasks_section(rows))
    lines = replace_matching_section(
        lines,
        "## ",
        " By Subtask",
        f"## {model_display_name(model)} By Subtask",
        make_subtask_table(rows),
    )
    lines = replace_matching_section(
        lines,
        "## Sorted By ",
        " Rank",
        f"## Sorted By {rank_display_name(model)} Rank",
        make_sorted_rank_table(rows),
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--template-md",
        type=Path,
        default=None,
        help=(
            "Markdown template path. Default: tmp_elo_subtask_flux2_klein_9b_table.md "
            "next to the input HTML."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    html_path = args.html.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else html_path.with_suffix(".md")
    template_path = (
        args.template_md.expanduser().resolve()
        if args.template_md
        else html_path.parent / DEFAULT_TEMPLATE_NAME
    )

    rows = build_summary_rows(html_path, args.model)
    output_path.write_text(render_markdown(rows, args.model, template_path), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
