"""
Calculate ELO scores for each subtask using all 3 dimensions jointly.
Usage: python elo_subtask_score.py --result-files <jsonl_paths> --bootstrap 500
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from common_utils.elo_score import (
    calculate_joint_leaderboard,
    print_leaderboard,
    parse_paths,
    parse_csv_values,
    _parse_match_key,
)


def load_result_files(path_str: str):
    """Load all jsonl files and return combined list with source dimension."""
    all_data = []
    for path in parse_paths(path_str):
        path = path.strip()
        if path and os.path.exists(path):
            path_lower = path.lower()
            if "eval_vc" in path_lower or "_vc_" in path_lower:
                dim_name = "Visual Consistency"
            elif "eval_vq" in path_lower or "_vq_" in path_lower:
                dim_name = "Visual Quality"
            elif "eval_if" in path_lower or "_if_" in path_lower:
                dim_name = "Instruction Following"
            else:
                dim_name = "unknown"
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    record["_dimension"] = dim_name
                    all_data.append(record)
    return all_data


def _extract_subtask_type(cluster_id: str) -> str:
    """Extract subtask type from cluster_id, e.g. 'background_change_000014' -> 'background_change'"""
    parts = cluster_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 6:
        return parts[0]
    return cluster_id


def group_by_subtask(data_list):
    """Group pairwise comparisons by subtask TYPE, and split by dimension."""
    subtask_data = defaultdict(lambda: defaultdict(list))
    for item in data_list:
        try:
            cluster_id, _, _ = _parse_match_key(item["key"])
            subtask_type = _extract_subtask_type(cluster_id)
            dim_name = item.get("_dimension", "unknown")
            if dim_name == "unknown":
                continue
            subtask_data[subtask_type][dim_name].append(item)
        except Exception:
            continue
    return subtask_data


def _collect_samples_per_model(dim_data_list, model_names):
    """Count unique prompt clusters per model across all dimensions."""
    model_to_clusters = defaultdict(set)
    for data_list in dim_data_list:
        for item in data_list:
            try:
                cluster_id, model_a, model_b = _parse_match_key(item["key"])
            except Exception:
                continue
            model_to_clusters[model_a].add(cluster_id)
            model_to_clusters[model_b].add(cluster_id)
    return {m: len(model_to_clusters.get(m, set())) for m in model_names}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Calculate ELO scores for each subtask using all 3 dimensions jointly."
    )
    parser.add_argument(
        "--result-files",
        type=str,
        required=True,
        help="Comma-separated paths to JSONL files containing pairwise match results.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="Number of bootstrap samples for CI estimation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap.",
    )
    parser.add_argument(
        "--exclude-models",
        type=str,
        default=None,
        help="Comma-separated model names to exclude.",
    )
    parser.add_argument(
        "--table-output",
        type=str,
        default=None,
        help="Output path for HTML table (e.g. subtask_elo_table.html).",
    )
    return parser.parse_args()


def _export_subtask_tables_to_html(output_path: str, subtask_results: list) -> None:
    """Export per-subtask leaderboards with overall + per-dimension scores to HTML."""
    all_rows_html = []
    for subtask_name, overall_df, dim_tables, sample_counts in subtask_results:
        model_order = overall_df["Model"].tolist()

        # Build combined table: Model | Samples | VC ELO | VC CI | VQ ELO | VQ CI | IF ELO | IF CI | Overall ELO | Overall CI | Rank
        header = "<thead><tr><th>Model</th><th>Samples</th><th>VC ELO</th><th>VC 95% CI</th><th>VQ ELO</th><th>VQ 95% CI</th><th>IF ELO</th><th>IF 95% CI</th><th>Overall ELO</th><th>Overall 95% CI</th><th>Rank</th></tr></thead>"
        body_rows = []
        for rank, model in enumerate(model_order, 1):
            row = [model]
            row.append(sample_counts.get(model, 0))

            # Per-dimension scores
            for dim_name, dim_df in dim_tables:
                if model in dim_df["Model"].values:
                    model_row = dim_df[dim_df["Model"] == model].iloc[0]
                    row.append(model_row["Score"])
                    row.append(model_row["CI_String"])
                else:
                    row.append("-")
                    row.append("-")

            # Overall
            model_row = overall_df[overall_df["Model"] == model].iloc[0]
            row.append(model_row["Score"])
            row.append(model_row["CI_String"])
            row.append(rank)
            body_rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")

        table_html = f'<table class="elo-detail">{header}<tbody>{"".join(body_rows)}</tbody></table>'
        all_rows_html.append(f"<h2>{subtask_name}</h2>\n{table_html}")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Subtask ELO Leaderboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin: 0 0 12px; }}
    h2 {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; margin: 12px 0 20px; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #9aa5b1; padding: 6px 10px; text-align: center; white-space: nowrap; }}
    th {{ background: #d9e2f3; font-weight: 700; }}
    td:first-child, th:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Subtask ELO Leaderboard</h1>
  {''.join(all_rows_html)}
</body>
</html>"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    args = parse_args()

    exclude_models = set(parse_csv_values(args.exclude_models) if args.exclude_models else [])

    all_data = load_result_files(args.result_files)
    print(f"Total records loaded: {len(all_data)}")

    # Group by subtask TYPE, then by dimension
    subtask_groups = group_by_subtask(all_data)
    print(f"Subtask types found: {len(subtask_groups)}")
    for name in sorted(subtask_groups.keys()):
        dim_counts = {d: len(v) for d, v in subtask_groups[name].items()}
        print(f"  - {name}: {dim_counts}")

    dim_names = ["Visual Consistency", "Visual Quality", "Instruction Following"]
    subtask_results = []

    for subtask_name, dim_data in sorted(subtask_groups.items()):
        print(f"\n{'='*60}")
        print(f"Subtask: {subtask_name}")
        print(f"{'='*60}")

        dim_data_list = [dim_data.get("Visual Consistency", []),
                         dim_data.get("Visual Quality", []),
                         dim_data.get("Instruction Following", [])]

        # Skip if any dimension is missing
        if any(len(d) == 0 for d in dim_data_list):
            missing = [n for n, d in zip(dim_names, dim_data_list) if len(d) == 0]
            print(f"  Skipped (missing dimensions: {missing})")
            continue

        # Compute per-dimension leaderboards
        dim_tables = []
        for idx, (dim_name, dim_d) in enumerate(zip(dim_names, dim_data_list)):
            dim_seed = None if args.seed is None else args.seed + idx + 1
            dim_df = calculate_joint_leaderboard(
                [dim_d],
                n_bootstrap=args.bootstrap,
                dimension_names=[dim_name],
                alpha=1.0,
                dimension_weighting="balanced",
                random_seed=dim_seed,
                exclude_models=exclude_models,
            )
            dim_tables.append((dim_name, dim_df))
            print_leaderboard(f"  {dim_name}", dim_df)

        # Compute joint overall leaderboard
        overall_df = calculate_joint_leaderboard(
            dim_data_list,
            n_bootstrap=args.bootstrap,
            dimension_names=dim_names,
            alpha=1.0,
            dimension_weighting="balanced",
            random_seed=args.seed,
            exclude_models=exclude_models,
        )

        if overall_df.empty:
            print(f"  Skipped (insufficient data)")
            continue

        print_leaderboard(f"Joint ELO for {subtask_name}", overall_df)

        # Collect sample counts
        model_names = overall_df["Model"].tolist()
        sample_counts = _collect_samples_per_model(dim_data_list, model_names)

        subtask_results.append((subtask_name, overall_df, dim_tables, sample_counts))

    # Export HTML table if requested
    if args.table_output and subtask_results:
        _export_subtask_tables_to_html(args.table_output, subtask_results)
        print(f"\nLeaderboard HTML saved to: {args.table_output}")


if __name__ == "__main__":
    main()