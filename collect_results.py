"""
실험 결과를 수집하여 Excel + PNG 그래프로 저장.

사용법:
  python collect_results.py \\
    --results-jsons ./results/pair_0.8B/eval_results.json \\
                    ./results/pair_2B/eval_results.json \\
    --output routing_results.xlsx

각 JSON 파일 형식 (evaluate.py --output-json 출력):
  {
    "weak_model": "Qwen/Qwen3.5-2B",
    "strong_model": "Qwen/Qwen3.5-9B",
    "weak_only_accuracy": 62.4,
    "strong_only_accuracy": 81.3,
    "results": [
      {"method": "random", "threshold": 0.1, "strong_percentage": 10.0, "accuracy": 65.2},
      ...
    ]
  }
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill, Alignment


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

def load_result(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    weak = data["weak_model"].split("/")[-1]   # "Qwen3.5-2B"
    strong = data["strong_model"].split("/")[-1]
    label = f"{weak} vs {strong}"
    df = pd.DataFrame(data["results"])
    df["pair"] = label
    df["weak_model"] = data["weak_model"]
    df["strong_model"] = data["strong_model"]
    return {
        "label": label,
        "weak_acc": data["weak_only_accuracy"],
        "strong_acc": data["strong_only_accuracy"],
        "weak_model": data["weak_model"],
        "strong_model": data["strong_model"],
        "df": df,
    }


# ─────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────

METHOD_STYLE = {
    "random":   {"color": "#888888", "linestyle": "--", "linewidth": 1.5, "marker": None},
    "mf":       {"color": "#2196F3", "linestyle": "-",  "linewidth": 2.0, "marker": "o"},
    "uniroute": {"color": "#FF9800", "linestyle": "-",  "linewidth": 2.0, "marker": "s"},
}
METHOD_LABEL = {
    "random":   "Random",
    "mf":       "MF Router",
    "uniroute": "UniRoute (K-Means)",
}


def make_graphs(pair_data_list: list[dict], output_png: str) -> str:
    n = len(pair_data_list)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, pdata in zip(axes, pair_data_list):
        df = pdata["df"].copy()
        weak_acc = pdata["weak_acc"]
        strong_acc = pdata["strong_acc"]

        # Plot each method
        for method in ["random", "mf", "uniroute"]:
            df_m = df[df["method"] == method].sort_values("strong_percentage")
            if df_m.empty:
                continue
            style = METHOD_STYLE.get(method, {})
            ax.plot(
                df_m["strong_percentage"],
                df_m["accuracy"],
                label=METHOD_LABEL.get(method, method),
                color=style.get("color", "black"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 1.5),
                marker=style.get("marker"),
                markersize=5,
            )

        # Baselines
        ax.axhline(weak_acc, color="#555555", linestyle=":", linewidth=1.2,
                   label=f"Weak only ({weak_acc:.1f}%)")
        ax.axhline(strong_acc, color="#C62828", linestyle=":", linewidth=1.2,
                   label=f"Strong only ({strong_acc:.1f}%)")

        ax.set_xlabel("Strong Model Calls (%)", fontsize=11)
        ax.set_ylabel("Pass Rate (%)", fontsize=11)
        ax.set_title(pdata["label"], fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 102)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph → {output_png}")
    return output_png


# ─────────────────────────────────────────────
# Excel helpers
# ─────────────────────────────────────────────

def _header_style(ws, row_idx: int, n_cols: int):
    fill = PatternFill("solid", fgColor="1E3A5F")
    font = Font(color="FFFFFF", bold=True)
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=row_idx, column=col)
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal="center")


def write_sheet1(wb, pair_data_list: list[dict]):
    ws = wb.create_sheet("Deferral Curves")
    headers = ["Pair", "Method", "Threshold", "Pass Rate (%)", "Weak (%)", "Strong (%)"]
    ws.append(headers)
    _header_style(ws, 1, len(headers))

    for pdata in pair_data_list:
        df = pdata["df"].sort_values(["method", "strong_percentage"])
        for _, row in df.iterrows():
            ws.append([
                pdata["label"],
                row["method"],
                round(float(row["threshold"]), 4),
                round(float(row["accuracy"]), 2),
                round(100.0 - float(row["strong_percentage"]), 2),
                round(float(row["strong_percentage"]), 2),
            ])

    # Auto width
    for col in ws.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 4


def write_sheet2(wb, pair_data_list: list[dict]):
    ws = wb.create_sheet("Summary")
    headers = ["Pair", "Weak Model", "Strong Model", "Weak-only (%)", "Strong-only (%)"]
    ws.append(headers)
    _header_style(ws, 1, len(headers))

    for pdata in pair_data_list:
        ws.append([
            pdata["label"],
            pdata["weak_model"],
            pdata["strong_model"],
            round(pdata["weak_acc"], 2),
            round(pdata["strong_acc"], 2),
        ])

    for col in ws.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 4


def write_sheet_graph(wb, png_path: str):
    ws = wb.create_sheet("Graphs")
    img = XLImage(png_path)
    img.anchor = "A1"
    ws.add_image(img)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect routing experiment results → Excel + graphs")
    parser.add_argument(
        "--results-jsons",
        nargs="+",
        required=True,
        help="Paths to JSON files produced by evaluate.py --output-json (one per model pair)",
    )
    parser.add_argument("--output", default="routing_results.xlsx", help="Output Excel path")
    args = parser.parse_args()

    pair_data_list = [load_result(p) for p in args.results_jsons]

    # PNG graph
    output_dir = str(Path(args.output).parent)
    png_path = os.path.join(output_dir, "routing_curves.png")
    make_graphs(pair_data_list, png_path)

    # Excel
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # remove default sheet

    write_sheet1(wb, pair_data_list)
    write_sheet2(wb, pair_data_list)
    write_sheet_graph(wb, png_path)

    wb.save(args.output)
    print(f"Saved Excel → {args.output}")


if __name__ == "__main__":
    main()
