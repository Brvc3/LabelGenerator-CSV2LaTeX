
import pandas as pd
import numpy as np
import argparse
import re

# 默认指标优化方向
DEFAULT_DIRECTION = {
    'TPR': 'max',
    'AUC': 'max',
    'FPR': 'min',
    'EER': 'min'
}

def extract_metrics_methods_ordered(columns):
    method_metric_pairs = [col.split('_') for col in columns if '_' in col]
    seen_methods = []
    seen_metrics = []

    for metric, method in method_metric_pairs:
        if method not in seen_methods:
            seen_methods.append(method)
        if metric not in seen_metrics:
            seen_metrics.append(metric)

    return seen_metrics, seen_methods

def infer_metrics_and_methods(columns):
    metrics = set()
    methods = set()
    for col in columns:
        match = re.match(r"(.+?)_(.+)", col)
        if match:
            metrics.add(match.group(1))
            methods.add(match.group(2))
    return sorted(metrics), sorted(methods)

def highlight_row_cell(val, best_val, worst_val, nan_present):
    if pd.isna(val):
        return "\\textcolor{red}{-}"
    elif val == best_val:
        return f"\\textcolor{{green}}{{{val:.4f}}}"
    elif not nan_present and val == worst_val:
        return f"\\textcolor{{red}}{{{val:.4f}}}"
    else:
        return f"{val:.4f}"

def generate_latex_table(df, direction_map):
    metrics, methods = extract_metrics_methods_ordered(df.columns)

    header = (
        "\\begin{tabular}{" + "c|" + "|".join(["c" * len(metrics)] * len(methods)) + "}\n"
        "\hline\n"
        "Method & " + " & ".join([f"\multicolumn{{{len(metrics)}}}{{c|}}{{{m}}}" for m in methods[:-1]]) +
        f" & \multicolumn{{{len(metrics)}}}{{c}}{{{methods[-1]}}} \\\\\n"
        "\hline\n"
        "Metrics & " + " & ".join(metrics * len(methods)) + " \\\\\n"
        "\hline\n"
    )

    latex_rows = []
    for idx, row in df.iterrows():
        row_tex = [idx.replace('_', '\_')]
        for method in methods:
            method_values = []
            for metric in metrics:
                col = f"{metric}_{method}"
                val = row.get(col, np.nan)
                method_values.append(val)

            # 全体该 metric 的比较（所有方法）
            for i, metric in enumerate(metrics):
                candidates = [row.get(f"{metric}_{m}", np.nan) for m in methods]
                valid_vals = [v for v in candidates if pd.notna(v)]

                if not valid_vals:
                    row_tex.append("\\textcolor{red}{-}")
                    continue

                direction = direction_map.get(metric, 'max')
                best_val = max(valid_vals) if direction == 'max' else min(valid_vals)
                worst_val = min(valid_vals) if direction == 'max' else max(valid_vals)
                nan_present = len(valid_vals) < len(methods)

                val = method_values[i]
                row_tex.append(highlight_row_cell(val, best_val, worst_val, nan_present))

        latex_rows.append(" & ".join(row_tex) + " \\\\")

    footer = "\hline\n\end{tabular}\n"
    return header + "\n".join(latex_rows) + "\n" + footer

def main(input_csv, output_tex):
    df = pd.read_csv(input_csv, index_col=0)
    latex_code = generate_latex_table(df, DEFAULT_DIRECTION)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"LaTeX 表格已保存至: {output_tex}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LaTeX 表格生成器（多方法/多指标/高亮/缺失处理）")
    parser.add_argument("-i", "--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出 LaTeX 文件路径")
    args = parser.parse_args()
    main(args.input, args.output)
