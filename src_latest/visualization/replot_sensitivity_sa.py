import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_ground_truth_daily(
    ground_truth_path: str,
    county_name: str = "Chautauqua County",
    start_date: str = "2021-01-01",
    end_date: str = "2022-05-15",
) -> pd.DataFrame:
    df = pd.read_csv(ground_truth_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df[df["Recip_County"] == county_name].copy()
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["Dose1_Recip_pop_pct_MA"] = (
        pd.to_numeric(df["Dose1_Recip_pop_pct"], errors="coerce")
        .rolling(window=7, min_periods=1)
        .mean()
    )

    start = df["Date"].min()
    df["day"] = (df["Date"] - start).dt.days
    return df[["Date", "day", "Dose1_Recip_pop_pct_MA"]].copy()


def load_sim_step_curve(step_csv: str) -> pd.DataFrame:
    df = pd.read_csv(step_csv, usecols=["tick", "vax_rate"], low_memory=False)
    df["tick"] = pd.to_numeric(df["tick"], errors="coerce")
    df["vax_rate"] = pd.to_numeric(df["vax_rate"], errors="coerce")
    df = df.dropna(subset=["tick", "vax_rate"]).copy()
    df = (
        df.groupby("tick", as_index=False)["vax_rate"]
        .mean()
        .sort_values("tick")
        .reset_index(drop=True)
    )
    df["day"] = df["tick"] * 7.0
    return df


def calc_metrics_day(sim_df: pd.DataFrame, gt_daily: pd.DataFrame) -> Dict[str, float]:
    sim_plot = sim_df[["day", "vax_rate"]].dropna().sort_values("day").copy()
    gt_plot = gt_daily[["day", "Dose1_Recip_pop_pct_MA"]].dropna().sort_values("day").copy()

    if sim_plot.empty or gt_plot.empty or len(sim_plot) < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n_points": 0}

    common_days = gt_plot[
        (gt_plot["day"] >= sim_plot["day"].min())
        & (gt_plot["day"] <= sim_plot["day"].max())
    ]["day"].to_numpy(dtype=float)

    if len(common_days) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n_points": 0}

    sim_interp_pct = np.interp(
        common_days,
        sim_plot["day"].to_numpy(dtype=float),
        sim_plot["vax_rate"].to_numpy(dtype=float),
    ) * 100.0

    gt_interp_pct = np.interp(
        common_days,
        gt_plot["day"].to_numpy(dtype=float),
        gt_plot["Dose1_Recip_pop_pct_MA"].to_numpy(dtype=float),
    )

    err = sim_interp_pct - gt_interp_pct
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((gt_interp_pct - np.mean(gt_interp_pct)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {"mae": mae, "rmse": rmse, "r2": r2, "n_points": int(len(common_days))}


def load_case_curve_and_metrics(
    output_root: str,
    case_dir: str,
    gt_daily: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    step_csv = os.path.join(output_root, case_dir, "dataframes", "step_by_step_data.csv")
    if not os.path.exists(step_csv):
        raise FileNotFoundError(f"Missing step csv: {step_csv}")

    sim_df = load_sim_step_curve(step_csv)
    metrics = calc_metrics_day(sim_df, gt_daily)
    return sim_df, metrics


def metrics_table_axis(ax, rows: list, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=10)

    table_df = pd.DataFrame(rows)
    for col in ["MAE", "RMSE", "R2"]:
        table_df[col] = table_df[col].map(lambda x: "nan" if pd.isna(x) else f"{x:.2f}")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)


def plot_belief_combined(
    output_root: str,
    gt_daily: pd.DataFrame,
    out_dir: str,
) -> pd.DataFrame:
    bt_cases = [0.5, 1.0, 1.5, 2.0]
    bt_dirs = {bt: f"sensitivity_bt_{bt}_no_split" for bt in bt_cases}
    colors = {
        0.5: "#1f77b4",
        1.0: "#2ca02c",
        1.5: "#ff7f0e",
        2.0: "#9467bd",
    }
    linestyles = {
        0.5: "-",
        1.0: "-.",
        1.5: ":",
        2.0: (0, (9, 3)),
    }
    style_text = {
        0.5: "solid",
        1.0: "dash-dot",
        1.5: "dotted",
        2.0: "long-dash",
    }

    fig, (ax_plot, ax_tbl) = plt.subplots(
        1,
        2,
        figsize=(15, 6.8),
        dpi=170,
        gridspec_kw={"width_ratios": [4.7, 1.9]},
    )

    ax_plot.plot(
        gt_daily["day"],
        gt_daily["Dose1_Recip_pop_pct_MA"] / 100.0,
        color="#d62728",
        linestyle="--",
        linewidth=2.2,
        label="Observed (7-day MA)",
    )

    rows = []
    records = []

    for bt in bt_cases:
        sim_df, metrics = load_case_curve_and_metrics(output_root, bt_dirs[bt], gt_daily)
        ax_plot.plot(
            sim_df["day"],
            sim_df["vax_rate"],
            color=colors[bt],
            linestyle=linestyles[bt],
            linewidth=2.1,
            label=f"bt={bt} ({style_text[bt]})",
        )

        rows.append(
            {
                "Case": f"bt={bt}",
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "R2": metrics["r2"],
            }
        )
        records.append(
            {
                "case": f"bt={bt}",
                "mae_percent_day": metrics["mae"],
                "rmse_percent_day": metrics["rmse"],
                "r2_day": metrics["r2"],
                "n_points": metrics["n_points"],
            }
        )

    ax_plot.set_title("Belief Threshold Sensitivity", fontsize=15)
    ax_plot.set_xlabel("Days", fontsize=12)
    ax_plot.set_ylabel("Cumulative Vaccination Rate", fontsize=12)
    ax_plot.set_ylim(0, 1.0)
    ax_plot.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_plot.legend(loc="lower right", fontsize=10)

    metrics_table_axis(ax_tbl, rows, "Metrics (day + percent)")

    fig.tight_layout()
    save_path = os.path.join(out_dir, "belief_threshold_combined_5lines.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(os.path.join(out_dir, "belief_threshold_metrics_day_percent.csv"), index=False)
    return metrics_df


def plot_rw_two_panels(
    output_root: str,
    gt_daily: pd.DataFrame,
    out_dir: str,
) -> pd.DataFrame:
    rw_cases = [0.1, 0.9]
    rw_dirs = {0.1: "sensitivity_rw_0.1_bt0.5", 0.9: "sensitivity_rw_0.9_bt0.5"}
    colors = {0.1: "#1f77b4", 0.9: "#17becf"}
    linestyles = {0.1: "-", 0.9: (0, (9, 3))}
    style_text = {0.1: "solid", 0.9: "long-dash"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=170, sharex=True, sharey=True)
    records = []

    for idx, rw in enumerate(rw_cases):
        ax = axes[idx]
        sim_df, metrics = load_case_curve_and_metrics(output_root, rw_dirs[rw], gt_daily)

        ax.plot(
            sim_df["day"],
            sim_df["vax_rate"],
            color=colors[rw],
            linestyle=linestyles[rw],
            linewidth=2.2,
            label=f"Simulated rw={rw} ({style_text[rw]})",
        )
        ax.plot(
            gt_daily["day"],
            gt_daily["Dose1_Recip_pop_pct_MA"] / 100.0,
            color="#d62728",
            linestyle="--",
            linewidth=2.0,
            label="Observed (7-day MA)",
        )

        text = f"MAE={metrics['mae']:.2f}\nRMSE={metrics['rmse']:.2f}\nR2={metrics['r2']:.2f}"
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.78},
        )

        ax.set_title(f"rw={rw}", fontsize=13)
        ax.set_xlabel("Days", fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="lower right", fontsize=9)

        records.append(
            {
                "case": f"rw={rw}",
                "mae_percent_day": metrics["mae"],
                "rmse_percent_day": metrics["rmse"],
                "r2_day": metrics["r2"],
                "n_points": metrics["n_points"],
            }
        )

    axes[0].set_ylabel("Cumulative Vaccination Rate", fontsize=11)
    axes[0].set_ylim(0, 1.0)
    fig.suptitle("Resonance Weight Sensitivity (bt=0.5)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(out_dir, "rw_bt05_two_panels.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(os.path.join(out_dir, "rw_bt05_metrics_day_percent.csv"), index=False)
    return metrics_df


def plot_rw_combined(
    output_root: str,
    gt_daily: pd.DataFrame,
    rw_metrics_df: pd.DataFrame,
    out_dir: str,
) -> None:
    rw_dirs = {0.1: "sensitivity_rw_0.1_bt0.5", 0.9: "sensitivity_rw_0.9_bt0.5"}
    colors = {0.1: "#1f77b4", 0.9: "#17becf"}
    linestyles = {0.1: "-", 0.9: (0, (9, 3))}
    style_text = {0.1: "solid", 0.9: "long-dash"}

    fig, (ax_plot, ax_tbl) = plt.subplots(
        1,
        2,
        figsize=(15, 6.8),
        dpi=170,
        gridspec_kw={"width_ratios": [4.7, 1.9]},
    )

    ax_plot.plot(
        gt_daily["day"],
        gt_daily["Dose1_Recip_pop_pct_MA"] / 100.0,
        color="#d62728",
        linestyle="--",
        linewidth=2.2,
        label="Observed (7-day MA)",
    )

    for rw in [0.1, 0.9]:
        sim_df, _ = load_case_curve_and_metrics(output_root, rw_dirs[rw], gt_daily)
        ax_plot.plot(
            sim_df["day"],
            sim_df["vax_rate"],
            color=colors[rw],
            linestyle=linestyles[rw],
            linewidth=2.2,
            label=f"Simulated rw={rw} ({style_text[rw]})",
        )

    ax_plot.set_title("Resonance Weight Sensitivity (bt=0.5)", fontsize=15)
    ax_plot.set_xlabel("Days", fontsize=12)
    ax_plot.set_ylabel("Cumulative Vaccination Rate", fontsize=12)
    ax_plot.set_ylim(0, 1.0)
    ax_plot.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_plot.legend(loc="lower right", fontsize=10)

    rows = []
    for _, row in rw_metrics_df.iterrows():
        rows.append(
            {
                "Case": str(row["case"]),
                "MAE": row["mae_percent_day"],
                "RMSE": row["rmse_percent_day"],
                "R2": row["r2_day"],
            }
        )
    metrics_table_axis(ax_tbl, rows, "Metrics (day + percent)")

    fig.tight_layout()
    save_path = os.path.join(out_dir, "rw_bt05_combined_3lines.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-visualize sensitivity analysis outputs with day+percent metrics."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root folder that contains sensitivity output folders.",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=str,
        default=None,
        help="Path to 00_NYS_County_vax_rate_by_age.csv.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Folder to save plots and metrics csv files.",
    )
    parser.add_argument(
        "--county-name",
        type=str,
        default="Chautauqua County",
        help="County name in ground truth csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    output_root = args.output_root or os.path.join(project_root, "data", "output")
    ground_truth_csv = args.ground_truth_csv or os.path.join(
        project_root, "data", "input", "00_NYS_County_vax_rate_by_age.csv"
    )
    save_dir = args.save_dir or os.path.join(output_root, "sensitivity_replots")
    ensure_dir(save_dir)

    gt_daily = load_ground_truth_daily(ground_truth_csv, county_name=args.county_name)

    bt_metrics = plot_belief_combined(output_root, gt_daily, save_dir)
    rw_metrics = plot_rw_two_panels(output_root, gt_daily, save_dir)
    plot_rw_combined(output_root, gt_daily, rw_metrics, save_dir)

    bt_metrics.to_csv(os.path.join(save_dir, "belief_threshold_metrics_day_percent.csv"), index=False)

    print("Generated files:")
    print(f"- {os.path.join(save_dir, 'belief_threshold_combined_5lines.png')}")
    print(f"- {os.path.join(save_dir, 'rw_bt05_two_panels.png')}")
    print(f"- {os.path.join(save_dir, 'rw_bt05_combined_3lines.png')}")
    print(f"- {os.path.join(save_dir, 'belief_threshold_metrics_day_percent.csv')}")
    print(f"- {os.path.join(save_dir, 'rw_bt05_metrics_day_percent.csv')}")


if __name__ == "__main__":
    main()
