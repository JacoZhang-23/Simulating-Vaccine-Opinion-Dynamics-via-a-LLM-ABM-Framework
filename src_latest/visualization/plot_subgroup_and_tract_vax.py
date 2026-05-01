import argparse
import os
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_ground_truth_daily(ground_truth_path: str, county_name: str = "Chautauqua County") -> pd.DataFrame:
    df = pd.read_csv(ground_truth_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df[df["Recip_County"] == county_name].copy()
    df = df[(df["Date"] >= "2021-01-01") & (df["Date"] <= "2022-05-15")].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    gt_cols = [
        "Dose1_Recip_pop_pct",
        "Dose1_Recip_5_11_pct",
        "Dose1_Recip_12_17_pct",
        "Dose1_Recip_18_64_pct",
        "Dose1_Recip_65Plus_pct",
    ]
    for col in gt_cols:
        df[f"{col}_MA"] = df[col].rolling(window=7, min_periods=1).mean()

    start_date = df["Date"].min()
    df["day"] = (df["Date"] - start_date).dt.days
    return df[["Date", "day"] + [f"{c}_MA" for c in gt_cols]].copy()


def load_simulation_data(step_path: str, final_path: str, pop_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_df = pd.read_csv(step_path)
    required_final_cols = [
        "unique_id",
        "age",
        "urban",
        "geoid",
        "is_vaccinated",
        "tick_vaccinated",
        "health_insurance",
        "if_employed",
        "education",
        "personal_income",
    ]
    available_final_cols = pd.read_csv(final_path, nrows=0).columns.tolist()
    use_final_cols = [c for c in required_final_cols if c in available_final_cols]
    final_df = pd.read_csv(final_path, usecols=use_final_cols, low_memory=False)
    for col in required_final_cols:
        if col not in final_df.columns:
            final_df[col] = np.nan

    pop_cols = ["reindex", "id", "tractid", "countyid"]
    pop_df = pd.read_csv(pop_path, usecols=pop_cols, low_memory=False)
    merged = final_df.merge(pop_df, left_on="unique_id", right_on="reindex", how="left")

    merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
    merged["tick_vaccinated"] = pd.to_numeric(merged["tick_vaccinated"], errors="coerce")
    merged["health_insurance"] = pd.to_numeric(merged["health_insurance"], errors="coerce")
    merged["if_employed"] = pd.to_numeric(merged["if_employed"], errors="coerce")
    merged["education"] = pd.to_numeric(merged["education"], errors="coerce")
    merged["personal_income"] = pd.to_numeric(merged["personal_income"], errors="coerce")

    merged["urban"] = merged["urban"].astype(str).str.lower().map(
        {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }
    )

    # Build tract GEOID for map join: STATEFP(36) + COUNTYFP(3) + TRACTCE(6).
    if "id" in merged.columns:
        merged["tract_geoid"] = merged["id"].astype(str).str.slice(0, 11)
    else:
        merged["tract_geoid"] = np.nan

    return step_df, merged


def load_tick_tract_rates(
    trajectory_csv: str,
    final_df: pd.DataFrame,
    target_ticks: list,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    agent_to_tract = final_df[["unique_id", "tract_geoid"]].dropna().drop_duplicates()
    wanted = set(int(t) for t in target_ticks)
    selected_parts = []

    for chunk in pd.read_csv(
        trajectory_csv,
        usecols=["Tick", "AgentID", "VaxStatus"],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["Tick"] = pd.to_numeric(chunk["Tick"], errors="coerce").astype("Int64")
        hit = chunk[chunk["Tick"].isin(wanted)]
        if not hit.empty:
            selected_parts.append(hit)

    if not selected_parts:
        return pd.DataFrame(columns=["Tick", "tract_geoid", "sim_vax_rate", "n_agents"])

    selected = pd.concat(selected_parts, ignore_index=True)
    selected["AgentID"] = pd.to_numeric(selected["AgentID"], errors="coerce")
    selected["VaxStatus"] = pd.to_numeric(selected["VaxStatus"], errors="coerce")

    merged = selected.merge(agent_to_tract, left_on="AgentID", right_on="unique_id", how="left")
    merged = merged.dropna(subset=["tract_geoid", "Tick", "VaxStatus"])

    rates = (
        merged.groupby(["Tick", "tract_geoid"], as_index=False)
        .agg(sim_vax_rate=("VaxStatus", "mean"), n_agents=("AgentID", "count"))
        .sort_values(["Tick", "tract_geoid"])
    )
    return rates


def subgroup_curve(final_df: pd.DataFrame, tick_values: np.ndarray, mask: pd.Series) -> pd.DataFrame:
    subgroup = final_df[mask].copy()
    if subgroup.empty:
        return pd.DataFrame({"tick": tick_values, "vax_rate": np.nan})

    vaccinated = subgroup["is_vaccinated"].fillna(False).astype(bool).to_numpy()
    tv = subgroup["tick_vaccinated"].to_numpy()
    max_tick = int(np.nanmax(tick_values))
    tv = np.where(vaccinated, np.nan_to_num(tv, nan=max_tick), np.inf)

    rates = []
    for t in tick_values:
        rates.append(np.mean(tv <= t))
    return pd.DataFrame({"tick": tick_values, "vax_rate": rates})


def calc_metrics_tick(sim_df: pd.DataFrame, gt_daily: pd.DataFrame, gt_col_ma: str) -> dict:
    sim_tick = sim_df[["tick", "vax_rate"]].dropna().copy()
    sim_tick["week"] = pd.to_numeric(sim_tick["tick"], errors="coerce").astype("Int64")
    sim_tick = sim_tick.dropna(subset=["week", "vax_rate"])
    sim_tick["week"] = sim_tick["week"].astype(int)

    gt_weekly = gt_daily[["day", gt_col_ma]].dropna().copy()
    gt_weekly["week"] = (pd.to_numeric(gt_weekly["day"], errors="coerce") // 7).astype("Int64")
    gt_weekly = gt_weekly.dropna(subset=["week"]).copy()
    gt_weekly["week"] = gt_weekly["week"].astype(int)
    gt_weekly = (
        gt_weekly.groupby("week", as_index=False)[gt_col_ma]
        .mean()
        .rename(columns={gt_col_ma: "gt_rate_pct"})
    )

    cmp_df = sim_tick.merge(gt_weekly, on="week", how="inner")
    if cmp_df.empty:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n_ticks": 0}

    err = cmp_df["vax_rate"].astype(float) * 100.0 - cmp_df["gt_rate_pct"].astype(float)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    gt_vals = cmp_df["gt_rate_pct"].astype(float).to_numpy()
    ss_tot = float(np.sum((gt_vals - np.mean(gt_vals)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "n_points": int(len(cmp_df))}


def calc_metrics_day(sim_df: pd.DataFrame, gt_daily: pd.DataFrame, gt_col_ma: str) -> dict:
    sim_plot = sim_df[["tick", "vax_rate"]].dropna().copy()
    sim_plot["day"] = pd.to_numeric(sim_plot["tick"], errors="coerce") * 7
    sim_plot = sim_plot.dropna(subset=["day", "vax_rate"]).sort_values("day")

    gt_plot = gt_daily[["day", gt_col_ma]].dropna().copy()
    gt_plot = gt_plot.sort_values("day")
    gt_plot["gt_rate_pct"] = gt_plot[gt_col_ma].astype(float)

    if sim_plot.empty or gt_plot.empty or len(sim_plot) < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n_points": 0}

    common_days = gt_plot[
        (gt_plot["day"] >= sim_plot["day"].min()) & (gt_plot["day"] <= sim_plot["day"].max())
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
        gt_plot["gt_rate_pct"].to_numpy(dtype=float),
    )

    err = sim_interp_pct - gt_interp_pct
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((gt_interp_pct - np.mean(gt_interp_pct)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "n_points": int(len(common_days))}


def calc_metrics(sim_df: pd.DataFrame, gt_daily: pd.DataFrame, gt_col_ma: str, metric_time_unit: str) -> dict:
    if metric_time_unit == "day":
        return calc_metrics_day(sim_df, gt_daily, gt_col_ma)
    return calc_metrics_tick(sim_df, gt_daily, gt_col_ma)


def plot_compare(
    sim_df: pd.DataFrame,
    gt_daily: pd.DataFrame,
    gt_col_ma: str,
    title: str,
    save_path: str,
    metrics_rows: list,
    group_name: str,
    sim_color: str,
    gt_mode: str,
    metric_time_unit: str,
) -> None:
    sim_plot = sim_df.copy()
    sim_plot["day"] = sim_plot["tick"] * 7

    gt_plot = gt_daily.copy()
    gt_plot["gt_rate"] = gt_plot[gt_col_ma] / 100.0

    metrics = calc_metrics(sim_plot, gt_daily, gt_col_ma, metric_time_unit)
    metrics_rows.append(
        {
            "group": group_name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "n_points_compared": metrics.get("n_points", 0),
            "metric_unit": "percent",
            "metric_time_unit": metric_time_unit,
            "gt_mode": gt_mode,
        }
    )

    if gt_mode == "age_specific":
        gt_color = "#d62728"
        gt_label = "Observed Age-specific Rate (7-day MA)"
    else:
        gt_color = "#ff7f0e"
        gt_label = "Observed County Overall Rate (7-day MA)"

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.set_box_aspect(4 / 5)
    ax.plot(sim_plot["day"], sim_plot["vax_rate"], color=sim_color, label="Simulated Rate", lw=2.4)
    ax.plot(
        gt_plot["day"],
        gt_plot["gt_rate"],
        color=gt_color,
        label=gt_label,
        ls="--",
        lw=2.0,
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Cumulative Vaccination Rate", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    metrics_text = f"MAE = {metrics['mae']:.2f}\nRMSE = {metrics['rmse']:.2f}\nR2 = {metrics['r2']:.3f}"
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.legend()
    ensure_dir(os.path.dirname(save_path))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_subgroup_panels(
    group_curves: list,
    gt_daily: pd.DataFrame,
    gt_col_ma: str,
    title: str,
    save_path: str,
    metrics_rows: list,
    gt_mode: str,
    metric_time_unit: str,
    ncols: int = 2,
) -> None:
    gt_plot_base = gt_daily.copy()
    gt_plot_base["gt_rate"] = gt_plot_base[gt_col_ma] / 100.0

    if gt_mode == "age_specific":
        gt_color = "#d62728"
        gt_label = "Observed Age-specific"
    else:
        gt_color = "#ff7f0e"
        gt_label = "Observed County Overall"

    n = len(group_curves)
    nrows = int(np.ceil(n / ncols))
    # Keep output figure compact with a consistent 5:4 aspect ratio.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        dpi=240,
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    legend_handles = [
        Line2D([0], [0], color=gt_color, lw=2.0, ls="--", label=gt_label),
    ]

    for idx, item in enumerate(group_curves):
        ax = axes[idx]
        ax.set_box_aspect(4 / 5)
        sim_df = item["sim_df"].copy()
        sim_df["day"] = sim_df["tick"] * 7

        ax.plot(sim_df["day"], sim_df["vax_rate"], lw=2.2, color=item["color"])
        ax.plot(gt_plot_base["day"], gt_plot_base["gt_rate"], color=gt_color, ls="--", lw=1.8)

        metrics = calc_metrics(sim_df, gt_daily, gt_col_ma, metric_time_unit)
        metrics_rows.append(
            {
                "group": item["group_key"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "n_points_compared": metrics.get("n_points", 0),
                "metric_unit": "percent",
                "metric_time_unit": metric_time_unit,
                "gt_mode": gt_mode,
            }
        )
        ax.text(
            0.02,
            0.98,
            f"MAE={metrics['mae']:.2f}\nRMSE={metrics['rmse']:.2f}\nR2={metrics['r2']:.3f}",
            transform=ax.transAxes,
            fontsize=9.5,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.78),
        )
        ax.set_title(item["label"], fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax.set_ylim(0, 1)

        legend_handles.append(Line2D([0], [0], color=item["color"], lw=2.2, label=f"{item['label']} Simulated"))

        single_save = os.path.join(os.path.dirname(save_path), f"comparison_{item['group_key']}.png")
        plot_compare(
            item["sim_df"],
            gt_daily,
            gt_col_ma,
            f"{title} [{item['label']}]",
            single_save,
            metrics_rows=[],
            group_name=item["group_key"],
            sim_color=item["color"],
            gt_mode=gt_mode,
            metric_time_unit=metric_time_unit,
        )

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=16, y=0.995)
    fig.text(0.5, 0.055, "Days", ha="center", fontsize=12)
    fig.text(0.025, 0.5, "Cumulative Vaccination Rate", va="center", rotation="vertical", fontsize=12)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.095),
        ncol=min(4, len(legend_handles)),
        frameon=True,
    )
    fig.tight_layout(rect=[0.04, 0.14, 1, 0.95], h_pad=0.01, w_pad=0.025)

    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_age_panels(
    panel_data: list,
    gt_daily: pd.DataFrame,
    gt_map: dict,
    save_path: str,
    metrics_rows: list,
    metric_time_unit: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=160, sharex=True, sharey=True)
    axes = axes.flatten()

    legend_handles = [
        Line2D([0], [0], color="#d62728", lw=1.8, ls="--", label="Observed Age-specific"),
    ]

    for idx, item in enumerate(panel_data):
        ax = axes[idx]
        ax.set_box_aspect(4 / 5)
        key = item["key"]
        sim_df = item["sim_df"].copy()
        sim_df["day"] = sim_df["tick"] * 7
        gt_col = gt_map[key]

        gt_plot = gt_daily.copy()
        gt_plot["gt_rate"] = gt_plot[gt_col] / 100.0

        metrics = calc_metrics(sim_df, gt_daily, gt_col, metric_time_unit)
        metrics_rows.append(
            {
                "group": key,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "n_points_compared": metrics.get("n_points", 0),
                "metric_unit": "percent",
                "metric_time_unit": metric_time_unit,
                "gt_mode": "age_specific",
            }
        )

        ax.plot(sim_df["day"], sim_df["vax_rate"], color=item["color"], lw=2.2, label="Simulated")
        ax.plot(gt_plot["day"], gt_plot["gt_rate"], color="#d62728", ls="--", lw=1.8, label="Observed Age-specific")
        ax.set_title(item["title"], fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax.set_ylim(0, 1)
        ax.text(
            0.02,
            0.98,
            f"MAE={metrics['mae']:.2f}\nRMSE={metrics['rmse']:.2f}\nR2={metrics['r2']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.78),
        )

        legend_handles.append(
            Line2D([0], [0], color=item["color"], lw=2.2, label=f"{item['title']} Simulated")
        )

        single_save = os.path.join(os.path.dirname(save_path), f"comparison_{key}.png")
        plot_compare(
            item["sim_df"],
            gt_daily,
            gt_col,
            f"{item['title']}: Simulated vs Observed Vaccination Rate",
            single_save,
            metrics_rows=[],
            group_name=key,
            sim_color=item["color"],
            gt_mode="age_specific",
            metric_time_unit=metric_time_unit,
        )

    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=True)
    fig.suptitle("Age Subgroups: Simulated vs Observed Vaccination Rate", fontsize=16, y=0.995)
    fig.text(0.5, 0.055, "Days", ha="center", fontsize=12)
    fig.text(0.01, 0.5, "Cumulative Vaccination Rate", va="center", rotation="vertical", fontsize=12)
    fig.tight_layout(rect=[0.04, 0.14, 1, 0.95], h_pad=0.02, w_pad=0.05)

    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def create_tract_map(
    final_df: pd.DataFrame,
    tract_shp: str,
    county_fp: str,
    save_path: str,
    vmin: float = None,
    vmax: float = None,
) -> pd.DataFrame:
    df = final_df.copy()
    df["is_vaccinated"] = df["is_vaccinated"].fillna(False).astype(int)
    tract_rate = (
        df.dropna(subset=["tract_geoid"])
        .groupby("tract_geoid", as_index=False)
        .agg(sim_vax_rate=("is_vaccinated", "mean"), n_agents=("unique_id", "count"))
    )

    gdf = gpd.read_file(tract_shp)
    gdf["COUNTYFP"] = gdf["COUNTYFP"].astype(str)
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    county = gdf[gdf["COUNTYFP"] == county_fp].copy()

    merged = county.merge(tract_rate, left_on="GEOID", right_on="tract_geoid", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
    merged.plot(
        column="sim_vax_rate",
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.2,
        edgecolor="white",
        legend=True,
        missing_kwds={"color": "lightgray", "label": "No simulated agents"},
        ax=ax,
    )
    ax.set_title(
        "Simulated Vaccination Rate by Census Tract\nCountyFP=013 (New York)",
        fontsize=15,
    )
    ax.set_axis_off()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return merged[["GEOID", "sim_vax_rate", "n_agents"]].copy()


def create_tract_map_from_rates(
    tract_rate: pd.DataFrame,
    tract_shp: str,
    county_fp: str,
    save_path: str,
    title: str,
    vmin: float = None,
    vmax: float = None,
) -> pd.DataFrame:
    gdf = gpd.read_file(tract_shp)
    gdf["COUNTYFP"] = gdf["COUNTYFP"].astype(str)
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    county = gdf[gdf["COUNTYFP"] == county_fp].copy()

    merged = county.merge(tract_rate, left_on="GEOID", right_on="tract_geoid", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
    merged.plot(
        column="sim_vax_rate",
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.2,
        edgecolor="white",
        legend=True,
        missing_kwds={"color": "lightgray", "label": "No simulated agents"},
        ax=ax,
    )
    ax.set_title(title, fontsize=15)
    ax.set_axis_off()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return merged[["GEOID", "sim_vax_rate", "n_agents"]].copy()


def add_education_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["education_level"] = pd.Series(index=out.index, dtype="object")
    valid = out["education"].notna()
    out.loc[valid & (out["education"] <= 15), "education_level"] = "edu_low_hs_or_less"
    out.loc[valid & (out["education"] > 15) & (out["education"] <= 21), "education_level"] = "edu_mid_some_college"
    out.loc[valid & (out["education"] > 21), "education_level"] = "edu_high_bachelor_plus"
    return out


def run_all(args: argparse.Namespace) -> None:
    step_df, final_df = load_simulation_data(args.step_csv, args.final_csv, args.population_csv)
    final_df = add_education_level(final_df)
    gt_daily = load_ground_truth_daily(args.ground_truth_csv)
    ensure_dir(args.output_dir)

    max_tick = int(step_df["tick"].max())
    ticks = np.arange(1, max_tick + 1)

    gt_map = {
        "all": "Dose1_Recip_pop_pct_MA",
        "age_5_11": "Dose1_Recip_5_11_pct_MA",
        "age_12_17": "Dose1_Recip_12_17_pct_MA",
        "age_18_64": "Dose1_Recip_18_64_pct_MA",
        "age_65_plus": "Dose1_Recip_65Plus_pct_MA",
    }

    metrics_rows = []

    sim_all = subgroup_curve(final_df, ticks, final_df["age"].notna())
    metrics_day = calc_metrics(sim_all, gt_daily, gt_map["all"], "day")
    metrics_tick = calc_metrics(sim_all, gt_daily, gt_map["all"], "tick")

    if args.metric_time_unit == "auto":
        metric_time_unit = "day" if metrics_day["mae"] <= metrics_tick["mae"] else "tick"
    else:
        metric_time_unit = args.metric_time_unit

    pd.DataFrame(
        [
            {"metric_time_unit": "day", **metrics_day},
            {"metric_time_unit": "tick", **metrics_tick},
            {
                "metric_time_unit": "selected",
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "n_points": np.nan,
                "selected_unit": metric_time_unit,
            },
        ]
    ).to_csv(os.path.join(args.output_dir, "metric_time_unit_comparison.csv"), index=False)

    plot_compare(
        sim_all,
        gt_daily,
        gt_map["all"],
        "All Population: Simulated vs Observed Vaccination Rate",
        os.path.join(args.output_dir, "comparison_all.png"),
        metrics_rows,
        "all",
        sim_color="#1f77b4",
        gt_mode="county_overall",
        metric_time_unit=metric_time_unit,
    )

    age_panel_data = [
        {
            "key": "age_5_11",
            "title": "Age 5-11",
            "sim_df": subgroup_curve(final_df, ticks, final_df["age"].between(5, 11)),
            "color": "#1f77b4",
        },
        {
            "key": "age_12_17",
            "title": "Age 12-17",
            "sim_df": subgroup_curve(final_df, ticks, final_df["age"].between(12, 17)),
            "color": "#2ca02c",
        },
        {
            "key": "age_18_64",
            "title": "Age 18-64",
            "sim_df": subgroup_curve(final_df, ticks, final_df["age"].between(18, 64)),
            "color": "#9467bd",
        },
        {
            "key": "age_65_plus",
            "title": "Age 65+",
            "sim_df": subgroup_curve(final_df, ticks, final_df["age"] >= 65),
            "color": "#8c564b",
        },
    ]
    plot_age_panels(
        age_panel_data,
        gt_daily,
        gt_map,
        os.path.join(args.output_dir, "comparison_age_panels.png"),
        metrics_rows,
        metric_time_unit,
    )

    urban_groups = [
        {
            "group_key": "urban_0",
            "label": "Urban=0",
            "sim_df": subgroup_curve(final_df, ticks, final_df["urban"] == 0),
            "color": "#bcbd22",
        },
        {
            "group_key": "urban_1",
            "label": "Urban=1",
            "sim_df": subgroup_curve(final_df, ticks, final_df["urban"] == 1),
            "color": "#17becf",
        },
    ]
    plot_subgroup_panels(
        urban_groups,
        gt_daily,
        gt_map["all"],
        "Urban Subgroups: Simulated vs Observed Vaccination Rate",
        os.path.join(args.output_dir, "comparison_urban_panel.png"),
        metrics_rows,
        gt_mode="county_overall",
        metric_time_unit=metric_time_unit,
        ncols=2,
    )

    health_groups = []
    for hicov in sorted(final_df["health_insurance"].dropna().unique()):
        hicov_int = int(hicov)
        health_groups.append(
            {
                "group_key": f"health_insurance_{hicov_int}",
                "label": f"Health Insurance={hicov_int}",
                "sim_df": subgroup_curve(final_df, ticks, final_df["health_insurance"] == hicov_int),
                "color": "#4c78a8" if hicov_int == 1 else "#e45756",
            }
        )
    plot_subgroup_panels(
        health_groups,
        gt_daily,
        gt_map["all"],
        "Health Insurance Subgroups: Simulated vs Observed Vaccination Rate",
        os.path.join(args.output_dir, "comparison_health_insurance_panel.png"),
        metrics_rows,
        gt_mode="county_overall",
        metric_time_unit=metric_time_unit,
        ncols=2,
    )

    employment_groups = [
        {
            "group_key": "if_employed_0",
            "label": "Employment=0",
            "sim_df": subgroup_curve(final_df, ticks, final_df["if_employed"] == 0),
            "color": "#b07aa1",
        },
        {
            "group_key": "if_employed_1",
            "label": "Employment=1",
            "sim_df": subgroup_curve(final_df, ticks, final_df["if_employed"] == 1),
            "color": "#59a14f",
        },
    ]
    plot_subgroup_panels(
        employment_groups,
        gt_daily,
        gt_map["all"],
        "Employment Subgroups: Simulated vs Observed Vaccination Rate",
        os.path.join(args.output_dir, "comparison_employment_panel.png"),
        metrics_rows,
        gt_mode="county_overall",
        metric_time_unit=metric_time_unit,
        ncols=2,
    )

    # Additional heterogeneity variable 2: education level groups.
    edu_levels = [
        "edu_low_hs_or_less",
        "edu_mid_some_college",
        "edu_high_bachelor_plus",
    ]
    education_groups = []
    edu_label_map = {
        "edu_low_hs_or_less": "Education: HS or less",
        "edu_mid_some_college": "Education: Some college",
        "edu_high_bachelor_plus": "Education: Bachelor+",
    }
    for edu in edu_levels:
        edu_color = {
            "edu_low_hs_or_less": "#f28e2b",
            "edu_mid_some_college": "#76b7b2",
            "edu_high_bachelor_plus": "#edc948",
        }.get(edu, "#1f77b4")
        education_groups.append(
            {
                "group_key": f"education_{edu}",
                "label": edu_label_map.get(edu, edu),
                "sim_df": subgroup_curve(final_df, ticks, final_df["education_level"] == edu),
                "color": edu_color,
            }
        )
    plot_subgroup_panels(
        education_groups,
        gt_daily,
        gt_map["all"],
        "Education Subgroups: Simulated vs Observed Vaccination Rate",
        os.path.join(args.output_dir, "comparison_education_panel.png"),
        metrics_rows,
        gt_mode="county_overall",
        metric_time_unit=metric_time_unit,
        ncols=3,
    )

    # Intermediate step tract maps from agent trajectories.
    target_ticks = [int(t) for t in str(args.map_ticks).split(",") if str(t).strip()]
    tick_tract_rates = load_tick_tract_rates(args.trajectory_csv, final_df, target_ticks)

    # Build shared color limits so selected tick maps are directly comparable.
    final_tract_rates = (
        final_df.dropna(subset=["tract_geoid"])
        .assign(is_vaccinated_num=final_df["is_vaccinated"].fillna(False).astype(int))
        .groupby("tract_geoid", as_index=False)
        .agg(sim_vax_rate=("is_vaccinated_num", "mean"))
    )
    tick_scale_values = (
        tick_tract_rates["sim_vax_rate"].dropna() if not tick_tract_rates.empty else pd.Series(dtype=float)
    )
    shared_vmin, shared_vmax = None, None
    if not tick_scale_values.empty:
        shared_vmin = float(tick_scale_values.min())
        shared_vmax = float(tick_scale_values.max())
        if abs(shared_vmax - shared_vmin) < 1e-12:
            shared_vmax = shared_vmin + 1e-6

    # Final step map uses separate scale for better within-final contrast.
    final_vmin, final_vmax = None, None
    if not final_tract_rates.empty:
        final_vmin = float(final_tract_rates["sim_vax_rate"].min())
        final_vmax = float(final_tract_rates["sim_vax_rate"].max())
        if abs(final_vmax - final_vmin) < 1e-12:
            final_vmax = final_vmin + 1e-6

    # Tract-level choropleth map (final), using dedicated color scale.
    tract_png = os.path.join(args.output_dir, "tract_vaccination_map_36013.png")
    tract_summary = create_tract_map(
        final_df,
        args.tract_shp,
        args.county_fp,
        tract_png,
        vmin=final_vmin,
        vmax=final_vmax,
    )
    tract_summary.to_csv(os.path.join(args.output_dir, "tract_vax_rate_summary.csv"), index=False)

    step_summaries = []
    for t in target_ticks:
        one_tick = tick_tract_rates[tick_tract_rates["Tick"] == t].copy()
        step_png = os.path.join(args.output_dir, f"tract_vaccination_map_tick_{t}.png")
        title = f"Simulated Vaccination Rate by Census Tract\nTick={t}, CountyFP={args.county_fp} (New York)"
        step_summary = create_tract_map_from_rates(
            one_tick,
            args.tract_shp,
            args.county_fp,
            step_png,
            title,
            vmin=shared_vmin,
            vmax=shared_vmax,
        )
        step_summary["tick"] = t
        step_summaries.append(step_summary)

    if step_summaries:
        pd.concat(step_summaries, ignore_index=True).to_csv(
            os.path.join(args.output_dir, "tract_vax_rate_summary_selected_ticks.csv"),
            index=False,
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(args.output_dir, "comparison_metrics_summary.csv"), index=False)

    pd.DataFrame(
        [
            {
                "map_group": "selected_ticks_shared",
                "shared_colorbar_vmin": shared_vmin,
                "shared_colorbar_vmax": shared_vmax,
                "map_ticks": ",".join(str(x) for x in target_ticks),
            },
            {
                "map_group": "final_map_dedicated",
                "shared_colorbar_vmin": final_vmin,
                "shared_colorbar_vmax": final_vmax,
                "map_ticks": "final",
            },
        ]
    ).to_csv(os.path.join(args.output_dir, "tract_map_colorbar_scale.csv"), index=False)

    print("Saved outputs to:", args.output_dir)
    print("Generated files:")
    for name in sorted(os.listdir(args.output_dir)):
        if name.endswith(".png") or name.endswith(".csv"):
            print(" -", name)


def parse_args() -> argparse.Namespace:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Generate subgroup vaccination comparison plots and tract map.")
    parser.add_argument(
        "--step-csv",
        default=os.path.join(root, "data", "output", "dataframes", "step_by_step_data.csv"),
    )
    parser.add_argument(
        "--final-csv",
        default=os.path.join(root, "data", "output", "dataframes", "final_agent_state.csv"),
    )
    parser.add_argument(
        "--population-csv",
        default=os.path.join(root, "data", "input", "full_county_networks", "population.csv"),
    )
    parser.add_argument(
        "--ground-truth-csv",
        default=os.path.join(root, "data", "input", "00_NYS_County_vax_rate_by_age.csv"),
    )
    parser.add_argument(
        "--tract-shp",
        default=os.path.join(root, "tl_2022_36_tract", "tl_2022_36_tract.shp"),
    )
    parser.add_argument(
        "--trajectory-csv",
        default=os.path.join(root, "data", "output", "dataframes", "agent_trajectories.csv"),
    )
    parser.add_argument(
        "--map-ticks",
        default="12,15,20,49",
        help="Comma-separated tick values for intermediate tract maps.",
    )
    parser.add_argument(
        "--county-fp",
        default="013",
        help="Three-digit county code used in TIGER COUNTYFP.",
    )
    parser.add_argument(
        "--metric-time-unit",
        choices=["auto", "day", "tick"],
        default="auto",
        help="Metric alignment unit for MAE/RMSE/R2. auto selects day when day-MAE is smaller.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            root,
            "data",
            "output",
            "plots",
            f"subgroup_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_all(arguments)