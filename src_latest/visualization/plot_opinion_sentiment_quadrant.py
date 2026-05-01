import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from afinn import Afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


SUPPORTED_METHODS = ("vader", "afinn")


def score_vader(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    return float(analyzer.polarity_scores(str(text))["compound"])


def score_afinn(text: str, afinn: Afinn) -> float:
    # AFINN is unbounded; tanh keeps it in [-1, 1] and comparable to VADER.
    return float(np.tanh(float(afinn.score(str(text))) / 5.0))

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_tick_list(tick_text: str):
    raw = str(tick_text).strip().lower()
    if raw in {"all", "*", ""}:
        return None
    ticks = [int(t.strip()) for t in str(tick_text).split(",") if str(t).strip()]
    return sorted(set(ticks))


def sample_agent_ids(
    traj_csv: str,
    agent_sample_frac: float,
    random_seed: int,
    chunksize: int = 1_000_000,
) -> tuple[set, int, int]:
    all_ids = set()
    chunk_iter = pd.read_csv(traj_csv, usecols=["AgentID"], chunksize=chunksize, low_memory=False)
    for chunk in tqdm(chunk_iter, desc="Sampling agents", unit="chunk"):
        all_ids.update(chunk["AgentID"].dropna().astype("int64").unique().tolist())

    id_array = np.array(sorted(all_ids), dtype=np.int64)
    total_agents = len(id_array)
    sample_n = max(1, int(total_agents * agent_sample_frac))
    rng = np.random.default_rng(random_seed)
    sampled = rng.choice(id_array, size=sample_n, replace=False)
    return set(sampled.tolist()), total_agents, sample_n


def build_opinion_change(
    traj_csv: str,
    sampled_agent_ids: set,
    use_all_agents: bool,
    target_ticks,
    exclude_next_opinion_one: bool,
) -> pd.DataFrame:
    dtypes = {
        "Tick": "int32",
        "AgentID": "int64",
        "Belief": "float32",
    }
    needed_ticks = None
    if target_ticks is not None:
        needed_ticks = set(target_ticks) | set([t + 1 for t in target_ticks])
    parts = []
    chunk_iter = pd.read_csv(
        traj_csv,
        usecols=["Tick", "AgentID", "Belief"],
        dtype=dtypes,
        chunksize=1_000_000,
        low_memory=False,
    )
    for chunk in tqdm(chunk_iter, desc="Building opinion deltas", unit="chunk"):
        if use_all_agents:
            mask = np.ones(len(chunk), dtype=bool)
        else:
            mask = chunk["AgentID"].isin(sampled_agent_ids)
        if needed_ticks is not None:
            mask = mask & chunk["Tick"].isin(needed_ticks)
        chunk = chunk[mask]
        if not chunk.empty:
            parts.append(chunk)

    if not parts:
        return pd.DataFrame(columns=["tick", "agent_id", "opinion_change_next_step"])

    traj = pd.concat(parts, ignore_index=True)
    traj = traj.sort_values(["AgentID", "Tick"])
    traj["next_opinion"] = traj.groupby("AgentID")["Belief"].shift(-1)
    if exclude_next_opinion_one:
        traj = traj[~np.isclose(traj["next_opinion"], 1.0, atol=1e-8)]
    traj["opinion_change_next_step"] = traj["next_opinion"] - traj["Belief"]
    out = traj.dropna(subset=["next_opinion"])[["Tick", "AgentID", "opinion_change_next_step"]].copy()
    if target_ticks is not None:
        out = out[out["Tick"].isin(target_ticks)]
    out = out.rename(columns={"Tick": "tick", "AgentID": "agent_id"})
    return out


def aggregate_sentiment(
    dialogue_csv: str,
    sampled_agent_ids: set,
    use_all_agents: bool,
    target_ticks,
    sample_frac: float,
    random_seed: int,
    methods: tuple[str, ...],
    chunksize: int = 500_000,
) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer() if "vader" in methods else None
    afinn = Afinn() if "afinn" in methods else None

    rng = np.random.default_rng(random_seed)

    agg = {method: {} for method in methods}
    total_rows = 0
    kept_rows = 0

    chunk_iter = pd.read_csv(
        dialogue_csv,
        usecols=["tick", "receiver_id", "dialogue"],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in tqdm(chunk_iter, desc="Aggregating sentiment", unit="chunk"):
        total_rows += len(chunk)
        if use_all_agents:
            mask = np.ones(len(chunk), dtype=bool)
        else:
            mask = chunk["receiver_id"].isin(sampled_agent_ids)
        if target_ticks is not None:
            mask = mask & chunk["tick"].isin(target_ticks)
        chunk = chunk[mask].copy()

        if chunk.empty:
            continue

        if sample_frac < 1.0:
            keep_mask = rng.random(len(chunk)) < sample_frac
            chunk = chunk[keep_mask].copy()

        if chunk.empty:
            continue

        kept_rows += len(chunk)
        texts = chunk["dialogue"].astype(str)

        method_scores = {}
        if "vader" in methods and analyzer is not None:
            method_scores["vader"] = texts.apply(lambda x: score_vader(x, analyzer)).to_numpy(dtype=np.float32)
        if "afinn" in methods and afinn is not None:
            method_scores["afinn"] = texts.apply(lambda x: score_afinn(x, afinn)).to_numpy(dtype=np.float32)

        base_keys = chunk[["tick", "receiver_id"]].to_numpy()
        for method in methods:
            arr = method_scores[method]
            grouped_df = pd.DataFrame(
                {
                    "tick": base_keys[:, 0],
                    "receiver_id": base_keys[:, 1],
                    "sentiment_score": arr,
                }
            ).groupby(["tick", "receiver_id"], as_index=False).agg(
                sentiment_sum=("sentiment_score", "sum"),
                sentiment_count=("sentiment_score", "count"),
            )

            method_agg = agg[method]
            for row in grouped_df.itertuples(index=False):
                key = (int(row.tick), int(row.receiver_id))
                if key in method_agg:
                    method_agg[key][0] += float(row.sentiment_sum)
                    method_agg[key][1] += int(row.sentiment_count)
                else:
                    method_agg[key] = [float(row.sentiment_sum), int(row.sentiment_count)]

    rows = []
    for method, method_agg in agg.items():
        for (tick, receiver_id), (sent_sum, sent_cnt) in method_agg.items():
            rows.append(
                {
                    "method": method,
                    "tick": tick,
                    "agent_id": receiver_id,
                    "sentiment_score": sent_sum / max(sent_cnt, 1),
                    "dialogue_sample_count": sent_cnt,
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["total_rows"] = total_rows
    out.attrs["kept_rows"] = kept_rows
    return out


def quadrant_label(x: float, y: float) -> str:
    if x >= 0 and y >= 0:
        return "Q1 (+sentiment, +opinion)"
    if x < 0 and y >= 0:
        return "Q2 (-sentiment, +opinion)"
    if x < 0 and y < 0:
        return "Q3 (-sentiment, -opinion)"
    return "Q4 (+sentiment, -opinion)"


def _fit_stats(df: pd.DataFrame) -> dict:
    x = df["sentiment_score"].to_numpy(dtype=float)
    y = df["opinion_change_next_step"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return {"pearson_r": np.nan, "slope": np.nan, "intercept": np.nan}
    slope, intercept = np.polyfit(x[finite], y[finite], 1)
    pearson_r = np.corrcoef(x[finite], y[finite])[0, 1]
    return {
        "pearson_r": float(pearson_r),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def plot_quadrant(df: pd.DataFrame, save_path: str, title: str) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["quadrant"] = [quadrant_label(x, y) for x, y in zip(df["sentiment_score"], df["opinion_change_next_step"])]
    fit = _fit_stats(df)

    q_stats = (
        df["quadrant"]
        .value_counts()
        .rename_axis("quadrant")
        .reset_index(name="count")
        .sort_values("quadrant")
    )
    q_stats["ratio"] = q_stats["count"] / max(len(df), 1)

    plt.figure(figsize=(12, 9), dpi=160)
    plt.scatter(
        df["sentiment_score"],
        df["opinion_change_next_step"],
        s=8,
        alpha=0.20,
        c="#1f77b4",
        edgecolors="none",
    )
    plt.axvline(0, color="black", linestyle="--", linewidth=1.2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)

    x_min, x_max = np.nanquantile(df["sentiment_score"], [0.01, 0.99])
    y_min, y_max = np.nanquantile(df["opinion_change_next_step"], [0.01, 0.99])
    x_pad = (x_max - x_min) * 0.15 if x_max > x_min else 0.1
    y_pad = (y_max - y_min) * 0.15 if y_max > y_min else 0.1
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)

    plt.title(title, fontsize=16)
    plt.xlabel("Language Sentiment Score at Step t", fontsize=12)
    plt.ylabel("Opinion Change from Step t to t+1", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if np.isfinite(fit["slope"]) and np.isfinite(fit["intercept"]):
        line_x = np.array([x_min - x_pad, x_max + x_pad], dtype=float)
        line_y = fit["slope"] * line_x + fit["intercept"]
        plt.plot(line_x, line_y, color="#d62728", linewidth=2.0, label="Linear fit")
        plt.legend(loc="lower right", frameon=True)

    quadrant_text = "\n".join(
        [f"{r.quadrant}: {r.count:,} ({r.ratio:.1%})" for r in q_stats.itertuples(index=False)]
    )
    plt.text(
        0.02,
        0.98,
        quadrant_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    fit_text = f"Pearson r: {fit['pearson_r']:.4f}"
    plt.text(
        0.02,
        0.82,
        fit_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return q_stats, fit


def run(args: argparse.Namespace) -> None:
    print(f"Run config: agent_sample_frac={args.agent_sample_frac}, sample_frac={args.sample_frac}, "
          f"exclude_next_opinion_one={args.exclude_next_opinion_one}")
    target_ticks = parse_tick_list(args.target_ticks)
    sampled_agent_ids, total_agents, sampled_agents = sample_agent_ids(
        args.trajectory_csv,
        agent_sample_frac=args.agent_sample_frac,
        random_seed=args.random_seed,
    )

    opinion_df_full = build_opinion_change(
        args.trajectory_csv,
        sampled_agent_ids=set(),
        use_all_agents=True,
        target_ticks=target_ticks,
        exclude_next_opinion_one=args.exclude_next_opinion_one,
    )
    opinion_df_10 = opinion_df_full[opinion_df_full["agent_id"].isin(sampled_agent_ids)].copy()

    sentiment_df = aggregate_sentiment(
        args.dialogue_csv,
        sampled_agent_ids=set(),
        use_all_agents=True,
        target_ticks=target_ticks,
        sample_frac=args.sample_frac,
        random_seed=args.random_seed,
        methods=SUPPORTED_METHODS,
        chunksize=args.dialogue_chunksize,
    )

    ensure_dir(args.output_dir)
    summary_rows = []
    scenarios = [
        {
            "name": "vader_10pct",
            "method": "vader",
            "opinion_df": opinion_df_10,
            "restrict_agents": sampled_agent_ids,
        },
        {
            "name": "afinn_10pct",
            "method": "afinn",
            "opinion_df": opinion_df_10,
            "restrict_agents": sampled_agent_ids,
        },
        {
            "name": "afinn_full",
            "method": "afinn",
            "opinion_df": opinion_df_full,
            "restrict_agents": None,
        },
    ]

    for scenario in scenarios:
        method = scenario["method"]
        scenario_name = scenario["name"]
        method_sent = sentiment_df[sentiment_df["method"] == method].copy()
        if scenario["restrict_agents"] is not None:
            method_sent = method_sent[method_sent["agent_id"].isin(scenario["restrict_agents"])].copy()

        merged_full = method_sent.merge(scenario["opinion_df"], on=["tick", "agent_id"], how="inner")
        merged_full = merged_full.dropna(subset=["sentiment_score", "opinion_change_next_step"])

        if merged_full.empty:
            print(f"No merged data for scenario={scenario_name}, skipped.")
            continue

        fit_full = _fit_stats(merged_full)
        merged = merged_full

        if args.max_points > 0 and len(merged) > args.max_points:
            merged = merged.sample(n=args.max_points, random_state=args.random_seed)

        title = f"Language Sentiment vs Next-step Opinion Change (Four Quadrants) [{scenario_name}]"
        fig_path = os.path.join(args.output_dir, f"opinion_sentiment_quadrant_scatter_{scenario_name}.png")
        q_stats, fit_plot = plot_quadrant(merged, fig_path, title)
        q_stats.insert(0, "scenario", scenario_name)
        q_stats.insert(1, "method", method)

        merged_full.to_csv(
            os.path.join(args.output_dir, f"opinion_sentiment_merged_sample_{scenario_name}.csv"), index=False
        )
        q_stats.to_csv(
            os.path.join(args.output_dir, f"opinion_sentiment_quadrant_stats_{scenario_name}.csv"), index=False
        )

        q_map = dict(zip(q_stats["quadrant"], q_stats["ratio"]))
        summary_rows.append(
            {
                "method": method,
                "q1_ratio": q_map.get("Q1 (+sentiment, +opinion)", 0.0),
                "q2_ratio": q_map.get("Q2 (-sentiment, +opinion)", 0.0),
                "q3_ratio": q_map.get("Q3 (-sentiment, -opinion)", 0.0),
                "q4_ratio": q_map.get("Q4 (+sentiment, -opinion)", 0.0),
                "q1_plus_q3": q_map.get("Q1 (+sentiment, +opinion)", 0.0)
                + q_map.get("Q3 (-sentiment, -opinion)", 0.0),
                "q2_plus_q4": q_map.get("Q2 (-sentiment, +opinion)", 0.0)
                + q_map.get("Q4 (+sentiment, -opinion)", 0.0),
                "scenario": scenario_name,
                "pearson_r": fit_full.get("pearson_r", np.nan),
                "fit_slope": fit_full.get("slope", np.nan),
                "fit_intercept": fit_full.get("intercept", np.nan),
                "merged_points_total": len(merged_full),
                "merged_points_plotted": len(merged),
                "plot_pearson_r": fit_plot.get("pearson_r", np.nan),
            }
        )

    run_meta = pd.DataFrame(
        [
            {
                "agents_total": total_agents,
                "agents_sampled": sampled_agents,
                "agent_sample_frac": args.agent_sample_frac,
                "target_ticks": "all" if target_ticks is None else ",".join(str(t) for t in target_ticks),
                "exclude_next_opinion_one": args.exclude_next_opinion_one,
                "dialogue_rows_total": sentiment_df.attrs.get("total_rows", np.nan),
                "dialogue_rows_used_after_sampling": sentiment_df.attrs.get("kept_rows", np.nan),
                "dialogue_sample_frac": args.sample_frac,
                "sentiment_methods": "vader,afinn",
                "random_seed": args.random_seed,
            }
        ]
    )
    run_meta.to_csv(os.path.join(args.output_dir, "opinion_sentiment_run_meta.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, "opinion_sentiment_method_comparison.csv"), index=False
    )

    print("Saved outputs to:", args.output_dir)
    for name in ("vader_10pct", "afinn_10pct", "afinn_full"):
        print(f" - opinion_sentiment_quadrant_scatter_{name}.png")
        print(f" - opinion_sentiment_quadrant_stats_{name}.csv")
        print(f" - opinion_sentiment_merged_sample_{name}.csv")
    print(" - opinion_sentiment_method_comparison.csv")
    print(" - opinion_sentiment_run_meta.csv")


def parse_args() -> argparse.Namespace:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Plot sentiment-opinion next-step quadrant scatter.")
    parser.add_argument(
        "--dialogue-csv",
        default=os.path.join(root, "data", "output", "dataframes", "dialogue_history.csv"),
    )
    parser.add_argument(
        "--trajectory-csv",
        default=os.path.join(root, "data", "output", "dataframes", "agent_trajectories.csv"),
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional extra row sampling after agent/tick filtering (0-1).",
    )
    parser.add_argument(
        "--agent-sample-frac",
        type=float,
        default=0.10,
        help="Fraction of agents sampled from trajectories.",
    )
    parser.add_argument(
        "--exclude-next-opinion-one",
        action="store_true",
        default=True,
        help="Exclude records where Opinion at t+1 is directly 1.0.",
    )
    parser.add_argument(
        "--target-ticks",
        default="all",
        help="Comma-separated steps, or 'all' to use all steps.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--max-points",
        type=int,
        default=300000,
        help="Max points for plotting; use <=0 to disable plotting subsample.",
    )
    parser.add_argument("--dialogue-chunksize", type=int, default=500000)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            root,
            "data",
            "output",
            "plots",
            f"opinion_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)