import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_tick_list(tick_text: str):
    raw = str(tick_text).strip().lower()
    if raw in {"all", "*", ""}:
        return None
    ticks = [int(t.strip()) for t in str(tick_text).split(",") if str(t).strip()]
    return sorted(set(ticks))


def append_and_downsample(existing: pd.DataFrame, new_data: pd.DataFrame, max_keep: int, rng: np.random.Generator) -> pd.DataFrame:
    if new_data.empty:
        return existing
    if existing.empty:
        combined = new_data.copy()
    else:
        combined = pd.concat([existing, new_data], ignore_index=True)
    if max_keep > 0 and len(combined) > max_keep:
        idx = rng.choice(len(combined), size=max_keep, replace=False)
        combined = combined.iloc[idx].reset_index(drop=True)
    return combined


def build_step_features(
    trajectory_csv: str,
    final_agent_csv: str,
    agent_sample_frac: float,
    target_ticks,
    exclude_next_opinion_one: bool,
    random_seed: int,
) -> tuple[pd.DataFrame, set, dict]:
    traj = pd.read_csv(
        trajectory_csv,
        usecols=["Tick", "AgentID", "Belief", "SocialInfluence", "DialogueCount"],
        dtype={
            "Tick": "int32",
            "AgentID": "int64",
            "Belief": "float32",
            "SocialInfluence": "float32",
            "DialogueCount": "int32",
        },
        low_memory=False,
    )

    rng = np.random.default_rng(random_seed)
    all_agent_ids = np.sort(traj["AgentID"].dropna().unique())
    total_agents = len(all_agent_ids)

    sampled_agent_ids: set
    if agent_sample_frac >= 1.0:
        sampled_agent_ids = set(all_agent_ids.tolist())
    else:
        sample_n = max(1, int(total_agents * agent_sample_frac))
        sampled = rng.choice(all_agent_ids, size=sample_n, replace=False)
        sampled_agent_ids = set(sampled.tolist())

    traj = traj[traj["AgentID"].isin(sampled_agent_ids)].copy()

    needed_ticks = None
    if target_ticks is not None:
        needed_ticks = set(target_ticks) | set([t + 1 for t in target_ticks])
        traj = traj[traj["Tick"].isin(needed_ticks)].copy()

    traj = traj.sort_values(["AgentID", "Tick"])
    traj["next_belief"] = traj.groupby("AgentID")["Belief"].shift(-1)
    # Align social influence with the same transition used by delta_belief: t -> t+1.
    traj["next_social_influence"] = traj.groupby("AgentID")["SocialInfluence"].shift(-1)
    traj = traj.dropna(subset=["next_belief", "next_social_influence"]).copy()

    if exclude_next_opinion_one:
        traj = traj[~np.isclose(traj["next_belief"], 1.0, atol=1e-8)].copy()

    traj["delta_belief"] = traj["next_belief"] - traj["Belief"]
    traj["social_influence_aligned"] = traj["next_social_influence"]
    if target_ticks is not None:
        traj = traj[traj["Tick"].isin(target_ticks)].copy()

    alpha_df = pd.read_csv(
        final_agent_csv,
        usecols=["unique_id", "alpha"],
        dtype={"unique_id": "int64", "alpha": "float32"},
        low_memory=False,
    ).drop_duplicates(subset=["unique_id"], keep="last")

    step_df = traj.rename(
        columns={
            "Tick": "tick",
            "AgentID": "agent_id",
            "Belief": "belief_t",
            "social_influence_aligned": "social_influence",
            "DialogueCount": "dialogue_count",
        }
    )[["tick", "agent_id", "belief_t", "next_belief", "delta_belief", "social_influence", "dialogue_count"]].copy()

    step_df = step_df.merge(alpha_df, left_on="agent_id", right_on="unique_id", how="left")
    step_df = step_df.drop(columns=["unique_id"])
    step_df["alpha"] = step_df["alpha"].fillna(0.5).astype("float32")
    step_df["env_pull"] = step_df["social_influence"] - step_df["belief_t"]
    step_df["predicted_delta"] = step_df["alpha"] * step_df["env_pull"]

    predicted_next = np.clip(step_df["belief_t"] + step_df["predicted_delta"], -1.0, 1.0)
    step_df["predicted_delta_clipped"] = predicted_next - step_df["belief_t"]

    meta = {
        "agents_total": int(total_agents),
        "agents_sampled": int(len(sampled_agent_ids)),
        "agent_sample_frac": float(agent_sample_frac),
        "step_rows": int(len(step_df)),
    }
    return step_df, sampled_agent_ids, meta


def build_dialogue_micro_rows(
    dialogue_csv: str,
    step_df: pd.DataFrame,
    sampled_agent_ids: set,
    target_ticks,
    sample_frac: float,
    random_seed: int,
    chunksize: int,
    max_rows_keep: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(random_seed)

    step_lookup = step_df[["tick", "agent_id", "belief_t", "delta_belief", "alpha", "env_pull", "dialogue_count"]].copy()
    output_cols = [
        "tick",
        "receiver_id",
        "neighbor_id",
        "network_layer",
        "belief_t",
        "neighbor_belief",
        "alpha",
        "delta_belief",
        "env_pull",
        "dialogue_count",
        "resonance_weight",
        "sum_resonance_layer",
        "norm_resonance",
        "opinion_diff",
        "dialogue_impact_est",
        "weighted_Oj",
    ]
    kept = pd.DataFrame(columns=output_cols)

    stats = {
        "dialogue_rows_total": 0,
        "dialogue_rows_after_filter": 0,
        "dialogue_rows_merged": 0,
        "dialogue_rows_kept_sample": 0,
    }

    chunk_iter = pd.read_csv(
        dialogue_csv,
        usecols=["tick", "receiver_id", "neighbor_id", "resonance_weight", "neighbor_belief", "network_layer"],
        dtype={
            "tick": "int32",
            "receiver_id": "int64",
            "neighbor_id": "int64",
            "resonance_weight": "float32",
            "neighbor_belief": "float32",
            "network_layer": "string",
        },
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk in tqdm(chunk_iter, desc="Building dialogue micro rows", unit="chunk"):
        stats["dialogue_rows_total"] += int(len(chunk))

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

        stats["dialogue_rows_after_filter"] += int(len(chunk))

        chunk = chunk.merge(
            step_lookup,
            left_on=["tick", "receiver_id"],
            right_on=["tick", "agent_id"],
            how="inner",
        )
        chunk = chunk.drop(columns=["agent_id"])

        if chunk.empty:
            continue

        stats["dialogue_rows_merged"] += int(len(chunk))

        chunk["sum_resonance_layer"] = chunk.groupby(["tick", "receiver_id", "network_layer"])["resonance_weight"].transform("sum")
        denom = chunk["sum_resonance_layer"].to_numpy(dtype=float)
        numer = chunk["resonance_weight"].to_numpy(dtype=float)
        chunk["norm_resonance"] = np.where(np.abs(denom) > 1e-12, numer / denom, 0.0)

        chunk["opinion_diff"] = chunk["neighbor_belief"] - chunk["belief_t"]
        chunk["dialogue_impact_est"] = chunk["alpha"] * chunk["norm_resonance"] * chunk["opinion_diff"]
        chunk["weighted_Oj"] = chunk["norm_resonance"] * chunk["neighbor_belief"]

        chunk = chunk[output_cols].dropna(subset=["opinion_diff", "dialogue_impact_est"])
        kept = append_and_downsample(kept, chunk, max_keep=max_rows_keep, rng=rng)

    stats["dialogue_rows_kept_sample"] = int(len(kept))
    return kept, stats


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return {"pearson_r": np.nan, "slope": np.nan, "intercept": np.nan}
    slope, intercept = np.polyfit(x[finite], y[finite], 1)
    pearson_r = np.corrcoef(x[finite], y[finite])[0, 1]
    return {"pearson_r": float(pearson_r), "slope": float(slope), "intercept": float(intercept)}


def clip_xy(df: pd.DataFrame, x_col: str, y_col: str, low_q: float = 0.01, high_q: float = 0.99) -> pd.DataFrame:
    x_low, x_high = np.nanquantile(df[x_col], [low_q, high_q])
    y_low, y_high = np.nanquantile(df[y_col], [low_q, high_q])
    return df[df[x_col].between(x_low, x_high) & df[y_col].between(y_low, y_high)].copy()


def plot_scheme_a_dialogue_scatter(df: pd.DataFrame, save_path: str) -> dict:
    plot_df = clip_xy(df, "opinion_diff", "dialogue_impact_est", low_q=0.005, high_q=0.995)
    fit = fit_linear(
        plot_df["opinion_diff"].to_numpy(dtype=float),
        plot_df["dialogue_impact_est"].to_numpy(dtype=float),
    )

    plt.figure(figsize=(12, 9), dpi=160)
    sc = plt.scatter(
        plot_df["opinion_diff"],
        plot_df["dialogue_impact_est"],
        c=plot_df["norm_resonance"],
        cmap="coolwarm",
        s=14,
        alpha=0.33,
        edgecolors="none",
        vmin=0.0,  # <-- Force min color to exactly 0.0
        vmax=1.0
    )
    plt.colorbar(sc, label="Weighted Resonance", ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.axvline(0, linestyle="--", linewidth=1.0, color="black")
    plt.axhline(0, linestyle="--", linewidth=1.0, color="black")
    plt.xlabel("Opinion Disparity (Neighbor - Self)")
    plt.ylabel("Opinion Change After Dialogue")
    plt.title("Influence Dynamics (Dialogue-level)")
    plt.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)

    if np.isfinite(fit["slope"]) and np.isfinite(fit["intercept"]):
        x_min, x_max = np.nanquantile(plot_df["opinion_diff"], [0.01, 0.99])
        line_x = np.array([x_min, x_max], dtype=float)
        line_y = fit["slope"] * line_x + fit["intercept"]
        plt.plot(line_x, line_y, color="#c62828", linewidth=2.0, label="Linear fit")
        plt.legend(loc="upper left")

    plt.text(
        0.02,
        0.91, 
        f"N={len(plot_df):,}\nPearson r={fit['pearson_r']:.4f}",
        transform=plt.gca().transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return {"rows_plot": int(len(plot_df)), **fit}


def plot_dialogue_weighted_neighbor_scatter(df: pd.DataFrame, save_path: str) -> dict:
    plot_df = clip_xy(df, "opinion_diff", "norm_resonance", low_q=0.005, high_q=0.995)

    plt.figure(figsize=(12, 9), dpi=160)
    plt.scatter(
        plot_df["opinion_diff"],
        plot_df["norm_resonance"],
        s=14,
        alpha=0.33,
        edgecolors="none",
        color="#1f77b4",
    )
    plt.axvline(0, linestyle="--", linewidth=1.0, color="black")
    plt.axhline(0, linestyle="--", linewidth=1.0, color="black")
    plt.xlabel("Opinion Difference (neighbor_belief - receiver_belief)")
    plt.ylabel("Normalized Resonance Weight (norm_resonance)")
    plt.title("Dialogue-level Opinion Difference vs Normalized Resonance Weight")
    plt.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)

    plt.text(
        0.02,
        0.885,
        f"N={len(plot_df):,}",
        transform=plt.gca().transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return {"rows_plot": int(len(plot_df)), "pearson_r": np.nan, "slope": np.nan, "intercept": np.nan}


def plot_scheme_c_agent_bubble(step_df: pd.DataFrame, save_path: str, max_points: int, random_seed: int) -> dict:
    if max_points > 0 and len(step_df) > max_points:
        plot_df = step_df.sample(n=max_points, random_state=random_seed).copy()
    else:
        plot_df = step_df.copy()

    plot_df = clip_xy(plot_df, "env_pull", "delta_belief", low_q=0.005, high_q=0.995)
    fit = fit_linear(
        plot_df["env_pull"].to_numpy(dtype=float),
        plot_df["delta_belief"].to_numpy(dtype=float),
    )

    dcount = plot_df["dialogue_count"].to_numpy(dtype=float)
    if len(dcount) == 0:
        return {"rows_plot": 0, **fit}

    d_hi = np.nanquantile(dcount, 0.95)
    d_clipped = np.clip(dcount, 0, d_hi)
    d_min = float(np.nanmin(d_clipped))
    d_max = float(np.nanmax(d_clipped))
    if d_max > d_min:
        size = 20 + 180 * (d_clipped - d_min) / (d_max - d_min)
    else:
        size = np.full_like(d_clipped, 60.0)

    plt.figure(figsize=(12, 9), dpi=160)
    sc = plt.scatter(
        plot_df["env_pull"],
        plot_df["delta_belief"],
        c=plot_df["alpha"],
        s=size,
        cmap="viridis",
        alpha=0.32,
        edgecolors="none",
        vmin=0.0,  # <-- Force min color to exactly 0.0
        vmax=1.0
    )
    plt.colorbar(sc, label="openness", ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.axvline(0, linestyle="--", linewidth=1.0, color="black")
    plt.axhline(0, linestyle="--", linewidth=1.0, color="black")
    plt.xlabel("Social Pull (Social Influence$_{t+1}$ - Opinion$_t$)")
    plt.ylabel("Opinion Change per Step")
    plt.title("Effect of Social Pull on Step-level Opinion Change")
    plt.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)

    if np.isfinite(fit["slope"]) and np.isfinite(fit["intercept"]):
        x_min, x_max = np.nanquantile(plot_df["env_pull"], [0.01, 0.99])
        line_x = np.array([x_min, x_max], dtype=float)
        line_y = fit["slope"] * line_x + fit["intercept"]
        plt.plot(line_x, line_y, color="#c62828", linewidth=2.0, label="Linear fit")
        plt.legend(loc="upper left")

    plt.text(
        0.02,
        0.91,
        f"N={len(plot_df):,}\nPearson r={fit['pearson_r']:.4f}",
        transform=plt.gca().transAxes,
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return {"rows_plot": int(len(plot_df)), **fit}


def save_layer_summary(dialogue_df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    if dialogue_df.empty:
        out = pd.DataFrame(
            columns=[
                "network_layer",
                "count",
                "opinion_diff_mean",
                "dialogue_impact_est_mean",
                "dialogue_impact_est_abs_mean",
                "corr_opdiff_impact",
            ]
        )
        out.to_csv(save_path, index=False)
        return out

    rows = []
    for layer, sub in dialogue_df.groupby("network_layer"):
        fit = fit_linear(
            sub["opinion_diff"].to_numpy(dtype=float),
            sub["dialogue_impact_est"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "network_layer": layer,
                "count": int(len(sub)),
                "opinion_diff_mean": float(sub["opinion_diff"].mean()),
                "dialogue_impact_est_mean": float(sub["dialogue_impact_est"].mean()),
                "dialogue_impact_est_abs_mean": float(sub["dialogue_impact_est"].abs().mean()),
                "corr_opdiff_impact": fit["pearson_r"],
            }
        )
    out = pd.DataFrame(rows).sort_values("network_layer")
    out.to_csv(save_path, index=False)
    return out


def run(args: argparse.Namespace) -> None:
    print(
        "Run config: "
        f"agent_sample_frac={args.agent_sample_frac}, "
        f"dialogue_sample_frac={args.sample_frac}, "
        f"target_ticks={args.target_ticks}, "
        f"exclude_next_opinion_one={args.exclude_next_opinion_one}"
    )

    target_ticks = parse_tick_list(args.target_ticks)

    step_df, sampled_agent_ids, step_meta = build_step_features(
        trajectory_csv=args.trajectory_csv,
        final_agent_csv=args.final_agent_csv,
        agent_sample_frac=args.agent_sample_frac,
        target_ticks=target_ticks,
        exclude_next_opinion_one=args.exclude_next_opinion_one,
        random_seed=args.random_seed,
    )

    dialogue_df, dialogue_meta = build_dialogue_micro_rows(
        dialogue_csv=args.dialogue_csv,
        step_df=step_df,
        sampled_agent_ids=sampled_agent_ids,
        target_ticks=target_ticks,
        sample_frac=args.sample_frac,
        random_seed=args.random_seed,
        chunksize=args.dialogue_chunksize,
        max_rows_keep=args.max_dialogue_rows_keep,
    )

    ensure_dir(args.output_dir)

    dialogue_scatter_stats = plot_scheme_a_dialogue_scatter(
        dialogue_df,
        os.path.join(args.output_dir, "dialogue_opinion_diff_vs_impact.png"),
    )
    weighted_neighbor_stats = plot_dialogue_weighted_neighbor_scatter(
        dialogue_df,
        os.path.join(args.output_dir, "dialogue_opinion_diff_vs_weighted_neighbor_belief.png"),
    )
    agent_bubble_stats = plot_scheme_c_agent_bubble(
        step_df,
        os.path.join(args.output_dir, "agent_step_env_pull_vs_delta_belief.png"),
        max_points=args.max_step_points,
        random_seed=args.random_seed,
    )

    dialogue_out_cols = [
        "tick",
        "receiver_id",
        "neighbor_id",
        "network_layer",
        "belief_t",
        "neighbor_belief",
        "alpha",
        "delta_belief",
        "env_pull",
        "dialogue_count",
        "resonance_weight",
        "norm_resonance",
        "opinion_diff",
        "dialogue_impact_est",
        "weighted_Oj",
    ]
    dialogue_df[dialogue_out_cols].to_csv(
        os.path.join(args.output_dir, "scheme_a_dialogue_micro_sample.csv"),
        index=False,
    )

    step_out_cols = [
        "tick",
        "agent_id",
        "belief_t",
        "next_belief",
        "delta_belief",
        "social_influence",
        "env_pull",
        "alpha",
        "dialogue_count",
        "predicted_delta",
        "predicted_delta_clipped",
    ]
    if args.max_step_points > 0 and len(step_df) > args.max_step_points:
        step_sample = step_df.sample(n=args.max_step_points, random_state=args.random_seed)
    else:
        step_sample = step_df
    step_sample[step_out_cols].to_csv(
        os.path.join(args.output_dir, "scheme_c_agent_step_sample.csv"),
        index=False,
    )

    layer_summary = save_layer_summary(
        dialogue_df,
        os.path.join(args.output_dir, "scheme_a_layer_summary.csv"),
    )

    summary = pd.DataFrame(
        [
            {
                "scenario": "dialogue_opinion_diff_vs_impact",
                **dialogue_scatter_stats,
            },
            {
                "scenario": "dialogue_opinion_diff_vs_weighted_neighbor_belief",
                **weighted_neighbor_stats,
            },
            {
                "scenario": "agent_step_env_pull_vs_delta_belief",
                **agent_bubble_stats,
            },
        ]
    )
    summary.to_csv(os.path.join(args.output_dir, "micro_influence_summary.csv"), index=False)

    run_meta = {
        **step_meta,
        **dialogue_meta,
        "target_ticks": "all" if target_ticks is None else ",".join(str(t) for t in target_ticks),
        "exclude_next_opinion_one": bool(args.exclude_next_opinion_one),
        "env_pull_alignment": "social_influence_t+1_minus_belief_t",
        "dialogue_sample_frac": float(args.sample_frac),
        "max_dialogue_rows_keep": int(args.max_dialogue_rows_keep),
        "max_step_points": int(args.max_step_points),
        "random_seed": int(args.random_seed),
        "layer_count": int(len(layer_summary)),
    }
    pd.DataFrame([run_meta]).to_csv(os.path.join(args.output_dir, "micro_influence_run_meta.csv"), index=False)

    print("Saved outputs to:", args.output_dir)
    print(" - dialogue_opinion_diff_vs_impact.png")
    print(" - dialogue_opinion_diff_vs_weighted_neighbor_belief.png")
    print(" - agent_step_env_pull_vs_delta_belief.png")
    print(" - scheme_a_dialogue_micro_sample.csv")
    print(" - scheme_c_agent_step_sample.csv")
    print(" - scheme_a_layer_summary.csv")
    print(" - micro_influence_summary.csv")
    print(" - micro_influence_run_meta.csv")


def parse_args() -> argparse.Namespace:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Micro-level opinion influence visualizations (dialogue-level and step-level).")
    parser.add_argument(
        "--dialogue-csv",
        default=os.path.join(root, "data", "output", "dataframes", "dialogue_history.csv"),
    )
    parser.add_argument(
        "--trajectory-csv",
        default=os.path.join(root, "data", "output", "dataframes", "agent_trajectories.csv"),
    )
    parser.add_argument(
        "--final-agent-csv",
        default=os.path.join(root, "data", "output", "dataframes", "final_agent_state.csv"),
    )
    parser.add_argument(
        "--agent-sample-frac",
        type=float,
        default=0.10,
        help="Fraction of agents used in micro analysis (0-1].",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.10,
        help="Row sampling fraction for dialogue_history after agent/tick filter (0-1].",
    )
    parser.add_argument(
        "--exclude-next-opinion-one",
        action="store_true",
        default=True,
        help="Exclude rows where next belief is exactly 1.0.",
    )
    parser.add_argument(
        "--target-ticks",
        default="all",
        help="Comma-separated ticks or 'all'.",
    )
    parser.add_argument(
        "--dialogue-chunksize",
        type=int,
        default=500000,
    )
    parser.add_argument(
        "--max-dialogue-rows-keep",
        type=int,
        default=300000,
        help="Max dialogue rows kept in memory/output sample.",
    )
    parser.add_argument(
        "--max-step-points",
        type=int,
        default=120000,
        help="Max step-level points for plotting/output sample.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            root,
            "data",
            "output",
            "plots",
            f"opinion_micro_influence_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())