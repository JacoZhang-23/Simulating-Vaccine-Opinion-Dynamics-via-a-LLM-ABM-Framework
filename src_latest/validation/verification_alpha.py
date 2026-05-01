"""
Verification: Alpha (Openness) Parameter

Test how alpha changes simulation outcomes.
The update rule is:
    b_i(t+1) = (1 - alpha_i) * b_i(t) + alpha_i * S_i

- alpha -> 0: agents change very slowly
- alpha -> 1: agents fully follow social influence
- alpha ~ U(0,1): heterogeneous openness (baseline)

Test setup:
- Fixed alpha in {0.0, 0.5, 1.0}
- Baseline: alpha ~ U(0, 1)
- Sample 0.02% seeds from full_county_networks and restore their neighbors
- Keep all other parameters at baseline values (belief_threshold=2.0, resonance_weight=0.3)

Outputs:
1. Vaccination-rate comparison plot for all alpha settings
2. Mean-belief trajectory plot for all alpha settings
3. Summary CSV
"""

import asyncio
import os
import sys
import time
import shutil
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger
from dotenv import load_dotenv
from datetime import datetime

# Make modules in src/ importable.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tools import (load_ground_truth_data, add_tick_field_to_population,
                   add_profile_embedding_to_population, enhance_profile_with_pums_features,
                   add_essential_worker_field, plot_vaccination_rate)
from model import VaxModel
from config import get_model_params, SIMULATION_PARAMS


# =========================================================================
# Sampling parameters
# =========================================================================
SEED_PROPORTION = 0.00025  # 0.025% seed
RANDOM_SEED = 42


def sample_with_complete_network(population_df, network_df,
                                  seed_proportion=SEED_PROPORTION):
    """
    Sample seed_proportion of agents from full_county_networks and restore 1-hop neighbors.

    Expected size: about 1% of the full population.
    """
    logger.info("--- Network-complete sampling ---")
    logger.info(f"  Full population: {len(population_df):,}")
    logger.info(f"  Full network edges: {len(network_df):,}")

    seed_size = max(int(len(population_df) * seed_proportion), 1)
    logger.info(f"  Seed size: {seed_size} ({seed_proportion*100:.3f}%)")

    seed_sample = population_df.sample(n=seed_size, random_state=RANDOM_SEED)
    seed_ids = set(seed_sample['reindex'].values)

    # Build the graph and expand to 1-hop neighbors.
    G = nx.Graph()
    for _, row in network_df.iterrows():
        G.add_edge(row['source_reindex'], row['target_reindex'])

    expanded_nodes = set()
    for node in seed_ids:
        if G.has_node(node):
            expanded_nodes.add(node)
            expanded_nodes.update(G.neighbors(node))

    logger.info(f"  Expanded to {len(expanded_nodes):,} nodes "
                f"({len(expanded_nodes)/len(population_df)*100:.2f}% of full)")

    # Induced subgraph.
    sampled_pop = population_df[population_df['reindex'].isin(expanded_nodes)].copy()
    sampled_net = network_df[
        network_df['source_reindex'].isin(expanded_nodes) &
        network_df['target_reindex'].isin(expanded_nodes)
    ].copy()

    # Quick summary stats.
    sampled_G = nx.Graph()
    for _, row in sampled_net.iterrows():
        sampled_G.add_edge(row['source_reindex'], row['target_reindex'])
    avg_deg = 2 * sampled_G.number_of_edges() / max(sampled_G.number_of_nodes(), 1)

    logger.info(f"  Sampled population: {len(sampled_pop):,}")
    logger.info(f"  Sampled edges: {len(sampled_net):,}")
    logger.info(f"  Avg degree: {avg_deg:.2f}")
    logger.info(f"  Components: {nx.number_connected_components(sampled_G)}")

    return sampled_pop, sampled_net


def prepare_data():
    """
    Load full_county_networks, sample seeds + neighbors, and run profile / embedding / tick preprocessing once.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_base_path = os.path.join(project_root, 'data', 'input')

    NETWORK_DATA_DIR = os.path.join(input_base_path, 'full_county_networks')
    POP_PATH = os.path.join(NETWORK_DATA_DIR, 'population.csv')
    NET_PATH = os.path.join(NETWORK_DATA_DIR, 'network_complete.csv')
    GROUND_TRUTH_PATH = os.path.join(input_base_path, '00_NYS_County_vax_rate_by_age.csv')

    for path, name in [(POP_PATH, "Population"), (NET_PATH, "Network")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found: {path}\n"
                "Run: python src_v3_3_local/generate_full_county_networks.py")

    logger.info("Loading full county network data...")
    full_pop = pd.read_csv(POP_PATH, low_memory=False)
    full_net = pd.read_csv(NET_PATH, low_memory=False)
    logger.info(f"  Full population: {len(full_pop):,}")
    logger.info(f"  Full network: {len(full_net):,} edges")

    # Sample the population.
    population_df, network_df = sample_with_complete_network(full_pop, full_net)

    # Check required fields.
    required_fields = ['id', 'age', 'gender', 'hhold', 'htype', 'wp', 'urban', 'reindex',
                       'GEOID_cty', 'if_employed', 'household_id', 'personal_income',
                       'education', 'occupation', 'health_insurance', 'FINCP', 'HHT',
                       'num_children', 'family_size']
    missing = [f for f in required_fields if f not in population_df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    population_df = add_essential_worker_field(population_df)

    logger.info("Generating profiles and embeddings once...")
    population_df['profile'] = population_df.apply(enhance_profile_with_pums_features, axis=1)
    population_df = add_profile_embedding_to_population(population_df)
    population_df = add_tick_field_to_population(population_df)

    ground_truth_df = load_ground_truth_data(GROUND_TRUTH_PATH)

    return population_df, network_df, ground_truth_df


async def run_single_alpha_experiment(alpha_setting, population_df, network_df,
                                       ground_truth_df, output_base):
    """
    Run one experiment for a given alpha setting.

    Args:
        alpha_setting: float 或 "uniform"
        population_df, network_df, ground_truth_df: 预处理好的数据
        output_base: 输出根目录
    """
    label = f"alpha_{alpha_setting}" if isinstance(alpha_setting, (int, float)) else "alpha_uniform"
    experiment_name = f"verification_{label}"
    output_path = os.path.join(output_base, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info(f"  Verification: {label}")
    logger.info("=" * 80)

    # Configure logging.
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)
    log_dir = os.path.join(output_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, f"{experiment_name}.log"),
               level="INFO", rotation="10 MB", colorize=False)

    load_dotenv()

    # Baseline model parameters.
    MODEL_PARAMS = get_model_params()
    MODEL_PARAMS.update({
        'belief_threshold': 2.0,
        'resonance_weight': 0.3,
        'use_batch_dialogue': True,
    })

    SIMULATION_STEPS = 36  # Half-length run is enough to see alpha differences.

    # Remove stale trajectory output.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_traj = os.path.join(project_root, "data", "output", "dataframes",
                                "agent_trajectories.csv")
    if os.path.exists(default_traj):
        os.remove(default_traj)

    # --- Initialize model ---
    sim_start = time.time()
    model = VaxModel(population_df.copy(), network_df.copy(), MODEL_PARAMS)

    # --- Override alpha ---
    if isinstance(alpha_setting, (int, float)):
        for agent in model.schedule.agents:
            agent.alpha = float(alpha_setting)
        logger.info(f"  Overrode all agents' alpha → {alpha_setting}")
    else:
        alphas = [a.alpha for a in model.schedule.agents]
        logger.info(f"  Baseline alpha ~ U(0,1): mean={np.mean(alphas):.3f}, "
                     f"std={np.std(alphas):.3f}")

    logger.info(f"  Agents: {len(model.schedule.agents):,}  |  Steps: {SIMULATION_STEPS}")

    # --- Run simulation ---
    await model.run_model(SIMULATION_STEPS)
    sim_duration = time.time() - sim_start
    logger.info(f"  Done in {sim_duration:.1f}s ({sim_duration / 60:.1f} min)")

    # --- Save results ---
    output_df_path = os.path.join(output_path, "dataframes")
    os.makedirs(output_df_path, exist_ok=True)

    if os.path.exists(default_traj):
        shutil.move(default_traj, os.path.join(output_df_path, "agent_trajectories.csv"))

    model.datacollector.to_csv(os.path.join(output_df_path, "step_by_step_data.csv"),
                                index=False)

    final_data = []
    for a in model.schedule.agents:
        final_data.append({
            'unique_id': a.unique_id, 'age': a.age, 'belief': a.belief,
            'alpha': a.alpha, 'is_vaccinated': a.is_vaccinated,
            'dialogue_count': len(a.dialogue_history),
        })
    final_df = pd.DataFrame(final_data)
    final_df.to_csv(os.path.join(output_df_path, "final_agent_state.csv"), index=False)

    metrics = plot_vaccination_rate(model.datacollector, ground_truth_df, output_path)

    if not model.client_session.closed:
        await model.client_session.close()

    vax_rate = final_df['is_vaccinated'].mean()
    avg_belief = final_df['belief'].mean()
    logger.info(f"  Vax rate: {vax_rate:.3f} | Avg belief: {avg_belief:.3f}")

    return {
        'alpha_setting': label,
        'alpha_value': alpha_setting if isinstance(alpha_setting, (int, float)) else 'U(0,1)',
        'final_vax_rate': vax_rate,
        'avg_belief': avg_belief,
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'r2': metrics['r2'],
        'duration_sec': sim_duration,
        'step_data': model.datacollector.copy(),
    }


def plot_alpha_comparison(all_results, output_base):
    """
    绘制两张对比图（各含 6 条线）：
      图 1: 疫苗接种率轨迹
      图 2: 平均 belief 演化
    """
    # 配色：从冷到暖，基线黑色虚线
    colors = {
        'alpha_0.0': '#2166ac',
        'alpha_0.5': '#fdae61',
        'alpha_1.0': '#d73027',
        'alpha_uniform': '#333333',
    }

    # ---- 图 1: 接种率轨迹 ----
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    for r in all_results:
        label = r['alpha_setting']
        disp = f"α = {r['alpha_value']}" if r['alpha_value'] != 'U(0,1)' else "α ~ U(0,1) [baseline]"
        color = colors.get(label, '#888888')
        ls = '--' if label == 'alpha_uniform' else '-'
        lw = 2.5 if label == 'alpha_uniform' else 1.6
        days = r['step_data']['tick'] * 7
        ax1.plot(days, r['step_data']['vax_rate'], color=color, ls=ls, lw=lw, label=disp)

    ax1.set_title("Vaccination Rate Trajectory by α", fontsize=14)
    ax1.set_xlabel("Days", fontsize=12)
    ax1.set_ylabel("Cumulative Vaccination Rate", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    path1 = os.path.join(output_base, "alpha_verification_vax_rate.png")
    fig1.savefig(path1, bbox_inches='tight')
    plt.close(fig1)
    logger.info(f"Plot saved: {path1}")

    # ---- 图 2: 平均 belief 演化 ----
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=150)
    for r in all_results:
        label = r['alpha_setting']
        disp = f"α = {r['alpha_value']}" if r['alpha_value'] != 'U(0,1)' else "α ~ U(0,1) [baseline]"
        color = colors.get(label, '#888888')
        ls = '--' if label == 'alpha_uniform' else '-'
        lw = 2.5 if label == 'alpha_uniform' else 1.6
        days = r['step_data']['tick'] * 7
        ax2.plot(days, r['step_data']['avg_belief'], color=color, ls=ls, lw=lw, label=disp)

    ax2.set_title("Average Belief Evolution by α", fontsize=14)
    ax2.set_xlabel("Days", fontsize=12)
    ax2.set_ylabel("Mean Belief", fontsize=12)
    ax2.set_ylim(-1, 1)
    ax2.axhline(0, color='gray', ls=':', lw=0.8)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    path2 = os.path.join(output_base, "alpha_verification_belief_evolution.png")
    fig2.savefig(path2, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"Plot saved: {path2}")


async def main():
    """运行完整的 alpha 参数 verification。"""
    logger.info("\n" + "=" * 80)
    logger.info("  VERIFICATION: Alpha (Openness) Parameter")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")

    alpha_settings = [0.0, 0.5, 1.0, "uniform"]

    logger.info(f"Alpha settings: {alpha_settings}")
    logger.info(f"Data source: full_county_networks, seed={SEED_PROPORTION*100:.3f}% + 1-hop expansion\n")

    # --- 加载 & 抽样（一次性） ---
    population_df, network_df, ground_truth_df = prepare_data()

    # --- 输出目录 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_base = os.path.join(project_root, "data", "output", "verification_alpha")
    os.makedirs(output_base, exist_ok=True)

    # --- 依次运行 ---
    all_results = []
    for i, alpha in enumerate(alpha_settings, 1):
        logger.info(f"\n--- Experiment {i}/{len(alpha_settings)}: alpha = {alpha} ---")
        try:
            result = await run_single_alpha_experiment(
                alpha, population_df, network_df, ground_truth_df, output_base)
            all_results.append(result)
        except Exception as e:
            logger.error(f"  [FAIL] alpha={alpha}: {e}")

    # --- 汇总 ---
    if all_results:
        plot_alpha_comparison(all_results, output_base)

        summary_rows = [{
            'alpha': r['alpha_value'],
            'final_vax_rate': f"{r['final_vax_rate']:.4f}",
            'avg_belief': f"{r['avg_belief']:.4f}",
            'MAE': f"{r['mae']:.4f}",
            'RMSE': f"{r['rmse']:.4f}",
            'R2': f"{r['r2']:.4f}",
            'runtime_min': f"{r['duration_sec']/60:.1f}",
        } for r in all_results]
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_base, "alpha_verification_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        logger.info("\n" + "=" * 80)
        logger.info("  SUMMARY")
        logger.info("=" * 80)
        print(summary_df.to_string(index=False))
        logger.info(f"\nSaved: {summary_path}")

    logger.info(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
