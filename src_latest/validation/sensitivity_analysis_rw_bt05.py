"""
Sensitivity analysis for resonance_weight with belief_threshold fixed at 0.5.

Under the restricted interaction setting, test how resonance_weight affects vaccination rates.
- resonance_weight controls the balance between background similarity and dialogue content
    - resonance = w * background_similarity + (1-w) * dialogue_content
    - smaller w means dialogue content matters more
    - larger w means background similarity matters more
- belief_threshold is fixed at 0.5
- all other parameters stay fixed
- data source: subsample_networks (10% sample)

Test setup:
- resonance_weight in {0.1, 0.9}
- belief_threshold = 0.5
"""

import asyncio
import os
import sys
import time
import shutil
import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv
import networkx as nx
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tools import (load_ground_truth_data, plot_vaccination_rate,
                   add_tick_field_to_population, add_profile_embedding_to_population,
                   enhance_profile_with_pums_features, add_essential_worker_field)
from model import VaxModel
from config import get_model_params, SIMULATION_PARAMS


async def run_single_experiment(resonance_weight):
    """运行单个 resonance_weight 值的实验（bt=0.5 固定）"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # LLMIP_new/
    output_base = os.path.join(project_root, "data", "output")

    experiment_name = f"sensitivity_rw_{resonance_weight}_bt0.5"
    output_base_path = os.path.join(output_base, experiment_name)
    os.makedirs(output_base_path, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info(f"🔬 Sensitivity Analysis: resonance_weight = {resonance_weight}, belief_threshold = 0.5")
    logger.info(f"📁 Output: {output_base_path}")
    logger.info("=" * 80 + "\n")

    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", colorize=True)
    log_path = os.path.join(output_base_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, f"simulation_{experiment_name}.log")
    logger.add(log_file, level="INFO", rotation="10 MB", colorize=False)

    # --- 1. 定义路径和参数 ---
    logger.info("--- Step 1: Loading Data ---")
    load_dotenv()

    input_base_path = os.path.join(project_root, 'data', 'input')
    NETWORK_DATA_DIR = os.path.join(input_base_path, 'subsample_networks')
    POP_PATH = os.path.join(NETWORK_DATA_DIR, 'population.csv')
    NET_PATH = os.path.join(NETWORK_DATA_DIR, 'network_complete.csv')
    GROUND_TRUTH_PATH = os.path.join(input_base_path, '00_NYS_County_vax_rate_by_age.csv')

    # 从配置文件获取模型参数
    MODEL_PARAMS = get_model_params()

    MODEL_PARAMS.update({
        'belief_threshold': 0.5,           # 固定为 0.5（限制性交互）
        'resonance_weight': resonance_weight,  # 🔬 敏感性分析：改变此参数
        'use_batch_dialogue': True,
    })

    SIMULATION_STEPS = SIMULATION_PARAMS['default_steps']

    # --- 1.5 清理旧的 agent_trajectories.csv ---
    default_trajectory_path = os.path.join(project_root, "data", "output", "dataframes",
                                           "agent_trajectories.csv")
    if os.path.exists(default_trajectory_path):
        os.remove(default_trajectory_path)
        logger.info("🧹 Removed stale agent_trajectories.csv")

    # --- 2. 加载预生成的子样本网络数据（10% subsample） ---
    logger.info(f"Loading subsample network data from: {NETWORK_DATA_DIR}")

    if not os.path.exists(POP_PATH):
        logger.error(f"Population file not found: {POP_PATH}")
        logger.error("Please run: python src_v3_3_local/generate_subsample_sm_network.py")
        raise FileNotFoundError(f"Missing population file: {POP_PATH}")

    if not os.path.exists(NET_PATH):
        logger.error(f"Network file not found: {NET_PATH}")
        raise FileNotFoundError(f"Missing network file: {NET_PATH}")

    logger.info(f"  Loading population: {POP_PATH}")
    population_df = pd.read_csv(POP_PATH, low_memory=False)
    logger.info(f"    ✅ Loaded {len(population_df):,} agents")

    logger.info(f"  Loading network: {NET_PATH}")
    network_df = pd.read_csv(NET_PATH, low_memory=False)
    logger.info(f"    ✅ Loaded {len(network_df):,} network edges")
    if 'Relation' in network_df.columns:
        logger.info(f"    Network layer breakdown: {dict(network_df['Relation'].value_counts())}")

    initial_sample_ids = set(population_df['reindex'].values)

    # 确保所有必需的字段都存在
    required_fields = ['id', 'age', 'gender', 'hhold', 'htype', 'wp', 'urban', 'reindex',
                       'GEOID_cty', 'if_employed', 'household_id', 'personal_income',
                       'education', 'occupation', 'health_insurance', 'FINCP', 'HHT',
                       'num_children', 'family_size']

    missing_fields = [field for field in required_fields if field not in population_df.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields in population data: {missing_fields}")

    population_df = add_essential_worker_field(population_df)

    logger.info("Generating enhanced profiles...")
    population_df['profile'] = population_df.apply(enhance_profile_with_pums_features, axis=1)

    logger.info("Adding profile embeddings...")
    population_df = add_profile_embedding_to_population(population_df)

    logger.info("Adding vaccination eligibility ticks...")
    population_df = add_tick_field_to_population(population_df)

    logger.info("Loading ground truth vaccination data...")
    ground_truth_df = load_ground_truth_data(GROUND_TRUTH_PATH)

    # --- 3. 初始化并运行模型 ---
    logger.info("\n" + "=" * 80)
    logger.info(f"🚀 Initializing VaxModel (resonance_weight={resonance_weight}, belief_threshold=0.5)...")
    logger.info("=" * 80)
    logger.info(f"📊 Configuration:")
    logger.info(f"   - Population: {len(population_df):,} agents")
    logger.info(f"   - Network edges: {len(network_df):,}")
    logger.info(f"   - Simulation steps: {SIMULATION_STEPS}")
    logger.info(f"   - 🎯 resonance_weight: {resonance_weight} (敏感性分析参数)")
    logger.info(f"   - belief_threshold: 0.5 (固定)")
    logger.info(f"   - 📊 权重解释: 背景相似度 {resonance_weight * 100:.0f}%, 对话内容 {(1 - resonance_weight) * 100:.0f}%")
    concurrent_batches = MODEL_PARAMS.get('concurrent_batches', 100)
    batch_size = MODEL_PARAMS.get('batch_size', 25)
    logger.info(f"   - API concurrency: {concurrent_batches} batches × {batch_size} agents = {concurrent_batches * batch_size} agents/round")
    logger.info(f"   - API keys: {len(MODEL_PARAMS.get('api_keys', []))}")

    sim_start = time.time()
    model = VaxModel(population_df, network_df, MODEL_PARAMS)
    logger.info(f"\n🏃 Running simulation for {SIMULATION_STEPS} steps...")
    logger.info("=" * 80 + "\n")
    await model.run_model(SIMULATION_STEPS)

    sim_duration = time.time() - sim_start
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Simulation completed in {sim_duration:.1f}s ({sim_duration / 60:.1f} minutes)")
    logger.info("=" * 80)

    # --- 4. 保存和分析结果 ---
    logger.info("Simulation finished. Saving results...")

    output_df_path = os.path.join(output_base_path, "dataframes")
    os.makedirs(output_df_path, exist_ok=True)

    # 迁移 agent_trajectories.csv
    default_trajectory_path = os.path.join(project_root, "data", "output", "dataframes",
                                           "agent_trajectories.csv")
    experiment_trajectory_path = os.path.join(output_df_path, "agent_trajectories.csv")

    if os.path.exists(default_trajectory_path):
        shutil.move(default_trajectory_path, experiment_trajectory_path)
        logger.info(f"✅ Moved agent_trajectories.csv to experiment directory")
    else:
        logger.warning("⚠️  agent_trajectories.csv not found at default location")

    # 保存最终 Agent 状态
    final_agent_data_list = []
    agents_with_dialogue = 0
    total_agents = len(model.schedule.agents)

    for a in model.schedule.agents:
        if len(a.dialogue_history) > 0:
            agents_with_dialogue += 1

        agent_data = {
            'profile': a.profile,
            'unique_id': a.unique_id,
            'age': a.age,
            'urban': a.urban,
            'geoid': a.geoid if hasattr(a, 'geoid') else None,
            'if_employed': a.if_employed if hasattr(a, 'if_employed') else None,
            'belief': a.belief,
            'alpha': a.alpha,
            'is_vaccinated': a.is_vaccinated,
            'tick': a.tick if hasattr(a, 'tick') else None,
            'tick_vaccinated': a.tick_vaccinated,
            'embedding': a.embedding.tolist() if hasattr(a, 'embedding') else None,
            'dialogue_count': len(a.dialogue_history),
            'avg_resonance': np.mean([d['resonance_weight'] for d in a.dialogue_history]) if a.dialogue_history else 0,
            'in_initial_sample': a.unique_id in initial_sample_ids,
            'personal_income': a.personal_income if hasattr(a, 'personal_income') else None,
            'education': a.education if hasattr(a, 'education') else None,
            'occupation': a.occupation if hasattr(a, 'occupation') else None,
            'health_insurance': a.health_insurance if hasattr(a, 'health_insurance') else None,
            'HHT': a.HHT if hasattr(a, 'HHT') else None,
            'FINCP': a.FINCP if hasattr(a, 'FINCP') else None,
            'num_children': a.num_children if hasattr(a, 'num_children') else None,
            'family_size': a.family_size if hasattr(a, 'family_size') else None,
        }
        final_agent_data_list.append(agent_data)

    final_agent_df = pd.DataFrame(final_agent_data_list)
    final_agent_df.to_csv(os.path.join(output_df_path, "final_agent_state.csv"), index=False)
    logger.info("Final agent data saved.")

    # 保存每步数据
    model.datacollector.to_csv(os.path.join(output_df_path, "step_by_step_data.csv"), index=False)
    logger.info("Step-by-step data saved.")

    # 保存对话统计
    dialogue_stats = model.get_dialogue_statistics()
    if dialogue_stats['total_dialogues'] > 0:
        dialogue_stats['dialogue_df'].to_csv(os.path.join(output_df_path, "dialogue_history.csv"), index=False)
        logger.info("Dialogue history saved.")

    # 绘制图表并获取评估指标
    metrics = plot_vaccination_rate(model.datacollector, ground_truth_df, output_base_path)

    # 输出统计信息
    logger.info("\n=== Discussion Participation ===")
    participation_rate = agents_with_dialogue / total_agents * 100
    logger.info(f"Agents who participated: {agents_with_dialogue} ({participation_rate:.2f}%)")
    logger.info(f"Agents who never participated: {total_agents - agents_with_dialogue} ({100 - participation_rate:.2f}%)")

    logger.info("\n" + "=" * 80)
    logger.info(f"📊 Final Metrics - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.2f}")
    logger.info(f"🎯 resonance_weight={resonance_weight}, belief_threshold=0.5")
    logger.info("=" * 80)

    return {
        'resonance_weight': resonance_weight,
        'belief_threshold': 0.5,
        'background_weight': resonance_weight,
        'dialogue_weight': 1 - resonance_weight,
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'r2': metrics['r2'],
        'final_vaccination_rate': final_agent_df['is_vaccinated'].mean(),
        'avg_belief': final_agent_df['belief'].mean(),
        'agents_with_dialogue': agents_with_dialogue,
        'total_agents': total_agents,
        'participation_rate': participation_rate,
        'duration_sec': sim_duration,
    }


async def main():
    """运行 rw 敏感性分析（bt=0.5 条件下）"""
    logger.info("\n" + "=" * 80)
    logger.info("🔬 SENSITIVITY ANALYSIS: resonance_weight (belief_threshold=0.5)")
    logger.info(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")

    resonance_weights = [0.1, 0.9]

    logger.info(f"📋 Testing resonance_weight values: {resonance_weights}")
    logger.info(f"📊 Fixed: belief_threshold = 0.5")
    logger.info(f"📊 Data source: subsample_networks (10% subsample)")
    logger.info(f"📊 Interpretation:")
    logger.info(f"   - rw=0.1: 90% dialogue content, 10% background similarity")
    logger.info(f"   - rw=0.9: 10% dialogue content, 90% background similarity\n")

    all_results = []

    for i, weight in enumerate(resonance_weights, 1):
        logger.info("\n" + "🔹" * 40)
        logger.info(f"Experiment {i}/{len(resonance_weights)}: resonance_weight = {weight}")
        logger.info(f"  Background: {weight * 100:.0f}%, Dialogue: {(1 - weight) * 100:.0f}%")
        logger.info("🔹" * 40 + "\n")

        try:
            result = await run_single_experiment(resonance_weight=weight)
            all_results.append(result)
            logger.info(f"✅ Experiment {i} completed successfully\n")
        except Exception as e:
            logger.error(f"❌ Experiment {i} failed: {e}")
            logger.exception(e)
            continue

    # 生成汇总报告
    if all_results:
        logger.info("\n" + "=" * 80)
        logger.info("📊 SENSITIVITY ANALYSIS SUMMARY")
        logger.info("=" * 80 + "\n")

        results_df = pd.DataFrame(all_results)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        summary_path = os.path.join(project_root, "data", "output",
                                    "sensitivity_analysis_summary_rw_bt05.csv")
        results_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to: {summary_path}\n")

        logger.info("Results Table:")
        print(results_df.to_string(index=False))

        logger.info("\n" + "=" * 80)
        logger.info("Key Findings:")
        logger.info(f"Best MAE: rw={results_df.loc[results_df['mae'].idxmin(), 'resonance_weight']:.1f} "
                     f"(MAE={results_df['mae'].min():.2f})")
        logger.info(f"Best R²: rw={results_df.loc[results_df['r2'].idxmax(), 'resonance_weight']:.1f} "
                     f"(R²={results_df['r2'].max():.2f})")
        logger.info("=" * 80)

    logger.info(f"\n📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("✨ Sensitivity analysis completed!\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("An error occurred during sensitivity analysis.")
