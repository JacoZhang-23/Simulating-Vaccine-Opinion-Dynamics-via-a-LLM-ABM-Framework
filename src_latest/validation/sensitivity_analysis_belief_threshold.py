"""
Sensitivity analysis for belief_threshold (no split version).

Tests how different belief_threshold values (1.5, 1.0, 0.5) affect vaccination rates.
- belief_threshold controls whether two agents can talk based on belief distance
- all other parameters stay fixed
- this version has no split mechanism; large requests return an empty dict

Goal: check whether large requests trigger JSONDecodeError and whether the code fails gracefully.
"""

import asyncio
import os
import sys
import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv
import networkx as nx
from datetime import datetime
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tools import (load_synthetic_data, plot_vaccination_rate, load_ground_truth_data, 
                      add_tick_field_to_population, add_profile_embedding_to_population,
                      synthesize_profile, add_pums_features_to_population, 
                      enhance_profile_with_pums_features, assign_pums_features_with_geographic_constraint,
                      add_essential_worker_field)
from model import VaxModel
from config import get_model_params, SIMULATION_PARAMS

def sample_with_complete_network(population_df, network_df, initial_sample_proportion=0.005):
    """Sample a fraction of the population and keep all connected neighbors."""
    logger.info(f"Starting network-complete sampling process...")
    logger.info(f"Initial population size: {len(population_df)}")
    logger.info(f"Initial network connections: {len(network_df)}")
    
    initial_sample_size = int(len(population_df) * initial_sample_proportion)
    logger.info(f"Initial sample size target: {initial_sample_size} ({initial_sample_proportion*100}%)")
    
    initial_sample = population_df.sample(n=initial_sample_size, random_state=42)
    initial_sample_ids = set(initial_sample['reindex'].values)
    logger.info(f"Selected {len(initial_sample_ids)} initial individuals")
    
    G = nx.Graph()
    for _, row in network_df.iterrows():
        G.add_edge(row['source_reindex'], row['target_reindex'])
    
    connected_nodes = set()
    for node in initial_sample_ids:
        if G.has_node(node):
            connected_nodes.add(node)
            connected_nodes.update(G.neighbors(node))
    
    logger.info(f"Total nodes after including connections: {len(connected_nodes)}")
    
    # Build the expanded population sample, keeping only nodes that are in the network.
    sampled_population_df = population_df[population_df['reindex'].isin(connected_nodes)].copy()
    
    sampled_network_df = network_df[
        (network_df['source_reindex'].isin(connected_nodes)) & 
        (network_df['target_reindex'].isin(connected_nodes))
    ].copy()
    
    sampled_G = nx.Graph()
    for _, row in sampled_network_df.iterrows():
        sampled_G.add_edge(row['source_reindex'], row['target_reindex'])
    
    initial_sample_degrees = [sampled_G.degree(n) for n in initial_sample_ids if n in sampled_G]
    avg_initial_degree = np.mean(initial_sample_degrees) if initial_sample_degrees else 0
    
    all_degrees = [d for _, d in sampled_G.degree()]
    avg_all_degree = np.mean(all_degrees) if all_degrees else 0
    
    logger.info("\n=== Sampled Network Statistics ===")
    logger.info(f"Initial sample size: {len(initial_sample_ids)}")
    logger.info(f"Final population size: {len(sampled_population_df)} ({len(sampled_population_df)/len(population_df)*100:.2f}% of original)")
    logger.info(f"Network connections: {len(sampled_network_df)}")
    logger.info(f"Average degree of initial sample: {avg_initial_degree:.2f}")
    logger.info(f"Average degree of all included individuals: {avg_all_degree:.2f}")
    logger.info(f"Network density: {nx.density(sampled_G):.4f}")
    logger.info(f"Connected components: {nx.number_connected_components(sampled_G)}")
    
    return sampled_population_df, sampled_network_df, initial_sample_ids


async def run_single_experiment(belief_threshold):
    """Run one experiment for a single belief_threshold value."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # LLMIP_new/
    output_base = os.path.join(project_root, "data", "output")

    experiment_name = f"sensitivity_bt_{belief_threshold}_no_split"
    output_base_path = os.path.join(output_base, experiment_name)
    os.makedirs(output_base_path, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info(f"🔬 Sensitivity Analysis (No Split): belief_threshold = {belief_threshold}")
    logger.info(f"📁 Output: {output_base_path}")
    logger.info("="*80 + "\n")
    
    # Configure logging and remove the previous sink to avoid stacking writes to multiple files.
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", colorize=True)  # Terminal: DEBUG level, show all intermediate steps.
    log_path = os.path.join(output_base_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, f"simulation_{experiment_name}.log")
    logger.add(log_file, level="INFO", rotation="10 MB", colorize=False)  # File: INFO level, record only important information.
    
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
    
    # 添加新的模型参数（基于对话的信念更新）
    MODEL_PARAMS.update({
        'belief_threshold': belief_threshold,  # 🔬 敏感性分析：改变此参数
        'resonance_weight': 0.3,  # 敏感性分析：使用0.3（对话内容权重70%）
        'use_batch_dialogue': True,  # 使用批量对话处理
    })
    
    SIMULATION_STEPS = SIMULATION_PARAMS['default_steps']

    # --- 1.5 清理旧的 agent_trajectories.csv（model.py 以 append 模式写入，必须提前清除）---
    default_trajectory_path = os.path.join(project_root, "data", "output", "dataframes", "agent_trajectories.csv")
    if os.path.exists(default_trajectory_path):
        os.remove(default_trajectory_path)
        logger.info(f"🧹 Removed stale agent_trajectories.csv to ensure clean data for this experiment")

    # --- 2. 加载预生成的子样本网络数据 ---
    logger.info(f"Loading subsample network data from: {NETWORK_DATA_DIR}")

    if not os.path.exists(POP_PATH):
        logger.error(f"Population file not found: {POP_PATH}")
        logger.error(f"Please run: python src_v3_3_local/generate_subsample_sm_network.py")
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

    # 所有 agents 均视为初始样本（预生成数据无需二次采样）
    initial_sample_ids = set(population_df['reindex'].values)

    # 确保所有必需的字段都存在
    required_fields = ['id', 'age', 'gender', 'hhold', 'htype', 'wp', 'urban', 'reindex',
                      'GEOID_cty', 'if_employed', 'household_id', 'personal_income',
                      'education', 'occupation', 'health_insurance', 'FINCP', 'HHT',
                      'num_children', 'family_size']

    missing_fields = [field for field in required_fields if field not in population_df.columns]
    if missing_fields:
        logger.warning(f"Missing required fields: {missing_fields}")
        raise ValueError(f"Missing required fields in population data: {missing_fields}")

    population_df = add_essential_worker_field(population_df)

    # 合成 profile 字段（使用增强版本，整合所有PUMS特征）
    logger.info("Generating enhanced profiles...")
    population_df['profile'] = population_df.apply(enhance_profile_with_pums_features, axis=1)

    # 添加 profile embedding 字段（使用BERT进行文本嵌入）
    logger.info("Adding profile embeddings...")
    population_df = add_profile_embedding_to_population(population_df)

    # 添加tick字段（疫苗接种资格时间）
    logger.info("Adding vaccination eligibility ticks...")
    population_df = add_tick_field_to_population(population_df)

    logger.info("Loading ground truth vaccination data...")
    ground_truth_df = load_ground_truth_data(GROUND_TRUTH_PATH)

    # --- 3. 初始化并运行模型 ---
    logger.info("\n" + "="*80)
    logger.info(f"🚀 Initializing VaxModel (belief_threshold={belief_threshold}, NO SPLIT)...")
    logger.info("="*80)
    logger.info(f"📊 Configuration:")
    logger.info(f"   - Population: {len(population_df):,} agents")
    logger.info(f"   - Network edges: {len(network_df):,}")
    logger.info(f"   - Simulation steps: {SIMULATION_STEPS}")
    logger.info(f"   - 🎯 belief_threshold: {belief_threshold} (敏感性分析参数)")
    logger.info(f"   - ⚠️  No split mechanism: Large requests will return empty dict on error")
    concurrent_batches = MODEL_PARAMS.get('concurrent_batches', 100)
    batch_size = MODEL_PARAMS.get('batch_size', 35)
    resonance_weight = MODEL_PARAMS.get('resonance_weight', 0.3)
    logger.info(f"   - API concurrency: {concurrent_batches} batches × {batch_size} agents = {concurrent_batches * batch_size} agents/round")
    logger.info(f"   - API keys: {len(MODEL_PARAMS.get('api_keys', []))}")
    logger.info(f"   - resonance_weight: {resonance_weight} (sensitivity analysis: 0.3)")
    
    import time
    sim_start = time.time()
    model = VaxModel(population_df, network_df, MODEL_PARAMS)
    logger.info(f"\n🏃 Running simulation for {SIMULATION_STEPS} steps...")
    logger.info("="*80 + "\n")
    await model.run_model(SIMULATION_STEPS)
    
    sim_duration = time.time() - sim_start
    logger.info("\n" + "="*80)
    logger.info(f"✅ Simulation completed in {sim_duration:.1f}s ({sim_duration/60:.1f} minutes)")
    logger.info("="*80)

    # --- 4. 保存和分析结果 ---
    logger.info("Simulation finished. Saving results...")
    
    # 创建输出目录
    output_df_path = os.path.join(output_base_path, "dataframes")
    os.makedirs(output_df_path, exist_ok=True)
    
    # --- 4.0 迁移 agent_trajectories.csv 到实验目录 ---
    # model.py 会将数据保存到 ../data/output/dataframes/agent_trajectories.csv
    # 我们需要将它移动到当前实验的输出目录中
    default_trajectory_path = os.path.join(project_root, "data", "output", "dataframes", "agent_trajectories.csv")
    experiment_trajectory_path = os.path.join(output_df_path, "agent_trajectories.csv")
    
    if os.path.exists(default_trajectory_path):
        shutil.move(default_trajectory_path, experiment_trajectory_path)
        logger.info(f"✅ Moved agent_trajectories.csv to experiment directory: {experiment_trajectory_path}")
    else:
        logger.warning(f"⚠️  agent_trajectories.csv not found at default location: {default_trajectory_path}")
    
    # --- 4.1 保存最终的Agent状态 ---
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
            # PUMS特征
            'personal_income': a.personal_income if hasattr(a, 'personal_income') else None,
            'education': a.education if hasattr(a, 'education') else None,
            'occupation': a.occupation if hasattr(a, 'occupation') else None,
            'health_insurance': a.health_insurance if hasattr(a, 'health_insurance') else None,
            # 家庭级别特征
            'HHT': a.HHT if hasattr(a, 'HHT') else None,
            'FINCP': a.FINCP if hasattr(a, 'FINCP') else None,
            'num_children': a.num_children if hasattr(a, 'num_children') else None,
            'family_size': a.family_size if hasattr(a, 'family_size') else None,
        }
        final_agent_data_list.append(agent_data)
    
    final_agent_df = pd.DataFrame(final_agent_data_list)
    final_agent_df.to_csv(os.path.join(output_df_path, "final_agent_state.csv"), index=False)
    logger.info(f"Final agent data saved.")

    # --- 4.2 保存每步数据 ---
    model.datacollector.to_csv(os.path.join(output_df_path, "step_by_step_data.csv"), index=False)
    logger.info(f"Step-by-step data saved.")

    # 保存对话统计信息
    dialogue_stats = model.get_dialogue_statistics()
    if dialogue_stats['total_dialogues'] > 0:
        dialogue_stats['dialogue_df'].to_csv(os.path.join(output_df_path, "dialogue_history.csv"), index=False)
        logger.info(f"Dialogue history saved.")

    # 绘制图表并获取评估指标
    metrics = plot_vaccination_rate(model.datacollector, ground_truth_df, output_base_path)

    # --- 4.3 输出统计信息 ---
    logger.info("\n=== Discussion Participation ===")
    participation_rate = agents_with_dialogue / total_agents * 100
    non_participation_rate = 100 - participation_rate
    logger.info(f"Agents who participated in discussions: {agents_with_dialogue} ({participation_rate:.2f}%)")
    logger.info(f"Agents who never participated: {total_agents - agents_with_dialogue} ({non_participation_rate:.2f}%)")
    
    logger.info("\n=== Sample Comparison ===")
    initial_sample = final_agent_df[final_agent_df['in_initial_sample']]
    logger.info("Initial Sample Stats:")
    logger.info(f"Average belief: {initial_sample['belief'].mean():.3f}")
    logger.info(f"Vaccination rate: {(initial_sample['is_vaccinated']==1).mean():.3f}")
    
    logger.info("\n" + "="*80)
    logger.info(f"📊 Final Evaluation Metrics - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.2f}")
    logger.info(f"🎯 belief_threshold = {belief_threshold} (No Split)")
    logger.info("="*80)
    
    return {
        'belief_threshold': belief_threshold,
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'r2': metrics['r2'],
        'final_vaccination_rate': final_agent_df['is_vaccinated'].mean(),
        'avg_belief': final_agent_df['belief'].mean(),
        'agents_with_dialogue': agents_with_dialogue,
        'total_agents': total_agents,
        'participation_rate': participation_rate
    }


async def main():
    """运行完整的敏感性分析（无Split版本）"""
    logger.info("\n" + "="*80)
    logger.info("🔬 SENSITIVITY ANALYSIS: belief_threshold Parameter (NO SPLIT)")
    logger.info(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")
    
    # 定义要测试的belief_threshold值
    belief_thresholds = [0.5, 1.0, 1.5, 2.0]

    logger.info(f"📋 Testing belief_threshold values: {belief_thresholds}")
    logger.info(f"⚠️  No split mechanism: Failed batches return empty dict, simulation continues\n")
    
    all_results = []
    
    # 依次运行每个实验
    for i, threshold in enumerate(belief_thresholds, 1):
        logger.info("\n" + "🔹"*40)
        logger.info(f"Experiment {i}/{len(belief_thresholds)}: belief_threshold = {threshold}")
        logger.info("🔹"*40 + "\n")
        
        try:
            result = await run_single_experiment(
                belief_threshold=threshold
            )
            all_results.append(result)
            logger.info(f"✅ Experiment {i} completed successfully\n")
            
        except Exception as e:
            logger.error(f"❌ Experiment {i} failed: {e}")
            logger.exception(e)
            continue
    
    # 生成汇总报告
    if all_results:
        logger.info("\n" + "="*80)
        logger.info("📊 SENSITIVITY ANALYSIS SUMMARY (NO SPLIT)")
        logger.info("="*80 + "\n")
        
        results_df = pd.DataFrame(all_results)
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_script_dir)
        summary_path = os.path.join(_project_root, "data", "output", "sensitivity_analysis_summary_no_split.csv")
        results_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to: {summary_path}\n")
        
        logger.info("Results Table:")
        print(results_df.to_string(index=False))
        
        logger.info("\n" + "="*80)
        logger.info("Key Findings:")
        logger.info(f"Best MAE: belief_threshold={results_df.loc[results_df['mae'].idxmin(), 'belief_threshold']} (MAE={results_df['mae'].min():.2f})")
        logger.info(f"Best RMSE: belief_threshold={results_df.loc[results_df['rmse'].idxmin(), 'belief_threshold']} (RMSE={results_df['rmse'].min():.2f})")
        logger.info(f"Best R²: belief_threshold={results_df.loc[results_df['r2'].idxmax(), 'belief_threshold']} (R²={results_df['r2'].max():.2f})")
        logger.info(f"Highest vaccination: belief_threshold={results_df.loc[results_df['final_vaccination_rate'].idxmax(), 'belief_threshold']} ({results_df['final_vaccination_rate'].max():.2%})")
        logger.info(f"Lowest vaccination: belief_threshold={results_df.loc[results_df['final_vaccination_rate'].idxmin(), 'belief_threshold']} ({results_df['final_vaccination_rate'].min():.2%})")
        logger.info("="*80)
    
    logger.info(f"\n📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("✨ Sensitivity analysis (No Split) completed!\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("An error occurred during sensitivity analysis.")
