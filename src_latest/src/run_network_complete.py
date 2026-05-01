# run_network_complete.py
import asyncio
import os
import json
import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv
import networkx as nx
from collections import defaultdict

# Make local modules importable.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Local imports.
from tools import (load_synthetic_data, plot_vaccination_rate, load_ground_truth_data, 
                      add_tick_field_to_population, add_profile_embedding_to_population,
                      synthesize_profile, add_pums_features_to_population, 
                      enhance_profile_with_pums_features, assign_pums_features_with_geographic_constraint,
                      add_essential_worker_field)
from model import VaxModel
from config import get_model_params, SIMULATION_PARAMS


def _sanitize_for_logging(value):
    """Recursively sanitize config values to keep logs readable and safe."""
    if isinstance(value, dict):
        return {k: _sanitize_for_logging(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_logging(v) for v in value]
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _mask_sensitive_params(params):
    """Mask sensitive fields like API keys before writing logs/snapshots."""
    masked = dict(params)
    if 'api_keys' in masked:
        keys = masked.get('api_keys') or []
        masked['api_keys'] = [
            (f"***{k[-6:]}" if isinstance(k, str) and len(k) >= 6 else "***")
            for k in keys
        ]
        masked['api_key_count'] = len(keys)
    return masked

def sample_with_complete_network(population_df, network_df, initial_sample_proportion=0.005):
    """
    从人口中抽样指定比例的个体，并包含这些个体的所有网络连接对象
    
    Args:
        population_df: 包含所有人口的DataFrame
        network_df: 包含网络连接的DataFrame
        initial_sample_proportion: 初始抽样比例
    
    Returns:
        sampled_population_df: 包含初始样本及其网络连接对象的人口DataFrame
        sampled_network_df: 对应的网络DataFrame
    """
    logger.info(f"Starting network-complete sampling process...")
    logger.info(f"Initial population size: {len(population_df)}")
    logger.info(f"Initial network connections: {len(network_df)}")
    
    # 计算初始样本量
    initial_sample_size = int(len(population_df) * initial_sample_proportion)
    logger.info(f"Initial sample size target: {initial_sample_size} ({initial_sample_proportion*100}%)")
    
    # 随机选择初始样本
    initial_sample = population_df.sample(n=initial_sample_size, random_state=42)
    initial_sample_ids = set(initial_sample['reindex'].values)
    logger.info(f"Selected {len(initial_sample_ids)} initial individuals")
    
    # 创建网络图来分析连接关系
    G = nx.Graph()
    for _, row in network_df.iterrows():
        G.add_edge(row['source_reindex'], row['target_reindex'])
    
    # 获取所有与初始样本有连接的节点
    connected_nodes = set()
    for node in initial_sample_ids:
        if G.has_node(node):
            # 添加节点本身
            connected_nodes.add(node)
            # 添加所有邻居节点
            connected_nodes.update(G.neighbors(node))
    
    logger.info(f"Total nodes after including connections: {len(connected_nodes)}")
    
    # 获取扩展后的人口样本
    sampled_population_df = population_df[population_df['reindex'].isin(connected_nodes)].copy()
    
    # 获取这些个体之间的所有网络连接
    sampled_network_df = network_df[
        (network_df['source_reindex'].isin(connected_nodes)) & 
        (network_df['target_reindex'].isin(connected_nodes))
    ].copy()
    
    # 计算和输出网络统计信息
    sampled_G = nx.Graph()
    for _, row in sampled_network_df.iterrows():
        sampled_G.add_edge(row['source_reindex'], row['target_reindex'])
    
    # 计算初始样本中个体的平均度
    initial_sample_degrees = [sampled_G.degree(n) for n in initial_sample_ids if n in sampled_G]
    avg_initial_degree = np.mean(initial_sample_degrees) if initial_sample_degrees else 0
    
    # 计算所有个体的平均度
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

async def main():
    """异步主函数，负责设置和运行整个模拟。"""
    # --- 0. Logging and environment ---
    # Resolve the script path so relative paths work from any working directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # LLMIP_new/
    output_base_path = os.path.join(project_root, "data", "output")
    log_file_path = os.path.join(output_base_path, "logs", "simulation_{time}.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger.add(log_file_path, rotation="5 MB", level=SIMULATION_PARAMS['log_level'])
    logger.info("--- Simulation Start ---")

    # 加载环境变量
    load_dotenv()

    # --- 1. Paths and parameters ---
    input_base_path = os.path.join(project_root, 'data', 'input')
    
    # ============================================================
    # Network mode:
    #   'subsample'  - data/input/subsample_networks/   (~12k agents, fast test)
    #   'full'       - data/input/full_county_networks/ (~128k agents, full run)
    # ============================================================
    NETWORK_MODE = 'full'   # Full-county run
    
    if NETWORK_MODE == 'subsample':
        NETWORK_DATA_DIR = os.path.join(input_base_path, 'subsample_networks')
    else:
        NETWORK_DATA_DIR = os.path.join(input_base_path, 'full_county_networks')
    
    POP_PATH = os.path.join(NETWORK_DATA_DIR, 'population.csv')
    NET_PATH = os.path.join(NETWORK_DATA_DIR, 'network_complete.csv')
    GROUND_TRUTH_PATH = os.path.join(input_base_path, '00_NYS_County_vax_rate_by_age.csv')

    # Load model parameters from config.
    MODEL_PARAMS = get_model_params()
    
    # Override the main dialogue parameters.
    MODEL_PARAMS.update({
        'belief_threshold': 1.0,
        'resonance_weight': 0.3,
        'use_batch_dialogue': True,
    })
    
    SIMULATION_STEPS = SIMULATION_PARAMS['default_steps']

    # --- 1.5 Parameter audit log ---
    logger.info("\n" + "=" * 80)
    logger.info("📋 Parameter Audit Snapshot")
    logger.info("=" * 80)

    runtime_config = {
        'network_mode': NETWORK_MODE,
        'network_data_dir': NETWORK_DATA_DIR,
        'population_path': POP_PATH,
        'network_path': NET_PATH,
        'ground_truth_path': GROUND_TRUTH_PATH,
        'simulation_steps': SIMULATION_STEPS,
    }

    effective_model_params = _sanitize_for_logging(_mask_sensitive_params(MODEL_PARAMS))
    effective_simulation_params = _sanitize_for_logging(dict(SIMULATION_PARAMS))

    logger.info("🔧 Key decision parameters:")
    logger.info(f"   - belief_threshold (bt): {MODEL_PARAMS.get('belief_threshold')}")
    logger.info(f"   - resonance_weight (rw): {MODEL_PARAMS.get('resonance_weight')}")
    logger.info(f"   - use_batch_dialogue: {MODEL_PARAMS.get('use_batch_dialogue')}")
    logger.info(f"   - batch_size: {MODEL_PARAMS.get('batch_size')}")
    logger.info(f"   - concurrent_batches: {MODEL_PARAMS.get('concurrent_batches')}")
    logger.info(f"   - max_concurrency_per_key: {MODEL_PARAMS.get('max_concurrency_per_key')}")

    logger.info("📦 Runtime Config (full):")
    logger.info(json.dumps(runtime_config, ensure_ascii=False, indent=2, sort_keys=True))

    logger.info("📦 Simulation Params (full):")
    logger.info(json.dumps(effective_simulation_params, ensure_ascii=False, indent=2, sort_keys=True))

    logger.info("📦 Model Params (full, masked):")
    logger.info(json.dumps(effective_model_params, ensure_ascii=False, indent=2, sort_keys=True))

    # Write a parameter snapshot for later checks.
    config_snapshot_dir = os.path.join(output_base_path, "config")
    os.makedirs(config_snapshot_dir, exist_ok=True)
    config_snapshot_path = os.path.join(config_snapshot_dir, "run_network_complete_config_snapshot.json")

    config_snapshot = {
        'runtime_config': runtime_config,
        'simulation_params': effective_simulation_params,
        'model_params': effective_model_params,
    }
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2, sort_keys=True)

    logger.info(f"📝 Config snapshot saved: {config_snapshot_path}")
    logger.info("=" * 80 + "\n")
    
    # --- 2. 加载预生成的网络数据 ---
    logger.info(f"Loading pre-generated network data from: {NETWORK_DATA_DIR}")
    
    if not os.path.exists(POP_PATH):
        logger.error(f"Population file not found: {POP_PATH}")
        logger.error(f"Please run the network generation script first:")
        if NETWORK_MODE == 'subsample':
            logger.error(f"  python src_v3_3_local/generate_subsample_sm_network.py")
        else:
            logger.error(f"  python src_v3_3_local/generate_full_county_networks.py")
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
    
    # 所有加载的agents都视为初始样本（预生成数据无需二次采样）
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

    # --- 2.5 清理旧的 agent_trajectories.csv（防止追加模式累积旧数据） ---
    trajectory_path = os.path.join(project_root, "data", "output", "dataframes", "agent_trajectories.csv")
    if os.path.exists(trajectory_path):
        os.remove(trajectory_path)
        logger.info(f"🧹 Removed old agent_trajectories.csv to ensure clean data")

    # --- 3. 初始化并运行模型 ---
    logger.info("\n" + "="*80)
    logger.info("🚀 Initializing VaxModel...")
    logger.info("="*80)
    logger.info(f"📊 Configuration:")
    logger.info(f"   - Population: {len(population_df):,} agents")
    logger.info(f"   - Network edges: {len(network_df):,}")
    logger.info(f"   - Simulation steps: {SIMULATION_STEPS}")
    concurrent_batches = MODEL_PARAMS.get('concurrent_batches', 18)
    batch_size = MODEL_PARAMS.get('batch_size', 8)
    logger.info(f"   - API concurrency: {concurrent_batches} batches × {batch_size} agents = {concurrent_batches * batch_size} agents/round")
    logger.info(f"   - API keys: {len(MODEL_PARAMS.get('api_keys', []))}")
    
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
            'tick': a.tick if hasattr(a, 'tick') else None,  # 添加资格时间用于调试
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
    logger.info("="*80)
    logger.info("--- Simulation End ---")

if __name__ == "__main__":
    # 使用 asyncio.run() 来启动异步主函数
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("An unhandled exception occurred during the simulation.")
