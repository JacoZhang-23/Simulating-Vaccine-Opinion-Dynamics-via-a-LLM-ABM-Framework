#!/usr/bin/env python3
"""
生成全县（36013）完整人口的WK和SM网络
===========================================

网络生成方法：
1. WK (Workplace) - 参考sample_by_household.py的工作网络生成逻辑
   - 使用Small-world网络（Newman-Watts-Strogatz）
   - workplace规模：10-50人
   - 参数：k=4, p=0.3

2. SM (Social Media) - 使用BA模型（与generate_reconstructed_sm_network.py相同）
   - Barabási-Albert模型 + 边修剪
   - 目标平均度：可调整（默认1.5）
   - 年龄范围：13-64岁

输出位置：data/input/full_county_networks/
"""
import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from loguru import logger
from collections import defaultdict

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'input' / 'full_county_networks'
POPULATION_CSV = PROJECT_ROOT / 'data' / 'output' / 'synthetic_population_36013.csv'

# 传统网络文件（含 hh, wk, sc, dc 关系）
TRADITIONAL_NET = PROJECT_ROOT / 'data' / 'input' / '20240408_reindex_ntwk_traditional.csv'

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# ============================================================================
# 参数配置
# ============================================================================
RANDOM_SEED = 42
WORKPLACE_SIZE_RANGE = (10, 50)  # 工作场所规模

# SM网络参数（BA模型，参照原始R脚本）
SM_AVG_DEGREE = 50   # 目标平均度（R脚本: sample_pa(n, m=25, power=0.5) → avg≈50）
                     # Poisson(λ=2)在_process_batch中控制每步实际LLM交互数，不改变网络结构
SM_MIN_AGE = 13      # SM用户最小年龄
SM_MAX_AGE = 64      # SM用户最大年龄


def create_edges_smallworld(node_list, g, k=4, p=0.3):
    """
    为一组节点创建小世界网络边（参考sample_by_household.py）
    
    Args:
        node_list: 节点ID列表
        g: NetworkX图对象
        k: 每个节点连接的最近邻数量
        p: 重新连接的概率
    """
    n = len(node_list)
    if n == 0:
        return
    
    if n <= 5:
        # 小团体：完全图
        subgraph = nx.complete_graph(n)
    else:
        # 大团体：小世界网络
        subgraph = nx.newman_watts_strogatz_graph(n, k=k, p=p, seed=RANDOM_SEED)
    
    # 重新标记节点为实际的reindex值
    mapping = dict(zip(range(n), node_list))
    subgraph = nx.relabel_nodes(subgraph, mapping)
    
    # 添加边到主图
    g.add_edges_from(subgraph.edges())


def generate_workplace_network(population_df):
    """
    生成全县的工作网络
    与原始 create_social_networks.py 完全对齐：
    - 使用人口中已有的 wp 字段识别工作人口（wp 包含 'w'）
    - 按原始 wp 值直接分组，不重新分配工作场所
    - 组内构建 NWS 小世界网络（k=4, p=0.3），≤5 人用完全图

    Returns:
        edges_df: 边DataFrame（source_reindex, target_reindex, Relation）
        population_df: 原始人口DataFrame（未修改）
    """
    logger.info("\n" + "="*80)
    logger.info("Step 1: Generating Workplace Network (aligned with create_social_networks.py)")
    logger.info("="*80)

    if 'wp' not in population_df.columns:
        logger.error("  'wp' column not found in population data")
        return pd.DataFrame(columns=['source_reindex', 'target_reindex', 'Relation']), population_df

    # 识别工作人口：wp 字段包含 'w'（与原始 create_social_networks.py 一致）
    working_pop = population_df[population_df['wp'].astype(str).str.contains('w', na=False)].copy()
    logger.info(f"  Working population (wp contains 'w'): {len(working_pop):,} ({len(working_pop)/len(population_df)*100:.1f}%)")

    if len(working_pop) == 0:
        logger.warning("  No working population found")
        return pd.DataFrame(columns=['source_reindex', 'target_reindex', 'Relation']), population_df

    # 创建网络
    G = nx.Graph()
    G.add_nodes_from(working_pop['reindex'].values)

    logger.info(f"  Building workplace network edges (groupby original wp)...")

    # 按原始 wp 分组，组内构建小世界或完全图（与原始代码对齐）
    grouped = working_pop.groupby('wp')
    for wp_id, group in grouped:
        node_list = group['reindex'].tolist()
        create_edges_smallworld(node_list, G, k=4, p=0.3)

    logger.info(f"    ✅ Workplace network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # 转换为 DataFrame
    edges_data = [{'source_reindex': u, 'target_reindex': v, 'Relation': 'wk'}
                  for u, v in G.edges()]
    edges_df = pd.DataFrame(edges_data)

    return edges_df, population_df


def generate_sm_network_ba(population_df, avg_degree=SM_AVG_DEGREE, min_age=SM_MIN_AGE, max_age=SM_MAX_AGE):
    """
    使用BA模型 + 边修剪生成SM网络（与generate_reconstructed_sm_network.py相同）
    
    方法：
    1. 识别SM用户（13-64岁）
    2. 使用Barabási-Albert模型生成scale-free网络
    3. 通过随机移除边来达到目标平均度
    
    Args:
        population_df: 人口DataFrame
        avg_degree: 目标平均度
        min_age: SM用户最小年龄
        max_age: SM用户最大年龄
    
    Returns:
        edges_df: 边DataFrame（source_reindex, target_reindex, Relation）
        stats: 统计信息字典
    """
    logger.info("\n" + "="*80)
    logger.info("Step 2: Generating SM Network (BA Model + Edge Pruning)")
    logger.info("="*80)
    
    logger.info(f"  Parameters:")
    logger.info(f"    - Target avg degree: {avg_degree}")
    logger.info(f"    - Age range: {min_age}-{max_age}")
    logger.info(f"    - Random seed: {RANDOM_SEED}")
    
    # 识别SM用户
    if 'age' not in population_df.columns:
        logger.error("    Age column not found in population data")
        return pd.DataFrame(columns=['source_reindex', 'target_reindex', 'Relation']), {}
    
    sm_users = population_df[
        (population_df['age'] >= min_age) & 
        (population_df['age'] <= max_age)
    ].copy()
    
    logger.info(f"  SM-eligible population: {len(sm_users):,} ({len(sm_users)/len(population_df)*100:.1f}%)")
    
    if len(sm_users) == 0:
        logger.warning("    No SM users found")
        return pd.DataFrame(columns=['source_reindex', 'target_reindex', 'Relation']), {}
    
    # 年龄段统计
    teens = sm_users[(sm_users['age'] >= 13) & (sm_users['age'] <= 17)]
    adults = sm_users[(sm_users['age'] >= 18) & (sm_users['age'] <= 64)]
    logger.info(f"    - Teens (13-17): {len(teens):,}")
    logger.info(f"    - Adults (18-64): {len(adults):,}")
    
    # 准备用户ID列表
    user_ids = sm_users['reindex'].values
    n = len(user_ids)
    
    # 生成BA网络
    logger.info(f"\n  Generating BA network...")
    m = max(1, int(avg_degree / 2))
    logger.info(f"    - BA parameter m: {m}")
    
    np.random.seed(RANDOM_SEED)
    G = nx.barabasi_albert_graph(n, m, seed=RANDOM_SEED)
    
    # 如果需要，移除边来达到目标度
    current_avg = 2 * G.number_of_edges() / n
    if current_avg > avg_degree:
        target_edges = int((avg_degree * n) / 2)
        edges_to_remove = G.number_of_edges() - target_edges
        
        logger.info(f"    - Current edges: {G.number_of_edges():,}, target: {target_edges:,}")
        logger.info(f"    - Removing {edges_to_remove:,} edges to reach target degree...")
        
        all_edges = list(G.edges())
        np.random.shuffle(all_edges)
        
        for u, v in all_edges[:edges_to_remove]:
            G.remove_edge(u, v)
        
        logger.info(f"    ✅ After pruning: {G.number_of_edges():,} edges")
    
    # 转换节点ID
    id_mapping = {i: user_ids[i] for i in range(n)}
    edges = list(G.edges())
    
    edges_df = pd.DataFrame([
        {'source_reindex': id_mapping[u], 'target_reindex': id_mapping[v], 'Relation': 'sm'}
        for u, v in edges
    ])
    
    # 计算统计信息
    degrees = [G.degree(node) for node in G.nodes()]
    avg_degree_actual = np.mean(degrees) if degrees else 0
    median_degree = np.median(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0
    
    # 度为0的节点
    nodes_with_edges = set(G.nodes())
    isolated_nodes = n - len(nodes_with_edges)
    
    stats = {
        'eligible_population': n,
        'nodes_in_graph': G.number_of_nodes(),
        'isolated_nodes': isolated_nodes,
        'edges': len(edges_df),
        'avg_degree_graph': avg_degree_actual,
        'avg_degree_total': (2 * len(edges_df)) / n if n > 0 else 0,
        'median_degree': median_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'density': nx.density(G),
        'n_components': nx.number_connected_components(G),
        'largest_cc': len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0
    }
    
    logger.info(f"\n  ✅ SM Network Statistics:")
    logger.info(f"    - Eligible population (13-64岁): {stats['eligible_population']:,}")
    logger.info(f"    - Nodes in graph: {stats['nodes_in_graph']:,}")
    logger.info(f"    - Isolated SM users: {stats['isolated_nodes']:,}")
    logger.info(f"    - Edges: {stats['edges']:,}")
    logger.info(f"    - Avg degree (graph nodes): {stats['avg_degree_graph']:.2f}")
    logger.info(f"    - Avg degree (total eligible): {stats['avg_degree_total']:.2f} (target: {avg_degree})")
    logger.info(f"    - Density: {stats['density']:.6f}")
    logger.info(f"    - Connected components: {stats['n_components']}")
    
    return edges_df, stats


def load_existing_networks(population_df):
    """
    从 20240408_reindex_ntwk_traditional.csv 分块加载 hh, sc, dc 网络
    （该文件有 source_reindex, target_reindex, Relation 格式，包含 hh/wk/sc/dc 关系）
    
    Note: wk 关系被忽略，因为 generate_workplace_network() 会重新生成工作网络
    
    Args:
        population_df: 36013县人口，用于过滤边（仅保留两端都在本县的边）
    
    Returns:
        dict: {network_type: edges_df}  (hh, sc, dc)
    """
    logger.info("\n" + "="*80)
    logger.info("Step 3: Loading hh/sc/dc Networks from Traditional Network (chunked)")
    logger.info("="*80)
    logger.info(f"  Source file: {TRADITIONAL_NET}")
    logger.info(f"  Note: 'wk' edges are skipped (regenerated separately)")
    
    if not TRADITIONAL_NET.exists():
        logger.error(f"  ❌ Traditional network not found: {TRADITIONAL_NET}")
        return {t: pd.DataFrame(columns=['source_reindex', 'target_reindex', 'Relation'])
                for t in ('hh', 'sc', 'dc')}
    
    pop_ids = set(population_df['reindex'].values)
    logger.info(f"  Filtering to {len(pop_ids):,} county-36013 agents...")
    
    CHUNK_SIZE = 2_000_000
    TARGET_RELATIONS = {'hh', 'sc', 'dc'}
    chunks_by_type = {t: [] for t in TARGET_RELATIONS}
    
    total_rows = 0
    for chunk in pd.read_csv(TRADITIONAL_NET, chunksize=CHUNK_SIZE, low_memory=False):
        total_rows += len(chunk)
        filtered = chunk[
            chunk['Relation'].isin(TARGET_RELATIONS) &
            chunk['source_reindex'].isin(pop_ids) &
            chunk['target_reindex'].isin(pop_ids)
        ]
        for rel_type, sub in filtered.groupby('Relation'):
            chunks_by_type[rel_type].append(sub[['source_reindex', 'target_reindex', 'Relation']])
        if total_rows % (CHUNK_SIZE * 5) < CHUNK_SIZE:
            logger.info(f"    ... processed {total_rows:,} rows")
    
    logger.info(f"  ✅ Finished scanning {total_rows:,} rows")
    
    networks = {}
    for rel_type in TARGET_RELATIONS:
        if chunks_by_type[rel_type]:
            df = pd.concat(chunks_by_type[rel_type], ignore_index=True)
        else:
            df = pd.DataFrame(columns=['source_reindex', 'target_reindex', 'Relation'])
        networks[rel_type] = df
        logger.info(f"    ✅ {rel_type}: {len(df):,} edges")
    
    return networks


def save_networks(wk_edges, sm_edges, existing_networks, population_df, sm_stats):
    """
    保存所有网络文件
    
    Args:
        wk_edges: 工作网络边
        sm_edges: SM网络边
        existing_networks: 已有网络字典
        population_df: 人口数据
        sm_stats: SM网络统计信息
    """
    logger.info("\n" + "="*80)
    logger.info("Step 4: Saving Networks")
    logger.info("="*80)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    
    # 保存人口数据
    pop_output = OUTPUT_DIR / 'population.csv'
    logger.info(f"\n  Saving population: {pop_output}")
    population_df.to_csv(pop_output, index=False)
    logger.info(f"    ✅ Saved {len(population_df):,} agents")
    
    # 保存各类网络
    network_data = {
        'wk': wk_edges,
        'sm': sm_edges,
        **existing_networks
    }
    
    for net_type, edges_df in network_data.items():
        net_output = OUTPUT_DIR / f'network_{net_type}.csv'
        logger.info(f"  Saving {net_type} network: {net_output}")
        edges_df.to_csv(net_output, index=False)
        logger.info(f"    ✅ Saved {len(edges_df):,} edges")
    
    # 保存完整网络（所有类型合并）
    complete_edges = pd.concat([edges_df for edges_df in network_data.values()], ignore_index=True)
    complete_output = OUTPUT_DIR / 'network_complete.csv'
    logger.info(f"\n  Saving complete network: {complete_output}")
    complete_edges.to_csv(complete_output, index=False)
    logger.info(f"    ✅ Saved {len(complete_edges):,} edges")
    
    # 打印网络统计
    logger.info(f"\n  Network Summary:")
    for net_type in ['hh', 'wk', 'sc', 'dc', 'sm']:
        if net_type in network_data:
            count = len(network_data[net_type])
            pct = count / len(complete_edges) * 100 if len(complete_edges) > 0 else 0
            logger.info(f"    - {net_type}: {count:,} ({pct:.1f}%)")
    logger.info(f"    - Total: {len(complete_edges):,}")
    
    return complete_edges


def generate_report(population_df, wk_edges, sm_edges, sm_stats, existing_networks, complete_edges):
    """
    生成网络统计报告
    """
    logger.info("\n" + "="*80)
    logger.info("Step 5: Generating Report")
    logger.info("="*80)
    
    report_path = OUTPUT_DIR / 'NETWORK_REPORT.md'
    
    # 构建图
    G = nx.Graph()
    for _, row in complete_edges.iterrows():
        G.add_edge(row['source_reindex'], row['target_reindex'])
    
    total_pop = len(population_df)
    graph_nodes = G.number_of_nodes()
    isolated = total_pop - graph_nodes
    
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values())) if degrees else 0
    
    # 写入报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Full County Network Generation Report\n\n")
        f.write(f"**Generated at**: {pd.Timestamp.now()}\n\n")
        
        f.write("## Parameters\n\n")
        f.write(f"- **Random Seed**: {RANDOM_SEED}\n")
        f.write(f"- **Workplace Size Range**: {WORKPLACE_SIZE_RANGE}\n")
        f.write(f"- **SM Target Avg Degree**: {SM_AVG_DEGREE}\n")
        f.write(f"- **SM Age Range**: {SM_MIN_AGE}-{SM_MAX_AGE}\n\n")
        
        f.write("## Population Statistics\n\n")
        f.write(f"- **Total Population**: {total_pop:,}\n")
        f.write(f"- **Nodes in Network**: {graph_nodes:,} ({graph_nodes/total_pop*100:.1f}%)\n")
        f.write(f"- **Isolated Nodes**: {isolated:,} ({isolated/total_pop*100:.1f}%)\n\n")
        
        f.write("## Network Statistics\n\n")
        f.write("| Network Type | Edges | Percentage |\n")
        f.write("|--------------|-------|------------|\n")
        for net_type in ['hh', 'wk', 'sc', 'dc', 'sm']:
            edges_df = existing_networks.get(net_type, pd.DataFrame()) if net_type not in ['wk', 'sm'] else (wk_edges if net_type == 'wk' else sm_edges)
            count = len(edges_df)
            pct = count / len(complete_edges) * 100 if len(complete_edges) > 0 else 0
            f.write(f"| {net_type} | {count:,} | {pct:.1f}% |\n")
        f.write(f"| **Total** | {len(complete_edges):,} | 100.0% |\n\n")
        
        f.write("## Overall Network Properties\n\n")
        f.write(f"- **Total Edges**: {G.number_of_edges():,}\n")
        f.write(f"- **Average Degree**: {avg_degree:.2f}\n")
        f.write(f"- **Network Density**: {nx.density(G):.6f}\n")
        f.write(f"- **Connected Components**: {nx.number_connected_components(G)}\n\n")
        
        f.write("## SM Network Details (BA Model)\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Eligible Population | {sm_stats['eligible_population']:,} |\n")
        f.write(f"| Nodes in Graph | {sm_stats['nodes_in_graph']:,} |\n")
        f.write(f"| Isolated Nodes | {sm_stats['isolated_nodes']:,} |\n")
        f.write(f"| Edges | {sm_stats['edges']:,} |\n")
        f.write(f"| Avg Degree (Graph) | {sm_stats['avg_degree_graph']:.2f} |\n")
        f.write(f"| Avg Degree (Total) | {sm_stats['avg_degree_total']:.2f} |\n")
        f.write(f"| Density | {sm_stats['density']:.6f} |\n")
        f.write(f"| Components | {sm_stats['n_components']} |\n\n")
        
        f.write("## Method Details\n\n")
        f.write("### Workplace Network (WK)\n")
        f.write("- **Method**: Small-world network (Newman-Watts-Strogatz)\n")
        f.write(f"- **Workplace size**: {WORKPLACE_SIZE_RANGE[0]}-{WORKPLACE_SIZE_RANGE[1]} people\n")
        f.write("- **Network parameters**: k=4, p=0.3\n\n")
        
        f.write("### Social Media Network (SM)\n")
        f.write("- **Method**: Barabási-Albert model + Edge pruning\n")
        f.write(f"- **Target avg degree**: {SM_AVG_DEGREE}\n")
        f.write(f"- **Age range**: {SM_MIN_AGE}-{SM_MAX_AGE}\n")
        f.write("- **Process**: Generate BA network → Randomly remove edges to reach target degree\n\n")
    
    logger.info(f"  ✅ Report saved: {report_path}")


def main():
    """主执行流程"""
    logger.info("\n" + "="*80)
    logger.info("GENERATE FULL COUNTY NETWORKS (36013)")
    logger.info("="*80)
    logger.info(f"Output: {OUTPUT_DIR}")
    
    # 加载人口数据
    logger.info(f"\n  Loading population: {POPULATION_CSV}")
    pop_df = pd.read_csv(POPULATION_CSV, low_memory=False)
    logger.info(f"    ✅ Loaded {len(pop_df):,} agents")
    
    # 生成WK网络
    wk_edges, pop_df = generate_workplace_network(pop_df)
    
    # 生成SM网络（BA模型）
    sm_edges, sm_stats = generate_sm_network_ba(pop_df)
    
    # 加载已有网络（hh/sc/dc从传统网络文件中提取，过滤出36013县的边）
    existing_networks = load_existing_networks(pop_df)
    
    # 保存所有网络
    complete_edges = save_networks(wk_edges, sm_edges, existing_networks, pop_df, sm_stats)
    
    # 生成报告
    generate_report(pop_df, wk_edges, sm_edges, sm_stats, existing_networks, complete_edges)
    
    logger.info("\n" + "="*80)
    logger.info("✅ FULL COUNTY NETWORKS GENERATED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == '__main__':
    main()
