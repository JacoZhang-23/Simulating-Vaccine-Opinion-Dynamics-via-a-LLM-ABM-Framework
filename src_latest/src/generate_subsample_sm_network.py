#!/usr/bin/env python3
"""
为子样本生成SM网络（BA模型）
==========================================

方法：
- 使用Barabási-Albert模型 + 边修剪（与generate_reconstructed_sm_network.py相同）
- 目标平均度：1.5
- 年龄范围：13-64岁

输出：data/input/subsample_networks/
"""
import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from loguru import logger

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SAMPLE_DIR = PROJECT_ROOT / 'data' / 'input' / 'sampled_household_based'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'input' / 'subsample_networks'
POPULATION_CSV = SAMPLE_DIR / 'population.csv'

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# ============================================================================
# 参数配置
# ============================================================================
RANDOM_SEED = 42
SM_AVG_DEGREE = 50   # 目标平均度（与原始R脚本一致: sample_pa(n, m=25, power=0.5) → avg≈50）
                     # Poisson(λ=2)在_process_batch中控制每步实际LLM交互数，不改变网络结构
SM_MIN_AGE = 13      # SM用户最小年龄
SM_MAX_AGE = 64      # SM用户最大年龄


def load_population():
    """
    加载子样本人口数据
    
    Returns:
        pd.DataFrame: 人口数据
    """
    logger.info("="*80)
    logger.info("Step 1: Loading Subsample Population")
    logger.info("="*80)
    
    logger.info(f"  Loading: {POPULATION_CSV}")
    pop_df = pd.read_csv(POPULATION_CSV)
    logger.info(f"    ✅ Total population: {len(pop_df):,}")
    
    # 检查必要字段
    if 'age' not in pop_df.columns:
        logger.error("    ❌ Age column not found")
        sys.exit(1)
    
    if 'reindex' not in pop_df.columns:
        logger.error("    ❌ reindex column not found")
        sys.exit(1)
    
    return pop_df


def generate_sm_network_ba(population_df, avg_degree=SM_AVG_DEGREE, min_age=SM_MIN_AGE, max_age=SM_MAX_AGE):
    """
    使用BA模型 + 边修剪生成SM网络
    
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
    sm_users = population_df[
        (population_df['age'] >= min_age) & 
        (population_df['age'] <= max_age)
    ].copy()
    
    logger.info(f"\n  SM-eligible population: {len(sm_users):,} ({len(sm_users)/len(population_df)*100:.1f}%)")
    
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


def save_networks(sm_edges, pop_df):
    """
    保存SM网络和更新完整网络
    
    Args:
        sm_edges: SM边DataFrame
        pop_df: 人口DataFrame
    """
    logger.info("\n" + "="*80)
    logger.info("Step 3: Saving Networks")
    logger.info("="*80)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    
    # 复制人口数据
    pop_output = OUTPUT_DIR / 'population.csv'
    logger.info(f"\n  Copying population: {pop_output}")
    pop_df.to_csv(pop_output, index=False)
    logger.info(f"    ✅ Saved {len(pop_df):,} agents")
    
    # 保存SM网络
    sm_output = OUTPUT_DIR / 'network_sm.csv'
    logger.info(f"\n  Saving SM network: {sm_output}")
    sm_edges.to_csv(sm_output, index=False)
    logger.info(f"    ✅ Saved {len(sm_edges):,} edges")
    
    # 加载其他网络类型
    logger.info(f"\n  Loading other network types from sample directory...")
    other_networks = {}
    for net_type in ['hh', 'wk', 'sc', 'dc']:
        net_file = SAMPLE_DIR / f'network_{net_type}.csv'
        if net_file.exists():
            df = pd.read_csv(net_file)
            other_networks[net_type] = df
            # 复制到输出目录
            output_file = OUTPUT_DIR / f'network_{net_type}.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"    ✅ Copied {net_type}: {len(df):,} edges")
        else:
            logger.warning(f"    ⚠️  {net_type} network not found")
    
    # 合并所有网络
    all_edges = [sm_edges]
    for df in other_networks.values():
        all_edges.append(df)
    
    complete_df = pd.concat(all_edges, ignore_index=True)
    
    # 保存完整网络
    complete_output = OUTPUT_DIR / 'network_complete.csv'
    logger.info(f"\n  Saving complete network: {complete_output}")
    complete_df.to_csv(complete_output, index=False)
    logger.info(f"    ✅ Saved {len(complete_df):,} edges")
    
    # 打印网络统计
    logger.info(f"\n  Network Summary:")
    for net_type in ['hh', 'wk', 'sc', 'dc', 'sm']:
        if net_type == 'sm':
            count = len(sm_edges)
        elif net_type in other_networks:
            count = len(other_networks[net_type])
        else:
            count = 0
        
        pct = count / len(complete_df) * 100 if len(complete_df) > 0 else 0
        logger.info(f"    - {net_type}: {count:,} ({pct:.1f}%)")
    logger.info(f"    - Total: {len(complete_df):,}")
    
    return complete_df


def calculate_complete_network_stats(complete_df, pop_df):
    """
    计算完整网络统计
    
    Returns:
        dict: 统计信息
    """
    logger.info("\n" + "="*80)
    logger.info("Step 4: Calculating Complete Network Statistics")
    logger.info("="*80)
    
    # 构建图
    G = nx.Graph()
    for _, row in complete_df.iterrows():
        G.add_edge(row['source_reindex'], row['target_reindex'])
    
    total_pop = len(pop_df)
    graph_nodes = G.number_of_nodes()
    isolated = total_pop - graph_nodes
    
    degrees = dict(G.degree())
    avg_degree_graph = np.mean(list(degrees.values())) if degrees else 0
    avg_degree_pop = (G.number_of_edges() * 2) / total_pop if total_pop > 0 else 0
    
    stats = {
        'total_pop': total_pop,
        'graph_nodes': graph_nodes,
        'isolated': isolated,
        'total_edges': G.number_of_edges(),
        'avg_degree_graph': avg_degree_graph,
        'avg_degree_pop': avg_degree_pop,
        'n_components': nx.number_connected_components(G)
    }
    
    logger.info(f"  Total population: {stats['total_pop']:,}")
    logger.info(f"  Nodes in graph: {stats['graph_nodes']:,} ({stats['graph_nodes']/stats['total_pop']*100:.1f}%)")
    logger.info(f"  Isolated nodes: {stats['isolated']:,} ({stats['isolated']/stats['total_pop']*100:.1f}%)")
    logger.info(f"  Total edges: {stats['total_edges']:,}")
    logger.info(f"  Avg degree (graph): {stats['avg_degree_graph']:.2f}")
    logger.info(f"  Avg degree (total pop): {stats['avg_degree_pop']:.2f}")
    logger.info(f"  Connected components: {stats['n_components']}")
    
    return stats


def generate_report(sm_stats, complete_stats, complete_df):
    """
    生成报告文件
    """
    logger.info("\n" + "="*80)
    logger.info("Step 5: Generating Report")
    logger.info("="*80)
    
    report_path = OUTPUT_DIR / 'NETWORK_REPORT.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Subsample Network Generation Report (BA Model)\n\n")
        f.write(f"**Generated at**: {pd.Timestamp.now()}\n\n")
        
        f.write("## Parameters\n\n")
        f.write(f"- **Random Seed**: {RANDOM_SEED}\n")
        f.write(f"- **SM Target Avg Degree**: {SM_AVG_DEGREE}\n")
        f.write(f"- **SM Age Range**: {SM_MIN_AGE}-{SM_MAX_AGE}\n")
        f.write(f"- **Method**: Barabási-Albert model + Edge pruning\n\n")
        
        f.write("## Population Statistics\n\n")
        f.write(f"- **Total Population**: {complete_stats['total_pop']:,}\n")
        f.write(f"- **Nodes in Network**: {complete_stats['graph_nodes']:,} ({complete_stats['graph_nodes']/complete_stats['total_pop']*100:.1f}%)\n")
        f.write(f"- **Isolated Nodes**: {complete_stats['isolated']:,} ({complete_stats['isolated']/complete_stats['total_pop']*100:.1f}%)\n\n")
        
        f.write("## SM Network Statistics (BA Model)\n\n")
        f.write("| Metric | Value | Notes |\n")
        f.write("|--------|-------|-------|\n")
        f.write(f"| Eligible Population (13-64岁) | {sm_stats['eligible_population']:,} | Total SM-eligible users |\n")
        f.write(f"| Nodes in Graph | {sm_stats['nodes_in_graph']:,} | {sm_stats['nodes_in_graph']/sm_stats['eligible_population']*100:.1f}% of eligible |\n")
        f.write(f"| Isolated SM Users | {sm_stats['isolated_nodes']:,} | {sm_stats['isolated_nodes']/sm_stats['eligible_population']*100:.1f}% of eligible |\n")
        f.write(f"| Edges | {sm_stats['edges']:,} | Total connections |\n")
        f.write(f"| Avg Degree (Graph) | {sm_stats['avg_degree_graph']:.2f} | Among connected nodes |\n")
        f.write(f"| Avg Degree (Total) | {sm_stats['avg_degree_total']:.2f} | Target: {SM_AVG_DEGREE} |\n")
        f.write(f"| Median Degree | {sm_stats['median_degree']:.0f} | - |\n")
        f.write(f"| Max Degree | {sm_stats['max_degree']} | - |\n")
        f.write(f"| Min Degree | {sm_stats['min_degree']} | - |\n")
        f.write(f"| Density | {sm_stats['density']:.6f} | - |\n")
        f.write(f"| Connected Components | {sm_stats['n_components']} | - |\n")
        f.write(f"| Largest Component | {sm_stats['largest_cc']:,} nodes | - |\n\n")
        
        f.write("## Complete Network Statistics\n\n")
        f.write("| Network Type | Edges | Percentage |\n")
        f.write("|--------------|-------|------------|\n")
        for net_type in ['hh', 'wk', 'sc', 'dc', 'sm']:
            edges = complete_df[complete_df['Relation'] == net_type]
            count = len(edges)
            pct = count / len(complete_df) * 100 if len(complete_df) > 0 else 0
            f.write(f"| {net_type} | {count:,} | {pct:.1f}% |\n")
        f.write(f"| **Total** | {complete_stats['total_edges']:,} | 100.0% |\n\n")
        
        f.write("## Overall Network Properties\n\n")
        f.write(f"- **Total Edges**: {complete_stats['total_edges']:,}\n")
        f.write(f"- **Average Degree (Graph)**: {complete_stats['avg_degree_graph']:.2f}\n")
        f.write(f"- **Average Degree (Total Pop)**: {complete_stats['avg_degree_pop']:.2f}\n")
        f.write(f"- **Connected Components**: {complete_stats['n_components']}\n\n")
        
        f.write("## Method: Barabási-Albert Model + Edge Pruning\n\n")
        f.write("The SM network is generated using a **scale-free model**:\n\n")
        f.write("1. **Generate BA network**: Use Barabási-Albert model with preferential attachment\n")
        f.write(f"2. **Edge pruning**: Randomly remove edges to reach target avg degree ({SM_AVG_DEGREE})\n")
        f.write("3. **Result**: Scale-free network with adjustable density\n\n")
        f.write("**Advantages**:\n")
        f.write("- Scale-free property (power-law degree distribution)\n")
        f.write("- Adjustable average degree\n")
        f.write("- Efficient generation for large networks\n\n")
    
    logger.info(f"  ✅ Report saved: {report_path}")


def main():
    """主执行流程"""
    logger.info("\n" + "="*80)
    logger.info("GENERATE SUBSAMPLE SM NETWORK (BA MODEL)")
    logger.info("="*80)
    logger.info(f"Target Avg Degree: {SM_AVG_DEGREE}")
    
    # 加载人口
    pop_df = load_population()
    
    # 生成SM网络
    sm_edges, sm_stats = generate_sm_network_ba(pop_df)
    
    # 保存网络
    complete_df = save_networks(sm_edges, pop_df)
    
    # 计算完整网络统计
    complete_stats = calculate_complete_network_stats(complete_df, pop_df)
    
    # 生成报告
    generate_report(sm_stats, complete_stats, complete_df)
    
    logger.info("\n" + "="*80)
    logger.info("✅ SUBSAMPLE SM NETWORK GENERATED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"  Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
