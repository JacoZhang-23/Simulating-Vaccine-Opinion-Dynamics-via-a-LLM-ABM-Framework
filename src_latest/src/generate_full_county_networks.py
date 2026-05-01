#!/usr/bin/env python3
"""
Generate WK and SM networks for the full county population (36,013).

Network generation plan:
1. WK (Workplace)
    - Use a small-world network (Newman-Watts-Strogatz)
    - Workplace size: 10-50 people
    - Parameters: k=4, p=0.3

2. SM (Social Media)
    - Use a BA model with edge pruning
    - Target average degree: configurable (default 1.5)
    - Age range: 13-64

Output directory: data/input/full_county_networks/
"""
import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from loguru import logger
from collections import defaultdict

# Set up paths.
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'input' / 'full_county_networks'
POPULATION_CSV = PROJECT_ROOT / 'data' / 'output' / 'synthetic_population_36013.csv'

# Traditional network file containing hh, wk, sc, and dc relations.
TRADITIONAL_NET = PROJECT_ROOT / 'data' / 'input' / '20240408_reindex_ntwk_traditional.csv'

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# ============================================================================
# Parameters
# ============================================================================
RANDOM_SEED = 42
WORKPLACE_SIZE_RANGE = (10, 50)  # Workplace size.

# SM network parameters (BA model, aligned with the original R script).
SM_AVG_DEGREE = 50   # Target average degree (sample_pa(n, m=25, power=0.5) -> avg≈50).
                     # Poisson(λ=2) in _process_batch limits per-step LLM calls without changing the network structure.
SM_MIN_AGE = 13      # Minimum SM user age.
SM_MAX_AGE = 64      # Maximum SM user age.


def create_edges_smallworld(node_list, g, k=4, p=0.3):
    """Create small-world edges for a group of nodes."""
    n = len(node_list)
    if n == 0:
        return
    
    if n <= 5:
        # Small groups use a complete graph.
        subgraph = nx.complete_graph(n)
    else:
        # Larger groups use a small-world graph.
        subgraph = nx.newman_watts_strogatz_graph(n, k=k, p=p, seed=RANDOM_SEED)
    
    # Relabel nodes to the actual reindex values.
    mapping = dict(zip(range(n), node_list))
    subgraph = nx.relabel_nodes(subgraph, mapping)
    
    # Add edges to the main graph.
    g.add_edges_from(subgraph.edges())


def generate_workplace_network(population_df):
    """Generate the full-county workplace network."""
    logger.info("\n" + "="*80)
    logger.info("Step 1: Generating Workplace Network (aligned with create_social_networks.py)")
    logger.info("="*80)

    if 'wp' not in population_df.columns:
        logger.error("  'wp' column not found in population data")
        return pd.DataFrame(columns=['source_reindex', 'target_reindex', 'Relation']), population_df

    # Identify working population: wp contains 'w'.
    working_pop = population_df[population_df['wp'].astype(str).str.contains('w', na=False)].copy()
    logger.info(f"  Working population (wp contains 'w'): {len(working_pop):,} ({len(working_pop)/len(population_df)*100:.1f}%)")

    if len(working_pop) == 0:
        logger.warning("  No working population found")
        return pd.DataFrame(columns=['source_reindex', 'target_reindex', 'Relation']), population_df

    # Create the graph.
    G = nx.Graph()
    G.add_nodes_from(working_pop['reindex'].values)

    logger.info(f"  Building workplace network edges (groupby original wp)...")

    # Group by original wp and build a small-world or complete graph within each group.
    grouped = working_pop.groupby('wp')
    for wp_id, group in grouped:
        node_list = group['reindex'].tolist()
        create_edges_smallworld(node_list, G, k=4, p=0.3)

    logger.info(f"    ✅ Workplace network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Convert to a DataFrame.
    edges_data = [{'source_reindex': u, 'target_reindex': v, 'Relation': 'wk'}
                  for u, v in G.edges()]
    edges_df = pd.DataFrame(edges_data)

    return edges_df, population_df


def generate_sm_network_ba(population_df, avg_degree=SM_AVG_DEGREE, min_age=SM_MIN_AGE, max_age=SM_MAX_AGE):
    """Generate the SM network with a BA model and edge pruning.

    Steps:
    1. Identify SM users (ages 13-64)
    2. Generate a scale-free network with the Barabási-Albert model
    3. Randomly remove edges until the target average degree is reached

    Args:
        population_df: population DataFrame
        avg_degree: target average degree
        min_age: minimum SM user age
        max_age: maximum SM user age

    Returns:
        edges_df: edge DataFrame with source_reindex, target_reindex, and Relation
        stats: summary statistics dictionary
    """
    logger.info("\n" + "="*80)
    logger.info("Step 2: Generating SM Network (BA Model + Edge Pruning)")
    logger.info("="*80)
    
    logger.info(f"  Parameters:")
    logger.info(f"    - Target avg degree: {avg_degree}")
    logger.info(f"    - Age range: {min_age}-{max_age}")
    logger.info(f"    - Random seed: {RANDOM_SEED}")
    
    # Identify SM users.
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
    
    # Age-range statistics.
    teens = sm_users[(sm_users['age'] >= 13) & (sm_users['age'] <= 17)]
    adults = sm_users[(sm_users['age'] >= 18) & (sm_users['age'] <= 64)]
    logger.info(f"    - Teens (13-17): {len(teens):,}")
    logger.info(f"    - Adults (18-64): {len(adults):,}")
    
    # Prepare the user ID list.
    user_ids = sm_users['reindex'].values
    n = len(user_ids)
    
    # Generate the BA network.
    logger.info(f"\n  Generating BA network...")
    m = max(1, int(avg_degree / 2))
    logger.info(f"    - BA parameter m: {m}")
    
    np.random.seed(RANDOM_SEED)
    G = nx.barabasi_albert_graph(n, m, seed=RANDOM_SEED)
    
    # Remove edges if needed to reach the target degree.
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
    
    # Convert node IDs.
    id_mapping = {i: user_ids[i] for i in range(n)}
    edges = list(G.edges())
    
    edges_df = pd.DataFrame([
        {'source_reindex': id_mapping[u], 'target_reindex': id_mapping[v], 'Relation': 'sm'}
        for u, v in edges
    ])
    
    # Compute summary statistics.
    degrees = [G.degree(node) for node in G.nodes()]
    avg_degree_actual = np.mean(degrees) if degrees else 0
    median_degree = np.median(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0
    
    # Nodes with degree 0.
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
    Load hh, sc, and dc networks from 20240408_reindex_ntwk_traditional.csv in chunks.
    The file uses source_reindex, target_reindex, and Relation columns and includes hh/wk/sc/dc relations.

    Note: wk relations are ignored because generate_workplace_network() rebuilds the workplace network.
    
    Args:
        population_df: population for county 36013, used to filter edges and keep only edges whose endpoints are both in the county.
    
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
    Save all network files.
    
    Args:
        wk_edges: workplace network edges
        sm_edges: SM network edges
        existing_networks: existing network dictionary
        population_df: population data
        sm_stats: SM network statistics
    """
    logger.info("\n" + "="*80)
    logger.info("Step 4: Saving Networks")
    logger.info("="*80)
    
    # Create the output directory.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    
    # Save the population data.
    pop_output = OUTPUT_DIR / 'population.csv'
    logger.info(f"\n  Saving population: {pop_output}")
    population_df.to_csv(pop_output, index=False)
    logger.info(f"    ✅ Saved {len(population_df):,} agents")
    
    # Save each network layer.
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
    
    # Save the combined full network.
    complete_edges = pd.concat([edges_df for edges_df in network_data.values()], ignore_index=True)
    complete_output = OUTPUT_DIR / 'network_complete.csv'
    logger.info(f"\n  Saving complete network: {complete_output}")
    complete_edges.to_csv(complete_output, index=False)
    logger.info(f"    ✅ Saved {len(complete_edges):,} edges")
    
    # Print network statistics.
    logger.info(f"\n  Network Summary:")
    for net_type in ['hh', 'wk', 'sc', 'dc', 'sm']:
        if net_type in network_data:
            count = len(network_data[net_type])
            pct = count / len(complete_edges) * 100 if len(complete_edges) > 0 else 0
            logger.info(f"    - {net_type}: {count:,} ({pct:.1f}%)")
    logger.info(f"    - Total: {len(complete_edges):,}")
    
    return complete_edges


def generate_report(population_df, wk_edges, sm_edges, sm_stats, existing_networks, complete_edges):
    """Generate the network summary report."""
    logger.info("\n" + "="*80)
    logger.info("Step 5: Generating Report")
    logger.info("="*80)
    
    report_path = OUTPUT_DIR / 'NETWORK_REPORT.md'
    
    # Build the graph.
    G = nx.Graph()
    for _, row in complete_edges.iterrows():
        G.add_edge(row['source_reindex'], row['target_reindex'])
    
    total_pop = len(population_df)
    graph_nodes = G.number_of_nodes()
    isolated = total_pop - graph_nodes
    
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values())) if degrees else 0
    
    # Write the report.
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
