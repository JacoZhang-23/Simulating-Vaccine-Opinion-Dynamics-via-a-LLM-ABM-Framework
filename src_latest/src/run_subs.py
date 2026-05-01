# run_subs.py - subsample simulation entry (~12k agents, fast test)
# Data directory: data/input/subsample_networks/
import asyncio
import os
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

async def main():
    """Async main function for the subsample simulation."""
    # --- 0. Logging and environment ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # LLMIP_new/
    output_base_path = os.path.join(project_root, "data", "output")
    log_file_path = os.path.join(output_base_path, "logs", "simulation_subs_{time}.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger.add(log_file_path, rotation="5 MB", level=SIMULATION_PARAMS['log_level'])
    logger.info("--- Simulation Start [SUBSAMPLE] ---")

    # Load environment variables.
    load_dotenv()

    # --- 1. Paths and parameters ---
    input_base_path = os.path.join(project_root, 'data', 'input')

    # Always use the subsample network.
    NETWORK_DATA_DIR = os.path.join(input_base_path, 'subsample_networks')

    POP_PATH = os.path.join(NETWORK_DATA_DIR, 'population.csv')
    NET_PATH = os.path.join(NETWORK_DATA_DIR, 'network_complete.csv')
    GROUND_TRUTH_PATH = os.path.join(input_base_path, '00_NYS_County_vax_rate_by_age.csv')

    # Load model parameters from config.
    MODEL_PARAMS = get_model_params()

    # Override the main dialogue parameters.
    MODEL_PARAMS.update({
        'belief_threshold': 2,
        'resonance_weight': 0.3,
        'use_batch_dialogue': True,
    })

    SIMULATION_STEPS = SIMULATION_PARAMS['default_steps']

    # --- 2. Load prebuilt network data ---
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

    # Treat all loaded agents as the initial sample.
    initial_sample_ids = set(population_df['reindex'].values)

    # Check required fields.
    required_fields = ['id', 'age', 'gender', 'hhold', 'htype', 'wp', 'urban', 'reindex',
                      'GEOID_cty', 'if_employed', 'household_id', 'personal_income',
                      'education', 'occupation', 'health_insurance', 'FINCP', 'HHT',
                      'num_children', 'family_size']

    missing_fields = [field for field in required_fields if field not in population_df.columns]
    if missing_fields:
        logger.warning(f"Missing required fields: {missing_fields}")
        raise ValueError(f"Missing required fields in population data: {missing_fields}")

    population_df = add_essential_worker_field(population_df)

    # Build the profile field.
    logger.info("Generating enhanced profiles...")
    population_df['profile'] = population_df.apply(enhance_profile_with_pums_features, axis=1)

    # Add profile embeddings.
    logger.info("Adding profile embeddings...")
    population_df = add_profile_embedding_to_population(population_df)

    # Add vaccination eligibility ticks.
    logger.info("Adding vaccination eligibility ticks...")
    population_df = add_tick_field_to_population(population_df)

    logger.info("Loading ground truth vaccination data...")
    ground_truth_df = load_ground_truth_data(GROUND_TRUTH_PATH)

    # --- 2.5 Remove stale agent_trajectories.csv ---
    trajectory_path = os.path.join(project_root, "data", "output", "dataframes", "agent_trajectories.csv")
    if os.path.exists(trajectory_path):
        os.remove(trajectory_path)
        logger.info(f"🧹 Removed old agent_trajectories.csv to ensure clean data")

    # --- 3. Initialize and run the model ---
    logger.info("\n" + "="*80)
    logger.info("🚀 Initializing VaxModel [SUBSAMPLE]...")
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

    # --- 4. Save and analyze results ---
    logger.info("Simulation finished. Saving results...")

    output_df_path = os.path.join(output_base_path, "dataframes")
    os.makedirs(output_df_path, exist_ok=True)

    # --- 4.1 Save final agent state ---
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
    logger.info(f"Final agent data saved.")

    # --- 4.2 Save step-level data ---
    model.datacollector.to_csv(os.path.join(output_df_path, "step_by_step_data.csv"), index=False)
    logger.info(f"Step-by-step data saved.")

    # Save dialogue statistics.
    dialogue_stats = model.get_dialogue_statistics()
    if dialogue_stats['total_dialogues'] > 0:
        dialogue_stats['dialogue_df'].to_csv(os.path.join(output_df_path, "dialogue_history.csv"), index=False)
        logger.info(f"Dialogue history saved.")

    # Plot figures and compute metrics.
    metrics = plot_vaccination_rate(model.datacollector, ground_truth_df, output_base_path)

    # --- 4.3 Print summary statistics ---
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
    logger.info("--- Simulation End [SUBSAMPLE] ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("An unhandled exception occurred during the simulation.")
