# src_latest

This folder contains the cleaned code set from `src_v3_3_local`.

## Folder layout

- `src/`: core simulation code and network-generation scripts
- `validation/`: verification and sensitivity-analysis scripts
- `visualization/`: plotting scripts and saved plot assets
- `data/`: only the small input datasets needed for the main simulation

## Included code

- Main simulation: `src/run_network_complete.py`, `src/run_subs.py`
- Network generation: `src/generate_full_county_networks.py`, `src/generate_subsample_sm_network.py`
- Verification: `validation/verification_alpha.py`
- Sensitivity analysis: `validation/sensitivity_analysis_belief_threshold.py`, `validation/sensitivity_analysis_rw_bt05.py`
- Visualizations: `visualization/replot_sensitivity_sa.py`, `visualization/plot_opinion_micro_influence_scatter.py`, `visualization/plot_opinion_sentiment_quadrant.py`, `visualization/plot_subgroup_and_tract_vax.py`
- Core modules: `src/agent.py`, `src/config.py`, `src/model.py`, `src/tools.py`
- Plot assets: `visualization/cyber_network_analysis_plots/`

## Data requirements

The code includes only the small datasets required for the main runs:

- `data/input/subsample_networks/`
- `data/input/full_county_networks/`
- `data/input/sampled_household_based/`

The raw reindex/full-population source files are intentionally excluded.
The ground-truth vaccination file `data/input/00_NYS_County_vax_rate_by_age.csv` is private and must not be uploaded.

## Privacy note

If you regenerate data locally, keep any large source CSVs out of the repository unless they are explicitly approved.
