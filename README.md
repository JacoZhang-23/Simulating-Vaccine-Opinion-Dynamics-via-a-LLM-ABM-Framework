# Simulating Vaccine Opinion Dynamics via a Large Language Model-Driven Agent-based Modeling Framework: A Case Study of Chautauqua County, New York

This repository contains the code for the paper above. The project combines agent-based modeling, social-network generation, demographic data, and large language model interactions to simulate vaccine opinion dynamics and vaccination uptake at both the micro and macro levels.

## Project Overview

The main code has been cleaned and organized under `src_latest/`.

- `src_latest/src/`: core simulation code, network generation, model logic, and utilities
- `src_latest/validation/`: verification and sensitivity-analysis scripts
- `src_latest/visualization/`: plotting scripts and generated figures
- `src_latest/data/`: only the small input datasets needed for the main simulation runs

## Main Components

- Core agent and model logic for vaccine opinion dynamics
- LLM-driven dialogue and belief-updating workflow
- County-level workplace and social network construction
- Verification and sensitivity analysis for model behavior
- Visualization scripts for simulation and validation outputs

## Data Notes

Only the small runtime datasets needed for the main experiments are included.
The private vaccination file `data/input/00_NYS_County_vax_rate_by_age.csv` is intentionally excluded.

## Requirements

The project uses Python and common scientific-computing libraries such as pandas, numpy, networkx, loguru, dotenv, and Mesa-related simulation components.

## Running the Code

Typical entry points are located in `src_latest/src/`.

Examples:

- `python src_latest/src/run_network_complete.py`
- `python src_latest/src/run_subs.py`
- `python src_latest/validation/verification_alpha.py`

The exact script depends on whether you want the full-county run, subsample run, or validation workflow.

## Citation

If you use this code in research, please cite the associated paper.
