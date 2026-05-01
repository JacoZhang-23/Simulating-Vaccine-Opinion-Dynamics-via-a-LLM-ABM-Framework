# src/tools.py

import pandas as pd
import numpy as np
import networkx as nx
from loguru import logger
import random
import os
import matplotlib.pyplot as plt
import datetime
import torch
from transformers import BertTokenizer, BertModel
import pickle
from tqdm import tqdm
import logging

MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# Move the model to GPU when available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

logger.info(f"BERT model loaded on device: {device}")

def text2embedding(sentence):
    """Convert one text string to an embedding."""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    # Move inputs to GPU.
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    sentence_embedding = last_hidden_states[:, 0, :]
    sentence_embedding = sentence_embedding.cpu().numpy().flatten()
    return sentence_embedding

def batch_text2embedding(texts, batch_size=32):
    """
    Batch text-to-embedding conversion with GPU acceleration.
    """
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc='Batch Embedding'):
        batch_texts = texts[i:i+batch_size]
        
        # Batch tokenize.
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128,
            padding=True  # Batch processing needs padding.
        )
        
        # Move inputs to GPU.
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the [CLS] embedding.
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    
    # Merge batch results.
    return np.vstack(embeddings)

def save_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f'{filename} saved!')

def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        print(f'{filename} loaded!')
        return data

def remove_substrings(input_string, substrings_to_remove):
    result = input_string
    for substring in substrings_to_remove:
        result = result.replace(substring, "")
    return result

def parse_id_for_geoid_and_employment(id_str):
    """
    Parse ID string to extract GEOID_cty (first 5 digits) and employment status.
    ID format example: 36001001100w1370 where 'w' indicates employed, 'h' indicates not employed.
    """
    if not isinstance(id_str, str):
        return None, None
    
    # Extract first 5 digits for GEOID_cty
    geoid_cty = None
    if len(id_str) >= 5:
        try:
            geoid_cty = int(id_str[:5])
        except ValueError:
            geoid_cty = None
    
    # Determine employment status based on 'w' or 'h' in ID
    if_employed = None
    if 'w' in id_str.lower():
        if_employed = 1
    elif 'h' in id_str.lower():
        if_employed = 0
    
    return geoid_cty, if_employed

# Deprecated helper methods.
# def create_user_embedding():
#     """
#     Deprecated: this method generated user embeddings and has been replaced by add_profile_embedding_to_population.
#     """
#     pass

# def infer_hht_and_children_from_csv(population_df, network_df):
#     """
#     Deprecated: this method inferred HHT and child counts from CSV, but has been replaced by add_csv_based_features_to_population.
#     """
#     pass

# def add_essential_worker_field(population_df, essential_worker_percentage=0.182):
#     """
#     已过时：此方法用于添加必要工作者字段，但已被新的数据处理逻辑取代。
#     """
#     pass

# def load_ground_truth_vax_data(ground_truth_path):
#     """
#     已过时：此方法用于加载疫苗接种数据，但已被load_ground_truth_data方法取代。
#     """
#     pass

def add_profile_embedding_to_population(population_df, batch_size=64):
    """
    用 profile 文本生成 BERT embedding，并添加到 DataFrame。
    【优化】使用批量处理，充分利用GPU加速
    
    Args:
        population_df: 包含profile列的DataFrame
        batch_size: 批量大小，根据GPU内存调整（建议32-128）
    """
    logger.info(f"正在为人口数据批量生成人口学embedding（batch_size={batch_size}，使用设备: {device}）...")
    
    # 提取所有profile文本
    profiles = population_df['profile'].tolist()
    
    # 批量生成embeddings
    embeddings = batch_text2embedding(profiles, batch_size=batch_size)
    
    # 将embeddings转换为列表并添加到DataFrame
    population_df['embedding'] = list(embeddings)
    
    logger.info(f"人口学embedding已添加到 population_df（共{len(embeddings)}个）.")
    return population_df
# Occupation-aware worker flag.
def add_essential_worker_field(population_df, essential_worker_percentage=0.182):
    """Add an essential-worker flag."""
    logger.info("Adding essential-worker flag...")
    population_df['if_essential_worker'] = 0
    employed_adults = population_df[(population_df['if_employed'] == 1) & (population_df['age'] >= 18)]
    if len(employed_adults) > 0:
        essential_workers = random.sample(employed_adults.index.tolist(), 
                                        int(len(employed_adults) * essential_worker_percentage))
        population_df.loc[essential_workers, 'if_essential_worker'] = 1
    logger.info(f"必要工作者比例: {population_df['if_essential_worker'].mean():.3f}")
    return population_df

def add_tick_field_to_population(population_df):
    """
    Add a tick field for vaccination eligibility.
    Eligibility is based on age and employment status.
    """
    logger.info("Adding vaccination eligibility tick field...")

    # Build the age-to-tick mapping.
    tick_mapping = create_tick_df()

    # Drop an existing tick column if present.
    if 'tick' in population_df.columns:
        logger.debug("Dropping existing tick column")
        population_df = population_df.drop(columns=['tick'])

    # Merge tick values for every age group.
    population_with_tick = population_df.merge(tick_mapping, on='age', how='left')

    # Fill any missing tick values defensively.
    if population_with_tick['tick'].isna().any():
        logger.warning(f"Found {population_with_tick['tick'].isna().sum()} missing tick values; filling with 999")
        population_with_tick['tick'] = population_with_tick['tick'].fillna(999)

    # Essential workers become eligible at tick=1.
    if 'if_essential_worker' in population_with_tick.columns:
        essential_worker_mask = population_with_tick['if_essential_worker'] == 1
        population_with_tick.loc[essential_worker_mask, 'tick'] = 1
        logger.info(f"Set tick=1 for {essential_worker_mask.sum()} essential workers")

    # Print tick distribution stats.
    tick_dist = population_with_tick['tick'].value_counts().sort_index().to_dict()
    logger.info(f"Tick field added; eligibility distribution: {tick_dist}")

    # Sanity check for teens.
    teens = population_with_tick[population_with_tick['age'].between(14, 17)]
    if len(teens) > 0:
        teen_ticks = teens['tick'].value_counts().to_dict()
        logger.info(f"Teen tick distribution: {teen_ticks}")
        if 14 not in teen_ticks and 1 not in teen_ticks:
            logger.error("Warning: teen tick values may be incorrect")
    
    return population_with_tick

def synthesize_profile(row):
    # Process age.
    age = int(row['age']) if not pd.isna(row['age']) and str(row['age']).strip() != "" else -1
    # Process urban.
    if not pd.isna(row['urban']) and str(row['urban']).strip() != "":
        urban = "urban" if int(row['urban']) == 1 else "rural"
    else:
        urban = "unknown"
    # Process employment.
    if not pd.isna(row['if_employed']) and str(row['if_employed']).strip() != "":
        employed = "employed" if int(row['if_employed']) == 1 else "unemployed"
    else:
        employed = "unknown"
    # Process GEOID_cty.
    geoid = int(row['GEOID_cty']) if not pd.isna(row['GEOID_cty']) and str(row['GEOID_cty']).strip() != "" else -1
    return f"{age} years old, {urban}, {employed}, GEOID {geoid}"


def load_synthetic_data(pop_path, traditional_net_path, cyber_net_path, sample_proportion=1.0, county_geoid=None):
    """
    Load synthetic population and network data in chunks.
    Supports county filtering and sampling.
    """
    logger.info("--- Loading synthetic population and networks (memory efficient) ---")

    if county_geoid is None:
        logger.error("County GEOID must be provided for memory-efficient loading.")
        raise ValueError("County GEOID is required for efficient loading.")

    logger.info(f"Filtering data for county GEOID: {county_geoid}")

    # 1. Efficiently load population data for the specified county in chunks
    chunk_size = 1_000_000  # Process 1 million rows at a time
    pop_chunks = []
    with pd.read_csv(pop_path, chunksize=chunk_size, low_memory=False) as reader:
        for chunk in reader:
            # Filter each chunk for the target county.
            county_chunk = chunk[chunk['GEOID_cty'] == county_geoid]
            if not county_chunk.empty:
                pop_chunks.append(county_chunk)
    
    if not pop_chunks:
        logger.error(f"No population data found for county GEOID {county_geoid}.")
        return pd.DataFrame(), pd.DataFrame()

    population = pd.concat(pop_chunks, ignore_index=True)
    logger.info(f"Loaded population for county GEOID {county_geoid}: {len(population)} agents.")

    # 2. Optionally sample the county population before loading the large network files.
    if sample_proportion < 1.0:
        logger.warning(f"Sampling {sample_proportion * 100:.4f}% of the county population.")
        population = population.sample(frac=sample_proportion, random_state=42).reset_index(drop=True)
        logger.info(f"Population after sampling: {len(population)} agents.")

    # Final agent IDs to look up in the network files.
    final_pop_ids = set(population['reindex'])

    # 3. Load network data for the sampled population in chunks.
    network_chunks = []
    for net_path in [traditional_net_path, cyber_net_path]:
        logger.info(f"Filtering network file: {os.path.basename(net_path)}")
        with pd.read_csv(net_path, chunksize=chunk_size, low_memory=False) as reader:
            for chunk in reader:
                # Keep edges where both endpoints are in the final population.
                filtered_chunk = chunk[
                    chunk['source_reindex'].isin(final_pop_ids) &
                    chunk['target_reindex'].isin(final_pop_ids)
                ]
                if not filtered_chunk.empty:
                    network_chunks.append(filtered_chunk)

    if not network_chunks:
        logger.warning("No network connections found for the sampled population.")
        networks_df = pd.DataFrame(columns=['source_reindex', 'target_reindex', 'Relation'])
    else:
        networks_df = pd.concat(network_chunks, ignore_index=True)
    
    logger.info(f"Loaded {len(networks_df)} network connections for the final population.")

    # 4. Final type conversions on the filtered data.
    population['age'] = pd.to_numeric(population['age'])
    population['urban'] = pd.to_numeric(population['urban'])
    population['reindex'] = population['reindex'].astype(int)
    networks_df['source_reindex'] = pd.to_numeric(networks_df['source_reindex'])
    networks_df['target_reindex'] = pd.to_numeric(networks_df['target_reindex'])

    # 5. Build the age eligibility table.
    df_tick_age = create_tick_df()
    population = population.merge(df_tick_age, on="age", how="left")

    return population, networks_df


def create_tick_df():
    """
    基于另一个项目的逻辑创建疫苗接种资格时间映射
    根据NYS疫苗接种计划确定不同年龄组的资格时间
    """
    # 时间分辨率：每周 (Laurin 2018)
    tick_day = 7
    
    # 创建时间轴：2021-01-01 到 2022-05-19
    df_tick = pd.DataFrame({
        "date": pd.date_range("2021-01-01", "2022-05-19", freq=f"{tick_day}D").strftime("%Y-%m-%d"),
        "tick": range(int(np.ceil(500 / tick_day))),
        "age_5": 0, "age_12": 0, "age_16": 0, "age_30": 0,
        "age_50": 0, "age_60": 0, "age_65": 0, "age_75": 0
    })
    
    # NYS疫苗接种管理计划 (NYS 2021a,b)
    # to do: change tick-age linkage
    # df_tick.loc[df_tick["date"] >= "2021-12-01", "age_5"] = 1
    # df_tick.loc[df_tick["date"] >= "2021-05-19", "age_12"] = 1
    df_tick.loc[df_tick["date"] >= "2021-12-01", "age_5"] = 1
    df_tick.loc[df_tick["date"] >= "2021-05-19", "age_12"] = 1
    df_tick.loc[df_tick["date"] >= "2021-04-06", "age_16"] = 1
    df_tick.loc[df_tick["date"] >= "2021-03-30", "age_30"] = 1
    df_tick.loc[df_tick["date"] >= "2021-03-22", "age_50"] = 1
    df_tick.loc[df_tick["date"] >= "2021-03-10", "age_60"] = 1
    df_tick.loc[df_tick["date"] >= "2021-01-23", "age_65"] = 1
    df_tick.loc[df_tick["date"] >= "2021-01-11", "age_75"] = 1
    
    # 为每个年龄确定疫苗接种资格时间
    df_tick_age = pd.DataFrame({
        "age": range(90),
        "tick": 999  # 默认值，表示无资格
    })
    
    # 根据年龄组确定资格时间
    for col in df_tick.iloc[:, 2:].columns:
        age_threshold = int(col.split("_")[1])
        eligible_rows = df_tick.loc[df_tick[col] > 0, "tick"]
        if not eligible_rows.empty:
            eligibility_tick = eligible_rows.min()
            df_tick_age.loc[df_tick_age['age'] >= age_threshold, "tick"] = eligibility_tick
    
    return df_tick_age

def load_ground_truth_vax_data(ground_truth_path):
    """
    加载真实疫苗接种数据，用于模型验证
    基于另一个项目的逻辑
    """
    logger.info(f'加载真实疫苗接种数据: {ground_truth_path}')
    df_vax_real = pd.read_csv(ground_truth_path)
    df_vax_real['Date'] = pd.to_datetime(df_vax_real['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    df_vax_real = df_vax_real.sort_values(by=['Date'])
    
    # 移动平均周期
    moving_average_period = 7
    
    # 过滤NYS州级数据
    df_vax_nys = df_vax_real.loc[df_vax_real['Recip_County'] == "Chautauqua County"]
    
    # 模拟时间周期：2021-01-01 到 2022-05-15
    df_vax_nys = df_vax_nys.loc[df_vax_nys['Date'] <= "2022-05-15"]
    df_vax_nys = df_vax_nys.loc[df_vax_nys['Date'] >= "2021-01-01"]
    df_vax_nys = df_vax_nys.reset_index(drop=True)
    df_vax_nys['x'] = df_vax_nys.index
    
    # 提取关键列
    df_ground_truth = df_vax_nys.loc[:, ('x', "Date", "Dose1_Recip_pop_pct", "Dose1_Recip_5_11_pct", 
                                        "Dose1_Recip_12_17_pct", "Dose1_Recip_18_64_pct", "Dose1_Recip_65Plus_pct")]
    
    # 计算移动平均
    df_ground_truth['pop_pct_MA'] = df_ground_truth['Dose1_Recip_pop_pct'].rolling(moving_average_period, min_periods=1).mean()
    df_ground_truth['pct_5_11_MA'] = df_ground_truth['Dose1_Recip_5_11_pct'].rolling(moving_average_period, min_periods=1).mean()
    df_ground_truth['pct_12_17_MA'] = df_ground_truth['Dose1_Recip_12_17_pct'].rolling(moving_average_period, min_periods=1).mean()
    df_ground_truth['pct_18_64_MA'] = df_ground_truth['Dose1_Recip_18_64_pct'].rolling(moving_average_period, min_periods=1).mean()
    df_ground_truth['pct_65plus_MA'] = df_ground_truth['Dose1_Recip_65Plus_pct'].rolling(moving_average_period, min_periods=1).mean()
    
    return df_ground_truth


def load_ground_truth_data(path):
    """加载并处理真实的NYS疫苗接种率数据。"""
    logger.info(f"Loading ground truth data from {path}")
    df_real = pd.read_csv(path)
    df_real['Date'] = pd.to_datetime(df_real['Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    df_real = df_real.sort_values(by=['Date']).reset_index(drop=True)

    # 筛选NYS级别和时间范围
    df_nys = df_real[df_real['Recip_County'] == "Chautauqua County"]
    df_nys = df_nys[(df_nys['Date'] >= "2021-01-01") & (df_nys['Date'] <= "2022-05-15")]

    # 计算7日移动平均
    df_nys['pop_pct_MA'] = df_nys['Dose1_Recip_pop_pct'].rolling(window=7, min_periods=1).mean()

    return df_nys


def plot_vaccination_rate(model_results_df, ground_truth_df, output_path):
    """
    绘制模拟结果与真实数据的对比图（x轴为天数，严格用Date列转为天数）。
    计算并显示MAE（Mean Absolute Error）。
    """
    import matplotlib.pyplot as plt
    import os
    import datetime

    plt.figure(figsize=(12, 7), dpi=150)

    # 避免修改外部传入DataFrame，确保函数副作用最小。
    model_plot_df = model_results_df.copy()
    gt_plot_df = ground_truth_df.copy()

    # 模型tick转为天数
    model_plot_df['day'] = pd.to_numeric(model_plot_df['tick'], errors='coerce') * 7
    model_plot_df['vax_rate'] = pd.to_numeric(model_plot_df['vax_rate'], errors='coerce')
    model_plot_df = model_plot_df.dropna(subset=['day', 'vax_rate']).sort_values('day')

    # ground_truth_df用Date列转为天数
    gt_plot_df['Date'] = pd.to_datetime(gt_plot_df['Date'], errors='coerce')
    gt_plot_df = gt_plot_df.dropna(subset=['Date']).copy()
    start_date = gt_plot_df['Date'].min()
    gt_plot_df['day'] = (gt_plot_df['Date'] - start_date).dt.days
    gt_plot_df['pop_pct_MA'] = pd.to_numeric(gt_plot_df['pop_pct_MA'], errors='coerce')
    gt_plot_df = gt_plot_df.dropna(subset=['day', 'pop_pct_MA']).sort_values('day')

    # 使用“day + 百分数”口径计算MAE/RMSE/R²：
    # 将模拟曲线插值到真实数据每日时间点，再在百分数空间比较。
    if model_plot_df.empty or gt_plot_df.empty or len(model_plot_df) < 2:
        mae = float('nan')
        rmse = float('nan')
        r2 = float('nan')
    else:
        common_days = gt_plot_df[
            (gt_plot_df['day'] >= model_plot_df['day'].min()) &
            (gt_plot_df['day'] <= model_plot_df['day'].max())
        ]['day'].to_numpy(dtype=float)

        if len(common_days) == 0:
            mae = float('nan')
            rmse = float('nan')
            r2 = float('nan')
        else:
            sim_interp_pct = np.interp(
                common_days,
                model_plot_df['day'].to_numpy(dtype=float),
                model_plot_df['vax_rate'].to_numpy(dtype=float),
            ) * 100.0
            gt_interp_pct = np.interp(
                common_days,
                gt_plot_df['day'].to_numpy(dtype=float),
                gt_plot_df['pop_pct_MA'].to_numpy(dtype=float),
            )

            err = sim_interp_pct - gt_interp_pct
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((gt_interp_pct - np.mean(gt_interp_pct)) ** 2))
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
    
    plt.plot(model_plot_df['day'], model_plot_df['vax_rate'], color="blue", label="Simulated Rate", lw=2)
    plt.plot(gt_plot_df['day'], gt_plot_df['pop_pct_MA'] / 100.0, color="red",
             label="Observed Rate (7-day MA)", ls='--')

    plt.title("Model vs. Observed Vaccination Rate", fontsize=16)
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Cumulative Vaccination Rate", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 在图上标注评估指标
    if not np.isnan(mae):
        metrics_text = f'MAE = {mae:.2f}%\nRMSE = {rmse:.2f}%\nR² = {r2:.2f}'
        plt.text(0.02, 0.98, metrics_text, 
                transform=plt.gca().transAxes, 
                fontsize=13, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.legend()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_vaccination_comparison.png"
    save_path = os.path.join(output_path, "plots", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Plot saved to {save_path}")
    logger.info(f"Evaluation Metrics (day, percent) - MAE: {mae:.2f}%, RMSE: {rmse:.2f}%, R²: {r2:.2f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def load_pums_data(household_path, person_path):
    """
    加载PUMS数据（家庭和个人级别），包括性别信息
    """
    logger.info("Loading PUMS data...")
    
    # 加载家庭数据
    household_df = pd.read_csv(household_path)
    # 加载个人数据，包括性别(SEX)列
    person_df = pd.read_csv(person_path)
    
    logger.info(f"Loaded {len(household_df)} households and {len(person_df)} persons")
    
    return household_df, person_df

def create_joint_distribution_sampler(household_df, person_df):
    """
    创建联合分布抽样器，考虑变量间的相关性
    基于scientific data论文的方法，以家庭为主的采样策略
    注意：HHT和孩子数量从CSV网络数据推断，此函数主要用于个体特征和家庭收入
    """
    logger.info("Creating joint distribution sampler (household-based)...")
    
    # 数据清洗和预处理 - 家庭级别
    cleaned_household = household_df.copy()
    
    # 处理家庭类型 (HHT)
    cleaned_household['HHT'] = pd.to_numeric(cleaned_household['HHT'], errors='coerce')
    # 只保留有效的家庭类型 (1-7)
    cleaned_household = cleaned_household[cleaned_household['HHT'].between(1, 7)]
    
    # 处理家庭收入 (FINCP)
    cleaned_household['FINCP'] = pd.to_numeric(cleaned_household['FINCP'], errors='coerce')
    # 移除负值和异常值
    cleaned_household = cleaned_household[cleaned_household['FINCP'] >= 0]
    cleaned_household = cleaned_household[cleaned_household['FINCP'] <= 1000000]  # 限制在合理范围内
    
    # 统计每个家庭的孩子数量
    person_df_copy = person_df.copy()
    person_df_copy['AGEP'] = pd.to_numeric(person_df_copy['AGEP'], errors='coerce')
    children_per_household = person_df_copy[person_df_copy['AGEP'] < 18].groupby('SERIALNO').size()
    cleaned_household['num_children'] = cleaned_household['SERIALNO'].map(children_per_household).fillna(0).astype(int)
    
    # 限制孩子数量在合理范围内 (0-6)
    cleaned_household = cleaned_household[cleaned_household['num_children'] <= 6]
    
    # 处理个人数据 - 个体级别特征
    cleaned_person = person_df.copy()
    
    # 处理个人收入 (PINCP)
    cleaned_person['PINCP'] = pd.to_numeric(cleaned_person['PINCP'], errors='coerce')
    cleaned_person = cleaned_person[cleaned_person['PINCP'] >= 0]
    cleaned_person = cleaned_person[cleaned_person['PINCP'] <= 1000000]
    
    # 处理教育水平 (SCHL)
    cleaned_person['SCHL'] = pd.to_numeric(cleaned_person['SCHL'], errors='coerce')
    cleaned_person = cleaned_person[cleaned_person['SCHL'].between(1, 24)]
    
    # 处理职业类型 (COW)
    cleaned_person['COW'] = pd.to_numeric(cleaned_person['COW'], errors='coerce')
    cleaned_person = cleaned_person[cleaned_person['COW'].between(1, 9)]
    
    # 处理医保覆盖 (HICOV)
    cleaned_person['HICOV'] = pd.to_numeric(cleaned_person['HICOV'], errors='coerce')
    cleaned_person = cleaned_person[cleaned_person['HICOV'].between(1, 2)]
    
    # 移除任何包含NaN的行
    cleaned_household = cleaned_household.dropna(subset=['HHT', 'FINCP', 'num_children'])
    cleaned_person = cleaned_person.dropna(subset=['PINCP', 'SCHL', 'COW', 'HICOV'])
    
    logger.info(f"Cleaned household data: {len(cleaned_household)} valid records")
    logger.info(f"Cleaned person data: {len(cleaned_person)} valid records")
    
    # 创建家庭级别联合分布表
    household_joint_dist = cleaned_household.groupby(['HHT', 'FINCP', 'num_children']).size().reset_index(name='count')
    household_joint_dist['probability'] = household_joint_dist['count'] / household_joint_dist['count'].sum()
    
    # 创建个体级别联合分布表
    person_joint_dist = cleaned_person.groupby(['PINCP', 'SCHL', 'COW', 'HICOV']).size().reset_index(name='count')
    person_joint_dist['probability'] = person_joint_dist['count'] / person_joint_dist['count'].sum()
    
    # 创建分层抽样策略
    def create_stratified_sampler():
        """创建分层抽样器，考虑主要相关性"""
        
        # 家庭级别相关性
        hht_income_dist = cleaned_household.groupby(['HHT', 'FINCP']).size().reset_index(name='count')
        hht_income_dist['probability'] = hht_income_dist['count'] / hht_income_dist['count'].sum()
        
        hht_children_dist = cleaned_household.groupby(['HHT', 'num_children']).size().reset_index(name='count')
        hht_children_dist['probability'] = hht_children_dist['count'] / hht_children_dist['count'].sum()
        
        # 个体级别相关性
        income_education_dist = cleaned_person.groupby(['PINCP', 'SCHL']).size().reset_index(name='count')
        income_education_dist['probability'] = income_education_dist['count'] / income_education_dist['count'].sum()
        
        income_occupation_dist = cleaned_person.groupby(['PINCP', 'COW']).size().reset_index(name='count')
        income_occupation_dist['probability'] = income_occupation_dist['count'] / income_occupation_dist['count'].sum()
        
        education_occupation_dist = cleaned_person.groupby(['SCHL', 'COW']).size().reset_index(name='count')
        education_occupation_dist['probability'] = education_occupation_dist['count'] / education_occupation_dist['count'].sum()
        
        return {
            'household_joint': household_joint_dist,
            'person_joint': person_joint_dist,
            'hht_income': hht_income_dist,
            'hht_children': hht_children_dist,
            'income_education': income_education_dist,
            'income_occupation': income_occupation_dist,
            'education_occupation': education_occupation_dist
        }
    
    sampler = create_stratified_sampler()
    
    logger.info("Joint distribution sampler created successfully")
    return sampler, cleaned_household, cleaned_person

def sample_pums_features(sampler, n_samples=1000):
    """
    使用联合分布抽样器生成特征样本
    考虑变量间的相关性，家庭和个体分离采样
    注意：HHT和孩子数量已从CSV数据推断，这里只生成个体特征和家庭收入
    """
    logger.info(f"Sampling {n_samples} feature combinations...")
    
    household_samples = []
    person_samples = []
    
    for i in range(n_samples):
        # 家庭级别采样 - 只保留家庭收入，HHT和孩子数量从CSV推断
        if len(sampler['household_joint']) < 100000:
            household_sample = sampler['household_joint'].sample(n=1, weights='probability').iloc[0]
            household_dict = {
                'FINCP': household_sample['FINCP']
            }
        else:
            # 分层抽样家庭特征 - 只保留家庭收入
            hht_income_sample = sampler['hht_income'].sample(n=1, weights='probability').iloc[0]
            household_dict = {
                'FINCP': hht_income_sample['FINCP']
            }
        
        # 个体级别采样
        if len(sampler['person_joint']) < 100000:
            person_sample = sampler['person_joint'].sample(n=1, weights='probability').iloc[0]
            person_dict = {
                'personal_income': person_sample['PINCP'],
                'education': person_sample['SCHL'],
                'occupation': person_sample['COW'],
                'health_insurance': person_sample['HICOV']
            }
        else:
            # 分层抽样个体特征
            income_edu_sample = sampler['income_education'].sample(n=1, weights='probability').iloc[0]
            person_dict = {
                'personal_income': income_edu_sample['PINCP'],
                'education': income_edu_sample['SCHL'],
                'occupation': np.random.choice(sampler['education_occupation'][
                    sampler['education_occupation']['SCHL'] == income_edu_sample['SCHL']
                ]['COW'].values),
                'health_insurance': np.random.choice([1, 2], p=[0.85, 0.15])
            }
        
        # 合并家庭和个体特征
        combined_dict = {**household_dict, **person_dict}
        household_samples.append(household_dict)
        person_samples.append(person_dict)
    
    logger.info(f"Generated {len(household_samples)} household samples and {len(person_samples)} person samples")
    return household_samples, person_samples

# def add_pums_features_to_population(population_df, household_path, person_path, sample_size=None):
#     """
#     为合成人口添加PUMS特征
#     基于s论文的联合分布抽样方法，家庭和个体分离
#     注意：HHT和孩子数量从CSV网络数据推断，此函数只添加个体特征和家庭收入
#     """
#     logger.info("Adding PUMS features to synthetic population...")
    
#     # 加载PUMS数据
#     household_df, person_df = load_pums_data(household_path, person_path)
    
#     # 创建联合分布抽样器
#     sampler, cleaned_household, cleaned_person = create_joint_distribution_sampler(household_df, person_df)
    
#     # 确定抽样大小
#     if sample_size is None:
#         sample_size = len(population_df)
    
#     # 生成特征样本
#     household_samples, person_samples = sample_pums_features(sampler, sample_size)
    
#     # 将特征添加到人口数据
#     population_with_features = population_df.copy()
    
#     # 添加PUMS特征
#     for i, (household_sample, person_sample) in enumerate(zip(household_samples, person_samples)):
#         if i < len(population_with_features):
#             # 家庭级别特征 - 只保留家庭收入，HHT和孩子数量从CSV推断
#             population_with_features.loc[i, 'FINCP'] = household_sample['FINCP']
            
#             # 个体级别特征
#             population_with_features.loc[i, 'personal_income'] = person_sample['personal_income']
#             population_with_features.loc[i, 'education'] = person_sample['education']
#             population_with_features.loc[i, 'occupation'] = person_sample['occupation']
#             population_with_features.loc[i, 'health_insurance'] = person_sample['health_insurance']
    
#     # 填充剩余行（如果样本数少于人口数）
#     if len(household_samples) < len(population_with_features):
#         remaining_household, remaining_person = sample_pums_features(
#             sampler, len(population_with_features) - len(household_samples)
#         )
#         for i, (household_sample, person_sample) in enumerate(zip(remaining_household, remaining_person)):
#             idx = len(household_samples) + i
#             if idx < len(population_with_features):
#                 # 家庭级别特征 - 只保留家庭收入，HHT和孩子数量从CSV推断
#                 population_with_features.loc[idx, 'FINCP'] = household_sample['FINCP']
                
#                 # 个体级别特征
#                 population_with_features.loc[idx, 'personal_income'] = person_sample['personal_income']
#                 population_with_features.loc[idx, 'education'] = person_sample['education']
#                 population_with_features.loc[idx, 'occupation'] = person_sample['occupation']
#                 population_with_features.loc[idx, 'health_insurance'] = person_sample['health_insurance']
    
#     logger.info("PUMS features added successfully")
#     logger.info(f"Feature statistics:")
#     logger.info(f"  Family income: mean={population_with_features['FINCP'].mean():.0f}, std={population_with_features['FINCP'].std():.0f}")
#     logger.info(f"  Personal income: mean={population_with_features['personal_income'].mean():.0f}, std={population_with_features['personal_income'].std():.0f}")
#     logger.info(f"  Education level: mean={population_with_features['education'].mean():.1f}")
#     logger.info(f"  Health insurance coverage: {(population_with_features['health_insurance'] == 1).mean():.1%}")
#     # 注意：HHT和孩子数量从CSV数据推断，不在这里统计
    
#     return population_with_features

def enhance_profile_with_pums_features(row):
    """
    使用PUMS特征增强个人档案描述，家庭类型直接使用重构数据中的htype字段，
    并提供具体的家庭类型含义描述。
    """
    profile_parts = []
    # 基本信息
    age = int(row['age']) if not pd.isna(row['age']) else -1
    urban = "urban" if row.get('urban', 0) == 1 else "rural"
    employed = "employed" if row.get('if_employed', 0) == 1 else "unemployed"
    profile_parts.append(f"{age} years old, {urban}, {employed}")
    
    # 家庭类型直接使用重构数据中的htype字段，并提供具体含义
    if 'htype' in row and not pd.isna(row['htype']):
        htype_val = int(row['htype'])
        # 家庭类型含义映射
        htype_meanings = {
            0: "married couple family, no children under 18",
            1: "married couple family, with children under 18", 
            2: "unmarried partner family, no children under 18",
            3: "unmarried partner family, with children under 18",
            4: "male householder, living alone (18-64)",
            5: "male householder, living alone (65+)",
            6: "male householder, single parent family (with children under 18)",
            7: "female householder, living alone (18-64)",
            8: "female householder, living alone (65+)",
            9: "female householder, single parent family (with children under 18)",
            10: "non-family household (multiple people)",
            11: "group quarters (institutional living)"
        }
        
        if htype_val in htype_meanings:
            household_desc = htype_meanings[htype_val]
            profile_parts.append(f"household type {htype_val}: {household_desc}")
        else:
            profile_parts.append(f"household type {htype_val} (unknown type)")
    else:
        profile_parts.append("unknown household type")
    # 家庭收入 (FINCP)
    if 'FINCP' in row and not pd.isna(row['FINCP']):
        family_income = row['FINCP']
        if family_income < 30000:
            family_income_level = "low family income"
        elif family_income < 80000:
            family_income_level = "middle family income"
        else:
            family_income_level = "high family income"
        profile_parts.append(family_income_level)
    # 个人收入
    if 'personal_income' in row and not pd.isna(row['personal_income']):
        income = row['personal_income']
        if income < 25000:
            income_level = "low personal income"
        elif income < 75000:
            income_level = "middle personal income"
        else:
            income_level = "high personal income"
        profile_parts.append(income_level)
    # 教育水平
    if 'education' in row and not pd.isna(row['education']):
        edu_level = row['education']
        if edu_level <= 15:
            education_desc = "high school or less"
        elif edu_level <= 20:
            education_desc = "some college"
        elif edu_level <= 21:
            education_desc = "associate degree"
        elif edu_level <= 22:
            education_desc = "bachelor degree"
        else:
            education_desc = "graduate degree"
        profile_parts.append(education_desc)
    # 职业类型
    if 'occupation' in row and not pd.isna(row['occupation']):
        occ = row['occupation']
        if occ in [1, 2]:
            work_type = "private sector"
        elif occ in [3, 4, 5]:
            work_type = "government"
        elif occ in [6, 7]:
            work_type = "self-employed"
        else:
            work_type = "other work"
        profile_parts.append(work_type)
    # 医保覆盖
    if 'health_insurance' in row and not pd.isna(row['health_insurance']):
        insurance = "with health insurance" if row['health_insurance'] == 1 else "without health insurance"
        profile_parts.append(insurance)
    # 孩子数量
    if 'num_children' in row and not pd.isna(row['num_children']):
        num_children = int(row['num_children'])
        if num_children == 0:
            children_desc = "no children"
        elif num_children == 1:
            children_desc = "with 1 child"
        else:
            children_desc = f"with {num_children} children"
        profile_parts.append(children_desc)
    return ", ".join(profile_parts)


# 已过时：此方法用于从CSV推断特征，但新的重构数据已包含htype等直接信息
# def add_csv_based_features_to_population(population_df, network_df):
#     """
#     已过时：此方法用于从CSV推断特征，但新的重构数据已包含htype等直接信息
#     """
#     logger.warning("add_csv_based_features_to_population is deprecated. Use direct htype from reconstructed data instead.")
#     return population_df

def preprocess_pums_data(household_path, person_path):
    """
    预处理PUMS数据，清洗并创建索引。
    """
    logger.info("Preprocessing PUMS data...")

    # 加载数据
    household_df = pd.read_csv(household_path)
    person_df = pd.read_csv(person_path)

    # 清洗家庭数据
    household_df['HHT'] = pd.to_numeric(household_df['HHT'], errors='coerce')
    household_df['FINCP'] = pd.to_numeric(household_df['FINCP'], errors='coerce')
    household_df = household_df.dropna(subset=['HHT', 'FINCP'])

    # 清洗个人数据
    person_df['AGEP'] = pd.to_numeric(person_df['AGEP'], errors='coerce')
    person_df['PINCP'] = pd.to_numeric(person_df['PINCP'], errors='coerce')
    person_df = person_df.dropna(subset=['AGEP', 'PINCP'])

    # 合并家庭和个人数据
    merged_df = person_df.merge(household_df, on='SERIALNO', how='left')

    logger.info("PUMS data preprocessing complete.")
    return household_df, person_df, merged_df

def add_pums_features_to_population(population_df, household_path, person_path):
    """
    为合成人口添加PUMS特征，采用"个体优先，家庭汇总"的策略，确保逻辑一致性。
    使用htype、gender和age属性进行PUMS数据匹配。
    """
    logger.info("Adding PUMS features with Hierarchical Conditional Matching using htype, gender, and age...")

    # --- 1. 预处理PUMS数据 ---
    # 加载并进行基础清洗，包括性别(SEX)信息
    household_df = pd.read_csv(household_path, usecols=['SERIALNO', 'HHT', 'NP', 'FINCP'])
    person_df = pd.read_csv(person_path, usecols=['SERIALNO', 'AGEP', 'ESR', 'PINCP', 'SCHL', 'COW', 'HICOV', 'SEX'])

    # 清理PUMS数据
    person_df['AGEP'] = pd.to_numeric(person_df['AGEP'], errors='coerce')   # 年龄
    person_df['PINCP'] = pd.to_numeric(person_df['PINCP'], errors='coerce')     # 个人收入
    person_df['ESR'] = pd.to_numeric(person_df['ESR'], errors='coerce') # 就业状态
    person_df['SEX'] = pd.to_numeric(person_df['SEX'], errors='coerce') # 性别 (1: male, 2: female)
    person_df = person_df.dropna(subset=['AGEP', 'PINCP', 'ESR', 'SEX'])
    
    # 将PUMS家庭类型合并到个人记录，以便按家庭情境筛选
    pums_merged = person_df.merge(household_df[['SERIALNO', 'HHT']], on='SERIALNO', how='left')
    pums_merged = pums_merged.dropna(subset=['HHT'])


    # --- 2. 预处理合成人口数据 ---
    # 确保每个家庭有一个唯一的ID，以便后续汇总
    # 我们使用连通分量来识别家庭，并分配household_id
    if 'household_id' not in population_df.columns:
        population_df['household_id'] = population_df.groupby(['HHT', 'family_size']).ngroup()


    # --- 3. 为每个个体进行条件性赋值 (整群匹配)---
    pums_features = []
    for _, row in tqdm(population_df.iterrows(), total=len(population_df), desc="Matching PUMS Features"):
        age = row['age']
        hht = row['htype']
        is_employed = row['if_employed'] == 1
        gender = row.get('gender', None)  # 获取性别信息

        # 初始化特征字典
        features = {
            'personal_income': 0,
            'education': None,
            'occupation': None,
            'health_insurance': None
        }

        # 根据年龄、性别和就业状态定义PUMS筛选条件
        age_filter = pums_merged['AGEP'].between(age - 2, age + 2)
        hht_filter = pums_merged['HHT'] == hht
        
        # 映射性别到PUMS代码: 'm' -> 1, 'f' -> 2
        gender_code = None
        if gender == 'm':
            gender_code = 1
        elif gender == 'f':
            gender_code = 2
        
        gender_filter = pd.Series(True, index=pums_merged.index)  # 默认无性别过滤
        if gender_code is not None:
            gender_filter = pums_merged['SEX'] == gender_code
        
        if age >= 18:
            if is_employed:
                # 寻找已就业的成年人
                employment_filter = pums_merged['ESR'].isin([1, 2, 4, 5])
            else:
                # 寻找未就业的成年人
                employment_filter = pums_merged['ESR'].isin([3, 6])
        else:
            employment_filter = pd.Series(True, index=pums_merged.index)

        # 尝试找到最佳匹配（包含性别过滤）
        candidates = pums_merged[age_filter & hht_filter & gender_filter & employment_filter]
        
        # 如果找不到，放宽家庭类型(HHT)的限制
        if candidates.empty:
            candidates = pums_merged[age_filter & gender_filter & employment_filter]
        
        # 进一步放宽年龄限制
        if candidates.empty and age >= 18:
            age_filter_relaxed = pums_merged['AGEP'].between(age - 5, age + 5)
            candidates = pums_merged[age_filter_relaxed & gender_filter & employment_filter]

        # 从候选人中随机抽取一个，并"整群"赋值所有特征
        if not candidates.empty:
            sampled_person = candidates.sample(1).iloc[0]
            features['personal_income'] = sampled_person['PINCP'] if age >= 18 and is_employed else 0
            features['occupation'] = sampled_person['COW'] if age >= 18 and is_employed else None
            features['education'] = sampled_person['SCHL']
            features['health_insurance'] = sampled_person['HICOV']
        
        pums_features.append(features)

    # Assign new features, overwriting existing placeholder columns from the CSV-based function
    features_df = pd.DataFrame(pums_features, index=population_df.index)
    for col in features_df.columns:
        population_df[col] = features_df[col]


    # --- 4. 汇总计算家庭收入 (FINCP) ---
    logger.info("Aggregating personal incomes to calculate family income (FINCP)...")
    # The result of groupby().sum() is a Series, which is the correct input for map()
    family_incomes = population_df.groupby('household_id')['personal_income'].sum()
    population_df['FINCP'] = population_df['household_id'].map(family_incomes)
    population_df['FINCP'] = population_df['FINCP'].fillna(0) # Fill NaN for single-person households if any

    logger.info("PUMS features added successfully using hierarchical matching with htype, gender, and age.")
    logger.info(f"  Avg. Personal Income: ${population_df['personal_income'].mean():,.0f}")
    logger.info(f"  Avg. Family Income (calculated): ${population_df['FINCP'].mean():,.0f}")
    
    return population_df

def load_tract_puma_mapping(tract_puma_path):
    """
    加载Tract-PUMA映射关系
    """
    logger.info(f"Loading Tract-PUMA mapping from {tract_puma_path}")
    
    if tract_puma_path.endswith('.csv'):
        tract_puma_map = pd.read_csv(tract_puma_path)
    elif tract_puma_path.endswith('.gpkg'):
        try:
            import geopandas as gpd
            tract_puma_map = gpd.read_file(tract_puma_path)
        except ImportError:
            raise ImportError("geopandas is required for reading .gpkg files")
    else:
        raise ValueError("Unsupported file format for tract-puma mapping")
    
    # 确保必要的列存在
    required_cols = ['TRACTCE', 'PUMA5CE', 'COUNTYFP']
    for col in required_cols:
        if col not in tract_puma_map.columns:
            raise ValueError(f"Required column '{col}' not found in tract-puma mapping")
    
    # 转换数据类型
    tract_puma_map['TRACTCE'] = tract_puma_map['TRACTCE'].astype(str)
    tract_puma_map['PUMA5CE'] = tract_puma_map['PUMA5CE'].astype(str)
    tract_puma_map['COUNTYFP'] = tract_puma_map['COUNTYFP'].astype(str)
    
    logger.info(f"Loaded {len(tract_puma_map)} tract-puma mappings")
    return tract_puma_map

def load_tract_population_data(tract_pop_path):
    """
    加载Tract人口数据
    """
    logger.info(f"Loading tract population data from {tract_pop_path}")
    
    if tract_pop_path.endswith('.csv'):
        tract_pop = pd.read_csv(tract_pop_path)
    elif tract_pop_path.endswith('.gpkg'):
        try:
            import geopandas as gpd
            tract_pop = gpd.read_file(tract_pop_path)
        except ImportError:
            raise ImportError("geopandas is required for reading .gpkg files")
    else:
        raise ValueError("Unsupported file format for tract population data")
    
    # 确保必要的列存在
    if 'TRACTCE' not in tract_pop.columns or 'POPULATION' not in tract_pop.columns:
        raise ValueError("Required columns 'TRACTCE' and 'POPULATION' not found in tract population data")
    
    # 转换数据类型
    tract_pop['TRACTCE'] = tract_pop['TRACTCE'].astype(str)
    tract_pop['POPULATION'] = pd.to_numeric(tract_pop['POPULATION'], errors='coerce')
    tract_pop = tract_pop.dropna(subset=['POPULATION'])
    
    logger.info(f"Loaded {len(tract_pop)} tract population records")
    return tract_pop

def assign_pums_features_with_geographic_constraint(population_df, pums_household_path, pums_person_path, tract_puma_path, tract_pop_path, target_county_id):
    """
    为特定县（county）的合成人口添加 PUMS 特征，采用地理约束的候选池匹配。
    基于县对应的PUMA代码构建候选池，考虑变量间的相关性，并基于个体背景特征进行匹配。
    
    Args:
        population_df (pd.DataFrame): 合成人口数据，必须包含 'countyid', 'tractid', 'age', 'gender', 'htype', 'if_employed' 等字段。
        pums_household_path (str): PUMS 家庭数据文件路径。
        pums_person_path (str): PUMS 个人数据文件路径。
        tract_puma_path (str): Tract-PUMA 交叉参考表路径。
        tract_pop_path (str): Tract 人口数据路径（保留参数，但不再使用）。
        target_county_id (str): 目标县的 FIPS 代码，例如 '36013'。
    
    Returns:
        pd.DataFrame: 增加了 PUMS 特征的合成人口 DataFrame。
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting geographic-constrained PUMS feature assignment for County: {target_county_id}")

    # --- 1. 加载和预处理数据 ---
    try:
        # 加载PUMS数据
        pums_household = pd.read_csv(pums_household_path)
        pums_person = pd.read_csv(pums_person_path)
        tract_puma_map = load_tract_puma_mapping(tract_puma_path)
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        return population_df
    
    # 清洗 PUMS 家庭数据
    pums_household['HHT'] = pd.to_numeric(pums_household['HHT'], errors='coerce')
    pums_household['FINCP'] = pd.to_numeric(pums_household['FINCP'], errors='coerce')
    pums_household = pums_household.dropna(subset=['HHT', 'FINCP'])
    
    # 清洗 PUMS 个人数据
    pums_person['AGEP'] = pd.to_numeric(pums_person['AGEP'], errors='coerce')
    pums_person['PINCP'] = pd.to_numeric(pums_person['PINCP'], errors='coerce')
    pums_person['SCHL'] = pd.to_numeric(pums_person['SCHL'], errors='coerce')
    pums_person['COW'] = pd.to_numeric(pums_person['COW'], errors='coerce')
    pums_person['HICOV'] = pd.to_numeric(pums_person['HICOV'], errors='coerce')
    pums_person['SEX'] = pd.to_numeric(pums_person['SEX'], errors='coerce')
    pums_person = pums_person.dropna(subset=['AGEP', 'PINCP', 'SCHL', 'COW', 'HICOV', 'SEX'])
    
    # 处理PUMA列 - 优先使用PUMA5CE，然后是PUMA10/PUMA20
    puma_col = None
    for col in ['PUMA5CE', 'PUMA10', 'PUMA20']:
        if col in pums_person.columns:
            puma_col = col
            break
    
    if puma_col is None:
        logger.error("No PUMA column found in PUMS person data")
        return population_df
    
    pums_person = pums_person[pums_person[puma_col] != -9].copy()
    pums_person['PUMA'] = pums_person[puma_col].astype(str)
    
    # 合并 PUMS 个人和家庭数据
    pums_merged = pums_person.merge(pums_household[['SERIALNO', 'HHT', 'FINCP']], on='SERIALNO', how='left')
    pums_merged = pums_merged.dropna(subset=['HHT'])
    
    # 映射性别代码
    pums_merged['SEX'] = pums_merged['SEX'].map({1: 'm', 2: 'f'})
    
    # 清理合成人口数据 - 确保有tractid和countyid
    if 'tractid' not in population_df.columns or 'countyid' not in population_df.columns:
        logger.info("Extracting tractid and countyid from id column...")
        population_df['countyid'] = population_df['id'].str[2:5]
        population_df['tractid'] = population_df['id'].str[5:11]
    
    # 筛选目标县的人口数据
    population_df = population_df[population_df['countyid'] == str(target_county_id)].copy()
    
    # --- 2. 构建地理约束的候选池 ---
    logger.info("Building geographically constrained candidate pool...")
    
    # 根据示例数据，我们知道格式是：COUNTYFP是'1'格式，PUMA5CE是'1700'格式
    # 从target_county_id (如'013')中提取实际的county编号
    simple_county_id = str(int(target_county_id))  # 移除前导零
    logger.info(f"Converting target_county_id '{target_county_id}' to simple format: '{simple_county_id}'")
    
    # 确保tract_puma_map中的列都是字符串类型
    tract_puma_map['COUNTYFP'] = tract_puma_map['COUNTYFP'].astype(str)
    tract_puma_map['PUMA5CE'] = tract_puma_map['PUMA5CE'].astype(str)
    
    logger.info(f"Unique COUNTYFP values in tract_puma_map: {tract_puma_map['COUNTYFP'].unique()}")
    logger.info(f"Unique PUMA5CE values in tract_puma_map: {tract_puma_map['PUMA5CE'].unique()}")
    
    # 直接使用简化的county ID进行匹配
    target_county_tracts = tract_puma_map[tract_puma_map['COUNTYFP'] == simple_county_id].copy()
    
    if target_county_tracts.empty:
        logger.warning(f"No tracts found for county ID {simple_county_id}")
        return population_df
    
    logger.info(f"Found {len(target_county_tracts)} tracts for county {simple_county_id}")
    
    if target_county_tracts is None or target_county_tracts.empty:
        logger.warning(f"No tracts found for county ID {target_county_id} using formats: {county_formats_to_try}")
        return population_df
    
    target_pumas = target_county_tracts['PUMA5CE'].astype(str).unique()
    logger.info(f"Found {len(target_pumas)} PUMAs for county {target_county_id}: {target_pumas}")
    
    # 从tract_puma_map中获取PUMA代码
    target_pumas = target_county_tracts['PUMA5CE'].unique()
    logger.info(f"Found PUMAs for target county: {target_pumas}")
    
    # 筛选出PUMS候选人池（属于目标县PUMA的记录）
    # 确保PUMA代码格式匹配
    pums_merged['PUMA'] = pums_merged['PUMA'].astype(str)
    
    # 尝试直接匹配
    pums_candidate_pool = pums_merged[pums_merged['PUMA'].isin(target_pumas)]
    
    if pums_candidate_pool.empty:
        logger.warning(f"No direct PUMA matches found. Target PUMAs: {target_pumas}")
        # 尝试添加前导零进行匹配
        padded_pumas = [puma.zfill(4) for puma in target_pumas]
        logger.info(f"Trying padded PUMA codes: {padded_pumas}")
        pums_candidate_pool = pums_merged[pums_merged['PUMA'].isin(padded_pumas)]
        
        if pums_candidate_pool.empty:
            logger.warning("Still no matches found with padded PUMA codes")
            return population_df
    
    logger.info(f"Created candidate pool with {len(pums_candidate_pool)} records from PUMAs: {target_pumas}")
    
    # --- 3. 为每个个体匹配PUMS特征 ---
    logger.info("Matching PUMS features for each individual...")
    
    # Use the same column names as the non-geographic version for consistency
    pums_feature_mapping = {
        'PINCP': 'personal_income',
        'SCHL': 'education', 
        'COW': 'occupation',
        'HICOV': 'health_insurance',
        'FINCP': 'FINCP'
    }
    
    features_to_assign = list(pums_feature_mapping.keys())
    target_columns = list(pums_feature_mapping.values())

    # Initialize feature columns with NaN if not present
    for column in target_columns:
        if column not in population_df.columns:
            population_df[column] = np.nan

    for idx, individual in tqdm(population_df.iterrows(), total=len(population_df), desc="Matching individuals"):
        age = individual['age']
        gender = individual['gender']
        htype = individual['htype']
        employed = individual['if_employed'] == 1
        
        # 筛选条件：年龄±2岁，相同性别，相同家庭类型
        filtered_candidates = pums_candidate_pool[
            (pums_candidate_pool['AGEP'].between(age - 2, age + 2)) &
            (pums_candidate_pool['SEX'] == gender) &
            (pums_candidate_pool['HHT'] == htype)
        ]
        
        # 如果找不到匹配的，放宽条件
        if filtered_candidates.empty:
            filtered_candidates = pums_candidate_pool[
                (pums_candidate_pool['AGEP'].between(age - 5, age + 5)) &
                (pums_candidate_pool['SEX'] == gender)
            ]
        
        # 如果还是找不到，使用整个候选池
        if filtered_candidates.empty:
            filtered_candidates = pums_candidate_pool
        
        # 随机选择一个候选人
        if not filtered_candidates.empty:
            selected_candidate = filtered_candidates.sample(1).iloc[0]
            
            # 赋值所有特征，使用目标列名
            population_df.at[idx, 'personal_income'] = selected_candidate['PINCP']
            population_df.at[idx, 'education'] = selected_candidate['SCHL']
            population_df.at[idx, 'occupation'] = selected_candidate['COW']
            population_df.at[idx, 'health_insurance'] = selected_candidate['HICOV']
            population_df.at[idx, 'FINCP'] = selected_candidate['FINCP']
    
    # --- 4. 汇总计算家庭收入 (FINCP) 基于个人收入总和 ---
    logger.info("Calculating family income (FINCP) as sum of personal incomes within each household...")
    
    # 确保有household_id列
    if 'household_id' not in population_df.columns:
        logger.info("Creating household_id from htype and family_size...")
        population_df['household_id'] = population_df.groupby(['htype', 'family_size']).ngroup()
    
    # 计算每个家庭的个人收入总和
    household_income = population_df.groupby('household_id')['personal_income'].sum().reset_index()
    household_income.rename(columns={'personal_income': 'FINCP_calculated'}, inplace=True)
    
    # 将计算的家庭收入合并回人口数据
    population_df = population_df.merge(household_income, on='household_id', how='left')
    
    # 使用计算的家庭收入覆盖直接从PUMS获取的FINCP
    population_df['FINCP'] = population_df['FINCP_calculated']
    population_df.drop('FINCP_calculated', axis=1, inplace=True)
    
    # --- 5. 处理缺失值 ---
    logger.info("Handling missing values...")
    
    # 对于没有匹配到的个体，使用整个PUMS数据的平均值
    for pums_col, target_col in pums_feature_mapping.items():
        if pums_col in pums_merged.columns and target_col != 'FINCP':  # FINCP已经通过计算得到
            mean_value = pums_merged[pums_col].mean()
            population_df[target_col] = population_df[target_col].fillna(mean_value)
    
    # 确保FINCP没有缺失值
    if population_df['FINCP'].isna().any():
        fincp_mean = pums_merged['FINCP'].mean()
        population_df['FINCP'] = population_df['FINCP'].fillna(fincp_mean)
    
    logger.info("PUMS feature assignment with geographic constraint completed.")
    logger.info(f"Assigned features: {features_to_assign}")
    for feature in features_to_assign:
        if feature in population_df.columns:
            logger.info(f"  {feature}: mean={population_df[feature].mean():.2f}, std={population_df[feature].std():.2f}")
    
    return population_df
