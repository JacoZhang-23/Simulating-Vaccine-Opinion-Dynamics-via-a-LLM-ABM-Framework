import asyncio
import aiohttp
import numpy as np
import networkx as nx
from mesa import Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from loguru import logger
import random
import json
import re
import os

from agent import VaxAgent
from openai import OpenAI
from tools import text2embedding

# --- Rejection keyword filter (legacy) ---
REJECTION_KEYWORDS = [
    "I can't provide", "I cannot provide", "I'm unable to", 
    "I'm not able to", "I'd rather not", "I'd prefer not",
    "My guidelines", "My purpose", "My role",
    "As an AI", "I cannot answer", "I apologize"
]

# --- API failure monitor ---
class ApiFailureMonitor:
    """
    Track API failures and trigger a pause/recovery policy.
    """
    def __init__(self, failure_threshold=0.95, check_window=18):
        self.consecutive_failures = 0
        self.total_calls = 0
        self.total_failures = 0
        self.failure_threshold = failure_threshold  # Failure-rate threshold.
        self.check_window = check_window  # Rolling window size.
        self.recent_results = []  # Recent call outcomes.
        self.is_paused = False
        self.pause_count = 0
        
    def record_success(self):
        """Record a successful API call."""
        self.consecutive_failures = 0
        self.total_calls += 1
        self.recent_results.append(True)
        if len(self.recent_results) > self.check_window:
            self.recent_results.pop(0)
    
    def record_failure(self):
        """Record a failed API call."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_calls += 1
        self.recent_results.append(False)
        if len(self.recent_results) > self.check_window:
            self.recent_results.pop(0)
    
    def should_pause(self) -> tuple[bool, int]:
        """
        Decide whether the system should pause.
        """
        # Check the recent failure rate once the window is full.
        if len(self.recent_results) >= self.check_window:
            recent_failure_rate = self.recent_results.count(False) / len(self.recent_results)
            
            if recent_failure_rate >= 0.98:
                # Pause only when almost all keys are failing.
                self.pause_count += 1
                # Progressive wait time.
                wait_time = min(30 * (2 ** (self.pause_count - 1)), 120)
                return True, wait_time
        
        return False, 0
    
    def get_stats(self) -> str:
        """Return summary stats."""
        if self.total_calls == 0:
            return "No API calls yet"
        
        success_rate = (self.total_calls - self.total_failures) / self.total_calls * 100
        recent_failure_rate = 0
        if self.recent_results:
            recent_failure_rate = self.recent_results.count(False) / len(self.recent_results) * 100
        
        return f"Total: {self.total_calls}, Failed: {self.total_failures} ({100-success_rate:.1f}%), " \
               f"Recent: {recent_failure_rate:.1f}% failure, Consecutive: {self.consecutive_failures}"

# --- API key manager with semaphore, per-key stats, and isolation ---
class ApiKeyProvider:
    """
    API key manager with concurrency control and per-key statistics.
    """
    def __init__(self, api_keys: list, max_concurrency_per_key: int = 8):
        if not api_keys:
            raise ValueError("API keys list cannot be empty.")
        self.api_keys = api_keys
        self.max_concurrency_per_key = max_concurrency_per_key
        
        # Store available key slots in a queue.
        self._available_keys = asyncio.Queue()
        for key in self.api_keys:
            # Add one slot per allowed concurrent use.
            for _ in range(self.max_concurrency_per_key):
                self._available_keys.put_nowait(key)
        
        # Per-key stats.
        self.key_stats = {
            key: {
                'total': 0,
                'success': 0,
                'timeout': 0,
                'error': 0,
                'total_time': 0.0,
                'recent_failures': 0,  # 最近连续失败次数
                'is_isolated': False,   # 是否被隔离
                'isolation_until': 0,   # 隔离到什么时候（timestamp）
                'last_6_suffix': key[-6:] if len(key) >= 6 else key  # 用于日志显示
            } for key in self.api_keys
        }

    async def acquire(self) -> str:
        """
        Acquire an available API key.
        Wait if all keys are at capacity.
        """
        import time
        while True:
            key = await self._available_keys.get()
            stats = self.key_stats.get(key)
            
            # Skip isolated keys.
            if stats and stats['is_isolated']:
                if time.time() >= stats['isolation_until']:
                    # Isolation expired.
                    stats['is_isolated'] = False
                    stats['recent_failures'] = 0
                    logger.info(f"🔓 Key ...{stats['last_6_suffix']} 隔离期满，恢复使用")
                    return key
                else:
                    # Still isolated; put it back and try another key.
                    self._available_keys.put_nowait(key)
                    # Avoid a busy loop if every key is isolated.
                    active_keys = sum(1 for s in self.key_stats.values() if not s['is_isolated'])
                    if active_keys == 0:
                        wait_time = min(stats['isolation_until'] - time.time(), 10)
                        logger.warning(f"⚠️  所有key都被隔离，等待 {wait_time:.1f} 秒...")
                        await asyncio.sleep(max(1, wait_time))
                    continue
            
            return key

    def release(self, key: str):
        """Release an API key back to the pool."""
        self._available_keys.put_nowait(key)

    def get_total_capacity(self) -> int:
        """Return total concurrency capacity."""
        return len(self.api_keys) * self.max_concurrency_per_key
    
    def record_result(self, key: str, success: bool, is_timeout: bool, elapsed_time: float):
        """Record an API call result and isolate bad keys."""
        import time
        
        if key in self.key_stats:
            stats = self.key_stats[key]
            stats['total'] += 1
            stats['total_time'] += elapsed_time
            
            if success:
                stats['success'] += 1
                stats['recent_failures'] = 0  # Reset consecutive failures.
            elif is_timeout:
                stats['timeout'] += 1
                stats['recent_failures'] += 1
            else:
                stats['error'] += 1
                stats['recent_failures'] += 1
            
            # Isolate a key after 5 consecutive failures.
            if stats['recent_failures'] >= 5 and not stats['is_isolated']:
                stats['is_isolated'] = True
                stats['isolation_until'] = time.time() + 60  # 60-second isolation.
                failure_rate = (stats['timeout'] + stats['error']) / stats['total'] * 100 if stats['total'] > 0 else 0
                logger.warning(
                    f"🔒 Key ...{stats['last_6_suffix']} 连续{stats['recent_failures']}次失败 "
                    f"(总失败率{failure_rate:.1f}%)，隔离60秒"
                )
    
    def get_key_stats_summary(self) -> str:
        """Summarize stats for all keys."""
        lines = ["Per-Key Performance:"]
        for key, stats in self.key_stats.items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total'] * 100
                avg_time = stats['total_time'] / stats['total']
                status = "🔒 ISOLATED" if stats['is_isolated'] else ""
                lines.append(
                    f"  Key ...{stats['last_6_suffix']}: "
                    f"{stats['total']} calls, {success_rate:.1f}% success, "
                    f"{stats['timeout']} timeouts, avg {avg_time:.1f}s {status}"
                )
        return "\n".join(lines)


class VaxModel(Model):
    def __init__(self, population_df: pd.DataFrame, network_df: pd.DataFrame, params: dict):
        super().__init__()
        
        # --- Path setup (works from any working directory) ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(script_dir)  # LLMIP_new/
        
        # --- Model parameters ---
        self.params = params
        self.belief_threshold = params.get('belief_threshold', 2.0)
        self.resonance_weight = params.get('resonance_weight', 0.5)
        self.api_keys = params.get('api_keys', [])
        self.api_base_url = params.get('api_base_url', None)
        self.max_concurrency_per_key = params.get('max_concurrency_per_key', 8)
        
        # Local API configuration.
        self.llm_model_name = params.get('llm_model_name', 'Qwen/Qwen3-8B')
        
        # Batch dialogue parameters.
        self.batch_size = params.get('batch_size', 10)
        # Default concurrent batch count equals total API capacity.
        default_concurrent_batches = self.max_concurrency_per_key * len(self.api_keys)
        self.concurrent_batches = params.get('concurrent_batches', default_concurrent_batches)
        
        # --- Mesa components ---
        self.schedule = BaseScheduler(self)
        self.running = True
        self.agent_map = {}
        # Use a DataFrame so results can be written directly to CSV.
        self.datacollector = pd.DataFrame()
        
        # Micro-level agent tracking.
        self.agent_daily_log = []

        # --- 异步和API相关设置 ---
        if not self.api_keys:
            logger.error("No API Keys provided in model_params! Dialogue generation will fail.")
            # 在没有key的情况下，创建一个空的provider，避免后续代码出错
            self.api_key_provider = ApiKeyProvider(["DUMMY_KEY"], 1)
        else:
            # 使用从参数中获取的单key并发限制来初始化Provider
            self.api_key_provider = ApiKeyProvider(self.api_keys, self.max_concurrency_per_key)
            logger.info(f"ApiKeyProvider initialized with {len(self.api_keys)} keys and max concurrency per key: {self.max_concurrency_per_key}.")
            logger.info(f"Total API concurrency capacity: {self.api_key_provider.get_total_capacity()}")
            logger.info(f"Concurrent batch processing: concurrent_batches={self.concurrent_batches} (each batch = 1 API call = 1 concurrency slot)")
            logger.info(f"Expected max agents per step: {self.concurrent_batches} batches × {self.batch_size} agents/batch = {self.concurrent_batches * self.batch_size} agents")

        # Create OpenAI client instances (one per API key) that talk to the local base_url
        self.clients = []
        self._clients_by_key = {}
        for key in self.api_keys:
            # Use base_url from params for local proxy
            if self.api_base_url:
                client = OpenAI(api_key=key, base_url=self.api_base_url, timeout=None)
            else:
                client = OpenAI(api_key=key, timeout=None)
            self.clients.append(client)
            try:
                self._clients_by_key[key] = client
            except Exception:
                pass

        # Provide an async wrapper to call the sync OpenAI client from asyncio using to_thread
        async def api_call_async(client: OpenAI, model_name: str, messages: list, temperature: float = 0.75):
            def _call():
                return client.chat.completions.create(model=model_name, messages=messages, temperature=temperature)
            return await asyncio.to_thread(_call)

        self._client_index = 0
        self._api_call_async = api_call_async

        # Keep an aiohttp session available for any other HTTP needs (not used for LLM)
        self.client_session = aiohttp.ClientSession()
        logger.info(f"Using local API at {self.api_base_url} with model {self.llm_model_name}")

        # --- API失败监控器 ---
        # 动态设置check_window以匹配实际并发批次数
        self.failure_monitor = ApiFailureMonitor(
            failure_threshold=0.95,
            check_window=self.concurrent_batches  # 动态匹配并发数
        )
        
        # --- 初始化网络和代理 ---
        self._build_networks(network_df)  # 先构建网络
        self._create_agents(population_df)  # 再创建智能体

    # --- 数据收集 ---
    # 数据收集只在每个step结束后调用
    def collect_data(self):
        tick_data = {
            'tick': self.schedule.time,
            'vax_rate': np.mean([1 if agent.is_vaccinated else 0 for agent in self.schedule.agents]),
            'avg_belief': np.mean([agent.belief for agent in self.schedule.agents]),
            'avg_alpha': np.mean([agent.alpha for agent in self.schedule.agents]),
            'total_dialogues': sum(len(agent.dialogue_history) for agent in self.schedule.agents),
            'avg_resonance': np.mean([
                np.mean([d['resonance_weight'] for d in agent.dialogue_history]) 
                if agent.dialogue_history else 0 
                for agent in self.schedule.agents
            ])
        }
        self.datacollector = pd.concat([self.datacollector, pd.DataFrame([tick_data])], ignore_index=True)
        
        # 【新增】微观层面的agent数据追踪
        for agent in self.schedule.agents:
            agent_record = {
                'Tick': self.schedule.time,
                'AgentID': agent.unique_id,
                'Belief': agent.belief,
                'SocialInfluence': agent.last_social_influence,
                'VaxStatus': 1 if agent.is_vaccinated else 0,
                'DialogueCount': len(agent.dialogue_history)
            }
            self.agent_daily_log.append(agent_record)
        
        # 【新增】分批写入CSV（防止内存溢出）
        # 每 5000 条记录或每 10 个 Tick 写入一次
        if len(self.agent_daily_log) >= 5000 or (self.schedule.time % 10 == 0 and len(self.agent_daily_log) > 0):
            log_path = os.path.join(self.project_root, "data", "output", "dataframes", "agent_trajectories.csv")
            df = pd.DataFrame(self.agent_daily_log)
            file_exists = os.path.isfile(log_path)
            df.to_csv(log_path, mode='a', header=not file_exists, index=False)
            logger.debug(f"Flushed {len(self.agent_daily_log)} agent records to {log_path}")
            self.agent_daily_log = []

    def _build_networks(self, network_df):
        self.networks = {}
        relation_map = {
            'hh': 'family', 'dc': 'work', 'sc': 'work',
            'wk': 'work', 'sm': 'social_media'
        }
        network_df['layer'] = network_df['Relation'].map(relation_map)
        for layer, df_layer in network_df.groupby('layer'):
            if not df_layer.empty:
                self.networks[layer] = nx.from_pandas_edgelist(
                    df_layer, source='source_reindex', target='target_reindex'
                )
        logger.info(f"Initialized network layers: {list(self.networks.keys())}")

    def _create_agents(self, population_df):
        logger.info(f"Creating {len(population_df)} agents...")
        for _, row in population_df.iterrows():
            agent_id = row['reindex']
            # 优先使用已有profile和embedding，否则自动生成
            profile = row['profile'] if 'profile' in row else ''

            # 修复embedding处理逻辑
            if 'embedding' in row and row['embedding'] is not None:
                try:
                    embedding = np.array(row['embedding'])
                    # 确保embedding是一维数组
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                except (ValueError, TypeError):
                    # 如果转换失败，重新生成embedding
                    embedding = text2embedding(profile).flatten()
            else:
                embedding = text2embedding(profile).flatten()
            
            #!!! alpha setting - uniform distribution [0,1]
            alpha = np.random.uniform(0, 1)
            
            row_data = dict(row)
            row_data['profile'] = profile
            row_data['embedding'] = embedding
            row_data['alpha'] = alpha
            
            # 确保PUMS特征存在
            pums_features = ['personal_income', 'education', 'occupation', 'health_insurance']
            # pums_features = ['personal_income', 'education', 'occupation', 'health_insurance', 'HHT', 'FINCP', 'num_children', 'family_size']
            for feature in pums_features:
                if feature not in row_data or pd.isna(row_data[feature]):
                    row_data[feature] = None
            
            agent = VaxAgent(agent_id, self, row_data)
            
            #!!! valid neighbor setting and weight setting
            agent.belief_threshold = self.belief_threshold
            agent.resonance_weight = self.resonance_weight
            # 统计邻居数量
            agent.n_family_neighbors = len([n for n in agent.get_neighbors() if 'family' in self.networks and n.unique_id in self.networks['family'].nodes])
            agent.n_work_neighbors = len([n for n in agent.get_neighbors() if 'work' in self.networks and n.unique_id in self.networks['work'].nodes])
            agent.n_smedia_neighbors = len([n for n in agent.get_neighbors() if 'social_media' in self.networks and n.unique_id in self.networks['social_media'].nodes])
            
            self.schedule.add(agent)
            self.agent_map[agent_id] = agent
            # 在 model.py 的 VaxModel.initialize_agents 最后
            # avg_degree = np.mean([len(list(agent.get_neighbors())) for agent in self.schedule.agents])
            # logger.info(f"Network initialized. Average neighbors per agent: {avg_degree:.2f}")
        
        # 调试: 检查青少年的tick分布
        teens = [a for a in self.schedule.agents if a.age is not None and 14 <= a.age < 18]
        if teens:
            teen_ticks = [a.tick for a in teens]
            from collections import Counter
            tick_dist = Counter(teen_ticks)
            logger.info(f"青少年(14-17岁)数量: {len(teens)}, tick分布: {dict(tick_dist)}")
            teen_with_tick_14 = sum(1 for t in teen_ticks if t == 14)
            logger.info(f"  其中tick=14的青少年: {teen_with_tick_14}人")
            if teen_with_tick_14 == 0:
                logger.error("❌ BUG确认: 所有青少年的tick都不是14!")

    async def run_model(self, n_steps):
        """【异步】运行模型 n 个步骤"""
        import time
        total_start = time.time()
        
        for i in range(n_steps):
            await self.step()
            
            # 每10步输出一次汇总
            if (i + 1) % 10 == 0 or (i + 1) == n_steps:
                elapsed = time.time() - total_start
                avg_time_per_step = elapsed / (i + 1)
                remaining_steps = n_steps - (i + 1)
                eta = remaining_steps * avg_time_per_step
                
                logger.info(f"\n{'='*80}")
                logger.info(f"📈 Progress: {i+1}/{n_steps} steps ({(i+1)/n_steps*100:.1f}%)")
                logger.info(f"⏱️  Elapsed: {elapsed/60:.1f}min | Avg: {avg_time_per_step:.1f}s/step | ETA: {eta/60:.1f}min")
                logger.info(f"{'='*80}\n")
        
        # 【新增】在关闭session前，将剩余的agent日志写入CSV
        if len(self.agent_daily_log) > 0:
            log_path = os.path.join(self.project_root, "data", "output", "dataframes", "agent_trajectories.csv")
            df = pd.DataFrame(self.agent_daily_log)
            file_exists = os.path.isfile(log_path)
            df.to_csv(log_path, mode='a', header=not file_exists, index=False)
            logger.info(f"Flushed remaining {len(self.agent_daily_log)} agent records to {log_path}")
            self.agent_daily_log = []
        
        logger.info("Saved agent trajectories.")
        
        # 关闭 aiohttp session
        await self.client_session.close()

    async def step(self):
        """
        【异步】模型的一个步骤。
        使用并发批量对话处理：多个batch并发执行，每个batch消耗1个API Key并发限制。
        
        1. 将eligible agents分成多个batch
        2. 并发处理多个batch（受API Key并发限制）
        3. 每个batch内 = 1个API调用，包含所有agent-neighbor对
        4. 同步推进agent状态、执行决策、收集数据
        """
        import time
        step_start = time.time()
        current_step = self.schedule.time + 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📍 Step {current_step} - Batch dialogue processing...")
        logger.info(f"{'='*80}")
        
        # 1. 并发批量对话处理
        await self.batch_dialogue_update(batch_size=self.batch_size, concurrent_batches=self.concurrent_batches)

        # 2. 推进信念状态
        logger.debug("Advancing agent beliefs...")
        for agent in self.schedule.agents:
            agent.advance()

        # 3. 执行行动决策
        logger.debug("Executing agent actions...")
        for agent in self.schedule.agents:
            agent.step()

        # 4. 推进tick并收集数据
        self.schedule.time += 1
        self.collect_data()
        
        # 统计本步信息
        step_duration = time.time() - step_start
        vax_count = sum(1 for a in self.schedule.agents if a.is_vaccinated)
        vax_rate = vax_count / len(self.schedule.agents)
        logger.info(f"✅ Step {current_step} completed in {step_duration:.1f}s | Vaccinated: {vax_count}/{len(self.schedule.agents)} ({vax_rate*100:.1f}%)")
        logger.info(f"📊 API Stats: {self.failure_monitor.get_stats()}\n")
    
    async def batch_dialogue_update(self, batch_size=20, concurrent_batches=8):
        """
        【核心】并发批量对话处理方法
        
        架构：
        - 将eligible agents分成多个batches（每个batch包含batch_size个agents）
        - 并发处理多个batches（每个batch消耗1个API Key并发限制）
        - 每个batch内部 = 1个API调用，包含该batch中所有agents的所有对话需求
        
        关键设计：
        - 每个batch = 1个 _generate_batch_dialogues() 调用 = 1个API调用
        - 每个API调用占用 1个API Key并发限制（来自api_key_provider）
        - concurrent_batches 决定最多并发处理多少个batch（受API Key总并发数限制）
        
        Args:
            batch_size: 每个batch的agent数量（默认20）
            concurrent_batches: 最多并发处理多少个batch（默认8，受API Key并发限制）
        """
        # 1. 筛选需要更新的agents（所有未接种的agents）
        # 注：14岁以下的小孩在tick前不参与对话，这在 _process_batch 中处理
        agents_for_update = [
            agent for agent in self.schedule.agents
            if not agent.is_vaccinated
        ]
        
        if not agents_for_update:
            logger.info("   ⚠️  No agents for dialogue update.")
            return
        
        logger.info(f"   👥 Agents for update: {len(agents_for_update)}")
        
        # 2. 将agents分成多个batches
        batches = []
        for i in range(0, len(agents_for_update), batch_size):
            batch = agents_for_update[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"   📦 Total {len(batches)} batches to process with dynamic pool (max_concurrent={concurrent_batches})")
        
        # 3. 动态并发池：完成一个batch就启动下一个
        # 关键改进：不再分组等待，而是维护一个动态的并发池
        # - 始终保持 concurrent_batches 个任务在运行
        # - 一个任务完成后，立即从队列中取下一个batch启动
        # - 这样可以最大化利用API并发能力，减少空闲等待
        
        pending_batches = list(batches)  # 待处理的batch队列
        running_tasks = set()  # 正在运行的任务集合
        completed_count = 0  # 已完成的batch数量
        
        # 启动初始的 concurrent_batches 个任务
        while pending_batches and len(running_tasks) < concurrent_batches:
            batch = pending_batches.pop(0)
            task = asyncio.create_task(self._process_batch(batch))
            running_tasks.add(task)
        
        logger.info(f"   🚀 Started initial {len(running_tasks)} batches, {len(pending_batches)} pending")
        
        # 动态调度：每完成一个任务就启动新的
        while running_tasks:
            # 等待任何一个任务完成
            done, running_tasks = await asyncio.wait(running_tasks, return_when=asyncio.FIRST_COMPLETED)
            
            completed_count += len(done)
            
            # 启动新的batch来填补完成的位置
            newly_started = 0
            while pending_batches and len(running_tasks) < concurrent_batches:
                batch = pending_batches.pop(0)
                task = asyncio.create_task(self._process_batch(batch))
                running_tasks.add(task)
                newly_started += 1
            
            # 定期汇报进度
            if completed_count % 10 == 0 or not pending_batches:
                logger.info(f"   📊 Progress: {completed_count}/{len(batches)} completed, "
                           f"{len(running_tasks)} running, {len(pending_batches)} pending")
                logger.debug(f"   📈 API Stats: {self.failure_monitor.get_stats()}")
                # 【新增】每500个batch或最后打印per-key统计
                if completed_count % 500 == 0 or (not pending_batches and len(running_tasks) == 0):
                    logger.info(f"\n{self.api_key_provider.get_key_stats_summary()}\n")
        
        logger.info(f"   ✓ All {len(batches)} batches completed")
        # 【新增】最终打印per-key统计
        logger.info(f"\n📊 Final Per-Key Statistics:\n{self.api_key_provider.get_key_stats_summary()}\n")
    
    async def _process_batch(self, batch_agents: list):
        """
        【核心】处理单个batch的对话生成和更新
        
        执行步骤：
        1. 收集batch中所有agents的agent-neighbor对（包括网络层信息）
           - 14岁以下的小孩：只有在 time >= tick 时才参与对话
           - 14岁及以上的成人：始终参与对话（只要未接种）
           - HH (family) 层和 Work 层：保持全量交互，有多少邻居就交互多少
           - SM (social_media) 层：使用Poisson分布(lambda=2)随机采样，减少LLM调用压力
        2. 调用 _generate_batch_dialogues()：
           - acquire 1个API Key的并发限制
           - 根据网络层选择合适的profile信息
           - 构造包含所有对话需求的prompt
           - 发送 1个 LLM API调用
           - 解析响应
           - release API Key
        3. 更新batch中agents的信念（只有被选中的neighbors会被影响）
        
        关键：该函数会消耗 api_key_provider 中的 1个并发限制
        """
        logger.debug(f"Processing batch with {len(batch_agents)} agents")
        
        # 1. 为这个batch收集所有agent-neighbor对（包括网络层）
        agent_neighbor_pairs_with_layer = []
        for agent in batch_agents:
            # 检查小孩是否符合对话条件
            # 14岁以下的小孩：只有在 time >= tick 时才参与对话
            # 14岁及以上的成人：始终参与对话
            if agent.age < 14 and self.schedule.time < agent.tick:
                continue  # 小孩在tick前不交互，跳过
            
            agent.update_network_weights()
            valid_neighbors = agent.get_valid_neighbors()
            
            if valid_neighbors:
                # 【分层处理】HH(family)和Work层：全量交互；SM层：Poisson(λ=2)采样
                sm_neighbors = []
                for neighbor in valid_neighbors:
                    layer = self._get_network_layer(agent, neighbor)
                    if layer in ('family', 'work'):
                        # HH和Work层：保持全量交互，不做采样
                        agent_neighbor_pairs_with_layer.append((agent, neighbor, layer))
                    elif layer == 'social_media':
                        sm_neighbors.append(neighbor)
                
                # SM层：使用Poisson分布(lambda=2)随机选择交互邻居，减少LLM压力
                if sm_neighbors:
                    k = int(np.random.poisson(lam=2))
                    k = min(k, len(sm_neighbors))
                    if k > 0:
                        sampled_sm = random.sample(sm_neighbors, k)
                        for neighbor in sampled_sm:
                            agent_neighbor_pairs_with_layer.append((agent, neighbor, 'social_media'))
        
        if not agent_neighbor_pairs_with_layer:
            logger.debug(f"Batch has no valid pairs")
            return
        
        logger.debug(f"Batch collected {len(agent_neighbor_pairs_with_layer)} agent-neighbor pairs (HH/Work: full, SM: Poisson λ=2 sampled)")
        
        # 2. 生成这个batch的所有对话（带重试机制）
        batch_dialogues = await self._generate_batch_dialogues_with_retry(agent_neighbor_pairs_with_layer)
        
        # 3. 更新这个batch中的agents
        self._update_agents_from_batch_dialogues(batch_agents, batch_dialogues)
    
    def _get_network_layer(self, agent, neighbor) -> str:
        """
        确定 neighbor 在哪个网络层
        
        返回：'family', 'work', 或 'social_media'
        """
        # 需要先检查 agent 是否在网络中，避免 KeyError
        if self.networks.get('family') and agent.unique_id in self.networks['family']:
            if neighbor.unique_id in self.networks['family'].neighbors(agent.unique_id):
                return 'family'
        
        if self.networks.get('work') and agent.unique_id in self.networks['work']:
            if neighbor.unique_id in self.networks['work'].neighbors(agent.unique_id):
                return 'work'
        
        if self.networks.get('social_media') and agent.unique_id in self.networks['social_media']:
            if neighbor.unique_id in self.networks['social_media'].neighbors(agent.unique_id):
                return 'social_media'
        
        return 'unknown'
    
    async def _generate_batch_dialogues_with_retry(self, agent_neighbor_pairs_with_layer, max_retries=3):
        """
        带拆分机制的批量对话生成（已移除外层重试）
        
        核心逻辑：
        - _generate_batch_dialogues 内部实现自动拆分重试
        - 当批次失败时，自动拆分为两半递归处理
        - 无需外层重试机制
        
        Args:
            agent_neighbor_pairs_with_layer: agent-neighbor对列表
            max_retries: 保留参数以兼容旧代码，但不使用
            
        Returns:
            Dict mapping (agent_id, neighbor_id) -> dialogue_result
        """
        try:
            result = await self._generate_batch_dialogues(agent_neighbor_pairs_with_layer)
            
            if result:
                self.failure_monitor.record_success()
                return result
            else:
                # 返回空字典，记录失败
                self.failure_monitor.record_failure()
                return {}
                    
        except Exception as e:
            self.failure_monitor.record_failure()
            logger.error(f"Exception in batch dialogue generation: {str(e)}")
            return {}
    
    async def _generate_batch_dialogues(self, agent_neighbor_pairs_with_layer):
        """
        构造批量prompt并调用LLM生成所有对话。
        实现"报错自动拆分重试"机制：当API调用或JSON解析失败时，自动将batch拆分为两半递归处理。
        
        【关键改进】根据不同网络层使用不同可见度的 profile
        
        Args:
            agent_neighbor_pairs_with_layer: List of (agent, neighbor, layer) tuples
            
        Returns:
            Dict mapping (agent_id, neighbor_id) -> dialogue_result
        """
        from agent import get_attitude_from_belief
        
        batch_size = len(agent_neighbor_pairs_with_layer)
        
        # 【核心逻辑】外层try-except处理异常时自动拆分
        api_key = None
        try:
            # 构造批量prompt
            conversations = []
            pair_mapping = []  # 记录索引到(agent, neighbor)的映射
            
            for idx, (agent, neighbor, network_layer) in enumerate(agent_neighbor_pairs_with_layer):
                neighbor_attitude = get_attitude_from_belief(neighbor.belief)
                
                # 【更新】Sender 使用完整 profile（全知视角），Receiver 使用分层可见的 profile（受限视角）
                sender_full_profile = neighbor.profile
                receiver_profile_limited = agent.get_profile_for_layer(network_layer)
                
                conversation_item = {
                    "conversation_id": idx,
                    "receiver_agent_id": agent.unique_id,
                    "sender_agent_id": neighbor.unique_id,
                    "sender_profile": sender_full_profile,  # 使用完整 profile（Sender 全知）
                    "sender_attitude": neighbor_attitude,
                    "receiver_profile": receiver_profile_limited,  # 使用分层可见的 profile（Receiver 受限）
                    "network_layer": network_layer  # 记录网络层
                }
                conversations.append(conversation_item)
                pair_mapping.append((agent, neighbor))
            
            # 【改进】构造第一人称完整句子格式的system prompt
            # 指导LLM理解不同网络层的信息可见度
            system_prompt = """You are an expert dialogue writer generating authentic, persuasive messages between people with different vaccine beliefs.

IMPORTANT:
- You act as the Sender. You know your own full background details (full Sender Profile).
- You are talking to a Receiver. You ONLY know what is publicly visible about them in the current network layer (limited Receiver Profile).

For each message:
1. Use FIRST PERSON (I, me, my).
2. Leverage your background (Sender Profile) to build credibility, BUT do not reveal private details (e.g., exact income, home address, family identifiers) unless socially appropriate for the current network layer (okay for Family; avoid for Social Media; be cautious at Work).
3. Address the Receiver based ONLY on their limited profile (Target Audience). Do not hallucinate private details about them.
4. Keep it around 25-50 words, natural and conversational.

Return a valid JSON array where each element has:
- "message_id": the number from the input (0, 1, 2, ...)
- "dialogue": your generated message text

Example format:
[
    {"message_id": 0, "dialogue": "message here..."},
    {"message_id": 1, "dialogue": "message here..."}
]"""

            # 【改进】构造user prompt，采用完整的第一人称语句结构
            user_prompt = f"""Generate {len(conversations)} first-person persuasive messages. You are the Sender (you know your full background). You only know layer-visible info about the Receiver.\n\n"""

            for conv in conversations:
                layer = conv.get('network_layer', 'unknown')
                user_prompt += (
                    f"""[Message {conv['conversation_id']}] Context: {layer} network\n"""
                    f"Sender (me) — Full Profile: {conv['sender_profile']}\n"
                    f"My vaccine stance: {conv['sender_attitude']}\n"
                    f"Receiver (target) — Limited to {layer}: {conv['receiver_profile']}\n"
                    f"Task: Write my persuasive message to them. Keep around 25-50 words. Avoid revealing my private details unless socially appropriate for {layer}. Only use receiver info visible at {layer}.\n\n"
                )
            
            user_prompt += "\nReturn all messages as a JSON array with message_id and dialogue fields."
            
            # 【改进】使用OpenAI client进行API调用
            # 根据api_key获取对应的client
            api_key = await self.api_key_provider.acquire()
            client = self._clients_by_key.get(api_key, self.clients[0] if self.clients else None)
            if not client:
                logger.error(f"No OpenAI client available for API key")
                return {}
            
            # 准备消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            logger.debug(f"Sending batch request with {len(conversations)} conversations using OpenAI client...")
            
            # 使用OpenAI client进行API调用
            result = await self._api_call_async(client, self.llm_model_name, messages, temperature=0.75)
            
            if hasattr(result, 'choices') and result.choices:
                content = result.choices[0].message.content.strip()
                
                # 解析JSON响应 - 保持原有的正则表达式不变
                # 【改进】处理包含 <think> 标签的响应
                content_cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                
                # 如果清理后仍为空，直接返回
                if not content_cleaned:
                    logger.error(f"Response is empty after removing <think> tags")
                    return {}
                
                # 尝试解析 JSON
                # 先尝试作为数组解析
                try:
                    dialogues_list = json.loads(content_cleaned)
                    if not isinstance(dialogues_list, list):
                        # 如果是单个对象，转换为数组
                        dialogues_list = [dialogues_list]
                except json.JSONDecodeError:
                    # 如果直接解析失败，尝试从响应中提取 JSON 数组
                    json_match = re.search(r'\[.*?\]', content_cleaned, re.DOTALL)
                    if json_match:
                        dialogues_list = json.loads(json_match.group(0))
                    else:
                        # 最后尝试找 JSON 对象并包装成数组
                        json_match = re.search(r'\{.*?\}', content_cleaned, re.DOTALL)
                        if json_match:
                            dialogues_list = [json.loads(json_match.group(0))]
                        else:
                            logger.error(f"Cannot find valid JSON in response: {content_cleaned[:200]}")
                            return {}

                # 构造返回字典，同时过滤拒绝回答
                batch_dialogues = {}
                rejected_count = 0
                for item in dialogues_list:
                    # 兼容 message_id 和 conversation_id
                    msg_id = item.get("message_id") or item.get("conversation_id")
                    dialogue_text = item.get("dialogue", "")
                    
                    if msg_id is not None and msg_id < len(pair_mapping):
                        agent, neighbor = pair_mapping[msg_id]
                        
                        # 【新增】检查是否包含拒绝词汇
                        if self._is_rejected_dialogue(dialogue_text):
                            rejected_count += 1
                            logger.debug(f"Rejected dialogue for pair ({agent.unique_id}, {neighbor.unique_id}): {dialogue_text[:80]}...")
                            # 不添加到结果中，这样在 _update_agents_from_batch_dialogues 中会被跳过
                        else:
                            # 有效对话，添加到结果
                            batch_dialogues[(agent.unique_id, neighbor.unique_id)] = dialogue_text
                
                if rejected_count > 0:
                    logger.info(f"⚠️  Filtered {rejected_count} rejected dialogues (contain refusal keywords)")
                logger.debug(f"Successfully generated {len(batch_dialogues)} valid dialogues (rejected: {rejected_count}).")

                return batch_dialogues
            else:
                logger.error(f"LLM API returned unexpected format: {result}")
                return {}
        
        except (Exception, json.JSONDecodeError) as e:
            # 不再拆分子批次，失败直接跳过，避免潜在的并发死锁
            logger.error(f"Batch failed (size={batch_size}), error: {type(e).__name__}: {str(e)[:100]}. Skipping this batch without split.")
            return {}
        finally:
            # 统一在 finally 释放 key，避免异常路径漏释放导致卡住
            if api_key is not None:
                self.api_key_provider.release(api_key)
    
    def _is_rejected_dialogue(self, dialogue_text: str) -> bool:
        """
        检测对话是否包含拒绝词汇（LLM拒绝回答）
        
        Args:
            dialogue_text: 对话文本
            
        Returns:
            True if rejected (contains refusal keywords), False otherwise
        """
        if not dialogue_text or not dialogue_text.strip():
            return False
        
        dialogue_lower = dialogue_text.lower()
        for keyword in REJECTION_KEYWORDS:
            if keyword.lower() in dialogue_lower:
                return True
        
        return False
    
    def _update_agents_from_batch_dialogues(self, selected_agents, batch_dialogues):
        """
        根据批量对话结果更新agents的信念
        
        Args:
            selected_agents: 被选中的agents列表
            batch_dialogues: Dict mapping (agent_id, neighbor_id) -> dialogue_text
        """
        from agent import cosine_similarity
        
        for agent in selected_agents:
            valid_neighbors = agent.get_valid_neighbors()
            
            if not valid_neighbors:
                agent.next_belief = agent.belief
                continue
            
            # 收集该agent的所有对话影响
            layer_influences = {'family': [], 'work': [], 'social_media': []}
            
            for neighbor in valid_neighbors:
                dialogue_key = (agent.unique_id, neighbor.unique_id)
                dialogue_text = batch_dialogues.get(dialogue_key)
                
                if dialogue_text:
                    # 计算语义共鸣
                    mu_ij = agent.calculate_semantic_resonance(neighbor, dialogue_text)
                    
                    # 确定网络层（使用正确的方法检查边）
                    layer = self._get_network_layer(agent, neighbor)
                    
                    if layer in layer_influences:
                        layer_influences[layer].append((mu_ij, neighbor.belief))
                    
                    # 记录对话历史
                    agent.dialogue_history.append({
                        'tick': self.schedule.time,
                        'neighbor_id': neighbor.unique_id,
                        'dialogue': dialogue_text,
                        'resonance_weight': mu_ij,
                        'neighbor_belief': neighbor.belief,
                        'network_layer': layer
                    })
            
            # 计算每个网络层的加权平均意见
            avg_layer_opinion = {}
            for layer, values in layer_influences.items():
                if values:
                    total_mu = sum(v[0] for v in values)
                    total_weighted_belief = sum(v[0] * v[1] for v in values)
                    avg_layer_opinion[layer] = total_weighted_belief / total_mu if total_mu > 0 else 0
                else:
                    avg_layer_opinion[layer] = 0
            
            # 计算最终的社会总影响
            social_influence = (
                agent.w_family * avg_layer_opinion['family'] +
                agent.w_work * avg_layer_opinion['work'] +
                agent.w_smedia * avg_layer_opinion['social_media']
            )
            
            if social_influence == 0:
                social_influence = agent.belief
            
            # 【新增】记录社会影响值用于数据追踪
            agent.last_social_influence = social_influence
            
            # 更新下一时刻的信念
            agent.next_belief = (1 - agent.alpha) * agent.belief + agent.alpha * social_influence
            agent.next_belief = np.clip(agent.next_belief, -1.0, 1.0)
        
        logger.debug(f"Updated beliefs for {len(selected_agents)} agents.")

    def get_dialogue_statistics(self):
        """获取并聚合所有agent的对话历史和统计信息"""
        all_dialogues = []
        for agent in self.schedule.agents:
            for dialogue in agent.dialogue_history:
                dialogue_info = dialogue.copy()
                dialogue_info['receiver_id'] = agent.unique_id
                all_dialogues.append(dialogue_info)
        
        if not all_dialogues:
            return {
                'total_dialogues': 0,
                'avg_resonance': 0,
                'dialogue_df': pd.DataFrame()
            }
            
        dialogue_df = pd.DataFrame(all_dialogues)
        avg_resonance = dialogue_df['resonance_weight'].mean()
        
        return {
            'total_dialogues': len(dialogue_df),
            'avg_resonance': avg_resonance,
            'dialogue_df': dialogue_df
        }
