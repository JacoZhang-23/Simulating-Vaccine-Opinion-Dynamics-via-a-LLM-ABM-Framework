
# --- LLM API Configuration ---
API_KEY = "d0d8503ed7e04d85944ec78b490aa1eb4349834e166f4961920915b0bd58d828"
API_URL = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"

# src/agent.py

from mesa import Agent
import random
import numpy as np
import time
import aiohttp
import asyncio
import json
from tools import text2embedding

# --- Belief to Attitude Mapping ---
BELIEF_PROMPT_MAPPING = {
    (-1.0, -0.6): "I am strongly against the vaccine and think it's harmful.",
    (-0.6, -0.2): "I am skeptical about the vaccine and don't trust it.",
    (-0.2, 0.2): "I am uncertain about the vaccine and see both pros and cons.",
    (0.2, 0.6): "I am leaning towards getting the vaccine, it seems beneficial.",
    (0.6, 1.0): "I strongly support the vaccine and believe everyone should get it."
}

def get_attitude_from_belief(belief: float) -> str:
    """Map a belief value to a discrete attitude statement."""
    if belief >= 1.0:
        return BELIEF_PROMPT_MAPPING[(0.6, 1.0)]
    
    for (lower_bound, upper_bound), attitude in BELIEF_PROMPT_MAPPING.items():
        if lower_bound <= belief < upper_bound:
            return attitude
            
    return "I have a neutral stance on the vaccine."

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)


class VaxAgent(Agent):
    def __init__(self, unique_id, model, initial_data_row):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model
        self.profile = initial_data_row.get('profile', '')
        self.embedding = self._init_embedding(initial_data_row)
        self.belief = random.uniform(-1, 1)
        self.alpha = initial_data_row.get('alpha', 0.5)
        self.is_vaccinated = False
        self.next_belief = self.belief
        self.resonance_weight = 0.4
        
        # Demographic features.
        self.age = initial_data_row.get('age', None)
        self.urban = initial_data_row.get('urban', None)
        self.geoid = initial_data_row.get('GEOID_cty', None)
        self.if_employed = initial_data_row.get('if_employed', None)
        self.tick = initial_data_row.get('tick', 999)
        self.tick_vaccinated = None
        self.personal_income = initial_data_row.get('personal_income', None)
        self.education = initial_data_row.get('education', None)
        self.occupation = initial_data_row.get('occupation', None)
        self.health_insurance = initial_data_row.get('health_insurance', None)
        self.HHT = initial_data_row.get('HHT', None)
        self.FINCP = initial_data_row.get('FINCP', None)
        self.num_children = initial_data_row.get('num_children', None)
        self.family_size = initial_data_row.get('family_size', None)
        
        # Network-related state.
        self.w_family, self.w_work, self.w_smedia = 0, 0, 0
        self.n_family_neighbors, self.n_work_neighbors, self.n_smedia_neighbors = 0, 0, 0
        self.dialogue_history = []
        self.belief_threshold = 2.0
        
        self.avg_family_yt = 0
        self.avg_work_yt = 0
        self.avg_smedia_yt = 0
        
        # Micro-level tracking.
        self.last_social_influence = 0.0

    # --- Basic methods ---
    def _init_embedding(self, row):
        """Initialize the profile embedding."""
        if 'embedding' in row and row['embedding'] is not None:
            return np.array(row['embedding'], dtype=np.float32)
        profile_text = row.get('profile', '')
        if profile_text:
            return text2embedding(profile_text)
        return np.zeros(768)

    def get_neighbors(self):
        """Get all neighbors."""
        neighbors = set()
        for layer in ['family', 'work', 'social_media']:
            if layer in self.model.networks and self.model.networks[layer].has_node(self.unique_id):
                neighbors.update(self.model.networks[layer].neighbors(self.unique_id))
        return [self.model.agent_map[nid] for nid in neighbors if nid in self.model.agent_map]

    def get_neighbors_by_layer(self, layer_name):
        """Get neighbors for one layer."""
        neighbors = []
        if layer_name in self.model.networks and self.model.networks[layer_name].has_node(self.unique_id):
            neighbor_ids = self.model.networks[layer_name].neighbors(self.unique_id)
            neighbors = [self.model.agent_map[nid] for nid in neighbor_ids if nid in self.model.agent_map]
        return neighbors

    def check_guardian_permission(self):
        """Check guardian permission."""
        if self.age is not None and self.age < 14:
            family_neighbors = self.get_neighbors_by_layer('family')
            adult_guardians = [n for n in family_neighbors if n.age is not None and n.age >= 18]
            return any(guardian.is_vaccinated for guardian in adult_guardians) if adult_guardians else False
        return True

    def update_network_weights(self):
        """Update network weights."""
        self.n_family_neighbors = len(self.get_neighbors_by_layer('family'))
        self.n_work_neighbors = len(self.get_neighbors_by_layer('work'))
        self.n_smedia_neighbors = len(self.get_neighbors_by_layer('social_media'))
        
        total_neighbors = self.n_family_neighbors + self.n_work_neighbors + self.n_smedia_neighbors
        if total_neighbors > 0:
            # Give the family layer extra weight.
            w_family = 3 if self.n_family_neighbors > 0 else 0
            w_work = 1 if self.n_work_neighbors > 0 else 0
            w_smedia = 1 if self.n_smedia_neighbors > 0 else 0
            total_weight = w_family + w_work + w_smedia
            if total_weight > 0:
                self.w_family = w_family / total_weight
                self.w_work = w_work / total_weight
                self.w_smedia = w_smedia / total_weight

    def get_valid_neighbors(self):
        """Get valid communication neighbors."""
        all_neighbors = self.get_neighbors()
        return [n for n in all_neighbors if abs(self.belief - n.belief) <= self.belief_threshold]

    def calculate_semantic_resonance(self, neighbor, dialogue_text):
        """Compute semantic resonance for a dialogue."""
        if dialogue_text is None:
            return 0
        
        sim_V = cosine_similarity(self.embedding, neighbor.embedding)
        dialogue_embedding = text2embedding(dialogue_text)
        sim_M = cosine_similarity(self.embedding, dialogue_embedding)
        return self.resonance_weight * sim_V + (1 - self.resonance_weight) * sim_M

    def get_profile_for_layer(self, layer_name: str) -> str:
        """
        Return the layer-specific profile text.

        Information visibility varies by layer:
        family: full profile
        work: medium-detail profile
        social_media: limited public profile
        """
        if layer_name == 'family':
            # Full profile.
            return self.profile
        
        elif layer_name == 'work':
            # Work layer sees more detail.
            parts = []
            if self.occupation:
                parts.append(f"works as a {self.occupation}")
            if self.age:
                parts.append(f"age {self.age}")
            if self.education:
                parts.append(f"education: {self.education}")
            if self.personal_income:
                parts.append(f"income: {self.personal_income}")
            if self.if_employed is not None:
                employment_status = "employed" if self.if_employed else "unemployed"
                parts.append(f"{employment_status}")
            if self.urban is not None:
                location = "urban" if self.urban else "rural"
                parts.append(f"living in {location} area")
            
            if parts:
                return "I " + ", ".join(parts)
            return self.profile
        
        elif layer_name == 'social_media':
            # Social media sees limited public information.
            parts = []
            if self.occupation:
                parts.append(f"works as a {self.occupation}")
            if self.age:
                parts.append(f"age {self.age}")
            
            if parts:
                return "I " + ", ".join(parts)
            # Fallback when there is no occupation or age.
            return f"I am age {self.age}" if self.age else "I am on social media"
        
        else:
            # Default to full profile.
            return self.profile

    def advance(self):
        """Advance the agent belief to the next state."""
        # Vaccinated agents stay at belief = 1.0.
        if self.is_vaccinated:
            return
        self.belief = self.next_belief

    def step(self):
        """Perform one agent action step."""
        if self.is_vaccinated:
            return
            
        if self.model.schedule.time >= self.tick and self.check_guardian_permission():
            if self.belief > 0:
                vaccination_prob = self.belief
                if random.random() < vaccination_prob:
                    self.is_vaccinated = True
                    self.belief = 1.0
                    self.tick_vaccinated = self.model.schedule.time
