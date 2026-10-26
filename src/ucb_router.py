"""UCB-based LLM Router for FunSearch.

Strategies
----------
ucb : UCB1 — mean_utility + beta * sqrt(log(t) / n_pulls)

Baseline modes
--------------
evolving    : baseline = running global-best score, grows each step.
elite_median: baseline = median of top-E scores once pool full, else global-best.
"""

import heapq
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from omegaconf import DictConfig
from hydra.utils import instantiate


class UCBRouter:
    """UCB router for selecting LLMs in FunSearch."""

    def __init__(
        self,
        llm_clients: Dict[str, Any],
        beta: float = 1.0,
        initial_best_score: float = -np.inf,
        baseline_mode: str = "elite_median",
        elite_pool_size: int = 50,
    ):
        self.active_model_id = None
        self.hf_models_list = ['gemma']

        self.llm_clients = llm_clients
        self.llm_names = list(llm_clients.keys())
        self.beta = beta
        self.best_score = initial_best_score

        # ── Baseline ───────────────────────────────────────────────────────
        self.baseline_mode = baseline_mode
        self.elite_pool_size = elite_pool_size
        self.elite_pool: List[float] = []

        # ── Core statistics ────────────────────────────────────────────────
        self.n_pulls = {name: 0 for name in self.llm_names}
        self.mean_utility = {name: 0.0 for name in self.llm_names}
        self.iteration = 0

        print(f"[UCB Router] Initialized with {len(self.llm_names)} LLMs: {self.llm_names}")
        print(f"[UCB Router] Beta={beta}, Initial Best={initial_best_score}")
        print(f"[UCB Router] Baseline={baseline_mode}, ElitePoolSize={elite_pool_size}")

    # ─────────────────────────────────────────────────────────────────────────
    # Baseline helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_baseline(self) -> float:
        if self.baseline_mode == "evolving":
            return self.best_score
        # elite_median
        if len(self.elite_pool) >= self.elite_pool_size:
            return float(np.median(self.elite_pool))
        return self.best_score

    def _update_elite_pool(self, score: float) -> None:
        if self.baseline_mode != "elite_median" or not np.isfinite(score):
            return
        if len(self.elite_pool) < self.elite_pool_size:
            heapq.heappush(self.elite_pool, score)
        elif score > self.elite_pool[0]:
            heapq.heapreplace(self.elite_pool, score)

    def _compute_reward(self, score: float, baseline: float) -> float:
        return float(max(0.0, score - baseline))

    # ─────────────────────────────────────────────────────────────────────────
    # UCB
    # ─────────────────────────────────────────────────────────────────────────

    def compute_ucb_scores(self) -> Dict[str, float]:
        scores = {}
        for name in self.llm_names:
            exploit = self.mean_utility[name]
            if self.n_pulls[name] == 0:
                explore = float('inf')
            else:
                explore = self.beta * np.sqrt(np.log(self.iteration) / self.n_pulls[name])
            scores[name] = exploit + explore
        return scores

    def compute_scores(self) -> Dict[str, float]:
        return self.compute_ucb_scores()

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle management for local HF models
    # ─────────────────────────────────────────────────────────────────────────

    def _get_or_instantiate_client(self, name: str) -> Any:
        is_config = isinstance(self.llm_clients[name], DictConfig)
        if is_config:
            if self.active_model_id and self.active_model_id != name:
                active_client = self.llm_clients[self.active_model_id]
                if hasattr(active_client, 'cleanup'):
                    saved_config = active_client._config
                    active_client.cleanup()
                    self.llm_clients[self.active_model_id] = saved_config
            print(f"[Router] Instantiating {name}...")
            self.llm_clients[name] = instantiate(self.llm_clients[name], _config=self.llm_clients[name])
        return self.llm_clients[name]

    # ─────────────────────────────────────────────────────────────────────────
    # Selection
    # ─────────────────────────────────────────────────────────────────────────

    def select_llm(self) -> Tuple[str, Any]:
        scores = self.compute_ucb_scores()
        selected_name = max(scores, key=scores.get)
        print(f"\n[Router] Iteration {self.iteration}")
        print(f"[Router] Pull counts: {self.n_pulls}")
        print(f"[Router] UCB scores: {scores}")
        print(f"[Router] Selected: {selected_name}")
        client = self._get_or_instantiate_client(selected_name)
        self.active_model_id = selected_name
        return selected_name, client

    # ─────────────────────────────────────────────────────────────────────────
    # Utility computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_utility(
        self,
        scores: List[float],
    ) -> Tuple[float, List[float], float, List[float]]:
  
        rewards = []
        local_best = self.best_score
        baseline = self._get_baseline()

        for score in scores:
            reward = self._compute_reward(score, baseline)
            rewards.append(reward)
            local_best = max(local_best, score)

        aggregated_reward = float(np.mean(rewards))
        return aggregated_reward, local_best, rewards

    # ─────────────────────────────────────────────────────────────────────────
    # Update
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        selected_llm_name: str,
        scores: List[float],
        costs: List[float],
    ):
        aggregated_utility, local_best, rewards = self.compute_utility(scores)

        for s in scores:
            self._update_elite_pool(s)
        self.best_score = local_best

        self.n_pulls[selected_llm_name] += 1
        n = self.n_pulls[selected_llm_name]

        old_mean = self.mean_utility[selected_llm_name]
        self.mean_utility[selected_llm_name] = old_mean + (aggregated_utility - old_mean) / n

        self.iteration += 1

        print(f"[UCB Router] Updated {selected_llm_name}:")
        print(f"  - Aggregated Utility: {aggregated_utility:.4f}")
        print(f"  - Individual Rewards: {[f'{r:.4f}' for r in rewards]}")
        print(f"  - New Mean Utility: {self.mean_utility[selected_llm_name]:.4f}")
        print(f"  - Global Best Score: {self.best_score:.4f}")

        return rewards, rewards

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics & persistence
    # ─────────────────────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'iteration': self.iteration,
            'best_score': self.best_score,
            'n_pulls': self.n_pulls.copy(),
            'mean_utility': self.mean_utility.copy(),
            'llm_names': self.llm_names,
        }

    def save_checkpoint(self, path: str) -> None:
        import pickle
        state = {
            'n_pulls': self.n_pulls,
            'mean_utility': self.mean_utility,
            'iteration': self.iteration,
            'best_score': self.best_score,
            'active_model_id': self.active_model_id,
            'elite_pool': self.elite_pool,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_checkpoint(self, path: str) -> None:
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)

        for name in self.llm_names:
            if name in state['n_pulls']:
                self.n_pulls[name] = state['n_pulls'][name]
                self.mean_utility[name] = state['mean_utility'][name]

        self.iteration = state['iteration']
        self.best_score = state['best_score']
        self.active_model_id = state.get('active_model_id')
        if 'elite_pool' in state:
            self.elite_pool = state['elite_pool']
        print(f"[UCB Router] Restored from checkpoint at iteration {self.iteration}")
