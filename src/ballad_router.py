"""Ballad Router.

Combines a bootstrapped Extreme Value estimator for the base score with the
BFS Lineage credit and a fixed-horizon UCB exploration bonus.

Scoring (per arm a at step t):
    boots     = n_bootstrap bootstrap resamplings of obs[a], each of size n
    q_tau     = quantile(tau, max of each resample)          # tail-mass estimator
    max_r     = max(obs[a])                                    # running maximum
    base      = (1 - lam) * q_tau  +  lam * max_r             # hybrid extreme-value
    alpha_t   = alpha_lineage * max(0, 1 - t / T)             # decaying lineage weight
    lineage   = alpha_t * L̄(a)                                 # evolutionary guide
    bonus     = beta * sqrt(log(T) / n_pulls[a])              # BAI exploration driver
    score[a]  = base + lineage + bonus

Lineage: BFS over new_func.parent_funcs up to D_max; each ancestor arm is
credited with gamma^d * r where r = max(0, s - b_t) and b_t is the
elite-pool median. L̄(a) = lineage_sum[a] / log(max(e, n_pulls[a])).
"""

import heapq
import math
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from omegaconf import DictConfig
from hydra.utils import instantiate
from collections import deque

class BalladRouter:
    """Bootstrapped-QoMax router with BFS Lineage credit and UCB bonus."""

    def __init__(
        self,
        llm_clients: Dict[str, Any],
        # ── UCB / budget ──────────────────────────────────────────────────
        beta: float = 1.0,
        initial_best_score: float = -np.inf,
        total_steps: int = 1000,
        min_pulls: int = 3,
        samples_per_prompt: int = 1,
        # ── Bootstrap extreme-value estimator ─────────────────────────────
        n_bootstrap: int = 200,        # number of bootstrap resamples
        tau: float = 0.9,            # upper quantile of bootstrap maxima
        lam: float = 0.3,              # blend weight on running max vs q_tau
        rng_seed: Optional[int] = None,
        # ── Elite baseline pool ───────────────────────────────────────────
        elite_pool_size: int = 5,
        # ── Lineage BFS credit ────────────────────────────────────────────
        gamma: float = 0.9,
        D_max: int = 4,
        alpha_lineage: float = 1.0,
        use_lineage: bool = False,
        alpha_decay: bool = False,
        window_size: int = 25,
        use_log_t: bool = False,   # True → log(t) classic UCB1; False → log(T) fixed-horizon
    ):
        self.llm_clients = llm_clients
        self.llm_names   = list(llm_clients.keys())

        # UCB / budget
        self.beta          = beta
        self.best_score    = initial_best_score
        self.iteration     = 0
        self.active_model_id: Optional[str] = None

        # Per-arm stats
        self.n_pulls      = {n: 0   for n in self.llm_names}
        self.mean_utility = {n: 0.0 for n in self.llm_names}
        self._bqm_obs     = {n: []  for n in self.llm_names}  # raw per-sample scores

        # Elite-median baseline (b_t)
        self.elite_pool_size = elite_pool_size
        self.elite_pool: List[float] = []   # min-heap of top-E scores

        # Schedule counters (no k-annealing; kept for bonus / decay maths)
        self.t  = 0                                      # steps taken so far
        self.T  = total_steps                            # fixed horizon
        self.min_pulls = min_pulls
        self._samples_per_prompt = max(1, samples_per_prompt)

        # Bootstrap extreme-value estimator
        self.n_bootstrap = n_bootstrap
        self.tau         = tau
        self.lam         = lam
        self.rng         = np.random.default_rng(rng_seed)

        # Lineage BFS
        self.gamma         = gamma
        self.D_max         = D_max
        self.alpha_lineage = alpha_lineage
        self.alpha_decay   = alpha_decay
        self.use_lineage   = use_lineage
        self.window_size   = window_size
        self.use_log_t     = use_log_t
        # self._lineage_sum    = {n: 0.0 for n in self.llm_names}
        # self._lineage_weight = {n: 0.0 for n in self.llm_names}
        self._lineage_history = {n: deque() for n in self.llm_names}

        print(
            f"[UCBHybridQoMax] Arms={self.llm_names}  beta={beta}  "
            f"T={total_steps}  min_pulls={min_pulls}\n"
            f"                 n_bootstrap={n_bootstrap}  tau={tau}  lam={lam}  "
            f"gamma={gamma}  D_max={D_max}  alpha_lineage={alpha_lineage}  "
            f"use_lineage={use_lineage}"
        )

    # ── Baseline ───────────────────────────────────────────────────────────

    def _get_baseline(self) -> float:
        if len(self.elite_pool) >= self.elite_pool_size:
            return float(np.median(self.elite_pool))
        return self.best_score

    def _update_elite_pool(self, score: float) -> None:
        if not np.isfinite(score):
            return
        if len(self.elite_pool) < self.elite_pool_size:
            heapq.heappush(self.elite_pool, score)
        elif score > self.elite_pool[0]:
            heapq.heapreplace(self.elite_pool, score)

    # ── Lineage ────────────────────────────────────────────────────────────

    def _lineage_score(self, name: str) -> float:
        """Windowed intensity-based fertility score."""
        history = self._lineage_history[name]
        if not history:
            return 0.0

        # Prune expired credits (older than window_size pulls)
        cutoff = self.iteration - self.window_size
        while history and history[0][0] <= cutoff:
            history.popleft()

        # Sum remaining valid credits
        valid_sum = sum(credit for timestamp, credit in history)
        
        n = self.n_pulls[name]
        if n == 0:
            return 0.0
            
        return valid_sum / math.log(max(math.e, n))

    def _bfs_lineage_attribution(self, new_func, r: float) -> None:
        """Intensity-based BFS attribution with Sliding Window support."""
        if r <= 0.0 or not getattr(new_func, 'parent_funcs', None):
            return

        current = list(new_func.parent_funcs)
        depth   = 1

        while current and depth <= self.D_max:
            disc = self.gamma ** depth
            nxt  = []
            for f in current:
                arm = getattr(f, 'arm', None)
                if arm and arm in self._lineage_history:
                    # Append tuple of (current_iteration, discounted_reward)
                    self._lineage_history[arm].append((self.iteration, disc * r))
                for p in getattr(f, 'parent_funcs', []):
                    nxt.append(p)
            current = nxt
            depth  += 1

    def _print_lineage(self, new_func) -> None:
        """Print the ancestry DAG of new_func up to depth D_max."""
        func_id    = getattr(new_func, 'id',    'unknown')[:8]
        func_arm   = getattr(new_func, 'arm',   'None')
        func_score = getattr(new_func, 'score', float('nan'))
        print(f"  [Lineage] {'─'*50}")
        print(f"  [Lineage] Child  depth=0  id={func_id}  arm={func_arm}  score={func_score:.4f}")

        visited: set = {getattr(new_func, 'id', None)}
        current_level = []
        for pf in getattr(new_func, 'parent_funcs', []):
            pid = getattr(pf, 'id', None)
            if pid and pid not in visited:
                visited.add(pid)
                current_level.append(pf)

        depth = 1
        while current_level and depth <= self.D_max:
            p_prime_d  = len(current_level)
            next_level = []
            entries    = []
            for func in current_level:
                fid   = getattr(func, 'id',    'unknown')[:8]
                arm   = getattr(func, 'arm',   'None')
                score = getattr(func, 'score', float('nan'))
                entries.append(f"id={fid}  arm={arm}  score={score:.4f}")
                for parent in getattr(func, 'parent_funcs', []):
                    pid = getattr(parent, 'id', None)
                    if pid and pid not in visited:
                        visited.add(pid)
                        next_level.append(parent)
            print(f"  [Lineage] depth={depth}  p'={p_prime_d}  →  " + " | ".join(entries))
            current_level = next_level
            depth        += 1
        print(f"  [Lineage] {'─'*50}")

    # ── Scoring ────────────────────────────────────────────────────────────

    def compute_scores(self) -> Dict[str, float]:
        """Return the UCB-Hybrid-QoMax score for every arm (no side-effects).

        Bootstrap Extreme Value estimator:
            boots = n_bootstrap resamples of obs[m] (size n, with replacement)
            q_tau = quantile(tau, max of each resample)
            max_r = running maximum of obs[m]
            base  = (1 - lam) * q_tau  +  lam * max_r

        Lineage (optional, time-decayed):
            alpha_t = alpha_lineage * max(0, 1 - t / T)
            lineage = alpha_t * L̄(m)

        Exploration bonus:
            bonus   = beta * sqrt(log(T) / n_pulls[m])   # use_log_t=False (default)
            bonus   = beta * sqrt(log(t) / n_pulls[m])   # use_log_t=True
        """
        scores: Dict[str, float] = {}

        for m in self.llm_names:
            obs_arr = np.asarray(self._bqm_obs[m], dtype=float)
            n = len(obs_arr)

            # 1. Warm-up fallback
            if n < self.min_pulls:
                scores[m] = float('inf')
                continue

            # 2. Bootstrapped Extreme Value Estimator
            boots = self.rng.choice(obs_arr, size=(self.n_bootstrap, n), replace=True)
            q_tau = float(np.quantile(boots.max(axis=1), self.tau))
            max_r = float(np.max(obs_arr))

            # Base robust score
            score = (1.0 - self.lam) * q_tau + self.lam * max_r

            # 3. High-Density Lineage (The Evolutionary Guide)
            if self.use_lineage:
                score += self.alpha_lineage * self._lineage_score(m)

            # 4. Fixed-Horizon Exploration (The BAI Driver)
            log_arg = max(1, self.t) if self.use_log_t else self.T
            bonus   = self.beta * np.sqrt(np.log(log_arg) / self.n_pulls[m])
            score += bonus

            scores[m] = score

        return scores

    # ── Selection ──────────────────────────────────────────────────────────

    def _select_arm(self) -> str:
        self.t += 1

        # Round-robin warm-up: pull each arm until it has min_pulls observations
        for name in self.llm_names:
            if self.n_pulls[name] < self.min_pulls:
                return name

        scores   = self.compute_scores()
        best_arm = max(scores, key=lambda k: scores[k])
        return best_arm

    # ── HF model lifecycle ─────────────────────────────────────────────────

    def _get_or_instantiate_client(self, name: str) -> Any:
        is_config = isinstance(self.llm_clients[name], DictConfig)
        if is_config:
            if self.active_model_id and self.active_model_id != name:
                active_client = self.llm_clients[self.active_model_id]
                if hasattr(active_client, 'cleanup'):
                    saved_config = active_client._config
                    active_client.cleanup()
                    self.llm_clients[self.active_model_id] = saved_config
            print(f"[UCBHybridQoMax] Instantiating {name}...")
            self.llm_clients[name] = instantiate(
                self.llm_clients[name], _config=self.llm_clients[name])
        return self.llm_clients[name]

    def select_llm(self) -> Tuple[str, Any]:
        print(f"\n[UCBHybridQoMax] iter={self.iteration}  pulls={self.n_pulls}")
        selected = self._select_arm()
        print(f"[UCBHybridQoMax] Selected {selected}  t={self.t}")
        client = self._get_or_instantiate_client(selected)
        self.active_model_id = selected
        return selected, client

    # ── Update ─────────────────────────────────────────────────────────────

    def update_with_lineage(
        self,
        selected_llm_name: str,
        new_functions: List[Any],
        scores: List[float],
        costs: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Update arm statistics and propagate lineage credit via BFS.

        Propagated signal: r_i = max(0, s_i - b_t)  (baseline-clipped reward).

        Returns
        -------
        utilities : per-sample utility values (equal to rewards; no cost penalty)
        rewards   : per-sample baseline-clipped rewards
        """
        baseline   = self._get_baseline()
        utilities: List[float] = []
        rewards:   List[float] = []
        local_best = self.best_score

        for s in scores:
            r_clip = max(0.0, s - baseline) if np.isfinite(s) else 0.0
            rewards.append(r_clip)
            utilities.append(r_clip)
            if np.isfinite(s):
                local_best = max(local_best, s)

        for s in scores:
            self._update_elite_pool(s)
        self.best_score = local_best

        # Direct pull update
        self.n_pulls[selected_llm_name] += 1
        n   = self.n_pulls[selected_llm_name]
        agg = float(np.mean(utilities)) if utilities else 0.0
        old = self.mean_utility[selected_llm_name]
        self.mean_utility[selected_llm_name] = old + (agg - old) / n

        # Extend observation buffer with finite raw scores
        finite_scores = [s for s in scores if np.isfinite(s)]
        self._bqm_obs[selected_llm_name].extend(finite_scores)

        # Lineage BFS attribution (per-sample, baseline-clipped)
        if self.use_lineage and new_functions is not None:
            for nf, s in zip(new_functions, scores):
                if nf is None or not np.isfinite(s):
                    continue
                r_i = max(0.0, float(s) - float(baseline))
                if r_i > 0.0:
                    self._bfs_lineage_attribution(nf, r_i)

        self.iteration += 1

        print(
            f"[UCBHybridQoMax] Updated {selected_llm_name}  "
            f"agg_util={agg:.4f}  b_t={baseline:.4f}  best={self.best_score:.4f}"
        )
        print(
            f"[UCBHybridQoMax] L̄: "
            f"{ {n: round(self._lineage_score(n), 4) for n in self.llm_names} }"
        )

        return utilities, rewards

    # ── Diagnostics / persistence ──────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            'iteration':      self.iteration,
            'best_score':     self.best_score,
            'n_pulls':        self.n_pulls.copy(),
            'mean_utility':   self.mean_utility.copy(),
            'lineage_score':  {n: self._lineage_score(n) for n in self.llm_names},
            'llm_names':      self.llm_names,
            't':              self.t,
            'T':              self.T,
        }
        return stats

    def save_checkpoint(self, path: str) -> None:
        state = {
            'n_pulls':          self.n_pulls,
            'mean_utility':     self.mean_utility,
            'iteration':        self.iteration,
            'best_score':       self.best_score,
            'active_model_id':  self.active_model_id,
            'elite_pool':       self.elite_pool,
            '_bqm_obs':         self._bqm_obs,
            't':                self.t,
            '_lineage_history': self._lineage_history
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"[UCBHybridQoMax] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        for name in self.llm_names:
            if name in state.get('n_pulls', {}):
                self.n_pulls[name]      = state['n_pulls'][name]
                self.mean_utility[name] = state['mean_utility'][name]
            for attr in ('_bqm_obs', '_lineage_history'):
                saved = state.get(attr, {})
                if name in saved:
                    getattr(self, attr)[name] = saved[name]
        self.iteration       = state.get('iteration', 0)
        self.best_score      = state.get('best_score', self.best_score)
        self.active_model_id = state.get('active_model_id')
        self.t               = state.get('t', 0)
        self.elite_pool      = state.get('elite_pool', self.elite_pool)
        print(f"[UCBHybridQoMax] Restored at iteration {self.iteration}")