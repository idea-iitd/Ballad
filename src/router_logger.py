"""
Logger for tracking router decisions, costs, utilities, and periodic diagnostics.

RouterLogger wraps the base Logger and adds:
  - per-step decision logging (log_router_decision)
  - rolling window diagnostics every N rounds (record_round → _log_diagnostics)
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class RouterLogger:
    """Logs UCB router decisions and emits periodic diagnostic reports."""

    def __init__(
        self,
        base_logger,
        router_enabled: bool = False,
        strategy: str = "ucb",
        router=None,
        diag_log_path: Optional[str] = None,
        diag_window_size: int = 4,
        resume: bool = False,
    ):
        """
        Args:
            base_logger:      Existing Logger instance.
            router_enabled:   Whether the router is active.
            strategy:         Bandit strategy name (for display).
            router:           The router instance (needed for diagnostics).
            diag_log_path:    Absolute path for the diagnostics text file.
            diag_window_size: How many arm-pull rounds between diagnostic reports.
            resume:           If True, append to existing diag file instead of clearing.
        """
        self.base_logger    = base_logger
        self.router_enabled = router_enabled
        self.strategy       = strategy
        self.router         = router

        # ── Per-step decision log ─────────────────────────────────────────
        self.router_data: Dict[str, List] = {
            'iterations':        [],
            'selected_llm':      [],
            'ucb_scores':        [],
            'mean_utilities':    [],
            'n_pulls':           [],
            'costs':             [],
            'rewards':           [],
            'utilities':         [],
            'best_score_history':[],
            'timestamp':         [],
        }

        # ── Rolling diagnostic window ─────────────────────────────────────
        self._diag_log_path    = diag_log_path
        self._diag_window_size = diag_window_size
        # resumed_iters = router.iteration if (resume and router is not None) else 0
        self._diag_window_num  = 0
        self._round_offset     = 0
        self._window_num_at_offset = self._diag_window_num
        self._diag_buffer: List[Dict] = []

        if not resume and diag_log_path and os.path.exists(diag_log_path):
            os.remove(diag_log_path)

        # Write (or append a resume marker to) the diagnostics header
        if router_enabled and diag_log_path and router is not None:
            self._write_diag_header(resume=resume)

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics header
    # ─────────────────────────────────────────────────────────────────────────
    def sync_from_router(self) -> None:
        """Re-sync window number and round offset from the router's current iteration.

        Must be called after router.load_checkpoint() so that resumed_iters
        reflects the restored state rather than the zero-initialized value that
        existed when RouterLogger was first constructed.
        """
        if self.router is None:
            return
        resumed_iters = self.router.iteration
        self._diag_window_num = resumed_iters // self._diag_window_size
        self._window_num_at_offset = self._diag_window_num
        self._round_offset = resumed_iters

    def _write_diag_header(self, resume: bool = False) -> None:
        """Write strategy configuration to the top of the diagnostics file.

        Called once at construction (fresh run) or once more when resuming
        (appends a RESUMED marker with the current timestamp so the file
        stays a coherent record of all runs that touched it).
        """
        if not self._diag_log_path or self.router is None:
            return

        router   = self.router
        strategy = getattr(router, 'strategy', self.strategy)
        SEP      = "=" * 72
        SEP2     = "-" * 72

        lines: List[str] = []

        if resume:
            lines.append(f"\n{SEP}")
            lines.append(
                f"RESUMED  strategy={strategy}  "
                f"time={time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            lines.append(f"{SEP}\n")
        else:
            lines.append(SEP)
            lines.append(
                f"ROUTER DIAGNOSTICS  strategy={strategy}  "
                f"arms={getattr(router, 'llm_names', [])}  "
                f"time={time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            lines.append(SEP2)

            # ── Shared params present on most routers ──────────────────────
            shared = []
            for attr, label in [
                ('beta',           'beta'),
                ('lam',            'lam'),
                ('tau',            'tau'),
                ('use_log_t',      'use_log_t'),
                ('T',              'total_steps'),
                ('min_pulls',      'min_pulls'),
                ('elite_pool_size','elite_pool_size'),
            ]:
                val = getattr(router, attr, None)
                if val is not None:
                    # Avoid printing the same logical param twice via alias
                    if not any(s.startswith(f'{label}=') for s in shared):
                        shared.append(f"{label}={val}")
            if shared:
                lines.append("Shared params   : " + "  ".join(shared))

            # ── Strategy-specific params ───────────────────────────────────
            if strategy == 'ballad':
                lines.append(
                    "Strategy params : "
                    f"n_bootstrap={getattr(router, 'n_bootstrap', '?')}  "
                    f"tau={getattr(router, 'tau', '?')}  "
                    f"lam={getattr(router, 'lam', '?')}"
                    # rng_seed is not stored as a scalar after __init__; omitted here
                )

            # ── Lineage params (shared across strategies that support it) ──
            if getattr(router, 'use_lineage', False):
                gamma_attr = (
                    'gamma' if hasattr(router, 'gamma') else
                    'gamma_lineage' if hasattr(router, 'gamma_lineage') else None
                )
                gamma_val = getattr(router, gamma_attr, '?') if gamma_attr else '?'
                lines.append(
                    "Lineage params  : "
                    f"use_lineage=True  "
                    f"alpha_lineage={getattr(router, 'alpha_lineage', '?')}  "
                    f"alpha_decay={getattr(router, 'alpha_decay', '?')}  "
                    f"gamma={gamma_val}  "
                    f"D_max={getattr(router, 'D_max', '?')}"
                )
            else:
                lines.append("Lineage params  : use_lineage=False")

            lines.append(SEP)
            lines.append("")   # blank line before first window

        with open(self._diag_log_path, 'a') as f:
            f.write("\n".join(lines) + "\n")

    def log_router_decision(
        self,
        iteration: int,
        selected_llm: str,
        ucb_scores: Dict[str, float],
        mean_utilities: Dict[str, float],
        n_pulls: Dict[str, int],
        cost: float,
        reward: float,
        utility: float,
        best_score: float,
    ) -> None:
        """Record a single router decision (called once per sample in the batch)."""
        if not self.router_enabled:
            return

        self.router_data['iterations'].append(iteration)
        self.router_data['selected_llm'].append(selected_llm)
        self.router_data['ucb_scores'].append(ucb_scores.copy())
        self.router_data['mean_utilities'].append(mean_utilities.copy())
        self.router_data['n_pulls'].append(n_pulls.copy())
        self.router_data['costs'].append(cost)
        self.router_data['rewards'].append(reward)
        self.router_data['utilities'].append(utility)
        self.router_data['best_score_history'].append(best_score)
        self.router_data['timestamp'].append(time.time())

    def save_router_logs(self, output_path: Optional[Path] = None) -> None:
        """Persist per-step decision log to JSON."""
        if not self.router_enabled:
            return

        if output_path is None:
            log_dir = Path(self.base_logger.results_dir) / f"router_{self.strategy}"
            log_dir.mkdir(parents=True, exist_ok=True)
            output_path = log_dir / f"router_logs_{self.base_logger.run_id}.json"

        with open(output_path, 'w') as f:
            json.dump(self.router_data, f, indent=2)

        print(f"[RouterLogger] Saved router logs to {output_path}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Summarise overall router performance across all logged decisions."""
        if not self.router_enabled or not self.router_data['iterations']:
            return {}

        selected_llms = self.router_data['selected_llm']
        llm_names     = sorted(set(selected_llms))
        costs         = self.router_data['costs']
        rewards       = self.router_data['rewards']
        utilities     = self.router_data['utilities']

        llm_stats = {}
        for llm in llm_names:
            idx          = [i for i, x in enumerate(selected_llms) if x == llm]
            llm_costs    = [costs[i]    for i in idx]
            llm_rewards  = [rewards[i]  for i in idx]
            llm_utilities= [utilities[i]for i in idx]
            llm_stats[llm] = {
                'selections':  len(idx),
                'total_cost':  sum(llm_costs),
                'avg_cost':    float(np.mean(llm_costs))    if llm_costs    else 0.0,
                'total_reward':sum(llm_rewards),
                'avg_reward':  float(np.mean(llm_rewards))  if llm_rewards  else 0.0,
                'avg_utility': float(np.mean(llm_utilities))if llm_utilities else 0.0,
                'improvements':sum(1 for r in llm_rewards if r > 0),
                'efficiency':  (sum(llm_rewards) / sum(llm_costs)
                                if sum(llm_costs) > 0 else 0.0),
            }

        positive_rewards = [r for r in rewards if r > 0]
        return {
            'total_iterations':  len(self.router_data['iterations']),
            'selection_counts':  {llm: selected_llms.count(llm) for llm in llm_names},
            'total_cost':        sum(costs),
            'avg_cost':          float(np.mean(costs))    if costs    else 0.0,
            'total_reward':      sum(rewards),
            'improvement_rate':  (len(positive_rewards) / len(rewards)
                                  if rewards else 0.0),
            'avg_utility':       float(np.mean(utilities)) if utilities else 0.0,
            'initial_best_score':self.router_data['best_score_history'][0],
            'final_best_score':  self.router_data['best_score_history'][-1],
            'llm_statistics':    llm_stats,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Rolling diagnostic window
    # ─────────────────────────────────────────────────────────────────────────

    def record_round(
        self,
        arm: str,
        rewards: List[float],
        scores: List[float],
        produced_best: bool,
        best_score_before: float,
        best_score_after: float,
    ) -> None:
        """Buffer one arm-pull round and flush diagnostics every window_size rounds.

        Args:
            arm:               Name of the LLM selected this round.
            rewards:           Per-sample rewards from router.update().
            scores:            Per-sample evaluation scores from this round.
            produced_best:     True if this round improved the global best score.
            best_score_before: Global best score at the start of this round.
            best_score_after:  Global best score after this round.
        """
        if not self.router_enabled or self.router is None or self._diag_log_path is None:
            return

        self._diag_buffer.append({
            'arm':               arm,
            'rewards':           list(rewards),
            'scores':            list(scores),
            'produced_best':     produced_best,
            'best_score_before': best_score_before,
            'best_score_after':  best_score_after,
        })

        if len(self._diag_buffer) >= self._diag_window_size:
            self._diag_window_num += 1
            self._log_diagnostics(self._diag_window_num)
            self._diag_buffer.clear()

    def _log_diagnostics(self, window_num: int) -> None:
        """Write a diagnostic summary for the current buffer to the diagnostics file.

        Logged metrics
        --------------
        Per-arm (window):
          selected            — how often the bandit chose this arm
          best wins           — times this arm produced the new best program
          avg improvement±std — mean max(0, score − baseline) and its std

        Global (window):
          score trajectory  — best score at start vs end of window
          improvement rate  — % samples where reward > 0
          selection entropy — 100% = uniform exploration, 0% = greedy

        Cumulative (since start):
          total pulls / mean utility / max utility per arm

        Strategy-specific state:
          ucb              — exploit + explore breakdown per arm
          ballad — bootstrap params and per-arm score breakdown
        """
        buf       = self._diag_buffer
        router    = self.router
        llm_names = router.llm_names
        strategy  = getattr(router, 'strategy', self.strategy)
        n_rounds  = len(buf)
        # round_start = (window_num - 1) * self._diag_window_size + 1 + self._round_offset
        round_start = self._round_offset + (window_num - self._window_num_at_offset - 1) * self._diag_window_size + 1
        round_end   = round_start + n_rounds - 1

        # ── 1. Arm selection counts ──────────────────────────────────────
        sel_counts = {n: 0 for n in llm_names}
        for e in buf:
            sel_counts[e['arm']] += 1

        # ── 2. Best-program wins per arm ─────────────────────────────────
        best_wins = {n: 0 for n in llm_names}
        for e in buf:
            if e['produced_best']:
                best_wins[e['arm']] += 1

        # ── 3. Per-arm reward stats in window ────────────────────────────
        arm_rewards = {n: [] for n in llm_names}
        for e in buf:
            arm_rewards[e['arm']].extend(e['rewards'])

        arm_reward_mean = {
            n: float(np.mean(arm_rewards[n])) if arm_rewards[n] else float('nan')
            for n in llm_names
        }
        arm_reward_std = {
            n: float(np.std(arm_rewards[n])) if len(arm_rewards[n]) > 1 else 0.0
            for n in llm_names
        }

        # ── 4. Selection entropy ─────────────────────────────────────────
        total_sel = sum(sel_counts.values())
        entropy = 0.0
        for c in sel_counts.values():
            if c > 0:
                p = c / total_sel
                entropy -= p * math.log(p)
        max_entropy = math.log(len(llm_names)) if len(llm_names) > 1 else 1.0
        entropy_pct = 100.0 * entropy / max_entropy if max_entropy > 0 else 0.0

        # ── 5. Score trajectory ──────────────────────────────────────────
        score_start = buf[0]['best_score_before']
        score_end   = buf[-1]['best_score_after']

        # ── 6. Improvement rate ──────────────────────────────────────────
        all_rewards = [r for e in buf for r in e['rewards']]
        improvement_rate = (
            sum(1 for r in all_rewards if r > 0) / len(all_rewards)
            if all_rewards else 0.0
        )

        # ── 7. Strategy-specific state ───────────────────────────────────
        col = max(len(n) for n in llm_names)
        strategy_lines = []

        if strategy == 'ucb':
            scores = router.compute_scores()
            for n in llm_names:
                np_ = router.n_pulls[n]
                t   = router.iteration
                explore = (
                    router.beta * math.sqrt(math.log(t) / np_)
                    if t > 0 and np_ > 0 else float('inf')
                )
                strategy_lines.append(
                    f"  {n:<{col}}  exploit={router.mean_utility[n]:+.4f}  "
                    f"explore={explore:.4f}  index={scores[n]:.4f}"
                )

        elif strategy == 'ballad':
            # ── Bootstrap params header ──────────────────────────────────
            t_now   = getattr(router, 't', 0)
            T_tot   = getattr(router, 'T', 1)
            nb      = getattr(router, 'n_bootstrap', '?')
            tau_val = getattr(router, 'tau', float('nan'))
            lm_val  = getattr(router, 'lam', float('nan'))
            strategy_lines.append(
                f"  t={t_now}/{T_tot}  n_bootstrap={nb}  "
                f"tau={tau_val}  lam={lm_val}"
            )

            # ── Per-arm score breakdown ──────────────────────────────────
            scores = router.compute_scores()
            use_lin = getattr(router, 'use_lineage', False)
            alpha_lineage = getattr(router, 'alpha_lineage', float('nan'))
            alpha_decay   = getattr(router, 'alpha_decay', False)
            # Compute decayed alpha_t as the router would
            alpha_t = alpha_lineage
            if alpha_decay and T_tot > 0:
                alpha_t = alpha_lineage * max(0.0, 1.0 - (t_now / T_tot))

            if use_lin:
                strategy_lines.append(
                    f"  Lineage ACTIVE  alpha_lineage={alpha_lineage}  "
                    f"alpha_t={alpha_t:.4f}  gamma={getattr(router, 'gamma', '?')}  "
                    f"D_max={getattr(router, 'D_max', '?')}"
                )

            strategy_lines.append("")
            if use_lin:
                hdr_lin = (
                    f"  {'Arm':<{col}}  {'n_pulls':>8}  {'q_tau':>8}  "
                    f"{'max_r':>8}  {'base':>8}  {'L̄(×αt)':>10}  "
                    f"{'bonus':>8}  {'score':>8}"
                )
            else:
                hdr_lin = (
                    f"  {'Arm':<{col}}  {'n_pulls':>8}  {'q_tau':>8}  "
                    f"{'max_r':>8}  {'base':>8}  {'bonus':>8}  {'score':>8}"
                )
            strategy_lines.append(hdr_lin)
            strategy_lines.append("  " + "-" * (col + (72 if use_lin else 62)))

            beta  = getattr(router, 'beta', float('nan'))
            T_log = math.log(T_tot) if T_tot > 1 else 1.0

            for n in llm_names:
                obs  = getattr(router, '_bqm_obs', {}).get(n, [])
                np_  = router.n_pulls[n]
                if obs and np_ > 0:
                    obs_arr  = np.asarray(obs, dtype=float)
                    max_r    = float(obs_arr.max())
                    # Re-derive q_tau from score, bonus, and lineage
                    # score = base + lineage_term + bonus
                    # base  = (1-lam)*q_tau + lam*max_r
                    # Recover q_tau from: base = score - lineage - bonus
                    bonus     = beta * math.sqrt(T_log / np_) if np_ > 0 else float('inf')
                    lin_score = router._lineage_score(n) if use_lin else 0.0
                    lin_term  = alpha_t * lin_score if use_lin else 0.0
                    base      = scores[n] - lin_term - bonus
                    # q_tau = (base - lam*max_r) / (1 - lam)  when lam < 1
                    lam_v     = getattr(router, 'lam', 0.3)
                    if abs(1.0 - lam_v) > 1e-9:
                        q_tau = (base - lam_v * max_r) / (1.0 - lam_v)
                    else:
                        q_tau = max_r  # lam==1 → pure running max
                else:
                    max_r = bonus = base = lin_term = q_tau = float('nan')
                    lin_score = 0.0

                if use_lin:
                    strategy_lines.append(
                        f"  {n:<{col}}  {np_:>8}  {q_tau:>8.4f}  "
                        f"{max_r:>8.4f}  {base:>8.4f}  "
                        f"{lin_term:>10.4f}  {bonus:>8.4f}  {scores[n]:>8.4f}"
                    )
                else:
                    strategy_lines.append(
                        f"  {n:<{col}}  {np_:>8}  {q_tau:>8.4f}  "
                        f"{max_r:>8.4f}  {base:>8.4f}  "
                        f"{bonus:>8.4f}  {scores[n]:>8.4f}"
                    )

        # ── Write ────────────────────────────────────────────────────────
        SEP  = "=" * 72
        SEP2 = "-" * 72
        w    = col + 2

        with open(self._diag_log_path, 'a') as f:
            f.write(f"\n{SEP}\n")
            f.write(
                f"DIAGNOSTIC WINDOW #{window_num}  "
                f"rounds {round_start}–{round_end}  |  "
                f"strategy={strategy}  |  "
                f"total_iterations={router.iteration}\n"
            )
            f.write(f"{SEP}\n\n")

            f.write(
                f"Score trajectory : {score_start:.4f} → {score_end:.4f}  "
                f"(Δ={score_end - score_start:+.4f})\n"
            )
            f.write(
                f"Improvement rate : {improvement_rate * 100:.1f}%  "
                f"(samples with reward > 0 out of {len(all_rewards)})\n"
            )
            f.write(
                f"Selection entropy: {entropy:.3f} / {max_entropy:.3f}  "
                f"({entropy_pct:.1f}% of max — 100%=uniform, 0%=greedy)\n"
            )
            f.write("\n")

            # Per-arm window table
            # "Avg improvement" = mean reward = mean max(0, score - baseline),
            # i.e. the average improvement over the running baseline per pull.
            f.write("Per-arm (this window)  [improvement = mean max(0, score − baseline)]\n")
            hdr = (
                f"{'Arm':<{w}}  {'Selected':>9}  "
                f"{'Best wins':>10}  {'Avg improvement':>16}  {'Improvement std':>16}\n"
            )
            f.write(hdr)
            f.write(SEP2 + "\n")
            for n in llm_names:
                f.write(
                    f"{n:<{w}}  {sel_counts[n]:>9}  "
                    f"{best_wins[n]:>10}  "
                    f"{arm_reward_mean[n]:>16.4f}  "
                    f"{arm_reward_std[n]:>16.4f}\n"
                )

            # Cumulative table
            f.write(f"\nCumulative (all {router.iteration} iterations):\n")
            mean_util_map = getattr(router, 'mean_utility', {})
            bqm_obs_map   = getattr(router, '_bqm_obs', None)
            show_max_raw  = bqm_obs_map is not None
            hdr2 = (
                f"{'Arm':<{w}}  {'Total pulls':>12}  {'Mean utility':>13}"
                + (f"  {'Max raw score':>14}" if show_max_raw else "")
                + "\n"
            )
            f.write(hdr2)
            f.write(SEP2 + "\n")
            for n in llm_names:
                mu = mean_util_map.get(n, float('nan')) if mean_util_map else float('nan')
                row = f"{n:<{w}}  {router.n_pulls[n]:>12}  {mu:>13.4f}"
                if show_max_raw:
                    obs = bqm_obs_map.get(n)
                    mx  = float(np.max(obs)) if obs else float('nan')
                    row += f"  {mx:>14.4f}"
                f.write(row + "\n")

            if strategy_lines:
                f.write(f"\nStrategy state ({strategy}):\n")
                f.write(SEP2 + "\n")
                for line in strategy_lines:
                    f.write(line + "\n")

            f.write("\n")

        print(
            f"[RouterLogger] Diagnostic window #{window_num} written to "
            f"{self._diag_log_path}"
        )