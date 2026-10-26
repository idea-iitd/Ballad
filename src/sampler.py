# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List
from src import programs_database
from src import evaluator
from src.logger import Logger
from src.ucb_router import UCBRouter
from src.ballad_router import BalladRouter
import time
import re
import pickle
import asyncio
import os
import random
import black
import ast

class LLM:
  def __init__(self, llm_client, samples_per_prompt: int):
    self.llm_client = llm_client
    self.samples_per_prompt = samples_per_prompt

  def draw_samples(self, prompt: str):
    return self.llm_client._draw_samples(prompt, self.samples_per_prompt)


def clean_code(sample: str) -> str:
    # Normalize/remove code fences (handles leading ```python and trailing ``` robustly)
    sample = re.sub(r'^\s*```(?:python)?\s*', '', sample, flags=re.IGNORECASE)
    sample = re.sub(r'\s*```\s*$', '', sample)

    # Remove any remaining inline fences (safety)
    sample = sample.replace('```', '')

    # Remove unwanted tokens or patterns
    sample = re.sub(r'\n*\*\*Explanation:.*', '', sample, flags=re.DOTALL)
    if 'Key improvements' in sample:
        sample = re.sub(r'\n*Key improvements.*', '', sample, flags=re.DOTALL)
    sample = re.sub(r'\n*\*\*Improvements.*', '', sample, flags=re.DOTALL)

    # Find the last function definition and then truncate the text at the end of its last `return` line.
    func_matches = list(re.finditer(r'def\s+[\w_]+\s*\(.*?\)\s*:', sample))
    if func_matches:
        last_func_start = func_matches[-1].start()

        # Find all 'return' occurrences after the last function start and pick the final one.
        return_matches = list(re.finditer(r'\breturn\b[^\n]*', sample[last_func_start:]))
        if return_matches:
            last_return = return_matches[-1]
            end_index = last_func_start + last_return.end()
            # Keep everything up to the end of the return line
            sample = sample[:end_index]
        else:
            # No return found in the last function: keep from the last function start to file end
            sample = sample[last_func_start:]

    # Final cleanup: strip any leftover multi-backtick blocks or trailing whitespace
    sample = re.sub(r'`{3,}.*$', '', sample, flags=re.DOTALL)
    sample = sample.rstrip() + "\n"

    # Keep existing behavior: remove anything before imports (assumes strip_before_import is defined)
    sample = strip_before_import(sample)
    return sample

def strip_before_import(text: str) -> str:
  """Remove everything before the first 'import' in a program string."""
  lines = text.splitlines()
  for i, line in enumerate(lines):
      if line.strip().startswith("import") or line.strip().startswith("from"):
          return "\n".join(lines[i:])
  return text  # return unchanged if no import found


def _extract_body_from_full_function(code: str) -> str:
    """
    Parses a string containing a single function and returns its body,
    preserving the original indentation and formatting.
    """
    try:
        tree = ast.parse(code.strip())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Ensure the function has a body
                if not node.body:
                    return ""

                # Get the original source code lines
                source_lines = code.strip().splitlines()
                
                # Find the start line of the first statement in the body
                # and the end line of the last statement.
                # AST line numbers are 1-based; list indices are 0-based.
                start_line_index = node.body[0].lineno - 1
                end_line_index = node.body[-1].end_lineno

                # Extract the slice of lines from the original source code
                body_lines = source_lines[start_line_index:end_line_index]
                
                return '\n'.join(body_lines)

    except Exception:
        # If parsing fails, return the original code
        return code
    
    # If no function definition is found, return the original code
    return code

def _process_llm_output(content: str) -> str:
    """
    Isolates the last function from the LLM's full output and returns its body.
    This handles cases where the prompt is copied before the generation.
    """
    # Use regex to find the start of the last function definition in the content
    matches = list(re.finditer(r'^\s*def\s+', content, re.MULTILINE))
    if matches:
        # If a function definition is found, slice the content from its start
        last_def_start = matches[-1].start()
        generation_only = content[last_def_start:]
    else:
        # If no 'def' is found, assume the entire content is the new generation
        generation_only = content

    # Now, extract the body from the isolated function code
    return _extract_body_from_full_function(generation_only)
   
class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
      best_score_list: List,
      best_score: float,
      best_program:Any,
      best_eval_time: float,
      best_func_timeouts: Any,
      best_function_id: str,
      problem_dir: Any, 
      priority_file: Any,
      llm_clients: Any,  # Changed from llm_client to llm_clients (dict)
      model_name: str,
      logger: Logger,
      log_dir_name:str,
      router_config: dict = None,  # New parameter for router configuration
      resume: bool = False 
  ) -> None:
    
    self._database = database
    self._evaluators = evaluators
    self.samples_per_prompt = samples_per_prompt
    # Initialize router if multiple LLMs provided
    if isinstance(llm_clients, dict) and len(llm_clients) > 1:
      self.use_router = True
      strategy = (router_config or {}).get('strategy', 'ucb')

      if strategy == 'ballad':
        self.router = BalladRouter(
            llm_clients=llm_clients,
            beta=router_config.get('beta', 1.0),
            initial_best_score=best_score,
            total_steps=router_config.get('total_steps', 1000),
            min_pulls=router_config.get('min_pulls', 3),
            samples_per_prompt=samples_per_prompt,
            n_bootstrap=router_config.get('n_bootstrap', 200),
            tau=router_config.get('tau', 0.9),
            lam=router_config.get('lam', 0.3),
            rng_seed=router_config.get('rng_seed', None),
            window_size=router_config.get('window_size', 25),
            elite_pool_size=(router_config.get('elite_pool_size') or 10),
            gamma=router_config.get('gamma_lineage', 0.9),
            D_max=router_config.get('D_max', 4),
            alpha_lineage=router_config.get('alpha_lineage', 1.0),
            use_lineage=router_config.get('use_lineage', False),
            alpha_decay=router_config.get('alpha_decay', False),
            use_log_t=router_config.get('use_log_t', False),
        )
        print(f"[Sampler] Using Ballad Router with {len(llm_clients)} LLMs")
      else:
        self.router = UCBRouter(
            llm_clients=llm_clients,
            beta=router_config.get('beta', 1.0),
            initial_best_score=best_score,
            elite_pool_size=router_config.get('elite_pool_size', 50),
        )

    else:
      self.use_router = False
      # Single LLM case (backward compatibility)
      if isinstance(llm_clients, dict):
        llm_client = list(llm_clients.values())[0]
      else:
        llm_client = llm_clients
      self._llm = LLM(llm_client, samples_per_prompt)
      print(f"[Sampler] Using single LLM (no router)")
    
    # from priority_v0
    self.best_score_list = best_score_list
    self.best_score = best_score 
    self.best_program = best_program
    self.best_eval_time = best_eval_time
    self.best_function_id = best_function_id
    self.best_func_timeouts = best_func_timeouts
    self.best_func_model = None #init program is not generated by any llm
    self.priority_file = priority_file
    self.logger = logger

    # Loop counters as instance variables so they survive checkpointing
    self._num_llm_calls = 0          
    self._count = 0                  

    # Checkpoint callback — set externally by funsearch.py after construction
    self.checkpoint_callback = None

    logs_dir = os.path.join(problem_dir, log_dir_name, f"{model_name}")
    os.makedirs(logs_dir, exist_ok=True)
    self.log_path = os.path.join(logs_dir, f"best_call{self.logger.run_id}.txt")

    # ── RouterLogger (needs logs_dir, so constructed here) ───────────────
    if self.use_router:
      from src.router_logger import RouterLogger
      _diag_dir = os.path.join(logs_dir, "diagnostics")
      os.makedirs(_diag_dir, exist_ok=True)
      _diag_log_path = os.path.join(
          _diag_dir, f"router_diagnostics_{self.logger.run_id}.txt"
      )
      self.router_logger = RouterLogger(
          logger,
          router_enabled=True,
          strategy=router_config.get('strategy', 'ucb'),
          router=self.router,
          diag_log_path=_diag_log_path,
          resume=resume,
      )

    # Load the threshold based on router or single LLM
    if self.use_router:
      # Use the max num_calls across all LLMs or a default
      self.thresh = max(client.num_calls for client in llm_clients.values())
    else:
      self.thresh = llm_client.num_calls
    
    if not resume:
      if os.path.exists(self.log_path):
        os.remove(self.log_path)

      with open(self.log_path, 'w') as f:
        f.write("Call 0:\n")
        f.write(f"Original Function Name: {self.best_program[0]}\n")
        f.write("Original Function Code:\n")

        try:
            formatted_code = black.format_str(self.best_program[1], mode=black.Mode())
        except Exception:
            formatted_code = self.best_program[1]  

        f.write(f"{formatted_code.strip()}\n")
        f.write(f"Original Function Score:{self.best_score}\n")
        f.write("Parent IDs: None\n")
        f.write(f"Original Function ID: {best_function_id}\n")
        f.write("=" * 40 + "\n\n")
      
      # log trivial program info
      self.logger.log(0, "", [], [], 0, "", "", "", 0, [], 0.0, 0.0, [],\
                      self.best_program[0], self.best_program[1], self.best_score_list, self.best_eval_time, self.best_function_id, self.best_func_timeouts, '', self.best_func_model)
    else:
      # Resume: log file already exists — we will append to it in sample()
      print(f"[Sampler] Resuming — skipping Call 0 init, appending to existing log.")

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""

    if not self.use_router:
      selected_llm_name = self._llm.llm_client.model_name 
      current_llm = self._llm

    _consecutive_prompt_failures = 0
    _MAX_CONSECUTIVE_PROMPT_FAILURES = 10

    while self._num_llm_calls < self.thresh + 1:
      try:
        prompt = self._database.get_prompt()
        _consecutive_prompt_failures = 0
      except Exception as _prompt_exc:
        _consecutive_prompt_failures += 1
        print(f'[Sampler] get_prompt failed ({_consecutive_prompt_failures}): {_prompt_exc}')
        if _consecutive_prompt_failures >= _MAX_CONSECUTIVE_PROMPT_FAILURES:
          raise RuntimeError(
              f'get_prompt failed {_MAX_CONSECUTIVE_PROMPT_FAILURES} times in a row — '
              'database may contain an unparseable function. '
              'Check programs_database._generate_prompt for details.'
          ) from _prompt_exc
        continue

      parent_ids = prompt.parent_ids
      parent_scores = prompt.parent_scores
      
      for parent_id, parent_score in zip(parent_ids, parent_scores):
        print(f"Parent ID: {parent_id}, Score: {parent_score}")

      # Select LLM using router or use single LLM
      if self.use_router:
        selected_llm_name, selected_llm_client = self.router.select_llm()
        current_llm = LLM(selected_llm_client, self.samples_per_prompt)
      
      print(f"============= Selected LLM: {selected_llm_name} =============")
      # Draw samples from selected LLM
      results = current_llm.draw_samples(prompt.code)
      
      # Storage for router update
      batch_scores = []
      batch_costs = []
      batch_new_functions = []  # Function objects for lineage BFS
      best_score_before_round = self.best_score
      
      for i, response in enumerate(results):
        sample            = response["text"]
        prompt_token_cnt  = response["input_tokens"]
        sample_token_cnt  = response["output_tokens"]
        sample_gen_time   = response["generation_time"]
        
        # You can now also access the cost data directly
        in_cost           = response["input_cost"]
        out_cost          = response["output_cost"]
        total_cost        = response["total_cost"]
        
        print(f"Prompt tokens: {prompt_token_cnt}")
        print(f"Sample tokens: {sample_token_cnt}")
        print(f"Sample generation time: {sample_gen_time:.2f} seconds")
        print(f"Cost: ${total_cost} | Tokens: {prompt_token_cnt + sample_token_cnt}")
       
        if sample is None:
          # If sample failed, record as worst case for router
          batch_scores.append(-np.inf)
          batch_costs.append(0)
          batch_new_functions.append(None)  # keep list aligned for lineage BFS
          continue

        print(f"before cleaning: \n {sample}")  
        proc_sample = clean_code(sample)
        print(f"after cleaning: \n {proc_sample}")  
        # exit(0)
        with open(self.priority_file, 'w') as file:
          file.write(proc_sample)
          file.flush()

        with open(self.priority_file, 'r') as file:
          content = file.read()

        function_body = _process_llm_output(content)

        chosen_evaluator = np.random.choice(self._evaluators)
        individual_scores, curr_avg_score, func_name, func_code, avg_eval_time, function_id, timed_out_list, new_function = chosen_evaluator.analyse(
            function_body, prompt.island_id, prompt.version_generated,
            prompt.parent_ids,
            arm=selected_llm_name if self.use_router else None,
            parent_funcs=prompt.parent_funcs)

        # Record score and cost for router
        batch_scores.append(curr_avg_score)
        batch_new_functions.append(new_function)  # collect for lineage BFS      
        batch_costs.append(total_cost)
        
        # Update best score and program if improved
        if curr_avg_score >= self.best_score:
          self.best_score_list = individual_scores
          self.best_score = curr_avg_score
          self.best_program = (func_name, func_code)
          self.best_eval_time = avg_eval_time
          self.best_function_id = function_id
          self.best_func_timeouts = timed_out_list
          self.best_func_model = selected_llm_name
          print("Best Score Updated:", self.best_score)
          print("Best Func Name:", func_name)
          print("Best Func Code:", func_code)
          print("Best Function ID:", function_id)
          print("Best Func Timeouts:", timed_out_list)

        # Textual log for keeping track (debugging purposes)
        with open(self.log_path, 'a') as f:
          call_num = self._num_llm_calls + i + 1
          best_func_name, best_func_code = self.best_program[0], self.best_program[1]
          score = self.best_score

          f.write(f"Call {call_num}:\n")
          if self.use_router:
            f.write(f"Best LLM: {self.best_func_model}\n")
          f.write(f"Best Function Name: {best_func_name}\n")
          f.write("Best Function Code:\n")

          try:
            formatted_code = black.format_str(best_func_code, mode=black.Mode())
          except Exception:
            formatted_code = best_func_code  

          f.write(f"{formatted_code.strip()}\n")
          f.write(f"Best Score: {score}\n")
          f.write(f"Best Function ID: {self.best_function_id}\n\n")
          f.write("." * 40 + "\n\n")
          if self.use_router:
            f.write(f"Selected LLM: {selected_llm_name}\n")
          f.write(f"Current Function Name: {func_name}\n")
          f.write(f"Current Function Code:\n")
          try:
            formatted_code = black.format_str(func_code, mode=black.Mode())
          except Exception:
            formatted_code = content
          f.write(f"{formatted_code.strip()}\n")
          f.write(f"Current Score: {curr_avg_score}\n")
          f.write(f"Current Function ID: {function_id}\n")
          f.write(f"Parent IDs: {parent_ids}\n")
          f.write(f"Parent Scores: {parent_scores}\n")
          f.write("=" * 40 + "\n\n")

        # log current and best program info
        self.logger.log(self._num_llm_calls + i + 1, prompt.code, parent_ids, parent_scores, prompt_token_cnt, func_name, func_code, \
                        function_id, sample_token_cnt, individual_scores, sample_gen_time, avg_eval_time, timed_out_list,\
                        self.best_program[0], self.best_program[1], self.best_score_list, self.best_eval_time, 
                        self.best_function_id, self.best_func_timeouts, selected_llm_name, self.best_func_model)
      
      # Update router after batch evaluation
      if self.use_router:
        # BalladRouter exposes update_with_lineage; UCBRouter uses update().
        # each respects its own use_lineage flag internally.
        if hasattr(self.router, 'update_with_lineage'):
          utilities, rewards = self.router.update_with_lineage(
              selected_llm_name, batch_new_functions, batch_scores, batch_costs)
        else:
          utilities, rewards = self.router.update(
              selected_llm_name, batch_scores, batch_costs)

        # ── Periodic diagnostics ───────────────────────────────────────────
        self.router_logger.record_round(
            arm=selected_llm_name,
            rewards=list(rewards),
            scores=list(batch_scores),
            produced_best=self.best_score > best_score_before_round,
            best_score_before=best_score_before_round,
            best_score_after=self.best_score,
        )

        # Per-sample decision log — only routers that track mean_utility /
        # compute_scores / best_score support this fine-grained log.
        can_log_decisions = (
            hasattr(self.router, 'compute_scores')
            and hasattr(self.router, 'mean_utility')
            and hasattr(self.router, 'best_score')
        )
        if can_log_decisions:
          for i, (utility, reward, cost) in enumerate(
              zip(utilities, rewards, batch_costs)):
            self.router_logger.log_router_decision(
              iteration=self._num_llm_calls + i + 1,
              selected_llm=selected_llm_name,
              ucb_scores=self.router.compute_scores(),
              mean_utilities=self.router.mean_utility.copy(),
              n_pulls=self.router.n_pulls.copy(),
              cost=cost,
              reward=reward,
              utility=utility,
              best_score=self.router.best_score)
            
      self._num_llm_calls += self.samples_per_prompt

      # Intermediate saving
      if self._count % 1 == 0:
        self.logger.save_logs()

        if self.checkpoint_callback is not None:       
          self.checkpoint_callback(self)               
        
        # Save router statistics if using router
        if self.use_router:
          router_stats = self.router.get_statistics()
          router_stats_path = os.path.join(os.path.dirname(self.log_path),
                                          f"router_stats_{self.logger.run_id}.txt")
          with open(router_stats_path, 'w') as f:
            f.write(f"Router Statistics (Iteration {router_stats['iteration']}):\n")
            if 'best_score' in router_stats:
              f.write(f"Global Best Score: {router_stats['best_score']:.4f}\n")
            if 'round' in router_stats:
              f.write(f"SDA Round: {router_stats['round']}\n")
            f.write(f"\nLLM Selection Counts:\n")
            for llm_name in router_stats['llm_names']:
              f.write(f"  {llm_name}: {router_stats['n_pulls'][llm_name]}\n")
            if 'mean_utility' in router_stats:
              f.write(f"\nMean Utilities:\n")
              for llm_name in router_stats['llm_names']:
                f.write(f"  {llm_name}: {router_stats['mean_utility'][llm_name]:.4f}\n")
            if 'max_score' in router_stats:
              f.write(f"\nMax Raw Scores:\n")
              for llm_name in router_stats['llm_names']:
                f.write(f"  {llm_name}: {router_stats['max_score'][llm_name]:.4f}\n")
            if 'lineage_mean' in router_stats:
              f.write(f"\nLineage Means:\n")
              for llm_name in router_stats['llm_names']:
                f.write(f"  {llm_name}: {router_stats['lineage_mean'][llm_name]:.4f}\n")
            if 'lineage_score' in router_stats:
              f.write(f"\nLineage Scores (intensity / log n):\n")
              for llm_name in router_stats['llm_names']:
                f.write(f"  {llm_name}: {router_stats['lineage_score'][llm_name]:.4f}\n")
          self.router_logger.save_router_logs()
      self._count += 1

  def get_checkpoint_state(self) -> dict:
    """Returns all sampler state needed to resume after a crash."""
    state = {
        'num_llm_calls': self._num_llm_calls,
        'count': self._count,
        'best_score': self.best_score,
        'best_score_list': self.best_score_list,
        'best_program': self.best_program,
        'best_eval_time': self.best_eval_time,
        'best_function_id': self.best_function_id,
        'best_func_timeouts': self.best_func_timeouts,
        'best_func_model': self.best_func_model,
    }
    if self.use_router:
      state['router_data'] = self.router_logger.router_data.copy()
    return state

  def restore_checkpoint_state(self, state: dict) -> None:
    """Restores sampler state from a checkpoint dict and sanitizes logs. Call before sample()."""
    self._num_llm_calls = state['num_llm_calls']
    self._count = state['count']
    self.best_score = state['best_score']
    self.best_score_list = state['best_score_list']
    self.best_program = state['best_program']
    self.best_eval_time = state['best_eval_time']
    self.best_function_id = state['best_function_id']
    self.best_func_timeouts = state['best_func_timeouts']
    self.best_func_model = state['best_func_model']
    
    if self.use_router and 'router_data' in state:
      self.router_logger.router_data = state['router_data']

    # --- NEW: Sanitize the text log to remove orphaned entries ---
    if os.path.exists(self.log_path):
      with open(self.log_path, 'r') as f:
          lines = f.readlines()

      # Find the line where the orphaned timeline starts
      cutoff_index = len(lines)
      target_marker = f"Call {self._num_llm_calls + 1}:\n"
      
      for i, line in enumerate(lines):
          if line == target_marker:
              cutoff_index = i
              break
              
      # Overwrite the file with the truncated history
      with open(self.log_path, 'w') as f:
          f.writelines(lines[:cutoff_index])
          
      print(f"[Sampler] Log file sanitized. Truncated {len(lines) - cutoff_index} orphaned lines.")

    # 2. Sanitize the STRUCTURED log (code_dict.pkl)
    if hasattr(self.logger, 'res_dict'):
      original_len = len(self.logger.res_dict)
      # Keep only logs where call_id is less than or equal to the checkpoint's call count
      self.logger.res_dict = [
          entry for entry in self.logger.res_dict 
          if entry.get("call_id", float('inf')) <= self._num_llm_calls
      ]
      
      # If we removed orphaned entries, overwrite the .pkl file on disk to sync it
      if len(self.logger.res_dict) < original_len:
        print(f"[Sampler] Sanitized code_dict: Truncated {original_len - len(self.logger.res_dict)} orphaned entries.")
        self.logger.save_logs()

    print(f"[Sampler] Restored from checkpoint — "
          f"resuming at call {self._num_llm_calls}, "
          f"best score so far: {self.best_score:.4f}")