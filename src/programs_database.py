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

"""A programs database that implements the evolutionary algorithm."""
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any
import ast
import textwrap
from absl import logging
import numpy as np
import scipy

from src import code_manipulation
from src import config as config_lib

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]

def _fix_function_indent(function_str: str) -> str:
    """Re-normalise a function definition string to 2-space body indentation.

    Called only when text_to_function raises (i.e. the absl 'Failed parsing'
    warning fires). Parses the string, extracts the body lines, dedents them,
    and re-indents with 2 spaces so that the body is compatible with the
    template's 2-space docstring when rendered by Function.__str__.
    """
    try:
        tree = ast.parse(function_str.strip())
    except SyntaxError:
        # Try trimming trailing lines until it parses (mirrors _trim_function_body)
        lines = function_str.strip().splitlines()
        tree = None
        while lines and tree is None:
            try:
                tree = ast.parse('\n'.join(lines))
            except SyntaxError as exc:
                lines = lines[:exc.lineno - 1] if exc.lineno and exc.lineno > 1 else []
        if not tree or not lines:
            return function_str

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        src_lines = function_str.strip().splitlines()
        body_start = node.body[0].lineno - 1
        body_end = node.body[-1].end_lineno
        body_raw = '\n'.join(src_lines[body_start:body_end])
        # Dedent to zero, then re-indent with 2 spaces
        dedented = textwrap.dedent(body_raw)
        fixed_body = '\n'.join(
            '  ' + ln if ln.strip() else '' for ln in dedented.splitlines()
        )
        # Reconstruct the function with the fixed body
        header = '\n'.join(src_lines[:body_start])
        return header + '\n' + fixed_body + '\n'

    return function_str

def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
  """Returns the tempered softmax of 1D finite `logits`."""
  if not np.all(np.isfinite(logits)):
    non_finites = set(logits[~np.isfinite(logits)])
    raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
  if not np.issubdtype(logits.dtype, np.floating):
    logits = np.array(logits, dtype=np.float32)

  result = scipy.special.softmax(logits / temperature, axis=-1)
  # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
  index = np.argmax(result)
  result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
  return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
  """Reduces per-test scores into a single score."""
  scores_array = np.array(list(scores_per_test.values()))
  average_score = np.mean(scores_array) if scores_array.size > 0 else 0
  # print("Average score from _reduce_score function: ", scores_array, average_score)
  return average_score

def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
  """Represents test scores as a canonical signature."""
  return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
  """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

  Attributes:
    code: The prompt, ending with the header of the function to be completed.
    version_generated: The function to be completed is `_v{version_generated}`.
    island_id: Identifier of the island that produced the implementations
       included in the prompt. Used to direct the newly generated implementation
       into the same island.
  """
  code: str
  version_generated: int
  island_id: int
  parent_ids: list[str] = dataclasses.field(default_factory=list) 
  parent_scores: list[float] = dataclasses.field(default_factory=list) 
  parent_funcs: list = dataclasses.field(default_factory=list, compare=False)  # Function refs for lineage BFS

class ProgramsDatabase:
  """A collection of programs, organized as islands."""

  def __init__(
      self,
      config: config_lib.ProgramsDatabaseConfig,
      template: code_manipulation.Program,
      function_to_evolve: str,
  ) -> None:
    self._config: config_lib.ProgramsDatabaseConfig = config
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve

    # Initialize empty islands.
    self._islands: list[Island] = []
    for _ in range(config.num_islands):
      self._islands.append(
          Island(template, function_to_evolve, config.functions_per_prompt,
                 config.cluster_sampling_temperature_init,
                 config.cluster_sampling_temperature_period))
    self._best_score_per_island: list[float] = (
        [-float('inf')] * config.num_islands)
    self._best_program_per_island: list[code_manipulation.Function | None] = (
        [None] * config.num_islands)
    self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
        [None] * config.num_islands)

    self._last_reset_time: float = time.time()

  def get_prompt(self) -> Prompt:
    """Returns a prompt containing implementations from one chosen island."""
    island_id = np.random.randint(len(self._islands))
    code, version_generated, parent_ids, parent_scores, parent_funcs = (
        self._islands[island_id].get_prompt())
    return Prompt(code, version_generated, island_id, parent_ids, parent_scores, parent_funcs)

  def _register_program_in_island(
      self,
      program: code_manipulation.Function,
      island_id: int,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `program` in the specified island."""
    self._islands[island_id].register_program(program, scores_per_test)
    score = _reduce_score(scores_per_test)
    program.score = score  # annotate function with its reduced score (for lineage BFS)
    if score > self._best_score_per_island[island_id]:
      self._best_program_per_island[island_id] = program
      self._best_scores_per_test_per_island[island_id] = scores_per_test
      self._best_score_per_island[island_id] = score
      logging.info('Best score of island %d increased to %s', island_id, score)

  def register_program(
      self,
      program: code_manipulation.Function,
      island_id: int | None,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Registers `program` in the database."""
    # In an asynchronous implementation we should consider the possibility of
    # registering a program on an island that had been reset after the prompt
    # was generated. Leaving that out here for simplicity.
    if island_id is None:
      # This is a program added at the beginning, so adding it to all islands.
      for island_id in range(len(self._islands)):
        self._register_program_in_island(program, island_id, scores_per_test)
    else:
      self._register_program_in_island(program, island_id, scores_per_test)

    # Check whether it is time to reset an island.
    if (time.time() - self._last_reset_time > self._config.reset_period):
      self._last_reset_time = time.time()
      self.reset_islands()

  def reset_islands(self) -> None:
    """Resets the weaker half of islands."""
    # We sort best scores after adding minor noise to break ties.
    indices_sorted_by_score: np.ndarray = np.argsort(
        self._best_score_per_island +
        np.random.randn(len(self._best_score_per_island)) * 1e-6)
    num_islands_to_reset = self._config.num_islands // 2
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    for island_id in reset_islands_ids:
      self._islands[island_id] = Island(
          self._template,
          self._function_to_evolve,
          self._config.functions_per_prompt,
          self._config.cluster_sampling_temperature_init,
          self._config.cluster_sampling_temperature_period)
      self._best_score_per_island[island_id] = -float('inf')
      founder_island_id = np.random.choice(keep_islands_ids)
      founder = self._best_program_per_island[founder_island_id]
      founder_scores = self._best_scores_per_test_per_island[founder_island_id]
      self._register_program_in_island(founder, island_id, founder_scores)
      
  def save_checkpoint(self, path: str) -> None:
    """Serializes full database state to disk."""
    import pickle
    import os
    
    state = {
        'islands': self._islands,
        'best_score_per_island': self._best_score_per_island,
        'best_program_per_island': self._best_program_per_island,
        'best_scores_per_test_per_island': self._best_scores_per_test_per_island,
        'last_reset_time': self._last_reset_time,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(state, f)
    logging.info('Database checkpoint saved to %s', path)

  def load_checkpoint(self, path: str) -> None:
      """Restores full database state from disk."""
      import pickle
      with open(path, 'rb') as f:
          state = pickle.load(f)
      self._islands = state['islands']
      self._best_score_per_island = state['best_score_per_island']
      self._best_program_per_island = state['best_program_per_island']
      self._best_scores_per_test_per_island = state['best_scores_per_test_per_island']
      self._last_reset_time = state['last_reset_time']
      logging.info('Database state restored from %s', path)


class Island:
  """A sub-population of the programs database."""

  def __init__(
      self,
      template: code_manipulation.Program,
      function_to_evolve: str,
      functions_per_prompt: int,
      cluster_sampling_temperature_init: float,
      cluster_sampling_temperature_period: int,
  ) -> None:
    self._template: code_manipulation.Program = template
    self._function_to_evolve: str = function_to_evolve
    self._functions_per_prompt: int = functions_per_prompt
    self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
    self._cluster_sampling_temperature_period = (
        cluster_sampling_temperature_period)

    self._clusters: dict[Signature, Cluster] = {}
    self._num_programs: int = 0

  def register_program(
      self,
      program: code_manipulation.Function,
      scores_per_test: ScoresPerTest,
  ) -> None:
    """Stores a program on this island, in its appropriate cluster."""
    signature = _get_signature(scores_per_test)
    if signature not in self._clusters:
      score = _reduce_score(scores_per_test)
      self._clusters[signature] = Cluster(score, program)
    else:
      self._clusters[signature].register_program(program)
    self._num_programs += 1

  def get_prompt(self) -> tuple[str, int, list[str]]:
    """Constructs a prompt containing functions from this island."""
    signatures = list(self._clusters.keys())
    cluster_scores = np.array(
        [self._clusters[signature].score for signature in signatures])

    # Convert scores to probabilities using softmax with temperature schedule.
    period = self._cluster_sampling_temperature_period
    temperature = self._cluster_sampling_temperature_init * (
        1 - (self._num_programs % period) / period)
    probabilities = _softmax(cluster_scores, temperature)

    # At the beginning of an experiment when we have few clusters, place fewer
    # programs into the prompt.
    functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

    # samples with replacement. In many prompts the same program is used twice. 
    try:
      idx = np.random.choice(
          len(signatures), size=functions_per_prompt, p=probabilities, replace=False)
    except ValueError:
      # When temperature is near 0, softmax collapses to a near-one-hot vector;
      # fall back to sampling with replacement.
      idx = np.random.choice(
          len(signatures), size=functions_per_prompt, p=probabilities, replace=True)
      
    chosen_signatures = [signatures[i] for i in idx]
    implementations = []
    scores = []
    parent_ids = []
    parent_funcs = []
    for signature in chosen_signatures:
      cluster = self._clusters[signature]
      sampled_program = cluster.sample_program()
      implementations.append(sampled_program)
      scores.append(cluster.score)
      parent_ids.append(sampled_program.id)
      parent_funcs.append(sampled_program)  # actual ref for lineage BFS

    indices = np.argsort(scores)
    sorted_implementations = [implementations[i] for i in indices]
    sorted_parent_ids   = [parent_ids[i]   for i in indices]
    sorted_parent_funcs = [parent_funcs[i] for i in indices]
    sorted_scores       = [scores[i]       for i in indices]
    version_generated = len(sorted_implementations) + 1
    return (self._generate_prompt(sorted_implementations),
            version_generated, sorted_parent_ids, sorted_scores, sorted_parent_funcs)
  

  def _generate_prompt(
      self,
      implementations: Sequence[code_manipulation.Function]) -> str:
    """Creates a prompt containing a sequence of function `implementations`."""
    implementations = copy.deepcopy(implementations)  # We will mutate these.

    # Format the names and docstrings of functions to be included in the prompt.
    versioned_functions: list[code_manipulation.Function] = []
    for i, implementation in enumerate(implementations):
      new_function_name = f'{self._function_to_evolve}_v{i}'
      implementation.name = new_function_name
      # Update the docstring for all subsequent functions after `_v0`.
      if i >= 1:
        implementation.docstring = (
            f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
      # If the function is recursive, replace calls to itself with its new name.
      implementation = code_manipulation.rename_function_calls(
          str(implementation), self._function_to_evolve, new_function_name)
      try:
        versioned_functions.append(
            code_manipulation.text_to_function(implementation))
      except:
        # The stored body uses different indentation than the template docstring,
        # causing an IndentationError in ast.parse. Re-normalise to 2-space and
        # retry; skip this function if it still fails.
        try:
          fixed = _fix_function_indent(implementation)
          versioned_functions.append(
              code_manipulation.text_to_function(fixed))
        except Exception:
          logging.warning(
              'Skipping unparseable function in prompt (even after indent fix): %.120s',
              implementation)

    # Create the header of the function to be generated by the LLM.
    next_version = len(implementations)
    new_function_name = f'{self._function_to_evolve}_v{next_version}'
    header = dataclasses.replace(
        implementations[-1],
        name=new_function_name,
        body='',
        docstring=('Improved version of '
                   f'`{self._function_to_evolve}_v{next_version - 1}`.'),
    )
    versioned_functions.append(header)

    # Replace functions in the template with the list constructed here.
    prompt = dataclasses.replace(self._template, functions=versioned_functions)
    return str(prompt)


class Cluster:
  """A cluster of programs on the same island and with the same Signature."""

  def __init__(self, score: float, implementation: code_manipulation.Function):
    self._score = score
    self._programs: list[code_manipulation.Function] = [implementation]
    self._lengths: list[int] = [len(str(implementation))]

  @property
  def score(self) -> float:
    """Reduced score of the signature that this cluster represents."""
    return self._score

  def register_program(self, program: code_manipulation.Function) -> None:
    """Adds `program` to the cluster."""
    self._programs.append(program)
    self._lengths.append(len(str(program)))

  def sample_program(self) -> code_manipulation.Function:
    """Samples a program, giving higher probability to shorther programs."""
    normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
        max(self._lengths) + 1e-6)
    probabilities = _softmax(-normalized_lengths, temperature=1.0)
    return np.random.choice(self._programs, p=probabilities)
