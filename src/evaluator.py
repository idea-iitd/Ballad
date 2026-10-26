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

"""Class for evaluating programs proposed by the Sampler."""
import ast
from collections.abc import Sequence
import copy
from tqdm import tqdm
from typing import Any, Tuple
from scipy.spatial import distance_matrix
from src import code_manipulation
from src import programs_database
from timeout_decorator import timeout
import subprocess
import tempfile
import textwrap
import json
import itertools
import numpy as np
import astunparse
import time
import signal
import dataclasses
import uuid
import inspect
import multiprocessing

class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    print("generated code is empty 1")
    return ''
  code = f'def fake_function_header():\n{generated_code}'
  print("From _trim_function_body: \n", code)
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
      print("After parsing: \n", code)
    except SyntaxError as e:
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    # Nothing could be saved from `generated_code`
    print("generated code is empty 2")
    return ''

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'

def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
    parent_ids: list[str] = None,
    arm: str | None = None,          # LLM arm that produced this sample
    parent_funcs: list = None,        # actual Function refs for lineage BFS
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function and the full runnable program."""
  body = _trim_function_body(generated_code)

  if body=='':
     print("empty body")
     return None,''
  
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = body
  evolved_function.id = str(uuid.uuid4()) #Assign new id for lineage tracking
  if parent_ids and len(parent_ids) > 0:
    evolved_function.parent_id = parent_ids[-1] 
  evolved_function.arm = arm
  evolved_function.parent_ids = list(parent_ids) if parent_ids else []
  evolved_function.parent_funcs = list(parent_funcs) if parent_funcs else []
  
  return evolved_function, str(program)

def extract_function_code(source: str, func_name: str) -> str:
    try:
      tree = ast.parse(source)
    except:
      return f"# Could not extract code for {func_name}"
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return textwrap.dedent(ast.get_source_segment(source, node))
    return f"# Could not extract code for {func_name}"

class Sandbox:
  """Sandbox for executing generated code."""
  def __init__(self, helper, priority_file: Any):
    self.helper = helper
    self.priority_file = priority_file

  def run(self, program, function_to_run, test_input, timeout_seconds):
    func_name = None
    func_code = None
    cost = -np.inf
    execution_succeeded = False
    timed_out = False

    if hasattr(test_input, "_fields"):
        code_to_execute = "result = evaluate(test_input, priority_file)"
    else:
        code_to_execute = "result = evaluate(*test_input, priority_file)"

    exec_globals = {
        "np": np,
        "copy": copy,
        "distance_matrix": distance_matrix,
    }
    exec_locals = {
        "priority": self.helper.priority,
        "solve": self.helper.solve,
        "evaluate": self.helper.evaluate,
        "result": None,
        "test_input": test_input,
        "priority_file": self.priority_file,
    }

    print("Priority file:", self.priority_file)
    print("Priority file type:", type(self.priority_file))

    def _worker(queue):
      try:
          exec(code_to_execute, exec_globals, exec_locals)
          result = exec_locals.get("result")

          if result is not None and isinstance(result, tuple) and len(result) == 3:
              queue.put(("ok", result))
          else:
              queue.put(("error", f"Unexpected result format: {result}", None, None))
      except Exception as e:
          func_code = exec_locals.get("result")
          # extract func_code from partial result if exec set it before crashing
          if isinstance(func_code, tuple) and len(func_code) == 3:
              _, _, func_code = func_code
          else:
              func_code = None
          queue.put(("error", f"{type(e).__name__}: {e}", func_code))

    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_worker, args=(queue,), daemon=True)

    try:
        start_time = time.time()
        proc.start()
        proc.join(timeout=timeout_seconds)

        if proc.is_alive():
            proc.kill()
            proc.join()
            timed_out = True
            print(f"[!] run() timed out after {timeout_seconds}s")
        else:
            if not queue.empty():
              status, *payload = queue.get_nowait()
              if status == "ok":
                  cost, func_name, func_code = payload[0]
                  execution_succeeded = True
                  print(f"Code executed successfully, cost= {cost}")
              else:
                  error_msg = payload[0]
                  func_code = payload[1] if len(payload) > 1 else None
                  print(f"[!] Execution error: {error_msg}")
                  if func_code:
                      print(f"[!] Erroneous code:\n{func_code}")
            else:
                print("[!] Worker returned no result.")

    except Exception as e:
        print(f"[!] Unexpected error in run(): {type(e).__name__}: {e}")
        func_code = str(e)

    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()

    return cost, func_name, func_code, execution_succeeded, timed_out
    
def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


class Evaluator:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      template: code_manipulation.Program,
      function_to_evolve: str,
      function_to_run: str,
      dataset_dict: dict,
      helper: Any,
      priority_file: Any,
      timeout_seconds: int = 60,
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = helper.get_inputs(dataset_dict)
    self._num_inputs = dataset_dict['n_instances']
    self._timeout_seconds = timeout_seconds
    self._sandbox = Sandbox(helper, priority_file)

  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
      parent_ids: list[str] = None,
      arm: str | None = None,      # LLM arm (for lineage tracking)
      parent_funcs: list = None,   # Function refs (for lineage BFS)
  ) -> None:
    """Compiles the sample into a program and executes it on test inputs."""
    new_function, program = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve,
        parent_ids, arm=arm, parent_funcs=parent_funcs)

    scores_per_test = {}
    timed_out_list = []
    start_time = time.time()
    for i in range(self._num_inputs):
      current_input = self._inputs[i]
      if not isinstance(current_input, tuple):
        current_input = (current_input,)

      print("\n----------------------------------------------")
      print("Input idx: ", i)
      test_output, func_name, func_code, runs_ok, timed_out = self._sandbox.run(
          program, self._function_to_run, current_input, self._timeout_seconds)     
      if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
          and test_output is not None):
        if not isinstance(test_output, (int, float)):
            raise ValueError('@function.run did not return an int/float score.')
        # scores_per_test[(current_input[0], current_input[1])] = test_output
        scores_per_test[i] = test_output
        print("Func Name: ", func_name)
        print("Func Code: \n", func_code)
        print("Function to run: ", self._function_to_run)
        print("Score: ", test_output)
        print("Timed out: ", timed_out)
      else:
        print(f"Function execution failed or called ancestor for test case {i}.")
        print("Func Name: ", func_name)
        print("Func Code: \n", func_code)
        print("Score: -inf")
        print("Timed out: ", timed_out)
        scores_per_test[i] = -np.inf    
      timed_out_list.append(timed_out)
    avg_eval_time = (time.time() - start_time)/self._num_inputs

    print("Timed out list: ", timed_out_list)
    if scores_per_test and not any(v == -np.inf for v in scores_per_test.values()) and (func_code is not None) and not any(v is None for v in scores_per_test.values()):
      self._database.register_program(new_function, island_id, scores_per_test)
      individual_scores = list(scores_per_test.values()) 
      curr_avg_score = np.mean(individual_scores)
      print("Current Avg Score: ", curr_avg_score)
    else:
      if not scores_per_test:
        individual_scores = [-np.inf] * self._num_inputs #[]
        curr_avg_score = -np.inf
      else:  # means some value(s) are -np.inf
        individual_scores = list(scores_per_test.values())
        curr_avg_score = -np.inf

    if new_function is None:
      function_id = str(uuid.uuid4())
    else:
      function_id = new_function.id

    return individual_scores, curr_avg_score, func_name, func_code, avg_eval_time, function_id, timed_out_list, new_function