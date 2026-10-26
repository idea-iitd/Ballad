# Copyright 2023 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0
"""A single-threaded implementation of the FunSearch pipeline (updated)."""
from collections.abc import Sequence
from typing import Any
import asyncio
from src import code_manipulation
from src import config as config_lib
from src import evaluator
from src import programs_database
from src import sampler
from src.logger import Logger
from src.checkpoint import CheckpointManager
import numpy as np
import requests
import argparse
import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
import importlib.util
import datetime
import re
import sys
import traceback
from hydra.utils import instantiate
import time

def _extract_function_names(specification: str) -> tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run."""
    run_functions = list(
        code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
    evolve_functions = list(
        code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
    return evolve_functions[0], run_functions[0]


def _import_helper(problem_name: str):
    """
    Import helpers.py from problems/<problem_name> as a package.
    Requires problems/ and each problem folder to have __init__.py
    """
    module_name = f"problems.{problem_name}.helpers"
    return importlib.import_module(module_name)

def init_priority(problem_name: str, model_name: str, specification: str, run_id: int):
    problem_dir = Path("problems") / problem_name
    priority_file = problem_dir / f"priority_{model_name}_{run_id}.py"

    # --- Find the evolve function ---
    pattern = r"@funsearch\.evolve\s*\ndef\s+(\w+)\s*\("
    match = re.search(pattern, specification)
    if not match:
        raise ValueError("No @funsearch.evolve function found in specification.")

    original_func_name = match.group(1)

    # Extract everything from @funsearch.evolve onwards
    evolve_start = specification.find("@funsearch.evolve")
    evolve_code = specification[evolve_start:]

    # Rename the evolved function to priority_v0
    evolve_code = evolve_code.replace(
        f"def {original_func_name}(", "def priority_v0("
    )

    # --- Write file ---
    with open(priority_file, "w") as f:
        f.write("import numpy as np\n")
        f.write("from copy import copy\n")
        f.write("from scipy.spatial import distance_matrix\n")
        f.write(evolve_code[len('@funsearch.evolve'):])

    print(f"[+] Wrote {priority_file}")

    return priority_file

@hydra.main(version_base=None, config_path="../cfg", config_name="config_router_ucb")
def main(cfg: DictConfig):
    start_funsearch = time.time()
    problem_name = cfg.problem.problem_name
    res_dir_name = cfg.res_dir_name
    log_dir_name = cfg.log_dir_name

    print(problem_name)
    problem_dir = Path("problems") / problem_name
    
    # --- Load specification.txt ---
    spec_file = problem_dir / "specification.txt"
    with open(spec_file, "r") as f:
        specification = f.read()
    
    # --- Load helper dynamically ---
    helper = _import_helper(problem_name)

    # --- Extract functions to evolve & run ---
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)

    # --- Prepare dataset ---
    dataset_dir = problem_dir / "dataset"
    dataset_outputs = helper.read_dataset(dataset_dir)
    
    if isinstance(dataset_outputs, dict):
        dataset_dict = dataset_outputs
    else:
        print("Check input format")
    print(dataset_dict.keys())

    # --- Setup database ---
    config = config_lib.Config()
    database = programs_database.ProgramsDatabase(
        config.programs_database, template, function_to_evolve)
    
    use_router = cfg.get('use_router', False)
    # --- Setup LLM clients ---
    # Check if we have multiple LLM clients (for router) or single client
    if hasattr(cfg, 'llm_clients') and use_router:
        # Multiple LLM arms (router mode)
        # suff = cfg.router.get('strategy', 'ucb')
        model_identifier = cfg.router.get('strategy', 'ucb')

        llm_clients = {}
        for group in cfg.llm_clients.values():
            for model_key, model_cfg in group.items():
                llm_clients[model_key] = instantiate(model_cfg)

        print(f"[FunSearch] Loaded {len(llm_clients)} LLM clients: {list(llm_clients.keys())}")
        # Extract router configuration
        router_config = {
            'strategy':             cfg.router.get('strategy', 'ucb'),
            'beta':                 cfg.router.get('beta', 1.0),
            'total_steps':          cfg.router.get('total_steps', 1000),
            'elite_pool_size':      cfg.router.get('elite_pool_size', 5),
            
            # Ballad params 
            'min_pulls':            cfg.router.get('min_pulls', 5),
            'n_bootstrap':          cfg.router.get('n_bootstrap', 200),
            'tau':                cfg.router.get('tau', 0.9),
            'lam':                  cfg.router.get('lam', 0.3), #controls weight of Max-Quantile Mixing
            'rng_seed':             cfg.router.get('rng_seed', None),
            'use_log_t':            cfg.router.get('use_log_t', False),
            'window_size':          cfg.router.get('window_size', 25),
            # Lineage BFS params (ballad only)
            'use_lineage':          cfg.router.get('use_lineage', False),
            'gamma_lineage':        cfg.router.get('gamma_lineage', 0.9), #discount factor
            'D_max':                cfg.router.get('D_max', 3),
            'alpha_lineage':        cfg.router.get('alpha_lineage', 1.0),
            'alpha_decay':          cfg.router.get('alpha_decay', False),
        }

        print(
            "[FunSearch] Router config: "
            f"strategy={router_config['strategy']}, "
            f"beta={router_config['beta']}, "
            f"total_steps={router_config['total_steps']}, "
            f"min_pulls={router_config['min_pulls']}, "
            f"elite_pool_size={router_config['elite_pool_size']}"
        )
        if router_config['strategy'] == 'ballad':
            print(
                "[FunSearch] UCB Hybrid QoMax config: "
                f"n_bootstrap={router_config['n_bootstrap']}, "
                f"tau={router_config['tau']}, "
                f"lam={router_config['lam']}, "
                f"rng_seed={router_config['rng_seed']}, "
                f"use_lineage={router_config['use_lineage']}, "
                f"gamma_lineage={router_config['gamma_lineage']}, "
                f"D_max={router_config['D_max']}, "
                f"alpha_lineage={router_config['alpha_lineage']}, "
                f"alpha_decay={router_config['alpha_decay']}"
            )
    else:
        # print(cfg.llm_clients)
        # 1. Get the path from config (e.g., "gemini_models.flash_lite")
        active_path = cfg.get('active_model')
        
        if not active_path:
            raise ValueError("use_router is False, but no 'active_model' is defined in config.")

        # 2. Split into group and model keys
        try:
            group_key, model_key = active_path.split('.')
        except ValueError:
            raise ValueError(f"active_model '{active_path}' must be in 'group.model' format.")

        # 3. Targeted Retrieval
        # Access: cfg.llm_clients -> gemini_models -> flash_lite
        group_cfg = cfg.llm_clients.get(group_key)
        if not group_cfg:
            raise ValueError(f"Group '{group_key}' not found in llm_clients. Check your defaults list.")
            
        model_cfg = group_cfg.get(model_key)
        if not model_cfg:
            raise ValueError(f"Model '{model_key}' not found in group '{group_key}'. Check your {group_key}.yaml")

        # 4. Instantiation
        model_identifier = model_cfg.model_name
        llm_clients = instantiate(model_cfg)
                
        router_config = None
        print(f"[FunSearch] Using single LLM client: {model_identifier}")

    logger = Logger(problem_name, model_identifier, res_dir_name)
    checkpoint_mgr = CheckpointManager(logger.results_dir, logger.run_id)

    resume_run_id = cfg.get('resume_run_id', None)  # add to your hydra config

    if str(resume_run_id).lower() in ['none', 'null', 'false', '']:
        resume_run_id = None

    is_resuming = False

    if resume_run_id is not None:
        try:
            resume_run_id = int(resume_run_id)
        except ValueError:
            pass
            
        logger = Logger(problem_name, model_identifier, res_dir_name, resume_run_id=resume_run_id)
        checkpoint_mgr = CheckpointManager(logger.results_dir, resume_run_id)
        
        if checkpoint_mgr.exists():
            print(f"[+] Resuming from checkpoint run {resume_run_id}")
            # Restore database
            database.load_checkpoint(checkpoint_mgr.db_path)
            # Restore router
            # if router is not None and Path(checkpoint_mgr.router_path).exists():
            #     router.load_checkpoint(checkpoint_mgr.router_path)
            # Restore sampler state
            saved = checkpoint_mgr.load()
            sampler_state = saved['sampler']
            is_resuming = True
        else:
            print(f"[!] No checkpoint found for run {resume_run_id}, starting fresh.")
            sampler_state = None
    else:
        sampler_state = None

    priority_file = init_priority(problem_name, model_identifier, specification, logger.run_id)
    print(priority_file, type(priority_file))

    # --- Evaluators ---
    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run, 
            dataset_dict,
            helper,
            priority_file
        ))

    if is_resuming and sampler_state is not None:
        print("[*] Resuming: Bypassing initial program evaluation. Loading baseline stats from checkpoint.")
        best_score_list = sampler_state['best_score_list']
        best_score = sampler_state['best_score']
        best_program = sampler_state['best_program']
        best_eval_time = sampler_state['best_eval_time']
        best_function_id = sampler_state['best_function_id']
        best_func_timeouts = sampler_state['best_func_timeouts']
    else:
        print("[*] Starting fresh: Evaluating initial implementation.")
        # --- Analyse initial implementation ---
        initial = template.get_function(function_to_evolve).body
        score_list, curr_avg_score, func_name, func_code, avg_eval_time, function_id, timed_out_list, _new_fn = evaluators[0].analyse(
            initial, island_id=None, version_generated=None, parent_ids=None)

        best_score_list = score_list
        best_score = curr_avg_score
        best_program = (func_name, func_code)
        best_eval_time = avg_eval_time
        best_function_id = function_id
        best_func_timeouts = timed_out_list

        print(f"[*] Initial {func_name} average score: {curr_avg_score:.4f} "
            f"with eval time: {avg_eval_time:.4f}s")

    # --- Samplers ---
    samplers = [
        sampler.Sampler(
            database, 
            evaluators, 
            config.samples_per_prompt,
            best_score_list, 
            best_score, 
            best_program, 
            best_eval_time, 
            best_func_timeouts, 
            best_function_id, 
            problem_dir, 
            priority_file, 
            llm_clients,  # Pass dict or single client
            model_identifier, 
            logger,
            log_dir_name,
            router_config=router_config,  # Pass router config
            resume=is_resuming
        )
        for _ in range(config.num_samplers)
    ]

    def make_checkpoint_callback(checkpoint_mgr, database):
        def callback(sampler_instance):
            current_router = getattr(sampler_instance, 'router', None)
            checkpoint_mgr.save(database, current_router, sampler_instance.get_checkpoint_state())
        return callback

    save_checkpoint = make_checkpoint_callback(checkpoint_mgr, database)

    for s in samplers:
        s.checkpoint_callback = save_checkpoint
        if sampler_state:
            s.restore_checkpoint_state(sampler_state)
            if hasattr(s, 'use_router') and s.use_router and Path(checkpoint_mgr.router_path).exists():
                s.router.load_checkpoint(checkpoint_mgr.router_path)
                if hasattr(s, 'router_logger'):
                    s.router_logger.sync_from_router()

    try:
        for s in samplers:
            s.sample()

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Exiting gracefully...")
        sys.exit(0)

    finally:
        logger.save_logs()
        print(f"Total time taken till completion/exit: {time.time() - start_funsearch:.3f} seconds")
        
        # Print final router statistics if using router
        if router_config and isinstance(llm_clients, dict):
            print("\n" + "="*60)
            # print("FINAL ROUTER STATISTICS")
            # print("="*60)
            # Router stats are saved in the sampler, but we can add a summary here
            print(f"Completed {config.samples_per_prompt} iterations")
            print("Check router_stats_*.txt files for detailed statistics")

if __name__ == "__main__":
    # check network once
    try:
        response = requests.get("https://google.com", timeout=5)
        print(f"Status Code: {response.status_code}")
    except Exception as e:
        print(f"Connection failed: {e}")
        
    main()