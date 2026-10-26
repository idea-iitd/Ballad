"""
Plots training stability: call number vs. average train score and standard deviation
of programs generated at that call, averaged over multiple runs.

[IMP]: Global min and max are now computed as the least and most average scores
across train instances that any run got across calls. The lines are colored based on
which model achieved the global min/max.

Can generate both individual model plots and comparative plots across models.

Saves:
- Individual plots in proc_results/{problem_name}/plots/
- Comparative plots in proc_results/{problem_name}/plots/

Usage:
    # Single model plot
    python source/best_train_scores.py --problem_name tsp --model_name gemini-2.5-flash-lite --runs "1 2 3"
    
    # Comparative plot across multiple models with different runs per model
    python source/best_train_scores.py --problem_name tsp --models gemini-2.5-flash-lite gemini-2.5-pro --runs "1 2 3" "1 2"
"""

import os
import pickle
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Mapping from problem_name to metric string
PROBLEM_METRICS = {
    "bpp_offline_aco": "No. of bins",
    "bpp_online": "Excess (%) bins over L1 Bound",
    "cvrp_aco": "Tour cost",
    "tsp_constructive": "Tour cost",
    "ged": "GED"
}

# Descriptive problem names
PROBLEM_NAMES = {
    "bpp_offline_aco": "Offline Bin Packing",
    "bpp_online": "Online Bin Packing",
    "cvrp_aco": "CVRP (ACO)",
    "tsp_constructive": "TSP (Constructive)",
}

def load_run_logs(results_dir, run_numbers=None):
    """
    Load specified run logs from a directory.
    
    Args:
        results_dir: Path to directory containing log files
        run_numbers: List of run numbers to load (e.g., [1, 2, 3]). 
                     If None, loads all available runs up to 3.
    
    Returns:
        List of log data, list of run numbers actually loaded
    """
    logs = []
    loaded_runs = []
    
    if run_numbers is None:
        # Auto-detect available runs (checks up to run 10)
        run_numbers = []
        for i in range(1, 11):
            file_path = results_dir / f"code_dict_run{i}.pkl"
            if file_path.exists():
                run_numbers.append(i)
    
    for run_num in run_numbers:
        file_path = results_dir / f"code_dict_run{run_num}.pkl"
        if file_path.exists():
            with open(file_path, "rb") as f:
                log_data = pickle.load(f)
                logs.append(log_data)
                loaded_runs.append(run_num)
                print(f"[+] Loaded run {run_num} with {len(log_data)} calls")
        else:
            print(f"[!] Warning: Run {run_num} not found at {file_path}")
    
    if not logs:
        print(f"[-] No log files found in {results_dir}")
    
    return logs, loaded_runs


def extract_training_data(logs, problem_name):
    """
    Extract training scores from logs.
    Only programs where ALL test instance scores are valid (finite) are included.
    Returns: dict mapping call_id -> list of scores (one per run)
    """
    # Determine if we need to negate scores (for minimization problems)
    negate = problem_name.lower() in ["cvrp_aco", "bpp_offline_aco", "tsp_constructive", "bpp_online", "ged"]
    
    call_scores = defaultdict(list)
    
    for run_idx, log in enumerate(logs):
        for entry in log:
            call_id = entry["call_id"]
            gen_scores = entry.get("best_gen_scores", [])
            
            # Only keep programs where ALL scores are valid (finite)
            # Filter out programs with any -inf or nan values
            valid_programs = [
                s for s in gen_scores 
                if s != -np.inf and not np.isnan(s)
            ]
            
            if valid_programs:
                avg_score = np.mean(valid_programs)
                if negate:
                    avg_score = -avg_score
                call_scores[call_id].append(avg_score)
            else:
                # All programs failed for this call
                call_scores[call_id].append(np.nan)
    
    return call_scores


def compute_statistics(call_scores):
    """
    Compute mean and std across runs for each call.
    Returns: calls (sorted), means, stds
    """
    calls = sorted(call_scores.keys())
    means = []
    stds = []
    
    for call in calls:
        scores = call_scores[call]
        # Filter out NaN values
        valid_scores = [s for s in scores if not np.isnan(s)]
        
        if valid_scores:
            means.append(np.mean(valid_scores))
            stds.append(np.std(valid_scores) if len(valid_scores) > 1 else 0.0)
        else:
            means.append(np.nan)
            stds.append(np.nan)
    
    return calls, np.array(means), np.array(stds)


def compute_global_extrema_from_calls(call_scores_dict):
    """
    Compute global min and max from call_scores dictionaries.
    
    Args:
        call_scores_dict: dict of {model_name: call_scores}
        where call_scores = {call_id: [scores from all runs]}
    
    Returns: (global_min, global_max, model_with_min, run_with_min, model_with_max, run_with_max)
    """
    model_extrema = {}
    
    # For each model, collect all scores across all call_ids and all runs, tracking run indices
    for model_name, call_scores in call_scores_dict.items():
        all_model_scores = []
        score_to_run = {}
        
        # Iterate through all call_ids
        for call_id, scores_list in call_scores.items():
            # scores_list contains scores from all runs for this call_id
            for run_idx, score in enumerate(scores_list):
                if not np.isnan(score):
                    all_model_scores.append(score)
                    score_to_run[float(score)] = run_idx + 1  # 1-indexed run number
        
        if all_model_scores:
            model_min = np.min(all_model_scores)
            model_max = np.max(all_model_scores)
            run_min = score_to_run.get(float(model_min), "Unknown")
            run_max = score_to_run.get(float(model_max), "Unknown")
            model_extrema[model_name] = (model_min, model_max, run_min, run_max)
    
    if not model_extrema:
        return None, None, None, None, None, None
    
    # Find global min and max across all models
    all_mins = [extrema[0] for extrema in model_extrema.values()]
    all_maxs = [extrema[1] for extrema in model_extrema.values()]
    
    global_min = np.min(all_mins)
    global_max = np.max(all_maxs)
    
    # Find which models and runs achieved these extrema
    min_info = [(model, extrema[2]) for model, extrema in model_extrema.items() 
                if extrema[0] == global_min][0]
    max_info = [(model, extrema[3]) for model, extrema in model_extrema.items() 
                if extrema[1] == global_max][0]
    
    model_with_min, run_with_min = min_info
    model_with_max, run_with_max = max_info
    
    return global_min, global_max, model_with_min, run_with_min, model_with_max, run_with_max


def plot_training_stability_single(calls, means, stds, problem_name, model_name, output_dir,
                                   loaded_runs, global_min, global_max, min_model, max_model):
    """
    Plot training stability for a single model with global min and max annotations.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter out NaN values
    valid_mask = ~np.isnan(means)
    calls_clean = np.array(calls)[valid_mask]
    means_clean = means[valid_mask]
    stds_clean = stds[valid_mask]
    
    # Plot mean line
    ax.plot(calls_clean, means_clean, linewidth=2.5, color='#2E86AB',
            label='Mean Score', marker='o', markersize=6, 
            markevery=max(1, len(calls_clean)//20))
    
    # Plot confidence band
    ax.fill_between(calls_clean, 
                    means_clean - stds_clean, 
                    means_clean + stds_clean,
                    alpha=0.3, color='#2E86AB', label='±1 Std Dev')
    
    # Add horizontal lines for global min and max with model colors
    if global_min is not None:
        min_color = 'green' if min_model == model_name else 'lightgreen'
        ax.axhline(y=global_min, color=min_color, linestyle='--', linewidth=2, 
                   label=f'Global Min: {global_min:.2f} [{min_model}]', alpha=0.7)
    
    if global_max is not None:
        max_color = 'orange' if max_model == model_name else 'lightyellow'
        ax.axhline(y=global_max, color=max_color, linestyle='--', linewidth=2, 
                   label=f'Global Max: {global_max:.2f} [{max_model}]', alpha=0.7)
    
    # Labels and title
    metric_name = PROBLEM_METRICS.get(problem_name.lower(), "Score")
    problem_display = PROBLEM_NAMES.get(problem_name.lower(), problem_name)
    
    ax.set_xlabel('Call Number', fontweight='bold')
    ax.set_ylabel(f'{metric_name}', fontweight='bold')
    
    runs_str = f"runs {','.join(map(str, loaded_runs))}" if loaded_runs else "all runs"
    ax.set_title(f'Best Score Convergence: {problem_display}\n[{model_name}] ({runs_str})', 
                 fontweight='bold', pad=20)
    
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    # output_path = output_dir / f"exp1_train_best_scores_{problem_name}_{model_name}.png"
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # print(f"[+] Saved plot to {output_path}")
    
    # Also save as PDF for publications
    output_path_pdf = output_dir / f"{problem_name}_{model_name}.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"[+] Saved PDF to {output_path_pdf}")
    
    plt.close()


def plot_training_stability_comparative(all_model_data, call_scores_dict, problem_name, output_dir):
    """
    Create comparative plot across multiple models.
    Handles unequal number of calls - plots each line till its end.
    
    Args:
        all_model_data: dict of {model_name: (calls, means, stds, loaded_runs)}
        call_scores_dict: dict of {model_name: call_scores} for extrema calculation
        problem_name: Name of the problem
        output_dir: Directory to save plots
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = sns.color_palette("husl", len(all_model_data))
    
    # Compute global min and max with model attribution from call_scores
    global_min, global_max, model_with_min, run_with_min, model_with_max, run_with_max = compute_global_extrema_from_calls(call_scores_dict)
    
    for idx, (model_name, (calls, means, stds, loaded_runs)) in enumerate(all_model_data.items()):
        # Filter out NaN values
        valid_mask = ~np.isnan(means)
        calls_clean = np.array(calls)[valid_mask]
        means_clean = means[valid_mask]
        stds_clean = stds[valid_mask]
        
        if len(calls_clean) == 0:
            print(f"[!] Warning: No valid data for {model_name}, skipping in comparative plot")
            continue
            
        # Plot mean line for the current model
        ax.plot(calls_clean, means_clean, linewidth=2.5, color=colors[idx],
                label=model_name, marker='o', markersize=4,
                markevery=max(1, len(calls_clean)//20))

        # Plot confidence band for the current model
        ax.fill_between(calls_clean,
                          means_clean - stds_clean,
                          means_clean + stds_clean,
                          alpha=0.2, color=colors[idx])
    
    # Calculate and plot global min and max with model colors
    if global_min is not None:
        # Get the color of the model that achieved global_min
        model_colors = {model: colors[idx] for idx, model in enumerate(all_model_data.keys())}
        min_line_color = model_colors.get(model_with_min, 'green')
        max_line_color = model_colors.get(model_with_max, 'orange')
        
        ax.axhline(y=global_min, color=min_line_color, linestyle='--', linewidth=2, 
                   label=f'Global Min: {global_min:.2f} [{model_with_min}, run{run_with_min}]', alpha=0.7)
        ax.axhline(y=global_max, color=max_line_color, linestyle='--', linewidth=2, 
                   label=f'Global Max: {global_max:.2f} [{model_with_max}, run{run_with_max}]', alpha=0.7)
                          
    # Labels and title
    metric_name = PROBLEM_METRICS.get(problem_name.lower(), "Score")
    problem_display = PROBLEM_NAMES.get(problem_name.lower(), problem_name)

    ax.set_xlabel('Call Number', fontweight='bold')
    ax.set_ylabel(f'{metric_name}', fontweight='bold')
    ax.set_title(f'Best Score Convergence Comparison: {problem_display}',
                 fontweight='bold', pad=20)

    # ax.legend(loc='best', frameon=True, shadow=True, title='Models')
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),   # center, below the plot
        ncol=4,                        # adjust based on number of models
        frameon=True,
        shadow=True,
        title='Models'
    )

    ax.grid(True, alpha=0.3, linestyle='--')

    # Tight layout
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Save
    models_str = '_vs_'.join(sorted(all_model_data.keys()))
    # output_path = output_dir / f"exp1_train_best_scores_{problem_name}_compare_{models_str}.png"
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # print(f"[+] Saved comparative plot to {output_path}")
    
    # Also save as PDF for publications
    # output_path_pdf = output_dir / f"exp1_train_best_scores_{problem_name}_compare_{models_str}.pdf"
    output_path_pdf = output_dir / f"{problem_name}.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"[+] Saved comparative PDF to {output_path_pdf}")

    plt.close()


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(description="Plot training stability from experiment logs.")
    parser.add_argument("--problem_name", type=str, required=True,
                        help="Name of the problem (e.g., tsp, cvrp).")
    
    # Mutually exclusive group for model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_name", type=str,
                             help="Name of a single model to plot.")
    model_group.add_argument("--models", type=str, nargs='+',
                             help="Names of multiple models to compare.")
    
    parser.add_argument("--runs", type=str, nargs='+',
                        help="Specific run numbers to include for each model as space-separated strings. "
                             "For single model: --runs \"1 2 3\". "
                             "For multiple models: --runs \"1 2 3\" \"1 2\" \"1 3 4\" (one string per model in order). "
                             "Number of run strings must match number of models.")
    
    args = parser.parse_args()

    # --- Setup Paths ---
    project_root = Path(__file__).resolve().parent.parent
    print("Project root: ", project_root)
    results_root_dir = project_root / "results" 
    output_dir = project_root / "proc" / args.problem_name / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Process run numbers per model ---
    models_to_process = [args.model_name] if args.model_name else args.models
    
    # Validate and parse run numbers for each model
    if args.runs is None:
        parser.error("--runs argument is required. Provide run numbers for each model.")
    
    if len(args.runs) != len(models_to_process):
        parser.error(f"Number of --runs strings ({len(args.runs)}) must match number of models ({len(models_to_process)}). "
                    f"Provide one space-separated string of run numbers per model.")
    
    # Parse run numbers from strings
    model_runs = {}
    for idx, model in enumerate(models_to_process):
        try:
            run_numbers = [int(r) for r in args.runs[idx].split()]
            model_runs[model] = run_numbers
        except ValueError:
            parser.error(f"Invalid run numbers in argument {idx+1}: '{args.runs[idx]}'. "
                        f"Run numbers must be space-separated integers.")

    # --- Data Loading and Processing ---
    all_model_data = {}
    call_scores_dict = {}  # NEW: Store raw call_scores for extrema computation
    
    for model_name in models_to_process:
        print(f"\n--- Processing model: {model_name} ---")
        model_results_dir = results_root_dir / args.problem_name / model_name
        
        runs_for_model = model_runs[model_name]
        logs, loaded_runs = load_run_logs(model_results_dir, runs_for_model)
        
        if not logs:
            print(f"[!] No logs found for {model_name}. Skipping.")
            continue
            
        call_scores = extract_training_data(logs, args.problem_name)
        calls, means, stds = compute_statistics(call_scores)
        
        all_model_data[model_name] = (calls, means, stds, loaded_runs)
        call_scores_dict[model_name] = call_scores  # NEW: Store for extrema calculation

    # --- Plotting ---
    if not all_model_data:
        print("\n[-] No data was processed. Exiting.")
        return

    if args.model_name:
        # Single model plot - still show global min/max if available
        model_name = args.model_name
        if model_name in all_model_data:
            calls, means, stds, loaded_runs = all_model_data[model_name]
            # For single model, compute global extrema from just this model
            global_min = np.nanmin(means) if not np.all(np.isnan(means)) else None
            global_max = np.nanmax(means) if not np.all(np.isnan(means)) else None
            plot_training_stability_single(calls, means, stds, args.problem_name,
                                           model_name, output_dir, loaded_runs,
                                           global_min, global_max, model_name, model_name)
    elif args.models:
        # Comparative plot
        plot_training_stability_comparative(all_model_data, call_scores_dict, args.problem_name, output_dir)


if __name__ == "__main__":
    main()
