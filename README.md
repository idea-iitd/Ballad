# Ballad
A routing algorithm for LLM-driven automated heuristic discovery pipelines.

## Environment Setup

Create and activate the conda environment from the provided YAML (tested on L40S):

```bash
conda env create -f environment_L40S.yml
conda activate router
```

## Gemini Keys

To use Gemini models, users must create a Google Cloud account and generate Gemini API keys through the Google AI / Vertex AI platform. Since the framework makes a large number of requests to Gemini models during execution, the free tier is typically insufficient; a paid billing-enabled account is required for stable usage and higher rate limits.

After obtaining the API keys, add them to the following configuration file:

```bash
~/cfg/llm_client/gemini.yaml
```

Ensure the keys are added correctly before running the pipeline.

## Training

Training is launched via `src/funsearch.py` using Hydra. The main config is `cfg/config_router_ucb.yaml`.

**Single model (no router):**
```bash
python -m src.funsearch \
    use_router=false \
    active_model=gemini_models.gemini_flash \
    problem=tsp_constructive
```

**Router mode (UCB or Ballad):**
```bash
python -m src.funsearch \
    use_router=true \
    router.strategy=ucb \
    problem=tsp_constructive
```

```bash
python -m src.funsearch \
    use_router=true \
    router.strategy=ballad \
    problem=tsp_constructive
```

**Supported problems:** `tsp_constructive`, `cvrp_aco`, `bpp_online`, `bpp_offline_aco`, `ged`

Override any config value as a Hydra argument (e.g. `router.beta=2.0`, `problem=cvrp_aco`). Results and logs are written to `results<suffix>/` and `logs<suffix>/`.

The optional suffix acts as a custom identifier for a specific experiment variant (e.g., ablations, hyperparameter settings, or repeated studies), allowing outputs to be separated into different directories. If no special variant tracking is needed, multiple runs can still be executed together. In that case, runs should be launched serially: start run i+1 only after the output file for run i has been created inside results<suffix>/<problem>/, preventing conflicts or overlapping writes.

To resume a previous run:
```bash
python -m src.funsearch \
    problem=tsp_constructive \
    use_router=true \
    router.strategy=ballad \
    resume_run_id=<run_id> \
```

## Inference

The inference pipeline runs validation eval → test metric eval and prints avg results across runs.

```bash
./run_inference_pipeline.sh \
    --problem tsp_constructive \
    --models gemini-2.5-flash gemma-3-12b-it gemma-3-27b-it \
    --runs "1 2 3" "1 2 3" "1 2"
```

With optional flags:
```bash
./run_inference_pipeline.sh \
    --problem tsp_constructive \
    --models gemini-2.5-flash gemma-3-12b-it \
    --runs "1 2 3" "1 2 3" \
    --use_parallel \
    --timeout 120
```

- `--use_parallel`: parallelize evaluation across instances. Suitable for CPU bound evaluations.
- `--use_gpu_batch`: batch GPU evaluation. Suitable for GPU bound ACO-based evaluations.
- `--timeout`: per-instance timeout in seconds (default: 60)

Individual inference scripts can also be run directly:
```bash
# Validation eval (selects best checkpoint per run)
python inference/eval_val.py \
    --problem_name tsp_constructive \
    --models gemini-2.5-flash gemma-3-12b-it \
    --runs "1 2 3" "1 2 3"

# Test eval on best-val checkpoint
python inference/eval_test_best_metric.py \
    --problem_name tsp_constructive \
    --models gemini-2.5-flash gemma-3-12b-it \
    --runs "1 2 3" "1 2 3"
```

## Evaluating Best Programs

`best_programs/` contains the best heuristic programs discovered per problem and run, named `{problem}_run{N}.py`. Use `eval_best_programs.py` to evaluate any of these directly on the val and/or test sets without needing training logs.

All scores are reported as the actual objective metric (lower is better), i.e. the cost is negated back from the internal representation.

**Evaluate all programs on both splits:**
```bash
python evals/eval_best_programs.py --all --mode both
```

**Evaluate all programs for one problem:**
```bash
python evals/eval_best_programs.py --problem tsp_constructive --mode test
python evals/eval_best_programs.py --problem cvrp_aco --mode val
```

**Evaluate specific files:**
```bash
python evals/eval_best_programs.py \
    --programs best_programs/tsp_constructive_run1.py \
               best_programs/tsp_constructive_run2.py \
    --mode both
```

**With optional flags:**
```bash
python evals/eval_best_programs.py --all --mode both \
    --use_parallel \
    --timeout 120

# GPU-batched evaluation (ACO-based CVRP and Offline BP only)
python evals/eval_best_programs.py --problem cvrp_aco --mode test --use_gpu_batch
```

The script prints a per-program result followed by a summary table of mean ± std across all evaluated programs per problem and split. Dataset splits and sizes are read from `cfg/problem/{problem}.yaml`.

## Analysis Scripts

**Training curves** — best score vs. call number, averaged across runs:
```bash
# Single model
python evals/best_train_scores.py \
    --problem_name tsp_constructive \
    --model_name gemini-2.5-flash-lite \
    --runs "1 2 3"

# Comparative plot across models
python evals/best_train_scores.py \
    --problem_name tsp_constructive \
    --models gemini-2.5-flash-lite gemini-2.5-flash gemma-3-12b-it \
    --runs "1 2 3" "1 2 3" "1 2 3"

# Batch: all problems and models defined in the script
./plot_best_scores_train.sh
```

Plots are saved to `proc_results/<problem>/plots/`.

**Cost statistics** — token usage and API cost per model/run:
```bash
# Single model
python evals/model_cost_stats.py \
    --problem_name tsp_constructive \
    --model_name gemini-2.5-flash-lite \
    --runs "1 2 3"

# Multiple models
python evals/model_cost_stats.py \
    --problem_name tsp_constructive \
    --models gemini-2.5-flash-lite gemini-2.5-flash ballad \
    --runs "1 2 3" "1 2 3" "1 2 3"
```

Outputs are saved to `proc/<problem>/cost_stats.txt` and `proc/<problem>/cost_stats.json`.

**Cost bar plots** — grid of cost plots across problems:
```bash
python evals/plot_cost_stats.py

python evals/plot_cost_stats.py \
    --problems tsp_constructive cvrp_aco bpp_online bpp_offline_aco ged\
    --out proc/cost_stats_bars.pdf
```

**Router selection share** — stacked-area plot of which LLM Ballad routes to over time:
```bash
python evals/plot_selection_share.py

python evals/plot_selection_share.py --window 50

python evals/plot_selection_share.py --problems tsp_constructive cvrp_aco
```

**Dataset sizes** — print train/val/test instance counts for all problems:
```bash
python print_dataset_sizes.py
```

## Running vLLM Servers (Gemma-3)

The Gemma-3 API endpoint is no longer available. You may use vLLM to serve the models locally before running any training or inference that uses these particular Gemma arms.

We tested the VLLM setup on 2 80 GB A100 GPUs. Create and activate the conda environment from the provided YAML.

```bash
conda env create -f environment_A100.yml
conda activate router
```

**To Serve LLMs via VLLM**:
```bash
chmod +x serve_llm.sh
./serve_llm.sh
```

The `base_url` values in `cfg/llm_client/gemma_vllm.yaml` (`http://127.0.0.1:8000/v1` and `http://127.0.0.1:8001/v1`) must match the ports as per your settings in the serve_llm.sh script. 

## Project Structure

```
cfg/                    Hydra config files (main config + per-LLM and per-problem overrides)
src/                    Core FunSearch + routing implementation
utils/llm_clients/      LLM client wrappers (Gemini, Gemma vLLM, Gemma API)
problems/               Problem specifications and datasets
evals/                  Analysis and plotting scripts
inference/              Val and test evaluation scripts
best_programs/          Best discovered programs per problem/run on val set
```
