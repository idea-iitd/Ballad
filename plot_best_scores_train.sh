#!/bin/bash

# Configuration structure: Each problem has models with their specific runs
# Format: PROBLEM -> MODEL1:RUNS1 MODEL2:RUNS2 ...

set -e  # Exit on error

# Define problems with model:runs pairs
# Format: "model1:run1 run2 run3|model2:run1 run2|..."

declare -A PROBLEM_CONFIG=(
    ["tsp_constructive"]="gemini-2.5-flash-lite: 1 2 3|gemini-2.5-flash: 2 4 5|gemma-3-12b-it: 1 2 3|gemma-3-27b-it: 1 2 3|ballad: 1 2 5|ucb: 1 2 3|"
    ["bpp_offline_aco"]="gemini-2.5-flash-lite: 1 2 3|gemini-2.5-flash: 1 2 3|gemma-3-12b-it: 1 2 3|gemma-3-27b-it: 1 2 3|ballad: 1 2 3|ucb: 1 2 3"
    ["bpp_online"]="gemini-2.5-flash-lite: 1 2 3|gemini-2.5-flash: 1 2 3|gemma-3-12b-it: 1 2 3|gemma-3-27b-it: 1 2 3|ballad: 1 2 3|ucb: 1 2 3"
    ["cvrp_aco"]="gemini-2.5-flash-lite: 1 2 3|gemini-2.5-flash: 1 2 3|gemma-3-12b-it: 1 2 3|gemma-3-27b-it: 1 2 3|ballad: 1 2 3|ucb: 1 2 3"
    ["ged"]="gemini-2.5-flash-lite: 1 2 4|gemini-2.5-flash: 1 2 3|gemma-3-12b-it: 1 2 3|gemma-3-27b-it: 1 2 3|ballad: 1 2 3|ucb: 1 2 3"
)

# Project root (adjust if needed)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Script path
SCRIPT_PATH="$PROJECT_ROOT/evals/best_train_scores.py"

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[-] Error: Script not found at $SCRIPT_PATH"
    exit 1
fi

echo "=========================================="
echo "Best Scores Training Plot Generation"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Script: $SCRIPT_PATH"
echo ""

# Track statistics
TOTAL_PROBLEMS=0
SUCCESSFUL_PROBLEMS=0
FAILED_PROBLEMS=0

# Iterate through each problem
for problem in "${!PROBLEM_CONFIG[@]}"; do
    TOTAL_PROBLEMS=$((TOTAL_PROBLEMS + 1))
    
    echo ""
    echo "=========================================="
    echo "Processing Problem: $problem"
    echo "=========================================="
    
    # Get configuration for this problem
    config=${PROBLEM_CONFIG[$problem]}
    
    # Parse model:runs pairs
    models=()
    runs_args=()
    
    IFS='|' read -ra MODEL_PAIRS <<< "$config"
    for pair in "${MODEL_PAIRS[@]}"; do
        IFS=':' read -r model runs <<< "$pair"
        models+=("$model")
        runs_args+=("$runs")
        echo "  Model: $model -> Runs: $runs"
    done
    
    echo ""
    echo "[*] Generating comparative plot across ${#models[@]} models..."
    echo "[*] Command: python $SCRIPT_PATH --problem_name $problem --models ${models[@]} --runs ${runs_args[@]}"
    echo ""
    
    # Run comparative plot
    if python "$SCRIPT_PATH" \
        --problem_name "$problem" \
        --models "${models[@]}" \
        --runs "${runs_args[@]}"; then
        
        echo "[+] Successfully generated comparative plot for $problem"
        SUCCESSFUL_PROBLEMS=$((SUCCESSFUL_PROBLEMS + 1))
    else
        echo "[-] Failed to generate comparative plot for $problem"
        FAILED_PROBLEMS=$((FAILED_PROBLEMS + 1))
        continue
    fi
    
    echo ""
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total Problems: $TOTAL_PROBLEMS"
echo "Successful: $SUCCESSFUL_PROBLEMS"
echo "Failed: $FAILED_PROBLEMS"
echo ""

if [ $FAILED_PROBLEMS -eq 0 ]; then
    echo "[+] All plots generated successfully!"
    exit 0
else
    echo "[-] Some plots failed to generate."
    exit 1
fi