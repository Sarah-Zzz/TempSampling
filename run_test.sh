#!/bin/bash

# Parse command line arguments
dataset=""
config=""
suffix=""
exp="both" # Default to 'both'

train_simple_args=() # Initialize an empty array for arguments to train_simple.py

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --) # Custom arguments for train_simple.py
            shift
            train_simple_args=("$@")
            break
            ;;
        --dataset) dataset="$2"; shift ;;
        --config) config="$2"; shift ;;
        --suffix) suffix="$2"; shift ;;
        --exp) exp="$2"; shift ;; # Parse --exp argument
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate arguments
if [ -z "$dataset" ] || [ -z "$config" ] || [ -z "$suffix" ]; then
    echo "Usage: $0 --dataset <dataset_name> --config <config_name> --suffix <suffix_value> [--exp <experiment_type>]"
    echo "  <dataset_name> can be: WIKI, WIKI_TALK, STACK_OVERFLOW, REDDIT, GDELT"
    echo "  <config_name> can be: TGN, APAN, TGAT"
    echo "  <experiment_type> can be: base, psfilter, both. Default is 'both'."
    exit 1
fi

# Validate exp argument
allowed_exps=("base" "psfilter" "both")
if ! [[ " ${allowed_exps[@]} " =~ " ${exp} " ]]; then
    echo "Error: Invalid experiment type '$exp'. Must be one of: ${allowed_exps[*]}"
    exit 1
fi

# Convert dataset and config to uppercase for validation
UPPER_DATASET=$(echo "$dataset" | tr '[:lower:]' '[:upper:]')
UPPER_CONFIG=$(echo "$config" | tr '[:lower:]' '[:upper:]')

# Allowed datasets and configs
allowed_datasets=("WIKI" "WIKI_TALK" "STACK_OVERFLOW" "REDDIT" "GDELT")
allowed_configs=("TGN" "APAN" "TGAT" "JODIE")

if ! [[ " ${allowed_datasets[@]} " =~ " ${UPPER_DATASET} " ]]; then
    echo "Error: Invalid dataset '$dataset'. Must be one of: ${allowed_datasets[*]}"
    exit 1
fi

# if ! [[ " ${allowed_configs[@]} " =~ " ${UPPER_CONFIG} " ]]; then
#     echo "Error: Invalid config '$config'. Must be one of: ${allowed_configs[*]}"
#     exit 1
# fi

# Construct config file path
config_file="./config/post_sample_filter/${UPPER_DATASET}/${UPPER_CONFIG}.yml"

# Check if config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file not found at '$config_file'"
    exit 1
fi

# Create results directory
results_dir="results_${suffix}"
[ -d "$results_dir" ] || mkdir -p "$results_dir"

# Process dataset name for filenames
lower_dataset_raw=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
lower_dataset=${lower_dataset_raw//_/} # Remove underscores

lower_config=$(echo "$config" | tr '[:upper:]' '[:lower:]')

echo "Starting runs for dataset: $dataset, config: $config, suffix: $suffix, experiment: $exp"
echo "Results will be saved in: $results_dir"
echo "Using config file: $config_file"

# echo "Activating environment"
# set -x
# conda activate /ihome/yzhang/sarahz/miniconda3/envs/cascade
# set +x

# --- Baseline Run ---
if [[ "$exp" == "base" || "$exp" == "both" ]]; then
    echo "--- Running Baseline ---"
    base_log_file="${results_dir}/${lower_dataset}_${lower_config}_base_${suffix}.log"
    base_output_log_file="${results_dir}/output_${lower_dataset}_${lower_config}_base_${suffix}.log"

    date | tee -a "${base_output_log_file}"
    echo "Command: python train_simple.py --extra_config ${config_file} --logfile ${base_log_file} ${train_simple_args[@]} 2>&1 | tee ${base_output_log_file}"
    set -x
    python train_simple.py --extra_config "${config_file}" --logfile "${base_log_file}" "${train_simple_args[@]}" 2>&1 | tee -a "${base_output_log_file}"
    set +x
    echo "Baseline run completed. Logs: ${base_log_file}, ${base_output_log_file}"
    date | tee -a "${base_output_log_file}"
fi

# --- Post-Sample-Filter Run ---
if [[ "$exp" == "psfilter" || "$exp" == "both" ]]; then
    echo "--- Running Post-Sample-Filter ---"
    psfilter_log_file="${results_dir}/${lower_dataset}_${lower_config}_psfilter_${suffix}.log"
    psfilter_output_log_file="${results_dir}/output_${lower_dataset}_${lower_config}_psfilter_${suffix}.log"

    date | tee -a "${psfilter_output_log_file}"
    echo "Command: python train_simple.py --extra_config ${config_file} --logfile ${psfilter_log_file} --post_sample_filter ${train_simple_args[@]} 2>&1 | tee ${psfilter_output_log_file}"
    set -x
    python train_simple.py --extra_config "${config_file}" --logfile "${psfilter_log_file}" --post_sample_filter "${train_simple_args[@]}" 2>&1 | tee -a "${psfilter_output_log_file}"
    set +x
    echo "Post-Sample-Filter run completed. Logs: ${psfilter_log_file}, ${psfilter_output_log_file}"
    date | tee -a "${psfilter_output_log_file}"
fi

echo "All runs finished."
