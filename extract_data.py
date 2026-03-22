#!/usr/bin/env python3
import sys
import re
import argparse

def extract_stable_rate(input_file):
    pattern = re.compile(r"Total number of nodes:\s*(\d+),\s*number of stable nodes:\s*(\d+),\s*percentage of stable nodes:\s*([\d.]+)%")
    accumulated_nodes = 0
    accumulated_stable = 0
    count = 0
    try:
        with open(input_file, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    num_nodes = int(match.group(1))
                    num_stable_nodes = int(match.group(2))
                    accumulated_nodes += num_nodes
                    accumulated_stable += num_stable_nodes
                    count += 1
    except FileNotFoundError:
        return ""

    if accumulated_nodes > 0:
        overall_stable_rate = (accumulated_stable / accumulated_nodes) * 100
        return f"{overall_stable_rate:.4f}"
    return ""

def main():
    parser = argparse.ArgumentParser(description='Extract data from log file and print in CSV format.')
    parser.add_argument('-i', '--input', required=True, help='Input log file')
    args = parser.parse_args()

    input_file = args.input

    # Data points to extract (initializing with empty strings or lists for 'last one' logic)
    date = ""
    model = ""
    dataset = ""
    mode = ""
    num_epochs = ""
    pre_and_post_filters = ""
    training_time = ""
    events_sampled = ""
    best_auc = ""
    avg_val_loss = ""
    min_val_loss = ""
    avg_stable_rate = extract_stable_rate(input_file)
    max_bs = ""
    similarity = ""
    history = ""
    tot_time_to_dgl_blocks_cuda = ""

    # Regex patterns
    model_pattern = re.compile(r"config/eval/([A-Z]+)_")
    dataset_pattern = re.compile(r"'data':\s*'([^']+)'")
    mode_pattern = re.compile(r"mode\s+(\S+)")
    epoch_pattern = re.compile(r"'epoch':\s*(\d+)")
    # post_sample_filter_pattern = re.compile(r"post-sample filter:\s*(\S+)")
    post_sample_filter_pattern = re.compile(r"pre_and_post_filters:\s*(\S+)")
    net_training_time_pattern = re.compile(r"net_training_time:\s*([\d.]+)")
    total_training_time_pattern = re.compile(r"Total training time:\s*([\d.]+)")
    events_sampled_pattern = re.compile(r"total_edges_sampled:\s*(\d+)")
    best_auc_pattern = re.compile(r"Best AUC:\s*([\d.]+)")
    avg_val_loss_pattern = re.compile(r"ave val loss\s*([\d.]+)")
    min_val_loss_pattern = re.compile(r"min val loss\s*([\d.]+)")
    max_bs_pattern1 = re.compile(r"max_batch_size:\s*(\d+)")
    max_bs_pattern2 = re.compile(r"max_batch_size overriden by command line:\s*(\d+)")
    max_bs_pattern3 = re.compile(r"'max_batch_size':\s*(\d+)")
    similarity_pattern1 = re.compile(r"adaptive_update_similarity:\s*([\d.]+)")
    similarity_pattern2 = re.compile(r"'adaptive_update_similarity'\s*=\s*([\d.]+)")
    history_pattern1 = re.compile(r"history:\s*(\d+)")
    history_pattern2 = re.compile(r"overriding history:\s*(\d+)")
    history_pattern3 = re.compile(r"'history':\s*(\d+)")
    tot_time_to_dgl_blocks_cuda_pattern = re.compile(r"tot_time_to_dgl_blocks_cuda:\s*([\d.]+)")

    try:
        with open(input_file, 'r') as f:
            for line in f:
                # model
                m = model_pattern.search(line)
                if m: model = m.group(1)
                
                # dataset
                m = dataset_pattern.search(line)
                if m: dataset = m.group(1)

                # mode
                m = mode_pattern.search(line)
                if m: mode = m.group(1)
                
                # num_epochs
                m = epoch_pattern.search(line)
                if m: num_epochs = m.group(1)

                # pre_and_post_filters
                m = post_sample_filter_pattern.search(line)
                if m: pre_and_post_filters = m.group(1)
                
                # training_time
                m1 = net_training_time_pattern.search(line)
                if m1: 
                    training_time = m1.group(1)
                else:
                    m2 = total_training_time_pattern.search(line)
                    if m2: training_time = m2.group(1)
                
                # events_sampled
                m = events_sampled_pattern.search(line)
                if m: events_sampled = m.group(1)
                
                # best_auc
                m = best_auc_pattern.search(line)
                if m: best_auc = m.group(1)
                
                # avg_val_loss
                m = avg_val_loss_pattern.search(line)
                if m: avg_val_loss = m.group(1)
                
                # min_val_loss
                m = min_val_loss_pattern.search(line)
                if m: min_val_loss = m.group(1)
                
                # max_bs
                m = max_bs_pattern1.search(line)
                if m: max_bs = m.group(1)
                m = max_bs_pattern2.search(line)
                if m: max_bs = m.group(1)
                m = max_bs_pattern3.search(line)
                if m: max_bs = m.group(1)
                
                # similarity
                m = similarity_pattern1.search(line)
                if m: similarity = m.group(1)
                m = similarity_pattern2.search(line)
                if m: similarity = m.group(1)
                
                # history
                m = history_pattern1.search(line)
                if m: history = m.group(1)
                m = history_pattern2.search(line)
                if m: history = m.group(1)
                m = history_pattern3.search(line)
                if m: history = m.group(1)

                # tot_time_to_dgl_blocks_cuda
                m = tot_time_to_dgl_blocks_cuda_pattern.search(line)
                if m: tot_time_to_dgl_blocks_cuda = m.group(1)

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    # Output CSV format
    fields = [
        # date,
        model,
        dataset,
        mode,
        num_epochs,
        pre_and_post_filters,
        training_time,
        events_sampled,
        best_auc,
        avg_val_loss,
        min_val_loss,
        avg_stable_rate,
        max_bs,
        similarity,
        history,
        tot_time_to_dgl_blocks_cuda,
        input_file
    ]
    print(",".join(fields))

if __name__ == "__main__":
    main()
