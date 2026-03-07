#!/usr/bin/env python3
import sys
import re

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_stable_rate.py <log_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    
    # Pattern to match lines like:
    # Total number of nodes: 9228, number of stable nodes: 150, percentage of stable nodes: 1.63%
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
                    # perc_stable = float(match.group(3)) # Extracted but not explicitly used for accumulation
                    
                    accumulated_nodes += num_nodes
                    accumulated_stable += num_stable_nodes
                    count += 1
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    if count == 0:
        print("No matching lines found.")
        return

    print(f"Total lines processed: {count}")
    print(f"Accumulated total_nodes: {accumulated_nodes}")
    print(f"Accumulated total_stable: {accumulated_stable}")
    
    if accumulated_nodes > 0:
        # The user requested total_nodes / total_stable, but usually stable rate is total_stable / total_nodes.
        # We will compute the percentage as (total_stable / total_nodes) * 100.
        overall_stable_rate = (accumulated_stable / accumulated_nodes) * 100
        print(f"Overall stable rate (percentage): {overall_stable_rate:.4f}%")
    else:
        print("Accumulated total_nodes is zero, cannot compute rate.")

if __name__ == "__main__":
    main()
