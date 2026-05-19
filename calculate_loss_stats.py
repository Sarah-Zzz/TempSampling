import re
import sys
import argparse

def calculate_stats(log_file):
    """
    Calculates statistics of stable and unstable loss from a log file.
    
    The log format is:
    Batch: [batch_id] stable_loss: [mean_loss] stable_cnt: [count] unstable_loss: [mean_loss] unstable_cnt: [count]
    
    The script calculates:
    - avg_stable_loss: Weighted average across all batches (sum of total batch losses / sum of total counts)
    - max_stable_loss: Maximum of batch-level mean losses
    - min_stable_loss: Minimum of non-zero batch-level mean losses
    ... and similarly for unstable losses.
    """
    # Regex to match the log lines
    pattern = re.compile(r"Batch:\s+\d+\s+stable_loss:\s+([\d.]+)\s+stable_cnt:\s+(\d+)\s+unstable_loss:\s+([\d.]+)\s+unstable_cnt:\s+(\d+)")

    total_weighted_stable_loss = 0.0
    total_stable_cnt = 0
    max_stable_loss = -1.0
    min_stable_loss = float('inf')

    total_weighted_unstable_loss = 0.0
    total_unstable_cnt = 0
    max_unstable_loss = -1.0
    min_unstable_loss = float('inf')

    found_any = False

    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    found_any = True
                    s_loss = float(match.group(1))
                    s_cnt = int(match.group(2))
                    u_loss = float(match.group(3))
                    u_cnt = int(match.group(4))

                    # Calculate stats for stable nodes
                    if s_cnt > 0:
                        # s_loss in log is the mean of the batch (from source code).
                        # To get the true global average, we use the weighted sum.
                        total_weighted_stable_loss += s_loss * s_cnt
                        total_stable_cnt += s_cnt
                        
                        if s_loss > max_stable_loss:
                            max_stable_loss = s_loss
                        if s_loss > 0 and s_loss < min_stable_loss:
                            min_stable_loss = s_loss

                    # Calculate stats for unstable nodes
                    if u_cnt > 0:
                        total_weighted_unstable_loss += u_loss * u_cnt
                        total_unstable_cnt += u_cnt
                        
                        if u_loss > max_unstable_loss:
                            max_unstable_loss = u_loss
                        if u_loss > 0 and u_loss < min_unstable_loss:
                            min_unstable_loss = u_loss

    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not found_any:
        print("No matching log lines found in the file.")
        return

    # Results calculation
    avg_stable_loss = total_weighted_stable_loss / total_stable_cnt if total_stable_cnt > 0 else 0.0
    avg_unstable_loss = total_weighted_unstable_loss / total_unstable_cnt if total_unstable_cnt > 0 else 0.0

    # Handle cases where no positive loss was found
    if min_stable_loss == float('inf'): min_stable_loss = 0.0
    if min_unstable_loss == float('inf'): min_unstable_loss = 0.0
    if max_stable_loss == -1.0: max_stable_loss = 0.0
    if max_unstable_loss == -1.0: max_unstable_loss = 0.0

    print(f"avg_stable_loss, {avg_stable_loss:.4f}")
    print(f"max_stable_loss, {max_stable_loss:.4f}")
    print(f"min_stable_loss, {min_stable_loss:.4f}")
    print(f"total_stable_cnt, {total_stable_cnt:.4f}")
    print(f"total_weighted_stable_loss, {total_weighted_stable_loss:.4f}")

    print(f"avg_unstable_loss, {avg_unstable_loss:.4f}")
    print(f"max_unstable_loss, {max_unstable_loss:.4f}")
    print(f"min_unstable_loss, {min_unstable_loss:.4f}")
    print(f"total_unstable_cnt, {total_unstable_cnt:.4f}")
    print(f"total_weighted_unstable_loss, {total_weighted_unstable_loss:.4f}")

    stable_cnt_ratio = total_stable_cnt / (total_stable_cnt + total_unstable_cnt)
    stable_loss_contrib = total_weighted_stable_loss / (total_weighted_stable_loss + total_weighted_unstable_loss)
    print(f"stable_cnt_ratio, {stable_cnt_ratio:.4f}")
    print(f"stable_loss_contrib, {stable_loss_contrib:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 calculate_loss_stats.py <log_file>")
        sys.exit(1)
    
    calculate_stats(sys.argv[1])
