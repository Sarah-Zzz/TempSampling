import re
import sys


def parse_log(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    time_pat = re.compile(
        r'total time:\s*([\d.]+)s\s+'
        r'sample time:\s*([\d.]+)s\s+'
        r'prep time:\s*([\d.]+)s\s+'
        r'model time:\s*([\d.]+)s'
    )
    epoch_stable_pat = re.compile(
        r'\[epoch stats\]\s+'
        r'stable_flag_update_time:\s*([\d.]+)s,\s+'
        r'stable_flag_get_time:\s*([\d.]+)s'
    )
    chunk_stable_pat = re.compile(
        r'\[chunk stats\]\s+'
        r'stable_flag_update_time:\s*([\d.]+)s,\s+'
        r'stable_flag_get_time:\s*([\d.]+)s'
    )
    epoch_pat = re.compile(r'>>>>>> Epoch (\d+):')

    epoch_data = {}
    current_epoch = -1

    for line in lines:
        ep = epoch_pat.search(line)
        if ep:
            current_epoch = int(ep.group(1))
            if current_epoch not in epoch_data:
                epoch_data[current_epoch] = {
                    'update_time': 0.0,
                    'get_time': 0.0,
                }

        tm = time_pat.search(line)
        if tm and current_epoch >= 0:
            d = epoch_data[current_epoch]
            d['total_time'] = float(tm.group(1))
            d['sample_time'] = float(tm.group(2))
            d['prep_time'] = float(tm.group(3))
            d['model_time'] = float(tm.group(4))

        sm = epoch_stable_pat.search(line)
        if sm and current_epoch >= 0:
            d = epoch_data[current_epoch]
            d['update_time'] = float(sm.group(1))
            d['get_time'] = float(sm.group(2))

        sc = chunk_stable_pat.search(line)
        if sc and current_epoch >= 0:
            d = epoch_data[current_epoch]
            d['update_time'] += float(sc.group(1))
            d['get_time'] += float(sc.group(2))

    epochs = []
    for eid in sorted(epoch_data.keys()):
        d = epoch_data[eid]
        if 'total_time' in d:
            d['epoch'] = eid
            epochs.append(d)

    return epochs


def is_base_run(logfile):
    """Check if all filter flags are False (base run without filtering)."""
    with open(logfile) as f:
        for line in f:
            line_lower = line.lower()
            if 'pre_and_post_filters' in line_lower or \
               'psf_identity' in line_lower or \
               'row_filter_only' in line_lower or \
               'sample_filter_only' in line_lower:
                val = line.split(':')[-1].strip()
                if val.lower() == 'true':
                    return False
    return True


def compute_stats(epochs):
    rows = []
    for ep in epochs:
        flag_sum = ep['update_time'] + ep['get_time']
        pct_prep = (flag_sum / ep['prep_time'] * 100) if ep['prep_time'] > 0 else 0
        pct_total = (flag_sum / ep['total_time'] * 100) if ep['total_time'] > 0 else 0
        rows.append((ep['epoch'], ep['total_time'], ep['sample_time'],
                      ep['prep_time'], ep['model_time'],
                      ep['update_time'], ep['get_time'],
                      flag_sum, pct_prep, pct_total))
    return rows


def print_table(epochs, base=False):
    if base:
        header = (
            f"{'epoch':>5s}  "
            f"{'total':>8s}  {'sample':>8s}  {'prep':>8s}  {'model':>8s}"
        )
    else:
        header = (
            f"{'epoch':>5s}  "
            f"{'total':>8s}  {'sample':>8s}  {'prep':>8s}  {'model':>8s}  "
            f"{'upd_time':>9s}  {'get_time':>9s}  {'upd+get':>9s}  "
            f"{'upd+get/prep':>11s}  {'upd+get/total':>12s}"
        )
    sep = '-' * len(header)
    rows = compute_stats(epochs)

    print(sep)
    print(header)
    print(sep)
    for r in rows:
        if base:
            print(f"{r[0]:5d}  {r[1]:8.4f}  {r[2]:8.4f}  {r[3]:8.4f}  {r[4]:8.4f}")
        else:
            print(
                f"{r[0]:5d}  "
                f"{r[1]:8.4f}  {r[2]:8.4f}  {r[3]:8.4f}  {r[4]:8.4f}  "
                f"{r[5]:9.6f}  {r[6]:9.6f}  {r[7]:9.6f}  "
                f"{r[8]:10.2f}%  {r[9]:11.2f}%"
            )
    print(sep)

    if rows:
        avg = tuple(sum(r[i] for r in rows) / len(rows) for i in range(1, 10))
        if base:
            print(f"{'avg':>5s}  {avg[0]:8.4f}  {avg[1]:8.4f}  {avg[2]:8.4f}  {avg[3]:8.4f}")
        else:
            print(
                f"{'avg':>5s}  "
                f"{avg[0]:8.4f}  {avg[1]:8.4f}  {avg[2]:8.4f}  {avg[3]:8.4f}  "
                f"{avg[4]:9.6f}  {avg[5]:9.6f}  {avg[6]:9.6f}  "
                f"{avg[7]:10.2f}%  {avg[8]:11.2f}%"
            )
        print(sep)


def print_csv(epochs, base=False):
    rows = compute_stats(epochs)
    out = sys.stdout

    if base:
        header = "epoch,total_time,sample_time,prep_time,model_time"
        print(header, file=out)
        for r in rows:
            print(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}", file=out)
    else:
        header = (
            "epoch,total_time,sample_time,prep_time,model_time,"
            "stable_flag_update_time,stable_flag_get_time,upd_get_sum,"
            "upd_get_prep_pct,upd_get_total_pct"
        )
        print(header, file=out)
        for r in rows:
            print(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},"
                  f"{r[5]:.6f},{r[6]:.6f},{r[7]:.6f},{r[8]:.2f},{r[9]:.2f}", file=out)

    if rows:
        avg = tuple(sum(r[i] for r in rows) / len(rows) for i in range(1, (5 if base else 10)))
        if base:
            print(f"avg,{avg[0]:.4f},{avg[1]:.4f},{avg[2]:.4f},{avg[3]:.4f}", file=out)
        else:
            print(f"avg,{avg[0]:.4f},{avg[1]:.4f},{avg[2]:.4f},{avg[3]:.4f},"
                  f"{avg[4]:.6f},{avg[5]:.6f},{avg[6]:.6f},{avg[7]:.2f},{avg[8]:.2f}", file=out)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} [--csv] <logfile>")
        sys.exit(1)

    csv_mode = '--csv' in sys.argv
    log_path = [a for a in sys.argv[1:] if not a.startswith('--')][0]

    epochs = parse_log(log_path)
    if not epochs:
        print("No timing data found in log file.")
        sys.exit(1)

    base = is_base_run(log_path)

    if csv_mode:
        print_csv(epochs, base=base)
    else:
        print_table(epochs, base=base)
