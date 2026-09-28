import re
import csv
import sys

epoch_pat = re.compile(r'>>>>>> Epoch (\d+):')
loss_pat = re.compile(
    r'train loss:([\d.]+).*?final_val loss:([\d.]+)'
)

def extract(log_path, out):
    rows = []
    current_epoch = None
    with open(log_path) as f:
        for line in f:
            m = epoch_pat.search(line)
            if m:
                current_epoch = int(m.group(1))
                continue
            m = loss_pat.search(line)
            if m and current_epoch is not None:
                train_loss = float(m.group(1))
                final_val_loss = float(m.group(2))
                # rows.append((train_loss, final_val_loss))
                rows.append((train_loss,))
    w = csv.writer(out)
    # w.writerow(['train_loss', 'final_val_loss'])
    w.writerows(rows)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python extract_loss.py <log_file>')
        sys.exit(1)
    log_path = sys.argv[1]
    extract(log_path, sys.stdout)
