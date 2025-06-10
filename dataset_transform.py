# import argparse
# import csv
# import os
# import pandas as pd
# from tqdm import tqdm


# parser=argparse.ArgumentParser()
# parser.add_argument('--data', type=str, help='dataset name')
# parser.add_argument('--add_reverse', default=False, action='store_true')
# args=parser.parse_args()


# data_dir = os.path.join('DATA', args.data)
# data_fn = os.listdir(data_dir)
# data_fn = [fn for fn in data_fn if fn.endswith('.txt')][0]
# data_fn = os.path.join(data_dir, data_fn)
# print('data_fn: ', data_fn)
# headers = ['src','dst','time','ext_roll']


# df = pd.DataFrame(columns=headers)
# with open(data_fn) as f:
#     lines = f.readlines()
#     # for line in tqdm(lines):
#     for line in lines:
#         src, dst, time = line.split()
#         src = int(src)
#         dst = int(dst)
#         time = int(time)
#         ext_roll = 0
#         # print('src: ', src, 'dst: ', dst, 'time: ', time, 'ext_roll: ', ext_roll)
#         # input("Press Enter to continue...")
#         df.loc[len(df)] = [src, dst, time, ext_roll]
#         if args.add_reverse:
#             df.loc[len(df)] = [dst, src, time, ext_roll]

# df.sort_values(by=['time'], inplace=True)
# train_end = int(len(df) * 0.7)
# val_end = int(len(df) * 0.85)
# df.loc[:train_end, 'ext_roll'] = 0
# df.loc[train_end:val_end, 'ext_roll'] = 1
# df.loc[val_end:, 'ext_roll'] = 2

# df.to_csv(os.path.join(data_dir, 'edges.csv'), index=True)

import argparse
import csv
import os
import pandas as pd
from tqdm import tqdm

# Argument parsing setup
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--add_reverse', default=False, action='store_true')
args = parser.parse_args()

# Construct data file path
data_dir = os.path.join('DATA', args.data)
data_fn = [fn for fn in os.listdir(data_dir) if fn.endswith('.txt')][0]
data_fn = os.path.join(data_dir, data_fn)
print('data_fn: ', data_fn)

# Prepare to load data
headers = ['src', 'dst', 'time', 'ext_roll']
data = []

# Efficient data loading
with open(data_fn) as f:
    for line in tqdm(f, desc="Reading file"):
        src, dst, time = map(int, line.split())
        data.append((src, dst, time, 0))
        if args.add_reverse:
            data.append((dst, src, time, 0))

# Create DataFrame from list
df = pd.DataFrame(data, columns=headers)

# Sort and split data
df.sort_values(by=['time'], inplace=True)
train_end = int(len(df) * 0.7)
val_end = int(len(df) * 0.85)
df['ext_roll'] = 2  # Default to 2
df.loc[:train_end, 'ext_roll'] = 0  # Set training data
df.loc[train_end:val_end, 'ext_roll'] = 1  # Set validation data

# Save to CSV
df.to_csv(os.path.join(data_dir, 'edges.csv'), index=True)
