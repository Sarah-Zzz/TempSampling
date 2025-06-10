import torch
import numpy as np
import pickle


FREEZE_THRESHOLD = 0.9
profile_dict = dict()
prev_profile_dict = dict()
node_update_batch_map = dict()
similarity = dict()

def load_similarities(similarity_path):
    global similarity
    print("Loading similarity from {}".format(similarity_path))
    with open(similarity_path, 'rb') as f:
        similarity = pickle.load(f)
    return similarity

def generate_update_batch(group_index, train_edge_end,df):
    global node_update_batch_map

    for i, rows in df[:train_edge_end].groupby(group_index):
        root_nodes = np.concatenate([rows.src.values, rows.dst.values,]).astype(np.int32)
        for node in root_nodes:
            if node not in node_update_batch_map:
                node_update_batch_map[node] = []
            node_update_batch_map[node].append(i)
    return node_update_batch_map

def reset_node_update_batch_map():
    global node_update_batch_map
    node_update_index_dict = dict()

def inter_batch_has_stable_input(src_idx, dst_idx, row_idx, sim_batch_size=900, threshold=0.9):
    """
    based on similarity, check if the input of src_idx and dst_idx is stable
    """
    global node_update_batch_map, similarity
    # which batch the update is in during profiling
    orig_batch_idx = row_idx // sim_batch_size
    if orig_batch_idx == 0:
        return True

    src_sim = similarity[src_idx]
    dst_sim = similarity[dst_idx]
    # which update this event is in
    src_cur_version_idx = node_update_batch_map[src_idx].index(orig_batch_idx)
    dst_cur_version_idx = node_update_batch_map[dst_idx].index(orig_batch_idx)

    # check if the input of src_idx and dst_idx is stable
    src_update_sim = similarity[src_idx][src_cur_version_idx-1]
    dst_update_sim = similarity[dst_idx][dst_cur_version_idx-1]

    if src_update_sim > threshold and dst_update_sim > threshold:
        return True
    else:
        return False



class Bifrost:
    """
    Bifrost is a tool to schedule the batch update of nodes in a dynamic graph.
    It is based on two criteria:
        1. the depedenency of updating events---coloring and scheduling
        2. the stability of the input of the updating events---freezing
    """
    def __init__(self, 
                init_batch_size=900, 
                recent_dependency=4, 
                history_window=4,
                color_threshold=0.9, 
                sim_threshold=0.9,
                node_num=1000,
                memory_size=1000,
                ):
        
        self.init_batch_size = init_batch_size

        self.color_threshold = color_threshold
        self.sim_threshold = sim_threshold
        self.recent_dependency = recent_dependency
        self.history_window = history_window

        