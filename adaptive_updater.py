import torch
import numpy as np
import pandas as pd



class Adaptive_Update_Controller():
    """
    controller for adaptive update
    we use a simple heuristic to determine whether to update the model or not, it has following functions:
    1. _init_: initialize the controller
    2. reset: reset the controller
    3. set_stable_record: set the stable record in size of [node_num], 1 means stable, 0 means unstable
    4. get_stable_record: get the stable record
    5. enable: enable the controller, set the stable record to all zeros and reset the stable number
    6. disable: disable the controller, set the stable record to all ones and set the stable number to node_num
    7. is_enabled: check if the controller is enabled
    8. elastic_filter: filter the nodes based on the stable record, return the indices of stable nodes. (and a mapping from old indices to new indices---NO NEED FOR NOW)
    9. elastic_recover: recover the results from the filtered indices, return the new node features---NO NEED FOR NOW

    the controller has following attributes:
    - node_num (int): number of nodes in the graph
    - freeze_threshold (float): threshold to determine whether a node is stable or not, default is 0.9---NO NEED FOR NOW
    - stable_record (np.ndarray): record of stable nodes, size is [node_num], 1 means stable, 0 means unstable
    - stable_num (int): number of stable nodes, initialized to 0
    - enable (bool): whether the controller is enabled or not, initialized to false
    """

    def __init__(self, node_num, freeze_threshold=0.9):
        """
        Args:
            node_num (int): number of nodes in the graph
            freeze_threshold (float): threshold to determine whether a node is stable or not, default is 0.9
        """
        if not isinstance(node_num, int) or node_num <= 0:
            raise ValueError("node_num must be a positive integer.")
        if not (0 <= freeze_threshold <= 1):
            raise ValueError("freeze_threshold must be in the range (-1, 1).")

        self.node_num = node_num
        self.freeze_threshold = freeze_threshold
        self.stable_num = 0
        self.stable_record = np.zeros((node_num,), dtype=np.int8)
        self.enable = False

    def reset(self):
        """
        Reset the controller to its initial state.
        This will set the stable record to all zeros and reset the stable number.
        """
        self.stable_num = 0
        self.stable_record.fill(0)

    def enable(self):
        """
        Enable the controller.
        This will set the stable record to all zeros and reset the stable number.
        """
        self.reset()
        self.enable = True
    
    def disable(self):
        """
        Disable the controller.
        This will set the stable record to all ones and set the stable number to node_num.
        """
        self.reset()
        self.enable = False

    def is_enabled(self):
        """
        Check if the controller is enabled.
        Returns:
            bool: True if the controller is enabled, False otherwise.
        """
        return self.enable

    def set_stable_record(self, stable_record):
        """
        Set the stable record for the controller.
        Args:
            stable_record (np.ndarray): A numpy array of shape [node_num] where each element is either 0 or 1.
                                        1 indicates the node is stable, 0 indicates it is unstable.
        """
        if stable_record.shape[0] != self.node_num:
            raise ValueError("Stable record size does not match node number.")
        self.stable_record = stable_record

    def get_stable_record(self):
        """
        Get the stable record of the controller.
        Returns:
            np.ndarray: A numpy array of shape [node_num] where each element is either 0 or 1.
                        1 indicates the node is stable, 0 indicates it is unstable.
        """
        return self.stable_record

    def elastic_row_filer(self, rows):
        """
        Filter the row indices based on the stable record.
        Args:
            row_indices (pd.dataframe): A pandas dataframe containing the row indices to be filtered.
        Returns:
            pd.DataFrame: A filtered dataframe containing only the stable nodes.
        """
        ##############################
        # TASK0---What you can try at the moment:
        # 1. Use the stable_record to filter the rows.
        # 2. Return a new dataframe with only the stable nodes.
        ##############################

        raise NotImplementedError("Elastic row filter is not implemented yet.")

    def elastic_filter(self, root_node):
        """
        Filter the nodes based on the stable record.
        args:
            root_node (int): The root node that need to be computed by the model.
        Returns:
            tuple: A tuple containing:
                - stable_indices (np.ndarray): Indices of stable nodes.
                - mapping (dict): A mapping from old indices to new indices.
        """
        raise NotImplementedError("Elastic filter is not implemented yet.")

    def elastic_recover(self, filtered_feats, mapping):
        raise NotImplementedError("Elastic recover is not implemented yet.")