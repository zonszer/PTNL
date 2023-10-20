from os.path import join as pjoin
from os.path import dirname as getdir
from os.path import basename as getbase
from os.path import splitext
from tqdm.auto import tqdm
import math, random
import numpy as np
import time
from glob import glob
import csv
import torch
from typing import List
import os
import re

import torch

class ClassLabelPool:
    """
    Store the average and current values for uncertainty of each class samples and the max capacity of the pool.
    """

    def __init__(self, max_capacity: int, cls_id):
        """
        Initialize the ClassLabelPool.
        Args:
            max_capacity (int): The maximum capacity of the pool.
            items_idx (torch.LongTensor): A tensor of item indices.
            items_unc (torch.Tensor): A tensor of item uncertainties.
        """
        self.pool_max_capacity = max_capacity
        self.is_freeze = False
        self.cls_id = cls_id
        self.device = 'cuda'
        self.unc_dtype = torch.float16
        self.reset()
        
    def _update_pool_attr(self):
        """
        Update the pool attributes.
        """
        # self.unc_avg = torch.mean(self.pool_unc)
        self.unc_max, self.unc_max_idx = torch.max(self.pool_unc, dim=0)
        assert self.pool_unc.shape == self.pool_unc.shape
        assert self.pool_unc.shape[0] <= self.pool_max_capacity
    
    def reset(self):
        """
        Reset the pool.
        """
        self.pool_idx = torch.LongTensor([])
        self.pool_unc = torch.Tensor([]).type(self.unc_dtype).to(self.device)
        self.popped_idx = torch.LongTensor([])
        self.popped_unc = torch.Tensor([]).type(self.unc_dtype).to(self.device)
        #info:
        # self.saved_logits = []
        # self.popped_img_feats = []
        # self.poped_logits = []
        #attribute:
        self.pool_capacity = 0
        self.unc_max = 1e-10
        self.popped_idx_past = None
        self.popped_unc_past = None

        assert self.is_freeze == False
        self.pool_unc_past = None
        self.pool_idx_past = None

    def scale_pool(self, next_capacity: int):
        """
        Enlarge the pool capacity. (if the pool capacity is smaller than the max_num given, remain unchanged)
        Args:
            enlarge_factor (int): The enlarge factor.
        """
        if self.pool_max_capacity <= next_capacity:
            self.pool_max_capacity = next_capacity
        else:
            pass
        return

    def freeze_stored_items(self):
        """
        Freeze the current items in pool. NOTE this method should only be called when the pool is not full.
        which means self.popped_idx.shape[0] is 0
        """
        assert self.popped_idx.shape[0] == 0
        self.pool_unc_past = self.pool_unc
        self.pool_idx_past = self.pool_idx

        # reset the pool：
        self.pool_idx = torch.LongTensor([])
        self.pool_unc = torch.Tensor([]).type(self.unc_dtype).to(self.device)
        self.popped_idx = torch.LongTensor([])
        self.popped_unc = torch.Tensor([]).type(self.unc_dtype).to(self.device)
        self.pool_max_capacity = self.pool_max_capacity - self.pool_capacity
        self.pool_capacity = 0
        self.unc_max = 1e-10

        self.is_freeze = True
    
    def unfreeze_stored_items(self):
        """
        Unfreeze the current items in pool. NOTE this method should only be called when the pool is not full.
        which means self.popped_idx.shape[0] is 0
        """
        assert self.is_freeze == True   
        self.pool_unc = torch.cat((self.pool_unc_past, self.pool_unc), dim=0)
        self.pool_idx = torch.cat((self.pool_idx_past, self.pool_idx), dim=0)

        # reset the pool：
        self.pool_max_capacity = self.pool_max_capacity + self.pool_idx_past.shape[0]
        self.pool_capacity = self.pool_idx.shape[0]
        self.unc_max = None
        self.pool_unc_past = None
        self.pool_idx_past = None

        self.is_freeze = False

    def recalculate_unc(self, logits_all, criterion, cal_poped_items=False):
        """
        Recalculate the uncertainty of the items in the pool.
        """
        # max_val, cav_pred = torch.max(logits_all[self.pool_idx], dim=1)
        unc = criterion(logits_all[self.pool_idx], 
                        torch.LongTensor([self.cls_id]).repeat(self.pool_idx.shape[0]).to(logits_all.device)) 
        self.pool_unc = unc
        if unc.shape[0] == 0:
            return 
        self.unc_max = unc.max()
        
        if cal_poped_items == True and self.popped_idx.shape[0] > 0:
            unc = criterion(logits_all[self.popped_idx], 
                            torch.LongTensor([self.cls_id]).repeat(self.popped_idx.shape[0]).to(logits_all.device)) 
            self.popped_unc = unc


    def pop_notinpool_items(self):
        """
        Get the popped items.
        Returns:
            tuple: A tuple containing the popped items (popped_idx, popped_unc).
        """
        self.popped_idx_past, self.popped_unc_past = self.popped_idx, self.popped_unc
        self.popped_idx = torch.LongTensor([])
        self.popped_unc = torch.Tensor([]).type(self.popped_unc.dtype).to(self.popped_unc.device)

        return self.popped_idx_past, self.popped_unc_past

    def update(self, feat_idx: torch.LongTensor, feat_unc: torch.Tensor, record_popped=True):
        """
        Update the pool with new values.
        Args:
            feat_idxs (torch.Tensor): A tensor of feature indices, better to be ascending order.
            feat_unc (torch.Tensor): A tensor of feature uncertainties.
        """
        if self.pool_capacity < self.pool_max_capacity:
            self.pool_idx = torch.cat((self.pool_idx, feat_idx.unsqueeze(0)))  # Interchanged positions
            self.pool_unc = torch.cat((self.pool_unc, feat_unc.unsqueeze(0)))  # Interchanged positions
            # self.saved_logits = torch.cat((self.saved_logits, feat_logit.unsqueeze(0)))  # Interchanged positions
            self.pool_capacity += 1
            in_pool = True
        else:
            assert self.pool_max_capacity != 0
            if self.unc_max <= feat_unc:
                if record_popped:
                    self.popped_idx = torch.cat((self.popped_idx, feat_idx.unsqueeze(0)))  # Interchanged positions
                    self.popped_unc = torch.cat((self.popped_unc, feat_unc.unsqueeze(0)))  # Interchanged positions
                in_pool = False
            else:
                if record_popped:
                    self.popped_idx = torch.cat((self.popped_idx, self.pool_idx[self.unc_max_idx].unsqueeze(0)))  # Interchanged positions
                    self.popped_unc = torch.cat((self.popped_unc, self.pool_unc[self.unc_max_idx].unsqueeze(0)))  # Interchanged positions
                    # self.popped_img_feats.append(info_dict['image_feat'])      #TODO debug the append is the sam ewith cat
                    # self.poped_logits.append(info_dict['logit'])
                
                self.pool_idx[self.unc_max_idx] = feat_idx
                self.pool_unc[self.unc_max_idx] = feat_unc
                # self.saved_logits[self.unc_max_idx] = feat_logit
                in_pool = True
                
        if in_pool:
            self._update_pool_attr()

        return in_pool

    def __str__(self):
        str_ = ''
        if hasattr(self, 'unc_avg'):
            str_ += f"unc_avg: {self.unc_avg:.4f}, "
        return str_ + f"unc_max: {self.unc_max:.4f}, pool_capacity: {self.pool_capacity}/{self.pool_max_capacity}"
    


def restore_pic(x):
    """
    A function to restore image data from a normalized and rearranged format back to its original format.
    Parameters:
        x (numpy array) : The input image data. Shape is (batch x C x H x W) and values are in range [0,1]
    Returns:
        numpy array : The restored image. Shape is (H x W x C) and values are in range [0,255]
    """
    x = x * 255
    x = x.clamp_(0, 255)
    x = jnp.round(x)
    x = x.astype(jnp.uint8)
    x = jnp.transpose(x, (0, 2, 3, 1))
    return x

def get_regular_weight(class_acc, beta_median:float) -> np.ndarray:
    """
    Read a file and extract accuracy values.
    Args:
        path (str): The path to the file.
    Returns:
        np.ndarray: An array of accuracy values.
    """
    # path = 'zero-shot_testdata_' + dataset + '.txt'
    # acc_values = []
    # with open(path, 'r') as f:
    #     lines = f.readlines()
    # # For each line, find the accuracy value and append it to the list
    # for line in lines:
    #     match = re.search(r'acc: (\d+\.\d+)%', line)
    #     if match:
    #         acc_value = float(match.group(1))
    #         acc_values.append(acc_value)
    acc_array = class_acc.half
    acc_array_ = -(beta_median / torch.median(acc_array)) * acc_array 
    beta_ = acc_array_ + (-acc_array_.min()-acc_array_.max())
    return beta_



def slice_idxlist(start, end, array_length, need_slice_remain=True):
    """
    Returns two lists of indices for the main slice and the remaining slice.
    The main slice contains indices from start to end, wrapping around the array if necessary.
    The remaining slice contains all other indices.

    Args:
    start (int): Start index of the main slice.
    end (int): End index of the main slice.
    array_length (int): Length of the array to be sliced.

    Returns:
    tuple: A tuple containing two lists of indices (main_slice_indices, remaining_slice_indices).
    """

    main_slice_indices = [i % array_length for i in range(start, end)]
    remaining_slice_indices = [i % array_length for i in range(end, start + array_length)]

    return main_slice_indices, remaining_slice_indices


def write_dict_to_csv(data: dict, file_path):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

def become_deterministic(seed=0):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)


def dict_add(dictionary: dict, key, value, acc='list'):
    """
    This function allows the addition of a value to an existing key in a dictionary or initialises a new key with a list or set.
    
    :param dictionary: dict. The main dictionary where the key-value pair will be added.
    :param key: The key which maps to the value.
    :param value: The value to be added for the associated key.
    :param acc: 'list' or 'set'. Determines whether a list or set should be initialized if the key isn't already present in dictionary.
    :return: None. The function modifies the dictionary in-place.
    :raises AssertionError: If acc parameter isn't 'list' or 'set'.
    """
    if key not in dictionary.keys():
        if acc=='list':
            dictionary[key] = []
        elif acc=='set':
            dictionary[key] = set()
        else:
            assert False, 'acc must either be "list" or "set"'
    dictionary[key].append(value)
    
class measure_time:
    """
    This class is used as a context manager to measure the execution time of a block of code.

    When used in a `with` statement, it will print the time elapsed since entering the block upon exiting the block.
    """
    def __init__(self):
        pass
    
    def __enter__(self):
        """
        Start the timer as the context is entered.
        """
        self.start_time = time.time()
        
    def __exit__(self, type, value, traceback):
        """
        Stop the timer as the context is exited and print the time elapsed.
        """
        print('time elapsed', time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time)))


def get_str_after_substring(text:str, substring:str):
    """
    This function gets a substring and the text after it from the original text string.
    
    :param text: str. The original text string from which a substring is to be identified.
    :param substring: str. The substring to be identified in the original text.
    :return: str/None. The substring and the text after it in the original text string, or None if the substring is not in the original text.
    """
    index = text.find(substring)
    if index >= 0:
        next_char = text[index + len(substring):]
        return substring + next_char
    else:
        return None

def fn_comb(kwargs: List):
    def comb(X):
        for fn in kwargs:
            X = fn(X)
        return X
    return comb



class printc:
    """colorful print, but now I want colorul logging to show the message"""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    BLACK = "\033[1;30m"
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    PURPLE = "\033[1;35m"
    CYAN = "\033[1;36m"
    WHITE = "\033[1;37m"

    @staticmethod
    def blue(*text):
        printc.uni(printc.BLUE, text)
    @staticmethod
    def green(*text):
        printc.uni(printc.GREEN, text)
    @staticmethod
    def yellow(*text):
        printc.uni(printc.YELLOW, text)
    @staticmethod
    def red(*text):
        printc.uni(printc.RED, text)
    @staticmethod
    def uni(color, text:tuple):
        print(color + ' '.join([str(x) for x in text]) + printc.END)

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


# utils performed by np
def accuracy(y_pred, y_test):
    """
    This function calculates the accuracy of mean prediction of Gaussian Process
    :param y_pred: np.ndarray. Prediction of Gaussian Process.
    :param y_test: np.ndarray. Ground truth label.
    :return: a float for accuracy.
    """
    return np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y_test, axis=-1))

# Data Loading
def _partial_flatten_and_normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    x = np.reshape(x, (x.shape[0], -1))
    return (x - np.mean(x)) / np.std(x)

def _flatten(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return np.reshape(x, (x.shape[0], -1))/255

def _normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return x / 255


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)