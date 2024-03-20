import random
import logging

import numpy as np


def get_spaced_randoms(n, start, end, m):
    """
    Generate n unique random numbers between start and end, 
    where each number is at least m steps away from the others.

    Args:
    n (int): The number of random numbers to generate.
    start (int): The start of the range in which to generate numbers.
    end (int): The end of the range in which to generate numbers.
    m (int): The minimum distance between any two numbers.

    Returns:
    list of int: The list of generated random numbers.
    """
    possible_nums = list(range(start, end + 1, m))
    if len(possible_nums) < n:
        raise ValueError(f"Cannot generate {n} numbers in the range {start}-{end} that are at least {m} apart.")
    random.seed(0)
    random_nums = random.sample(possible_nums, n)
    logging.info(f"[random_nums[0] : {random_nums}] Generated {n} random numbers in the range {start}-{end} that are at least {m} apart.")
    random_nums.sort()

    return random_nums


def split_indices(num_timestamps_train, test_size, ws, test_cont_ws=100):
    """
    Split the indices of the training set into train and test sets so that none of the test indices are within ws of train.
    """
    all_indices = set(range(num_timestamps_train))

    # Calculate the number of test samples
    num_test_samples = int(num_timestamps_train * test_size)

    test_start_indices = get_spaced_randoms(num_test_samples//test_cont_ws, 0, num_timestamps_train-test_cont_ws, ws)
    test_indices = []
    for i in test_start_indices:
        test_indices.extend(list(range(i, i+test_cont_ws)))

    train_indices = list(all_indices - set(test_indices))

    return sorted(train_indices), sorted(list(test_indices))


def get_sequences(data, label, fault_labels=None, win_size=32, horizon=1):
    # k, n, nf
    n_samples = data.shape[0]
    
    x_offsets = np.arange(-win_size+1, 1)
    y_offsets = np.arange(1, 1+horizon)
    
    xs, ys, ts, fl = [], [], [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(n_samples - abs(max(y_offsets)))  # Exclusive    
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        label_t = max(label[t + x_offsets]) # if any then marked as abnormal
        if fault_labels is not None:
            fault_label_t = max(fault_labels[t + x_offsets]) # if any then marked as abnormal
            fl.append(fault_label_t)
        xs.append(x_t)
        ys.append(y_t)
        ts.append(label_t)
    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    Label = np.stack(ts, axis=0)
    if fault_labels is not None:
        Fault = np.stack(fl, axis=0)
        return X, Y, Label, Fault
    return X, Y, Label, None


def get_discountinous_sequences(
        data, 
        label, 
        fault_labels=None, 
        sample_nums=None,
        win_size=15, 
        horizon=1, 
        stride=1
    ):
    n = data.shape[0]
    
    X_seq = []
    Y_seq = []
    Label_seq = []
    Fault_labels = []

    i = 0
    while i <= (n - win_size - horizon):
        # Check if all labels in the window are the same
        seq_has_same_label = len(set(label[i:i+win_size+horizon])) == 1
        if fault_labels is not None:
            seq_has_same_label = len(set(fault_labels[i:i+win_size+horizon])) == 1
            
        # Check if all sample numbers in the window are the same, if sample_nums is provided
        seq_has_same_sample = True  # Default to True
        if sample_nums is not None:
            seq_has_same_sample = len(set(sample_nums[i:i+win_size+horizon])) == 1
        
        if seq_has_same_label and seq_has_same_sample:
            X_seq.append(data[i:i+win_size])
            Y_seq.append(data[i+win_size:i+win_size+horizon])
            Label_seq.append(label[i])
            if fault_labels is not None:
                Fault_labels.append(fault_labels[i])
            i += stride
        else:
            # Skip ahead to the next non-overlapping window
            i += win_size

    Fault_labels = np.array(Fault_labels) if fault_labels is not None else None
    return np.array(X_seq), np.array(Y_seq), np.array(Label_seq), Fault_labels
