import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

from load_data import keys_to_numpy, get_keys_and_indices, remove_invalid
from experiments import *

from config import get_cfg_defaults

cfg = get_cfg_defaults()

experiment_files = {
    '1': "vanilla_HC.yaml",
    '2': "IS_error.yaml",
    '3': "IS_bool.yaml",
    '4': "IS_prob.yaml",
}
experiment_file = experiment_files[sys.argv[1]]


# Get keys and indices
keys, train_indices, val_indices, test_indices = get_keys_and_indices(cfg.DATA.NEON_PATH, cfg.DATA.TRAIN_VAL_TEST, num_tiles=cfg.DATA.NUM_TILES)

# Get train, validation, test sets with metadata
Y_train, M_train = keys_to_numpy(keys, train_indices, cfg.DATA.NEON_PATH, "lai", include_meta=True)
X_train = keys_to_numpy(keys, train_indices, cfg.DATA.NEON_PATH, "sat")
X_train, Y_train, M_train, num_removed = remove_invalid(X_train, Y_train, M_train)
print("Training set size: ")
print("X: ", X_train.shape)
print("Y: ", Y_train.shape)
print("#Removed: ", num_removed)

Y_val, M_val = keys_to_numpy(keys, val_indices, cfg.DATA.NEON_PATH, "lai", include_meta=True)
X_val = keys_to_numpy(keys, val_indices, cfg.DATA.NEON_PATH, "sat")
X_val, Y_val, M_val, num_removed = remove_invalid(X_val, Y_val, M_val)
print("Validation set size: ")
print("X: ", X_val.shape)
print("Y: ", Y_val.shape)
print("#Removed: ", num_removed)

Y_test, M_test = keys_to_numpy(keys, test_indices, cfg.DATA.NEON_PATH, "lai", include_meta=True)
X_test = keys_to_numpy(keys, test_indices, cfg.DATA.NEON_PATH, "sat")
X_test, Y_test, M_test, num_removed = remove_invalid(X_test, Y_test, M_test)
print("Test set size: ")
print("X: ", X_test.shape)
print("Y: ", Y_test.shape)
print("#Removed: ", num_removed)


# HPO

experiment_file = "experiments/" + experiment_file

cfg = get_cfg_defaults()
cfg.merge_from_file(experiment_file)

print("Running HPO for:", cfg.EXP.EXPERIMENT_NAME)
optimise_hyperparameters(cfg, overwrite=True, X_train=X_train, Y_train=Y_train, M_train=M_train, X_val=X_val, Y_val=Y_val, M_val=M_val)
print("Done. ")