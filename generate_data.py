import numpy as np

import os
import sys

from src.AutoIPQ import *
from shared_utils import set_default_parameters, load_data_synth, instantiate_sim, generate_lr_data


#######################################
# Setup
#######################################

project_dir = "/scratch/arp/AutoIPQ/"
num_instances = {"train":20000, "val":1000, "test":1000}
#num_instances = {"train":50, "val":50, "test":50}
overwrite = False


sim_params = {
    "prosail":  {
                        "parameters":['lai', 'cab', 'lidfa2', 'cw'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "prosail_2d":  {
                        "parameters":['lai', 'cab'],
                        "sim_parameters":['tts', 'tto', 'psi'],
                        "simdata_num_samples":1,
                    },
    "prosail_sim":  {
                        "parameters":['lai', 'cab', 'hspot', 'n', 'lidfa2', 'cw', 'cm'],
                        "sim_parameters":['tts', 'tto', 'psi'],
                        "simdata_num_samples":1,
                    },
    "prosail_real": {
                        "parameters":['lai', 'cab', 'hspot', 'n', 'lidfa2', 'cw', 'cm'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "mdgs1":  {
                        "parameters":['mu0', 'sigma0'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "mdgs2":  {
                        "parameters":['mu0', 'sigma0', "mu1", "sigma1"],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "mdgs5":  {
                        "parameters":sum([["mu" + str(i), "sigma" + str(i)] for i in range(5)], []),
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "mdgs10":  {
                        "parameters":sum([["mu" + str(i), "sigma" + str(i)] for i in range(10)], []),
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "mdgs20":  {
                        "parameters":sum([["mu" + str(i), "sigma" + str(i)] for i in range(20)], []),
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "mdgs50":  {
                        "parameters":sum([["mu" + str(i), "sigma" + str(i)] for i in range(50)], []),
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "lr2d":  {
                        "parameters":['w1', 'w2', 'b'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "lr5d":  {
                        "parameters":['w1', 'w2', 'w3', 'w4', 'w5', 'b'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "lr10d":  {
                        "parameters":['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'b'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "lr20d":  {
                        "parameters":['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 'w20', 'b'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "lr50d":  {
                        "parameters":['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 'w20', 'w21', 'w22', 'w23', 'w24', 'w25', 'w26', 'w27', 'w28', 'w29', 'w30', 'w31', 'w32', 'w33', 'w34', 'w35', 'w36', 'w37', 'w38', 'w39', 'w40', 'w41', 'w42', 'w43', 'w44', 'w45', 'w46', 'w47', 'w48', 'w49', 'w50', 'b'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "dpr_tp":       {
                        "parameters":['x1', 'x2', 'x3', 'x4'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "dpr_lorenz63": {
                        "parameters":['x1', 'x2', 'x3'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "sbi_gm":       {
                        "parameters":['x1', 'x2'],
                        "sim_parameters":[],
                        "simdata_num_samples":30, 
                    },
    "sbi_slcp":     {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_bglm":     {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_glu":      {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_tm":       {
                        "parameters":['x1', 'x2'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_gl":       {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_slcpd":      {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_bglmr":    {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
}

sim_outs = {
    "prosail": 11,
    "prosail_2d": 11,
    "prosail_sim": 11,
    "prosail_real": 11,
    "mdgs1": 2,
    "mdgs2": 4,
    "mdgs5": 10,
    "mdgs10": 20,
    "mdgs20": 40,
    "mdgs50": 100,
    "lr2d": 100,
    "lr5d": 100,
    "lr10d": 100,
    "lr20d": 100,
    "lr50d": 100,
    "dpr_tp": 4,
    "dpr_lorenz63": 3,
    "sbi_gm": 2,
    "sbi_slcp": 8,
    "sbi_bglm": 10,
    "sbi_glu": 10,
    "sbi_tm": 2,
    "sbi_gl": 10,
    "sbi_slcpd": 100,
    "sbi_bglmr": 100,
}



datasets = ["train", "val", "test"]
simulators = [
        #"prosail",
        #"prosail_2d",
        #"prosail_sim",
        #"prosail_real",
        #"mdgs1",
        #"mdgs2",
        #"mdgs5",
        #"mdgs10",
        #"mdgs20",
        #"mdgs50",
        "lr2d",
        #"lr5d",
        "lr10d",
        "lr20d",
        "lr50d",
        #"dpr_tp",
        #"dpr_lorenz63",
        #"sbi_gm",
        #"sbi_slcp",
        #"sbi_bglm",
        #"sbi_glu",
        #"sbi_tm",
        #"sbi_gl",
        #"sbi_slcpd",
        #"sbi_bglmr",
    ]


all_conds = {}
i = 1
for ds in datasets:
    for sim in simulators:
        all_conds[i] = {"dataset":ds, "simulator":sim}
        i += 1
        
conds = all_conds[int(sys.argv[1])]


currdir = project_dir + "data/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
    
currdir = currdir + "experiments/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
    
currdir = currdir + conds["simulator"] + "/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass

currdir = currdir + conds["dataset"] + "/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
    
    
path_X = currdir + "instances_X.csv"
if(overwrite or not(os.path.exists(path_X))):
    s = "instance"
    num_outputs = sim_outs[conds["simulator"]]
    outputs = ["o" + str(j) for j in range(num_outputs)]
    for output in outputs:
        s = s + "," + output
    s = s + "\n"
    with open(path_X, 'w') as fp:
        fp.write(s)
        
        
path_Y = currdir + "instances_Y.csv"
if(overwrite or not(os.path.exists(path_Y))):
    s = "instance"
    params = sim_params[conds["simulator"]]["parameters"]
    for param in params:
        s = s + "," + param
    s = s + "\n"
    with open(path_Y, 'w') as fp:
        fp.write(s)


path_M = currdir + "instances_M.csv"
if(overwrite or not(os.path.exists(path_M))):
    s = "instance"
    params = sim_params[conds["simulator"]]["sim_parameters"]
    for param in params:
        s = s + "," + param
    s = s + "\n"
    with open(path_M, 'w') as fp:
        fp.write(s)
        
        


#######################################
# Main dataset generation
#######################################

# Instantiate/set all relevant stuff
p = set_default_parameters()
p["simdata_simulator"] = conds["simulator"]
if(conds["simulator"] in ["prosail", "prosail_2d", "prosail_sim", "prosail_real"]):
    p["simdata_simulator"] = "prosail"
elif(conds["simulator"].startswith("mdgs")):
    p["simdata_simulator"] = "mdgs"
    p["simdata_dim"] = int(conds["simulator"][4:])
elif(conds["simulator"] == "lr5d"):
    if(not(os.path.exists("src/resources/data/lr5d.csv"))):
        generate_lr_data(dim=5)
p["exp_parameters"] = sim_params[conds["simulator"]]["parameters"]
p["exp_parameters_simulation"] = sim_params[conds["simulator"]]["sim_parameters"]
p["simdata_num_samples"] = sim_params[conds["simulator"]]["simdata_num_samples"]

p["exp_max_instances"] = num_instances[conds["dataset"]]
p["simdata_num_instances"] = num_instances[conds["dataset"]]
#p["exp_max_instances"] = 5 # TODO: remove, just testing
#p["simdata_num_instances"] = 5 # TODO: remove, just testing
p["method_budget"] = 20000 # For pre-computing global optimum

# TODO: set seed and pass to all relevant stuff
# This is just to make sure train, val and test use different seeds
i = 0
for ds in datasets:
    if(ds == conds["dataset"]):
        p["exp_seed"] = p["exp_seed"] + i
    i += 1
np.random.seed(p["exp_seed"])




# Generate dataset
data = load_data_synth(p)
X = data["X"]
Y = data["Y"]
M = data["M_table"]


for i in range(X.shape[0]):


    # Save stuff
    with open(path_X, 'a') as fp:
        s = str(i)
        row = X[i, :]
        for scalar in row:
            s = s + "," + str(scalar)
        s = s + "\n"
        fp.write(s)
        
    with open(path_Y, 'a') as fp:
        s = str(i)
        row = Y[i, :]
        for scalar in row:
            s = s + "," + str(scalar)
        s = s + "\n"
        fp.write(s)
        
    with open(path_M, 'a') as fp:
        s = str(i)
        row = M[i, :]
        for scalar in row:
            s = s + "," + str(scalar)
        s = s + "\n"
        fp.write(s)
    


# Done
print("Terminated successfully:", conds)