import numpy as np
import pandas as pd

import os
import sys

from src.AutoIPQ import *
from shared_utils import set_default_parameters, add_gaussian_noise_X, instantiate_sim


#######################################
# Setup
#######################################

project_dir = "/scratch/arp/AutoIPQ/"
dataset = "test"
max_instances = 200
epsilon_method = "percentile"
epsilon_percentile = 95
num_repeats = 20 # For stochastic simulators

overwrite = True


sim_params = {
    "prosail":  {
                        "parameters":['lai', 'cab', 'lidfa2', 'cw'],
                        "sim_parameters":[],
                    },
    "prosail_2d":  {
                        "parameters":['lai', 'cab'],
                        "sim_parameters":['tts', 'tto', 'psi'],
                    },
    "prosail_sim":  {
                        "parameters":['lai', 'cab', 'hspot', 'n', 'lidfa2', 'cw', 'cm'],
                        "sim_parameters":['tts', 'tto', 'psi'],
                    },
    "prosail_real": {
                        "parameters":['lai', 'cab', 'hspot', 'n', 'lidfa2', 'cw', 'cm'],
                        "sim_parameters":[],
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
                    },
    "sbi_slcp":     {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5'],
                        "sim_parameters":[],
                    },
    "sbi_bglm":     {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                    },
    "sbi_glu":      {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                    },
    "sbi_tm":       {
                        "parameters":['x1', 'x2'],
                        "sim_parameters":[],
                    },
    "sbi_gl":       {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                    },
    "sbi_slcpd":      {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5'],
                        "sim_parameters":[],
                    },
    "sbi_bglmr":    {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                    },
}

sim_outs = {
    "prosail": 11,
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


noise_levels = [0.1]
simulators = [
        "prosail",
        "prosail_2d",
        #"prosail_sim",
        #"prosail_real",
        "mdgs2",
        "mdgs5",
        "mdgs10",
        "mdgs20",
        "mdgs50",
        "dpr_tp",
        "dpr_lorenz63",
        "lr2d",
        "lr5d",
        "lr10d",
        "lr20d",
        "lr50d",
        "sbi_gm",
        "sbi_slcp",
        "sbi_bglm",
        "sbi_glu",
        "sbi_tm",
        "sbi_gl",
        #"sbi_slcpd",
        #"sbi_bglmr",
    ]


all_conds = {}
i = 1
for nl in noise_levels:
    for sim in simulators:
        all_conds[i] = {"noise_level":nl, "simulator":sim}
        i += 1
        
conds = all_conds[int(sys.argv[1])]

path_results = "results/epsilons.csv"
if(overwrite or not(os.path.exists(path_results))):
    s = "simulator,noise_level,epsilon_mean,epsilon_median,epsilon_std,epsilon_max,epsilon_68,epsilon_95\n"
    with open(path_results, 'w') as fp:
        fp.write(s)



p = set_default_parameters()
p["simdata_simulator"] = conds["simulator"]
if(conds["simulator"] == "prosail_2d"):
    p["simdata_simulator"] = "prosail"
elif(conds["simulator"].startswith("mdgs")):
    p["simdata_simulator"] = "mdgs"
    p["simdata_dim"] = int(conds["simulator"][4:])
p["exp_parameters"] = sim_params[conds["simulator"]]["parameters"]
p["exp_parameters_simulation"] = sim_params[conds["simulator"]]["sim_parameters"]


#######################################
# Calling func
#######################################


# Instantiate/set all relevant stuff
# Load data
suffix = ""
if(conds["noise_level"] > 0.0):
    suffix = "_noise" + str(conds["noise_level"])
data_path = project_dir + "data/experiments/" + conds["simulator"] + "/" + dataset + "/"
X_path = data_path + "instances_X.csv"
Y_path = data_path + "instances_Y.csv"
M_path = data_path + "instances_M.csv"
opt_path = data_path + "instances_opt" + suffix + ".csv"

X = pd.read_csv(X_path).values
X_ind = X[:,0]
X = X[:,1:]
Y = pd.read_csv(Y_path).values
Y_ind = Y[:,0]
O = pd.read_csv(opt_path).sort_values('instance').values
Y = Y[:,1:]
O_ind = O[:,0]
O = O[:,1:]

instance_order = O_ind.astype(int)


M = None
# TODO: not sure if this will work, assumes that malformed cvs can't be read, real check would be better
try:
    M = pd.read_csv(M_path).values[:,1:]
except:
    pass



# Get ready for loop
num_instances = min(max_instances, len(instance_order))


epsilons = np.zeros(len(instance_order))


O_i = 0
for XY_i in instance_order:

    #print(XY_i)

    # Compute loss for true solution
    true_vec = Y[XY_i, :]
    priors = {}
    if(M is not None):
        params_sim = sim_params[conds["simulator"]]["sim_parameters"]
        priors = {params_sim[j]:M[XY_i,j] for j in range(len(params_sim))}
    sim = instantiate_sim(p["simdata_simulator"], priors=priors, dim=p["simdata_dim"])
    
    auto = AutoIPQ(sim, X[XY_i, :], p["exp_parameters"], 10, optimiser=p["method_optimiser_1"])
    
    losses = np.zeros(num_repeats)
    for k in range(num_repeats):
        true_sim = auto.optimiser.forward_sim(true_vec, format="vec")
        losses[k] = auto.optimiser.f1(true_sim, X[XY_i, :])
    true_loss = np.median(losses)

    # Compare optimum loss to true loss
    
    opt_loss = O[O_i, 0]
    epsilons[O_i] = abs(opt_loss - true_loss)

    O_i += 1

# Compute epsilon

epsilon_mean = np.mean(epsilons)
epsilon_median = np.median(epsilons)
epsilon_std = np.std(epsilons)
epsilon_max = np.max(epsilons)
epsilon_68 = np.percentile(epsilons, 68)
epsilon_95 = np.percentile(epsilons, 95)

# Save stuff
s = conds["simulator"] + "," + str(conds["noise_level"]) + "," + str(epsilon_mean) + "," + str(epsilon_median) + "," + str(epsilon_std) + "," + str(epsilon_max) + "," + str(epsilon_68) + "," + str(epsilon_95) + "\n"

with open(path_results, 'a') as fp:
    fp.write(s)

# Done
print("Terminated successfully:", conds)