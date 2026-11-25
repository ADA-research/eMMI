import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import time

from src.AutoIPQ import *
from src.utils import *
from src.simulate import *
from src.ML import *
from src.ForwardModel import *
from shared_utils import instantiate_sim, set_default_parameters, load_parameters_file, add_gaussian_noise_X, load_dataset

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from autosklearn.classification import AutoSklearnClassifier





#######################################
# Setup
#######################################

project_path = "/scratch/arp/AutoIPQ/"
#total_instances = 200 # 1000
max_instances_data = 200
#total_instances = 2 # TODO: remove
global overwrite
overwrite = False

param_paths = {
    "prosail": "experiments/params_prosail.json",
    "prosail_2d": "experiments/params_prosail_2d.json",
    "prosail_sim": "experiments/params_prosail_sim.json",
    "prosail_real": "experiments/params_prosail_real.json",
    "mdgs2": "experiments/params_mdgs2.json",
    "mdgs5": "experiments/params_mdgs5.json",
    "mdgs10": "experiments/params_mdgs10.json",
    "mdgs20": "experiments/params_mdgs20.json",
    "mdgs50": "experiments/params_mdgs50.json",
    "lr2d": "experiments/params_lr2d.json",
    "lr5d": "experiments/params_lr5d.json",
    "lr10d": "experiments/params_lr10d.json",
    "lr20d": "experiments/params_lr20d.json",
    "lr50d": "experiments/params_lr50d.json",
    "dpr_tp": "experiments/params_dpr_tp.json",
    "dpr_lorenz63": "experiments/params_dpr_lorenz63.json",
    "sbi_gm": "experiments/params_sbi_gm.json",
    "sbi_slcp": "experiments/params_sbi_slcp.json",
    "sbi_bglm": "experiments/params_sbi_bglm.json",
    "sbi_bglmr": "experiments/params_sbi_bglmr.json",
    "sbi_tm": "experiments/params_sbi_tm.json",
    "sbi_gl": "experiments/params_sbi_gl.json",
    "sbi_glu": "experiments/params_sbi_glu.json",
    "sbi_slcpd": "experiments/params_sbi_slcpd.json",
}

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
        "lr2d",
        "lr5d",
        "lr10d",
        "lr20d",
        "lr50d",
        "dpr_tp", 
        "dpr_lorenz63",
        "sbi_gm",
        #"sbi_slcp",
        #"sbi_bglm",
        #"sbi_bglmr",
        "sbi_tm",
        "sbi_gl",
        #"sbi_glu",
        #"sbi_slcpd",
    ]
approaches = ["direct", "indirect", "uniform"]
approaches = ["uniform"]
datasets = ["test", "val"]
budget = 12000
budget_uniform = 50000

conds_all = {}
i = 1
for simulator in simulators:
    for approach in approaches:
        for ds in datasets:
            conds_all[i] = {"simulator":simulator, "approach":approach, "dataset":ds}
            i += 1
        
conds = conds_all[int(sys.argv[1])]



#######################################
# Functions
#######################################



def run_AutoIPQ(p, conds, data, i, sim, opt_i, opt_loss_i):

    def instantiate_boost():
        return(GradientBoostingClassifier())
        

    
    method_kwargs = {
                        "budget": p["method_budget"], 
                        "budget_split": p["method_budget_split"],
                        "num_solutions": p["method_population_size"],
                        "k": p["method_k"], 
                        "diversity_weight": p["method_diversity_weight"],
                        "approach": p["method_step2_approach"],
                        "epsilon2": p["method_epsilon2"],
                        "learning_rate": p["method_learning_rate"],
                        "init_noise": p["method_init_noise"], 
                        "constraint_penalty": p["method_constraint_penalty"],
                        "filter_density": p["method_filter_density"],
                        "density_filter_percentile": p["method_density_filter_percentile"],
                        "density_filter_same_class": p["method_density_filter_same_class"],
                        "return_opt": False, 
                        "store_XY": False, 
                        "store_test":True,
                        "train_clf": False,
                        "optimum": opt_i,
                    }
    
    instance_x = data["X"][i]
    
    # Run AutoIPQ to train classifier(s)
    auto = AutoIPQ(sim, instance_x, p["exp_parameters"], p["method_population_size"], optimiser=p["method_optimiser_1"], optimiser_2a=p["method_optimiser_2a"], optimiser_2b=p["method_optimiser_2b"], optimiser_2c=p["method_optimiser_2c"], optimiser_2d=p["method_optimiser_2d"])

    l_clf = auto.run(p["exp_epsilon"], instantiate_boost, **method_kwargs)
    
    sample_X = auto.sample_X # X for classifier, but ys (parameters) overall
    sample_Y = auto.sample_Y # labels for in/out, but only indication
    
    
    if(conds["approach"] == "indirect"):
        # Use previous points only to establish extremes, sample uniformly within (assume convexity)
        true_points = sample_X[sample_Y == 1, :]
        half_point = int(sample_X.shape[0])
        
        # Uniformly generate in samples based on extremes of previous sample
        in_samples = np.zeros((half_point, sample_X.shape[1]))
        for dim in range(sample_X.shape[1]):
            min_val = np.min(true_points[:, dim])
            max_val = np.max(true_points[:, dim])
            in_samples[:, dim] = np.random.uniform(low=min_val, high=max_val, size=half_point)
            
        # Generate out samples using random sampling over full space
        out_samples = np.zeros((half_point, sample_X.shape[1]))
        for dim in range(sample_X.shape[1]):
            min_val = sim.param_ranges[p["exp_parameters"][dim]][2]
            max_val = sim.param_ranges[p["exp_parameters"][dim]][3]
            out_samples[:, dim] = np.random.uniform(low=min_val, high=max_val, size=half_point)
            
        sample_X = np.concatenate((in_samples, out_samples), axis=0)
        
    # Pre-compute losses, label indication for efficiency later
    sample_Y = np.zeros(sample_X.shape[0]) # overwriting
    sample_L = np.zeros(sample_X.shape[0]) # Loss values
    epsilon_loss = opt_loss_i + p["exp_epsilon"]
    
    for s_i in range(0, sample_X.shape[0]):
        y = sample_X[s_i, :] # back to y being parameters for now
        sim_x = sim.simulate(parameters=auto.sampling_opt.coords_to_solution(y))
        l = auto.sampling_opt.f1(instance_x, sim_x)
        sample_L[s_i] = l
        sample_Y[s_i] = 1 if l<epsilon_loss else 0
            
    return(sample_X, sample_Y, sample_L)
        
        

def run_uniform_rejection(p, conds, data, i, sim, opt_i, opt_loss_i, budget=20000):
    epsilon_loss = opt_loss_i + p["exp_epsilon"]
    instance_x = data["X"][i]
    
    # Instantiate AutoIPQ for convenience in computing loss
    auto = AutoIPQ(sim, instance_x, p["exp_parameters"], 10)

    # Sample, combine into matrix
    column_vals = []
    for param in p["exp_parameters"]:
        range_min = sim.param_ranges[param][2]
        range_max = sim.param_ranges[param][3]
        param_vals = np.random.uniform(low=range_min, high=range_max, size=budget)
        column_vals.append(param_vals.reshape((len(param_vals), 1)))
    
    test_X = np.concatenate(column_vals, axis=1) # NOTE: X for clf is Y for problem (X is config, Y is valid/invalid)

    
    # Compute test_Y   
    test_Y = np.zeros(test_X.shape[0])
    test_L = np.zeros(test_X.shape[0])
    for sub_i in range(test_X.shape[0]):
        x_dict = {p["exp_parameters"][j]:test_X[sub_i][j] for j in range(test_X.shape[1])}
        sim_output = sim.simulate(x_dict)
        loss = auto.optimiser.f1(sim_output, instance_x) # x_i is 'real' instance observations
        test_L[sub_i] = loss
        if(loss <= epsilon_loss):
            test_Y[sub_i] = 1
            
    # Enforce balance
    X_true = test_X[test_Y == 1, :]
    Y_true = test_Y[test_Y == 1]
    L_true = test_L[test_Y == 1]
    
    X_false = test_X[test_Y == 0, :]
    Y_false = test_Y[test_Y == 0]
    L_false = test_L[test_Y == 0]
    
    minlen = min(len(Y_true), len(Y_false))
    X_true = X_true[:minlen, :]
    Y_true = Y_true[:minlen]
    L_true = L_true[:minlen]
    X_false = X_false[:minlen, :]
    Y_false = Y_false[:minlen]
    L_false = L_false[:minlen]
    
    X = np.concatenate((X_true, X_false), axis=0)
    Y = np.concatenate((Y_true, Y_false))
    L = np.concatenate((L_true, L_false))
    
    return(X, Y, L)


#######################################
# Big experiment function
#######################################


def generate_points(p, conds, data, project_path, overwrite, noise=None, budget_uniform=None):

    ###### SETUP ################################
    
    p["store_sample_path"] = p["exp_project_path"] + "data/experiments/" + conds["simulator"] + "/" + conds["dataset"] + "/"
    
    if(conds["approach"] == "direct"):
        currdir = p["store_sample_path"] + "clf_samples/"
    elif(conds["approach"] == "indirect"):
        currdir = p["store_sample_path"] + "clf_samples_indirect/"
    elif(conds["approach"] == "uniform"):
        currdir = p["store_sample_path"] + "clf_samples_uniform/"
        
    if(not(os.path.exists(currdir))):
        try:
            os.mkdir(currdir)
        except:
            pass

    
    ###### INFERENCE ############################
    
    num_instances = data[conds["dataset"]]["X"].shape[0]
    order = np.arange(num_instances)

    np.random.shuffle(order)
    np.random.seed(p["exp_seed"]) # Do this after order to allow multiple simultaneous runs
    
    suffix = ""
    if(noise is not None):
        suffix = "_noise" + str(noise)
    
    for i in order:
        # Check if result already exists for this instance
        
        path_Xi = currdir + "p" + str(i) + "_X" + suffix + ".csv"
        path_Yi = currdir + "p" + str(i) + "_Y" + suffix + ".csv"

        if(overwrite or not(os.path.exists(path_Xi)) or not(os.path.exists(path_Yi))):

            # Setup for sampling 
            x_i = data[conds["dataset"]]["X"][i]
            m_i = data[conds["dataset"]]["M"][i]
            opt_i = data[conds["dataset"]]["opt"][i]
            opt_loss_i = data[conds["dataset"]]["opt_loss"][i]
            
            priors = {p["exp_parameters_simulation"][k]:m_i[k] for k in range(len(p["exp_parameters_simulation"]))}
            sim = instantiate_sim(p["simdata_simulator"], priors=priors, dim=p["simdata_dim"])
            
            if(conds["approach"] in ["direct", "indirect"]):
                X, Y, L = run_AutoIPQ(p, conds, data[conds["dataset"]], i, sim, opt_i, opt_loss_i)
            elif(conds["approach"] == "uniform"):
                assert budget_uniform is not None
                X, Y, L = run_uniform_rejection(p, conds, data[conds["dataset"]], i, sim, opt_i, opt_loss_i, budget=budget_uniform)
            else:
                raise NotImplementedError()


            # Save stuff; check again in case other worker saved in the meantime (avoid mismatch)
            
            if(overwrite or not(os.path.exists(path_Xi)) or not(os.path.exists(path_Yi))):
                # X
                s = ""
                for param in p["exp_parameters"]:
                    s = s + "," + param
                s = s[1:] + "\n"
                for s_i in range(X.shape[0]):
                    s_local = ""
                    x = X[s_i, :]
                    for j in range(X.shape[1]):
                        s_local = s_local + "," + str(X[s_i, j])
                    s_local = s_local[1:] + "\n"
                    s = s + s_local
                with open(path_Xi, 'w') as fp:
                    fp.write(s)
                
                # Y+L
                s = "loss,label_indication\n"
                for s_i in range(X.shape[0]):
                    loss = L[s_i]
                    label = Y[s_i]
                    s_local = str(loss) + "," + str(label) + "\n"
                    s = s + s_local
                
                with open(path_Yi, 'w') as fp:
                    fp.write(s)

                

 
 
#######################################
# Running experiments
#######################################


if(__name__ == '__main__'):
    p = set_default_parameters()
    load_parameters_file(p, param_paths[conds["simulator"]])
    p["method_step2_approach"] = "split"
    p["method_budget_split"] = [1, 1, budget]
    data = load_dataset(conds["simulator"], project_path, max_instances=max_instances_data, noise=p["data_noise"], splits=[conds["dataset"]])

    generate_points(p, conds, data, project_path, overwrite, budget_uniform=budget_uniform)
    
    
    print("Terminated successfully:", conds)