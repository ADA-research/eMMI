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
from shared_utils import *
from shared_methods import *

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score





#######################################
# Setup
#######################################



# General steps:
    # 1: iterate over datasets with different dimensionality
        # 2: iterate over instances
        # 3: for every instance, run eMMI and save function evaluations
            # 4: iterate over function evaluation subsets in steps (first a small subset, then larger, etc)
                # 5: train classifier for this instance using sample subset, evaluate classification performance on validation points
                # 6: add performance to vector of performances for every instance
            # 7: Compute mean and std of instance performances, add to vectors of mean/std performances per budget
        # 8: Create line in plot for this dataset, optionally with std
    # 9: plot figure




project_path = "/scratch/arp/AutoIPQ/"
total_instances = 50 # 200 

max_budget = 1000 # 20000
budget_interval = 20 # 500
budgets = np.arange(0, max_budget, step=budget_interval)

simulator = "mdgs" # mdgs, lr


overwrite = False # TODO: if any issues in code, remember to change this to True



#autoipq_params = {
#    "solution_set_size": 20,
#    "k": 20,
#    "approach": "dual",
#    "budget_split": [0.33, 0.33, 0.33],
#    "optimiser": "HC", 
#    "optimiser_2c": "HC", 
#    "optimiser_2d": "HC", 
#    #"loss_func": "identity",
#}



dims = [2, 5, 10, 20, 50]
#dims = [2, 5, 10, 20] # TODO: remove

conds_all = {}
i = 1
for dim in dims:
    conds_all[i] = {"dim":dim}
    i += 1
        
conds = conds_all[int(sys.argv[1])]
dim = conds["dim"]


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


# Determine epsilon using oracle knowledge
#mean_arr = np.array([0.0 for _ in range(conds["dim"])])
#cov = np.identity(conds["dim"])
#from scipy.stats import multivariate_normal
#dist = multivariate_normal(mean_arr, cov)
#mean_pd = dist.pdf(mean_arr)
#mean_arr2 = np.array([0.0 for _ in range(conds["dim"])])
#mean_arr2[0] = 1.0
#per_pd = dist.pdf(mean_arr2)

#epsilon = mean_pd - per_pd


# Set up some dirs
currdir = project_path + "results/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
        
        
currdir = project_path + "budget_experiment_ " + simulator + "/"
if(not(os.path.exists(currdir))):
    try:
        os.mkdir(currdir)
    except:
        pass
            


def instantiate_boost():
    return(GradientBoostingClassifier())



########## HANDLING EXISTING RUNS #########

results_path = currdir

if(overwrite or not(os.path.exists(results_path + "means.csv"))):
    s = "dim,"
    for b in budgets:
        s = s + str(b) + ","
    s = s[:-1] + "\n"
    with open(results_path + "means.csv", 'w') as fp:
        fp.write(s)
    with open(results_path + "stds.csv", 'w') as fp:
        fp.write(s)


existing_runs_df = pd.read_csv(results_path + "means.csv")
existing_runs = existing_runs_df['dim'].values


########## RUN ##############


results_path = currdir

if(overwrite or not(os.path.exists(results_path + "means.csv"))):


    # List of vectors; these will be lines in plot
    dataset_mean_mat = np.zeros((len(dims), len(budgets)))
    dataset_std_mat = np.zeros((len(dims), len(budgets)))

    d = 0





#for dim in dims:

#print(type(dim))
#print(type(existing_runs[0]))


if(dim not in existing_runs):

    if(simulator == "lr"):

        sim_name = "lr" + str(dim) + "d"

        p = set_default_parameters()
        load_parameters_file(p, param_paths[sim_name])
        p = load_parameters_from_smac(p, sim_name, "AutoIPQ")

    elif(simulator == "mdgs"):
        sim_name = "mdgs" + str(dim)

        p = set_default_parameters()
        load_parameters_file(p, param_paths[sim_name])
        p = load_parameters_from_smac(p, sim_name, "AutoIPQ")

        p["simdata_simulator"] = "mdgs"
        p["simdata_dim"] = dim
        #param_list = []
        #for i in range(dim):
        #    param_list.append("mu"+str(i))
        #    param_list.append("sigma"+str(i))
        #p["exp_parameters"] = param_list
        #p["exp_parameters_simulation"] = []
        #p["simdata_num_samples"] = 1

        #p["exp_max_instances"] = total_instances
        #p["simdata_num_instances"] = total_instances



    # Instantiate simulator
    #sim = instantiate_sim(p["simdata_simulator"], dim=dim)
    sim = instantiate_sim(p["simdata_simulator"])

    # Create test set
    data = load_dataset(sim_name, project_path, max_instances=total_instances, noise=p["data_noise"])
    instances_X = data["test"]["X"]
    instances_Y = data["test"]["Y"]


    # Initialise matrix for this dataset of (n_i x b), where n_i is the number of instances and b is the number of budget steps
    perf_matrix = np.zeros((total_instances, len(budgets)))

    # Iterate over instances
    for i in range(instances_X.shape[0]):



        # Create AutoIPQ object
        #auto = AutoIPQ(sim, instances_X[i, :], list(sim.default_params.keys()),  **autoipq_params)
            
        #method_kwargs = {
        #                    "budget": max_budget, 
        #                    "num_solutions": 2,
        #                    "return_opt": False, 
        #                    "store_XY": True, 
        #                    "store_test":False,
        #                    "train_clf": False,
        #                }
        #l_clf = auto.run(epsilon, instantiate_boost, **method_kwargs)
        #clf_X = auto.Xs[0] # Parameters
        #clf_Y = auto.Ys[0] # Labels (in/out of epsilon manifold)

        #print(len(clf_X))
        #print(clf_X.shape) # testing
        #aslkdf

        #_, auto = run_AutoIPQ(p, data["test"], i, sim, None, variant=p["method_variant"], return_auto=True, return_preds=False, store_XY=True, warm_start_method='optimise')
        _, auto = run_AutoIPQ(p, data["test"], i, sim, None, variant=p["method_variant"], return_auto=True, return_preds=False, store_XY=True, warm_start_method='optimise', train_clf=False)
        clf_X = auto.Xs[0] # Parameters
        clf_Y = auto.Ys[0] # Labels (in/out of epsilon manifold)




        # Iterate over budget steps and train+evaluate classifier

        for j in range(len(budgets)):
            budget = budgets[j]
            train_X = np.concatenate(clf_X[:max(2, budget), :], axis=0)
            train_Y = np.concatenate(clf_Y[:max(2, budget)], axis=0)


            #print("train_X shape", train_X.shape)
            #print("train_Y shape", train_Y.shape)
            #print("train_Y sum:", np.sum(train_Y))

            # Evaluate on validation points

            opt_loss_i = data["test"]["opt_loss"][i]
            epsilon_loss = opt_loss_i + p["exp_epsilon"]

            # TODO: make adjustments to use p and normal parameter loading, configure conds
            test_X, test_Y = load_local_eval_sample(p, {"simulator":sim_name}, i, epsilon_loss, version="uniform")


            # Predict; if all 0s or 1s (happens sometimes), just predict all 0s or 1s
            label_sum = np.sum(train_Y)
            if(label_sum == 0):
                pred = np.zeros(len(test_Y))
            elif(label_sum == len(test_Y)):
                pred = np.ones(len(test_Y))
            else:
                # Real predictions
                clf = instantiate_boost()
                try:
                    clf.fit(train_X, train_Y)
                    pred = clf.predict(test_X)
                    #print("Successful training")

                except:
                    #print("failed training")
                    pred = np.zeros(len(test_Y))

            #print("pred:", pred)

            acc = accuracy_score(pred, test_Y)

            #print("acc:", acc)

            perf_matrix[i, j] = acc

    # Aggregate
    dset_mean_vec = np.nanmean(perf_matrix, axis=0) # Just because for high dim very rarely it seemed to go wrong
    dset_std_vec = np.nanstd(perf_matrix, axis=0)

    # Append to results
    s1 = str(dim) + ","
    s2 = str(dim) + ","
    for j in range(len(dset_mean_vec)):
        s1 = s1 + str(dset_mean_vec[j]) + ","
        s2 = s2 + str(dset_std_vec[j]) + ","
    s1 = s1[:-1] + "\n"
    s2 = s2[:-1] + "\n"
    with open(results_path + "means.csv", 'a') as fp:
        fp.write(s1)
    with open(results_path + "stds.csv", 'a') as fp:
        fp.write(s2)



#else:





# Load existing numerical results
means_df = pd.read_csv(results_path + "means.csv").sort_values('dim')
dataset_mean_mat = means_df.values[:, 1:]
std_df = pd.read_csv(results_path + "stds.csv").sort_values('dim')
dataset_std_mat = std_df.values[:, 1:]

existing_runs = means_df['dim'].values



    
# Create line plot


#print("\ndataset mean mat shape:", dataset_mean_mat.shape)
#print("dims len:", len(dims))
#print("dataset std mat shape:", dataset_std_mat.shape)

for d in range(dataset_mean_mat.shape[0]):

    plt.plot(budgets, dataset_mean_mat[d, :], label="d="+str(existing_runs[d]))
    plt.fill_between(budgets, dataset_mean_mat[d, :]-dataset_std_mat[d, :], dataset_mean_mat[d, :]+dataset_std_mat[d, :], alpha=0.3)

plt.title("Validation accuracy per budget")
plt.xlabel("Budget")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(results_path + "budget_lines.pdf", bbox_inches='tight')
plt.show()




    
print("Terminated successfully.")