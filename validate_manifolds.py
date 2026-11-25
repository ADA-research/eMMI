import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

from src.AutoIPQ import *
from src.utils import *
from src.simulate import *
from src.ML import *
from src.ForwardModel import *
from shared_utils import *
from shared_methods import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#######################################
# Setup
#######################################

project_path = "/scratch/arp/AutoIPQ/"
total_instances = 200 # 1000
#total_instances = 3 # TODO: remove
global overwrite
overwrite = True

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
    "lr5d": "experiments/params_lr5d.json",
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

methods = ["eM", "UC"] 
#methods = ["UC"] # TODO: remove
# Both "methods" are oracle methods; perfect epsilon manifold (assuming proper optimum found) and perfect std uncertainty

    
        
simulators = [
        "prosail",
        "prosail_2d",
        #"prosail_sim",
        #"prosail_real",
        #"mdgs2",
        #"mdgs5",
        #"mdgs10",
        #"mdgs20",
        #"mdgs50",
        "lr5d",
        "dpr_tp",
        "dpr_lorenz63",
        "sbi_gm",
        #"sbi_slcp",
        #"sbi_bglm",
        #"sbi_bglmr",
        "sbi_tm",
        #"sbi_gl",
        #"sbi_glu",
        #"sbi_slcpd",
    ]

#simulators = ["prosail_2d"] # TODO: remove

conds_all = {}
i = 1
for simulator in simulators:
    for method in methods:
        conds_all[i] = {"method":method, "simulator":simulator}
        #print(str(i) + ": " + str(conds_all[i]))
        i += 1
        
conds = conds_all[int(sys.argv[1])]



#######################################
# Functions
#######################################



def check_true_point_eM(p, x_i, y_true_i, epsilon_loss):

    # Create simulator
    sim = instantiate_sim(p["simdata_simulator"])

    # Perform simulation
    config = {}
    j = 0
    for param in p["exp_parameters"]:
        config[param] = y_true_i[j]
        j += 1
    sim_out = sim.simulate(config)

    # Compute loss; instantiate eMMI just to ensure all the loss functions etc match the rest of the experiments
    auto = AutoIPQ(sim, x_i, p["exp_parameters"], p["method_population_size"], optimiser=p["method_optimiser_1"], optimiser_2a=p["method_optimiser_2a"], optimiser_2b=p["method_optimiser_2b"], optimiser_2c=p["method_optimiser_2c"], optimiser_2d=p["method_optimiser_2d"])
    #print("epsilon loss:", epsilon_loss)
    loss = auto.optimiser.f1(x_i, sim_out)
    #print("loss:", loss)

    # Return 1 if loss for true point was smaller than epsilon loss, 0 otherwise
    res = 1 if loss<epsilon_loss else 0

    return(res)


def check_true_point_UC(p, val_X, val_Y, opt_i, y_true_i):
    #opt_std = optimise_std(p, val_X, val_Y, opt_i)
    opt_std = oracle_std(val_X, val_Y)

    # Check if true point is within opt +/- 3*std: result is 1 if so, 0 otherwise
    res = 1
    for j in range(len(y_true_i)):
        if(y_true_i[j] > opt_i[j]+opt_std[j] or y_true_i[j] < opt_i[j]-opt_std[j]):
            res=0

    # Return
    return(res)



def oracle_std(val_X, val_Y):
    # Compute std from in-samples in validation point set
    in_samples = val_X[val_Y == 1, :]
    stds = np.std(in_samples, axis=0)
    return(stds)



def optimise_std(p, val_X, val_Y, opt_i):
    # Try to correctly classify as many validation points as possible using only standard deviation-based confidence intervals
    # Find best std vector using this metric


    # Initialise std vector to some reasonable value (0.1*mean?)
    mean_vec = np.absolute(np.mean(val_X, axis=0))

    std_vec = np.zeros(val_X.shape[1])
    for j in range(len(std_vec)):
        std_vec[j] = 0.1 * np.mean(val_X[:, j])

    # Iterate over meta search budget
    best_score = -1
    for i in range(p["exp_uc_oracle_iterations"]):

        # Mutate std vector
        mutation_vec = np.zeros(len(std_vec))
        for j in range(len(mutation_vec)):
            mutation_vec[j] = np.random.normal(loc=0, scale=0.05*mean_vec[j])
        new_std_vec = std_vec + mutation_vec

        # Initialise inner result to vector of 1s (len sample points)
        inner_result = np.ones(val_X.shape[0])
        col_vecs = []

        # Iterate over parameters (column in sample points)
        for j in range(val_X.shape[1]):

            # Vectorised check which elements of param column are outside of parameter of opt +/- 3x std
            col_vec = np.ones(val_X.shape[0])
            col_vec[val_X[:, j] < opt_i[j]-3*new_std_vec[j]] = 0
            col_vec[val_X[:, j] > opt_i[j]+3*new_std_vec[j]] = 0
            col_vecs.append(col_vec)
        
        # Set inner result vector to 0 for all rows where any column is False
        col_matrix = np.stack(col_vecs, axis=1)
        sum_matrix = np.sum(col_matrix, axis=1)
        pred_vec = np.ones(val_X.shape[0])
        pred_vec[sum_matrix < 4] = 0

        # Create new inner result vector of 0s
        inner_result = np.zeros(val_X.shape[0])
        # Iterate over validation samples; should be 1 if both inner result and validation point are either 0 or 1
        try:
            sum_matrix2 = np.sum(np.stack([pred_vec, val_Y], axis=1), axis=1)
        except:
            print("something went wrong")
            print(pred_vec.shape)
            print(val_Y.shape)
            sum_matrix2 = np.ones(len(val_Y))
        inner_result[sum_matrix2 == 0] = 1
        inner_result[sum_matrix2 == 2] = 1
        sum_scalar = np.sum(inner_result)

        # If sum of inner result vector > existing best, replace existing std vector
        if(sum_scalar > best_score):
            best_score = sum_scalar
            std_vec = new_std_vec

    return(std_vec)

            

    
    
def setup_dirs(project_path, conds):

    # Set up some dirs
    currdir = project_path + "results/"
    if(not(os.path.exists(currdir))):
        try:
            os.mkdir(currdir)
        except:
            pass

    currdir = project_path + "manifold_validation/"
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
            
    currdir = currdir + conds["method"] + "/"
    if(not(os.path.exists(currdir))):
        try:
            os.mkdir(currdir)
        except:
            pass
            
            
    return(currdir)
    

#######################################
# Big experiment function
#######################################


def run_experiment(p, conds, data, data_noise, project_path, overwrite):

    ###### SETUP ################################
    
    currdir = setup_dirs(project_path, conds)
    res_path = currdir + "true_point_evals.csv"

    if(overwrite or not(os.path.exists(res_path))):
        s = "instance,result,true\n" # Result means the predicted result for true point (in manifold/3 stds); true is whether should be in
        with open(res_path, 'w') as fp:
            fp.write(s)

    
    
    ###### INFERENCE ############################
    
    num_instances = data["test"]["X"].shape[0]
    #np.random.shuffle(order)
    #print("order:", order)

    results = np.zeros(2*num_instances)
    trues = np.zeros(2*num_instances)
    trues[:num_instances] = 1

    bad_results = []


    for i in range(num_instances):

        # Setup for sampling
        x_i = data_noise["test"]["X"][i]
        y_true_i = data["test"]["Y"][i]
        opt_i = data_noise["test"]["opt"][i]
        opt_loss_i = data_noise["test"]["opt_loss"][i]
        epsilon_loss = opt_loss_i + p["exp_epsilon"]

        val_X, val_Y = load_local_eval_sample(p, conds, i, epsilon_loss, version="uniform")

        if(len(val_X) > 1): # Only run on instances that have at least 2 validation points (so 1 negative sample)

            # Run for true point (original noise-free point)
            if(conds["method"] == "eM"):
                result = check_true_point_eM(p, x_i, y_true_i, epsilon_loss)
            elif(conds["method"] == "UC"):
                print("i:", i)
                result = check_true_point_UC(p, val_X, val_Y, opt_i, y_true_i)
            else:
                raise NotImplementedError
            results[i] = result

            # Run for control point (false point from validation points)
            val_y = val_X[-1, :] # val_X contains parameters, therefore y in this context
            if(conds["method"] == "eM"):
                result2 = check_true_point_eM(p, x_i, val_y, epsilon_loss)
            elif(conds["method"] == "UC"):
                result2 = check_true_point_UC(p, val_X, val_Y, opt_i, val_y)
            else:
                raise NotImplementedError
            results[i] = result
            results[i+num_instances] = result2


        else:
            bad_results.append(i)



    for i in range(len(results)):
        with open(res_path, 'a') as fp:
            s = str(i) + "," + str(results[i]) + "," + str(trues[i]) + "\n"
            fp.write(s)

    # To avoid corrupting avg computation
    results = np.delete(results, bad_results)
    trues = np.delete(trues, bad_results)

    mae = np.mean(np.absolute(results-trues))
    avg_score = 1 - mae
    print("Average accuracy for " + str(conds) + ": ")
    print(str(avg_score) + " +/- " + str(np.std(np.absolute(results-trues))))
    print("Confusion matrix:")
    print(confusion_matrix(trues, results))

    confusion_matrix_display = ConfusionMatrixDisplay.from_predictions(trues, results, cmap="Blues", 
                                                                       display_labels=["Non-viable solution", "Viable solution"])
    confusion_matrix_display.figure_.savefig(currdir + "/confusion_matrix.pdf", bbox_inches='tight')




 
#######################################
# Running experiments
#######################################


if(__name__ == '__main__'):
    p = set_default_parameters()
    load_parameters_file(p, param_paths[conds["simulator"]])
    data = load_dataset(conds["simulator"], project_path, max_instances=total_instances, splits=["test"])
    data_noise = load_dataset(conds["simulator"], project_path, max_instances=total_instances, splits=["test"], noise=p["data_noise"])

    run_experiment(p, conds, data, data_noise, project_path, overwrite)
    
    
    print("Terminated successfully:", conds)