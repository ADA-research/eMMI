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
#from autosklearn.classification import AutoSklearnClassifier # not reconcilable with SMAC3 and new scikit learn that supports multi-output GP std :(

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn




#######################################
# Setup
#######################################

project_path = "/scratch/arp/AutoIPQ/"
total_instances = 200 # 1000
#total_instances = 3 # TODO: remove
clf_test_sample_size = 2000
global overwrite
overwrite = False

sample_local = sys.argv[2]

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

#methods_all = ["AutoIPQ", "RF", "GP", "classification", "auto-sklearn"]
#methods_all = ["AutoIPQ", "RF", "GP", "BNN", "classification"]
#methods_all = ["Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ", "RF", "GP", "BNN", "auto-sklearn"]
methods_all = ["AutoIPQ", "Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ", "ABCSMC", "RF", "GP", "BNN", "classification", "TabPFN"]
methods_own = ["AutoIPQ", "Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ", "classification"]
methods_bl = ["RF", "GP", "BNN", "ABCSMC", "TabPFN"]
#methods_all = ["AutoIPQ"] # TODO: remove



methods_spec = sys.argv[3]
if(methods_spec == "all"):
    methods = methods_all
elif(methods_spec == "own"):
    methods = methods_own
elif(methods_spec == "bl"):
    methods = methods_bl
    
        
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
        #"sbi_gl",
        #"sbi_glu",
        #"sbi_slcpd",
    ]
#simulators = ["dpr_tp"] # TODO: remove
budgets = [5000, 10000, 20000, 50000] # later
budgets = [5000, 10000, 20000]
budgets = [20000] # TODO: remove


methods = ["Conv-AutoIPQ", "Dual-AutoIPQ", "Seq-AutoIPQ", "classification"] # TODO: remove
simulators = ["dpr_tp", "dpr_tp", "dpr_tp", "dpr_tp", "dpr_tp", "dpr_tp", "dpr_tp", "dpr_tp", "dpr_tp", "dpr_tp", "dpr_tp"] # TODO: remove


conds_all = {}
i = 1
for simulator in simulators:
    for method in methods:
        for budget in budgets:
            conds_all[i] = {"method":method, "simulator":simulator, "budget":budget}
            #print(str(i) + ": " + str(conds_all[i]))
            i += 1
        
conds = conds_all[int(sys.argv[1])]



#######################################
# Functions
#######################################

def create_noise_free_training_set(p, sim, training_set_size=20000):
    gen = RandomLUTGenerator(sim, p["exp_parameters"], sim_parameters=p["exp_sim_parameters"])
    X, Y, M = gen.generate(training_set_size)
    return(X, Y, M)



    
    
def setup_dirs(project_path, conds):

    # Set up some dirs
    currdir = project_path + "results/"
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
            
    currdir = currdir + str(conds["budget"]) + "/"
    if(not(os.path.exists(currdir))):
        try:
            os.mkdir(currdir)
        except:
            pass
            
            
    dirname = p["exp_eval_sample_local"]
    if(dirname == "local"):
        if(p["exp_eval_sample_local_indirect"]):
            dirname = "local_indirect"
    currdir = currdir + dirname + "/"
    if(not(os.path.exists(currdir))):
        try:
            os.mkdir(currdir)
        except:
            pass
            
    
    path_rt = currdir + "running_times.csv"
    if(overwrite or not(os.path.exists(path_rt))):
        s = "instance,running_time\n"
        with open(path_rt, 'w') as fp:
            fp.write(s)
            
    return(currdir, path_rt)
    

#######################################
# Big experiment function
#######################################


def run_experiment(p, conds, data, project_path, overwrite, clf_test_sample_size=1000):

    ###### SETUP ################################
    
    currdir, path_rt = setup_dirs(project_path, conds)
    #if(p["data_store_sample_test"]):
    p["store_sample_path"] = p["exp_project_path"] + "data/experiments/" + conds["simulator"] + "/test/"
    
    ###### TRAINING #############################
    
    # If training, do that first
    if(conds["method"] in ["RF", "GP", "BNN", "Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ", "AutoIPQ", "TabPFN"]):
    
        # Saving training times just in case. Kind of pointless for CSV, but convenient
        path_rt2 = currdir + "training_time.csv"
        if(overwrite or not(os.path.exists(path_rt2))):
            s = "training_time\n"
            with open(path_rt2, 'w') as fp:
                fp.write(s)
                
        # If training, do that first
        if(conds["method"] in ["RF", "GP", "BNN", "Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ", "AutoIPQ", "TabPFN"]):
                    
            # Instantiate model        
            if(conds["method"] == "GP"):
                model, training_time = train_GP(p, data)
            elif(conds["method"] == "RF"):
                model, training_time = train_RF(p, data)
            elif(conds["method"] == "BNN"):
                model, training_time = train_BNN(p, data)
            elif(conds["method"] == "TabPFN"):
                regs, training_time = train_TabPFN(data)
            elif(conds["method"] in ["Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ", "AutoIPQ"]):
                warm_start_model, training_time = train_GP(p, data)
            else: 
                raise NotImplementedError()

            with open(path_rt2, 'a') as fp:
                fp.write(str(training_time) + "\n")
    
    
    ###### INFERENCE ############################
    
    num_instances = data["test"]["X"].shape[0]
    print("num_instances:", num_instances)
    order = np.arange(num_instances)
    print("order:", order)
    np.random.shuffle(order)
    #print("order:", order)
    np.random.seed(p["exp_seed"]) # Do this after order to allow multiple simultaneous runs
    
    for i in order:
        # Check if result already exists for this instance
        
        path_i = currdir + str(i) + ".csv"
        if(overwrite or not(os.path.exists(path_i))):

            # Setup for sampling
            x_i = data["test"]["X"][i]
            m_i = data["test"]["M"][i]
            opt_i = data["test"]["opt"][i]
            opt_loss_i = data["test"]["opt_loss"][i]
            epsilon_loss = opt_loss_i + p["exp_epsilon"]
            
            priors = {p["exp_parameters_simulation"][k]:m_i[k] for k in range(len(p["exp_parameters_simulation"]))}
            sim = instantiate_sim(p["simdata_simulator"], priors=priors)
            
            # Load previously stored local sample, or generate new uniform sample
            if(p["exp_eval_sample_local"] == "local"):
                test_X, test_Y = load_local_eval_sample(p, conds, i, epsilon_loss)
            elif(p["exp_eval_sample_local"] == "uniform"):
                test_X, test_Y = load_local_eval_sample(p, conds, i, epsilon_loss, version="uniform")
            elif(p["exp_eval_sample_local"] == "indirect"):
                test_X, test_Y = load_local_eval_sample(p, conds, i, epsilon_loss, version="indirect")
            else:
                test_X, test_Y = create_eval_sample(p, sim, x_i, epsilon_loss, clf_test_sample_size)
            
            to_run = True
            # Don't run on instances where a) saving went wrong, or b) ground truth is all 1 or all 0
            if(test_X.shape[0] != test_Y.shape[0] or len(test_Y) < 50):
                # Checked these files, weird malformed stuff going on, probably from two workers editing the same file, sometimes appends malformed lines. Most files are not affected in the first place. Skipping these instances
                #print("WARNING: too small (len: " + str(len(test_Y)) + ") or mismatch in shape for test_X and test_Y, conds: " + str(conds) + ", instance: " + str(i))
                to_run = False
                            
            label_sum = np.sum(test_Y)
            if(label_sum == len(test_Y) or label_sum == 0):
                #print("WARNING: bad validation sample with label sum " + str(label_sum) + " out of " + str(len(test_Y)) + ", for conds: " + str(conds) + ", instance: " + str(i))
                to_run = False
                
                
            if(to_run):

                # Call method functions
                st = time.time()
                if(conds["method"] == "AutoIPQ"):
                    preds = run_AutoIPQ(p, data["test"], i, sim, test_X, variant=p["method_variant"], wsm=warm_start_model)
                elif(conds["method"] == "Conv-AutoIPQ"):
                    preds = run_ConvAutoIPQ(p, data["test"], i, sim, test_X, wsm=warm_start_model)
                elif(conds["method"] == "Seq-AutoIPQ"):
                    preds = run_AutoIPQ(p, data["test"], i, sim, test_X, variant="Seq-AutoIPQ", wsm=warm_start_model)
                elif(conds["method"] == "Dual-AutoIPQ"):
                    preds = run_AutoIPQ(p, data["test"], i, sim, test_X, variant="Dual-AutoIPQ", wsm=warm_start_model)
                elif(conds["method"] == "ABCSMC"):
                    preds = run_ABCSMC(p, sim, data["val"]["X"][i], test_X, num_populations=p["bl_abcsmc_num_populations"])
                elif(conds["method"] == "TabPFN"):
                    preds = run_TabPFN(regs, data["val"]["X"][i], test_X)
                elif(conds["method"] == "GP"):
                    preds = run_GP(p, data["test"], i, model, test_X)
                elif(conds["method"] == "RF"):
                    preds = run_RF(p, data["test"], i, model, test_X)
                elif(conds["method"] == "BNN"):
                    preds = run_BNN(p, data["test"], i, model, test_X)
                elif(conds["method"] == "classification"):
                    try:
                        model = train_CLF(p, sim, x_i, epsilon_loss, model_type="GBC")
                        preds = run_CLF(p, model, test_X)
                    except ValueError:
                        #print("WARNING: failed classifier training for conds: " + str(conds) + ", instance: " + str(i) + ". Returning default vector of 1s (training set all 1s).")
                        preds = np.ones(test_Y.shape[0])
                elif(conds["method"] == "auto-sklearn"):
                    model = train_CLF(p, sim, x_i, epsilon_loss, model_type="auto-sklearn")
                    preds = run_CLF(p, model, test_X)
                # TODO: active learning classification
                else:
                    raise NotImplementedError()
                et = time.time()
                
                
                #print("conds:", conds)
                #print("preds len:", len(preds))
                #print("test_X shape:", test_X.shape)
                #print("test_Y shape:", test_Y.shape)
                
                # Save running time
                with open(path_rt, 'a') as fp:
                    s = str(i) + "," + str(et-st) + "\n"
                    fp.write(s)
                    
                # Save results
                s = "point_i,true_class,pred_class\n"
                for p_i in range(len(preds)):
                    s = s + str(p_i) + "," + str(test_Y[p_i]) + "," + str(preds[p_i]) + "\n"
                with open(path_i, 'w') as fp:
                    fp.write(s)
                    
 
 
#######################################
# Running experiments
#######################################


if(__name__ == '__main__'):
    p = set_default_parameters()
    p["exp_eval_sample_local"] = sample_local
    load_parameters_file(p, param_paths[conds["simulator"]])
    try:
        p = load_parameters_from_smac(p, conds["simulator"], conds["method"])
    except:
        pass
    data = load_dataset(conds["simulator"], project_path, max_instances=total_instances, noise=p["data_noise"])

    run_experiment(p, conds, data, project_path, overwrite, clf_test_sample_size=clf_test_sample_size)
    
    
    print("Terminated successfully:", conds)