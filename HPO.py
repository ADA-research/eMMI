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

from sklearn.metrics import f1_score

from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer
from ConfigSpace.conditions import InCondition
from smac import HyperparameterOptimizationFacade, Scenario




#######################################
# Setup
#######################################

project_path = "/scratch/arp/AutoIPQ/"
total_instances = 200
clf_test_sample_size = 1000
val_batch_size = 10 # Number of validation instances to evaluate on per configuration

#sample_local = True if sys.argv[2]=="True" else False
#sample_local_indirect = False
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

methods_all = ["AutoIPQ", "Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ", "RF", "GP", "BNN", "ABCSMC"]#, "classification"]#, "auto-sklearn"]
methods_own = ["AutoIPQ", "Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ"]
methods_bl = ["RF", "GP", "BNN", "ABCSMC"]

methods_spec = sys.argv[3]
if(methods_spec == "all"):
    methods = methods_all
elif(methods_spec == "own"):
    methods = methods_own
elif(methods_spec == "bl"):
    methods = methods_bl
    
methods = ["Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ"] # TODO: remove

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
        "sbi_gm",
        "dpr_tp",
        "dpr_lorenz63",
        #"sbi_slcp",
        #"sbi_bglm",
        #"sbi_bglmr",
        "sbi_tm",
        #"sbi_gl",
        #"sbi_glu",
        #"sbi_slcpd",
    ]
simulators = ["dpr_tp"] # TODO: remove
budgets = [2000, 6000, 12000]
budgets = [12000] # Don't need multiple budgets

conds_all = {}
i = 1
for simulator in simulators:
    for method in methods:
        for budget in budgets:
            conds_all[i] = {"method":method, "simulator":simulator, "budget":budget}
            i += 1
        
conds = conds_all[int(sys.argv[1])]

# Setup smac dir
currdir = "smac/"
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


#######################################
# Functions
#######################################


def create_search_space_AutoIPQ(p, variant="Dual-AutoIPQ"):

    if(variant == "Dual-AutoIPQ"):
        suffix = "_dual"
    elif(variant == "Seq-AutoIPQ"):
        suffix = "_seq"
    elif(variant == "Conv-AutoIPQ"):
        suffix = "_conv"
    else:
        raise NotImplementedError

    cs = ConfigurationSpace(seed=p["exp_seed"])
    
    # When interpreting, these should be divided by their sum (sum to 1.0)
    method_budget_split_a = Float("method"+suffix+"_budget_split_a", (0.0, 1.0), default=0.2)
    method_budget_split_b = Float("method"+suffix+"_budget_split_b", (0.0, 1.0), default=0.1)
    method_budget_split_c = Float("method"+suffix+"_budget_split_c", (0.0, 1.0), default=0.7)
    
    method_population_size = Integer("method"+suffix+"_population_size", (1, 100), default=20)
    method_k = Integer("method"+suffix+"_k", (1, 100), default=10) # Should be <= pop size when interpreting
    method_diversity_weight = Float("method"+suffix+"_diversity_weight", (0.0, 1.0), default=0.5)
    method_epsilon2 = Float("method"+suffix+"_epsilon2", (0.001, 1.0), default=0.1)
    method_learning_rate = Float("method"+suffix+"_learning_rate", (0.00001, 1.0), default=0.1)
    method_init_noise = Float("method"+suffix+"_init_noise", (0.00001, 1.0), default=0.01)
    method_constraint_penalty_inf = Categorical("constraint_penalty_inf", ["True", "False"], default="True") # Interpret as bool later, set to np.inf then
    method_constraint_penalty = Integer("method"+suffix+"_constraint_penalty", (1, 10), default=3) 
    method_filter_density = Categorical("method"+suffix+"_filter_density", ["True", "False"], default="False") # Interpret as bool later
    method_density_filter_percentile = Integer("method"+suffix+"_density_filter_percentile", (1, 99), default=10) 
    method_density_filter_same_class = Categorical("method"+suffix+"_density_filter_same_class", ["True", "False"], default="False") # interpret as bool
    method_random_exploration_proportion = Float("method"+suffix+"_random_exploration_proportion", (0.0, 1.0), default=0.1)
    
    method_optimiser_1 = Categorical("method"+suffix+"_optimiser_1", ["CMAES", "RS", "HC", "GD"], default="CMAES")
    method_optimiser_2a = Categorical("method"+suffix+"_optimiser_2a", ["RS", "HC", "GD"], default="HC")
    method_optimiser_2b = Categorical("method"+suffix+"_optimiser_2b", ["RS", "HC", "GD"], default="HC")
    method_optimiser_2c = Categorical("method"+suffix+"_optimiser_2c", ["CMAES", "RS", "HC", "GD"], default="GD")
    method_optimiser_2d = Categorical("method"+suffix+"_optimiser_2d", ["RS", "HC", "GD"], default="HC")

    cond_method_density_filter_same_class = InCondition(child=method_density_filter_same_class, parent=method_filter_density, values=["True"])
    cond_method_density_filter_percentile = InCondition(child=method_density_filter_percentile, parent=method_filter_density, values=["True"])
    cond_method_constraint_penalty = InCondition(child=method_constraint_penalty, parent=method_constraint_penalty_inf, values=["False"])
    
    cs.add([method_budget_split_a, method_budget_split_b, method_budget_split_c, method_population_size, method_k, method_learning_rate, method_init_noise, method_constraint_penalty_inf, method_constraint_penalty, method_optimiser_1, method_optimiser_2a, method_optimiser_2b, method_optimiser_2c, method_optimiser_2d, cond_method_constraint_penalty])
    
    # Variant specific hyperparameters
    if(variant == "Dual-AutoIPQ"):
        cs.add([method_random_exploration_proportion, method_diversity_weight, method_filter_density, method_density_filter_percentile, method_density_filter_same_class, cond_method_density_filter_same_class, cond_method_density_filter_percentile])
    elif(variant == "Seq-AutoIPQ"):
        cs.add([method_random_exploration_proportion, method_epsilon2, method_filter_density, method_density_filter_percentile, method_density_filter_same_class, cond_method_density_filter_same_class, cond_method_density_filter_percentile])
    elif(variant == "Conv-AutoIPQ"):
        suffix = "_conv"
    
    return(cs)


def create_search_space_AutoIPQ_unified(p):

    suffix = ""

    cs = ConfigurationSpace(seed=p["exp_seed"])

    # Selecting variant
    method_variant = Categorical("method"+suffix+"_variant", ["Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ", "uniform"], default="Dual-AutoIPQ") 
    cs.add([method_variant])
    
    # When interpreting, these should be divided by their sum (sum to 1.0)
    method_budget_split_a = Float("method"+suffix+"_budget_split_a", (0.0, 1.0), default=0.2)
    method_budget_split_b = Float("method"+suffix+"_budget_split_b", (0.0, 1.0), default=0.1)
    method_budget_split_c = Float("method"+suffix+"_budget_split_c", (0.0, 1.0), default=0.7)

    # If uniformly sampling, don't need most of these parameters, just variant, budget split and step 1 optimiser. But it's a hassle to set up conditions for all of these, let's just do it in the interpretation
    
    method_population_size = Integer("method"+suffix+"_population_size", (1, 100), default=20)
    method_k = Integer("method"+suffix+"_k", (1, 100), default=10) # Should be <= pop size when interpreting
    method_diversity_weight = Float("method"+suffix+"_diversity_weight", (0.0, 1.0), default=0.5)
    method_epsilon2 = Float("method"+suffix+"_epsilon2", (0.001, 1.0), default=0.1)
    method_learning_rate = Float("method"+suffix+"_learning_rate", (0.00001, 1.0), default=0.1)
    method_init_noise = Float("method"+suffix+"_init_noise", (0.00001, 1.0), default=0.01)
    method_constraint_penalty_inf = Categorical("constraint_penalty_inf", ["True", "False"], default="True") # Interpret as bool later, set to np.inf then
    method_constraint_penalty = Integer("method"+suffix+"_constraint_penalty", (1, 10), default=3) 
    method_filter_density = Categorical("method"+suffix+"_filter_density", ["True", "False"], default="False") # Interpret as bool later
    method_density_filter_percentile = Integer("method"+suffix+"_density_filter_percentile", (1, 99), default=10) 
    method_density_filter_same_class = Categorical("method"+suffix+"_density_filter_same_class", ["True", "False"], default="False") # interpret as bool
    method_random_exploration_proportion = Float("method"+suffix+"_random_exploration_proportion", (0.0, 1.0), default=0.1)
    
    method_optimiser_1 = Categorical("method"+suffix+"_optimiser_1", ["CMAES", "RS", "HC", "GD"], default="CMAES")
    method_optimiser_2a = Categorical("method"+suffix+"_optimiser_2a", ["RS", "HC", "GD"], default="HC")
    method_optimiser_2b = Categorical("method"+suffix+"_optimiser_2b", ["RS", "HC", "GD"], default="HC")
    method_optimiser_2c = Categorical("method"+suffix+"_optimiser_2c", ["CMAES", "RS", "HC", "GD"], default="GD")
    method_optimiser_2d = Categorical("method"+suffix+"_optimiser_2d", ["RS", "HC", "GD"], default="HC")

    cond_method_constraint_penalty = InCondition(child=method_constraint_penalty, parent=method_constraint_penalty_inf, values=["False"])

    

    
    cs.add([method_budget_split_a, method_budget_split_b, method_budget_split_c, method_population_size, method_k, method_learning_rate, 
            method_init_noise, method_constraint_penalty_inf, method_constraint_penalty, method_optimiser_1, method_optimiser_2a, 
            method_optimiser_2b, method_optimiser_2c, method_optimiser_2d, cond_method_constraint_penalty, method_random_exploration_proportion, 
            method_filter_density, method_density_filter_percentile, method_density_filter_same_class, method_diversity_weight, method_epsilon2])
    
    # Variant specific hyperparameters
    cond_method_random_exploration_proportion = InCondition(child=method_random_exploration_proportion, parent=method_variant, values=["Dual-AutoIPQ", "Seq-AutoIPQ"])
    cond_method_filter_density = InCondition(child=method_filter_density, parent=method_variant, values=["Dual-AutoIPQ", "Seq-AutoIPQ"])
    cond_method_density_filter_percentile = InCondition(child=method_density_filter_percentile, parent=method_filter_density, values=["True"])
    cond_method_density_filter_same_class = InCondition(child=method_density_filter_same_class, parent=method_filter_density, values=["True"])

    cond_method_diversity_weight = InCondition(child=method_diversity_weight, parent=method_variant, values=["Dual-AutoIPQ"])

    cond_method_epsilon2 = InCondition(child=method_epsilon2, parent=method_variant, values=["Seq-AutoIPQ"])

    cs.add([cond_method_random_exploration_proportion, cond_method_filter_density, cond_method_density_filter_percentile, cond_method_density_filter_same_class, 
            cond_method_diversity_weight, cond_method_epsilon2])

    
    return(cs)
    
    
def create_search_space_ABCSMC(p):
    cs = ConfigurationSpace(seed=p["exp_seed"])
    
    bl_abcsmc_num_populations = Integer("bl_abcsmc_num_populations", (1, 100), default=10)
    
    cs.add([bl_abcsmc_num_populations])
    
    return(cs)


def create_search_space_GP(p):
    cs = ConfigurationSpace(seed=p["exp_seed"])
    
    kernels = ["None", "ConstantKernel", "DotProduct", "ExpSineSquared", "Matern", "RBF", "RationalQuadratic", "WhiteKernel"]
    bl_gp_kernel = Categorical("bl_gp_kernel", kernels, default="None")
    bl_gp_optimizer = Categorical("bl_gp_optimizer", ["fmin_l_bfgs_b"], default="fmin_l_bfgs_b")
    bl_gp_restarts = Integer("bl_gp_restarts", (0, 10), default=0)
    bl_gp_normalize_y = Categorical("bl_gp_normalize_y", ["True", "False"], default="False")
    
    cs.add([bl_gp_kernel, bl_gp_optimizer, bl_gp_restarts, bl_gp_normalize_y])
    
    return(cs)
    
    
def create_search_space_RF(p):
    cs = ConfigurationSpace(seed=p["exp_seed"])
    
    bl_rf_n_estimators = Integer("bl_rf_n_estimators", (2, 1000), default=100)
    bl_rf_criterion = Categorical("bl_rf_criterion", ["gini", "entropy", "log_loss"], default="gini")
    bl_rf_use_max_depth = Categorical("bl_rf_use_max_depth", ["True", "False"], default="False")
    bl_rf_max_depth = Integer("bl_rf_max_depth", (2, 200), default=100) # conditional
    bl_rf_min_samples_split = Integer("bl_rf_min_samples_split", (2, 50), default=2)
    bl_rf_min_samples_leaf = Integer("bl_rf_min_samples_leaf", (1, 50), default=1)
    bl_rf_min_weight_fraction_leaf = Float("bl_rf_min_weight_fraction_leaf", (0.0, 0.5), default=0.0)
    bl_rf_max_features = Categorical("bl_rf_max_features", ["sqrt", "log2", "None"], default="sqrt")
    bl_rf_use_max_leaf_nodes = Categorical("bl_rf_use_max_leaf_nodes", ["True", "False"], default="False")
    bl_rf_max_leaf_nodes = Integer("bl_rf_max_leaf_nodes", (2, 1000), default=2) # conditional
    bl_rf_min_impurity_decrease = Float("bl_rf_min_impurity_decrease", (0.0, 1.0), default=0.0)
    bl_rf_bootstrap = Categorical("bl_rf_bootstrap", ["True", "False"], default="True")
    bl_rf_ccp_alpha = Float("bl_rf_ccp_alpha", (0.0, 1.0), default=0.0)

    cond_bl_rf_max_depth = InCondition(child=bl_rf_max_depth, parent=bl_rf_use_max_depth, values=["True"])
    cond_bl_rf_max_leaf_nodes = InCondition(child=bl_rf_max_leaf_nodes, parent=bl_rf_use_max_leaf_nodes, values=["True"])
    
    cs.add([bl_rf_n_estimators, bl_rf_criterion, bl_rf_use_max_depth, bl_rf_max_depth, bl_rf_min_samples_split, bl_rf_min_samples_leaf, bl_rf_min_weight_fraction_leaf, bl_rf_max_features, bl_rf_use_max_leaf_nodes, bl_rf_max_leaf_nodes, bl_rf_min_impurity_decrease,bl_rf_bootstrap, bl_rf_ccp_alpha, cond_bl_rf_max_depth,cond_bl_rf_max_leaf_nodes])
    
    return(cs)
    
    
def create_search_space_BNN(p):
    cs = ConfigurationSpace(seed=p["exp_seed"])
    
    bl_bnn_mu = Float("bl_bnn_prior_mu", (-10, 10), default=0)
    bl_bnn_sigma = Float("bl_bnn_prior_sigma", (0.0001, 10.0), default=0.1)
    bl_bnn_hidden_layers = Integer("bl_bnn_num_hidden_layers", (1, 5), default=1)
    bl_bnn_num_features = Integer("bl_bnn_num_features", (2, 500), default=100)
    bl_bnn_kl_weight = Float("bl_bnn_kl_weight", (0.001, 0.999), default=0.1)
    bl_bnn_lr = Float("bl_bnn_learning_rate", (0.0001, 0.1), default=0.01)
    
    cs.add([bl_bnn_mu, bl_bnn_sigma, bl_bnn_hidden_layers, bl_bnn_num_features, bl_bnn_kl_weight, bl_bnn_lr])
    
    return(cs)

    

#######################################
# Big HPO function
#######################################

#def run_experiment_smac(cfg, p_orig, conds, data, project_path, overwrite, clf_test_sample_size, val_batch_size):
def run_experiment_smac(cfg, seed=42):

    # Dumb hacky arg workaround stuff
    
    p = p_g.copy() # just in case of some shenanigans
    conds = conds_g
    data = data_g
    project_path = project_path_g
    clf_test_sample_size = clf_test_sample_size_g
    val_batch_size = val_batch_size_g
    
    ###### INTERPRET CFG ########################
    
    # General
    cfg_dict = cfg.get_dictionary()
    for k,v in cfg_dict.items():
        if(v == "True"):
            v = True
        if(v == "False"):
            v = False
        if(v == "None"):
            v = None
        p[k] = v
    
    # AutoIPQ specific
    if("method_budget_split_a" in cfg_dict):
        s = cfg_dict["method_budget_split_a"] + cfg_dict["method_budget_split_b"] + cfg_dict["method_budget_split_c"]
        l = [cfg_dict["method_budget_split_a"]/s, cfg_dict["method_budget_split_b"]/s, cfg_dict["method_budget_split_c"]/s]
        p["method_budget_split"] = l
    if("method_k" in cfg_dict):
        p["method_k"] = min(cfg_dict["method_k"], cfg_dict["method_population_size"])
        
    # GP specific
    if("bl_gp_kernel" in cfg_dict):
        from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, ExpSineSquared, Matern, RBF, RationalQuadratic, WhiteKernel
        kernel_s = cfg_dict["bl_gp_kernel"]
        if(kernel_s == "ConstantKernel"):
            p["bl_gp_kernel"] = ConstantKernel()
        elif(kernel_s == "DotProduct"):
            p["bl_gp_kernel"] = DotProduct()
        elif(kernel_s == "ExpSineSquared"):
            p["bl_gp_kernel"] = ExpSineSquared()
        elif(kernel_s == "Matern"):
            p["bl_gp_kernel"] = Matern()
        elif(kernel_s == "RBF"):
            p["bl_gp_kernel"] = RBF()
        elif(kernel_s == "RationalQuadratic"):
            p["bl_gp_kernel"] = RationalQuadratic()
        elif(kernel_s == "WhiteKernel"):
            p["bl_gp_kernel"] = WhiteKernel()
        
    # RF specific
    if("bl_rf_use_max_depth" in cfg_dict):
        p["bl_rf_max_depth"] = None
    if("bl_rf_use_max_leaf_nodes" in cfg_dict):
        p["bl_rf_max_leaf_nodes"] = None
    
    ###### TRAINING #############################
    
    # If training, do that first
    if(conds["method"] in ["RF", "GP", "BNN", "AutoIPQ", "Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ"]):
                
        # Instantiate model        
        if(conds["method"] == "GP"):
            model, training_time = train_GP(p, data)
        elif(conds["method"] == "RF"):
            model, training_time = train_RF(p, data)
        elif(conds["method"] == "BNN"):
            model, training_time = train_BNN(p, data)
        elif(conds["method"] in ["AutoIPQ", "Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ"]):
            warm_start_model, training_time = train_GP(p, data)
        else: 
            raise NotImplementedError()

    
    ###### INFERENCE ############################
    
    num_instances = data["val"]["X"].shape[0]
    order = np.arange(num_instances)[:val_batch_size]
    np.random.shuffle(order)
    np.random.seed(p["exp_seed"]) # Do this after order to allow multiple simultaneous runs
    
    pms = []
    
    for i in order:
        # Check if result already exists for this instance

        # Setup for sampling
        x_i = data["val"]["X"][i]
        m_i = data["val"]["M"][i]
        opt_i = data["val"]["opt"][i]
        epsilon_loss = data["val"]["opt_loss"][i] + p["exp_epsilon"]
        
        priors = {p["exp_parameters_simulation"][k]:m_i[k] for k in range(len(p["exp_parameters_simulation"]))}
        sim = instantiate_sim(p["simdata_simulator"], num_samples=p["simdata_num_samples"], priors=priors, dim=p["simdata_dim"])

        
        # Load previously stored local sample, or generate new uniform sample
        if(p["exp_eval_sample_local"] == "local"):
            test_X, test_Y = load_local_eval_sample(p, conds, i, epsilon_loss)
        elif(p["exp_eval_sample_local"] == "uniform"):
            test_X, test_Y = load_local_eval_sample(p, conds, i, epsilon_loss, version="uniform")
        else:
            test_X, test_Y = create_eval_sample(p, sim, x_i, epsilon_loss, clf_test_sample_size)
            
        if(test_X.shape[0] == len(test_Y) and len(test_Y) > 50):

            # Call method functions
            if(conds["method"] == "AutoIPQ"):
                preds = run_AutoIPQ(p, data["val"], i, sim, test_X, variant=p["method_variant"], wsm=warm_start_model)
            elif(conds["method"] == "Conv-AutoIPQ"):
                preds = run_ConvAutoIPQ(p, data["val"], i, sim, test_X, wsm=warm_start_model)
            elif(conds["method"] == "Seq-AutoIPQ"):
                preds = run_AutoIPQ(p, data["val"], i, sim, test_X, variant="Seq-AutoIPQ", wsm=warm_start_model)
            elif(conds["method"] == "Dual-AutoIPQ"):
                preds = run_AutoIPQ(p, data["val"], i, sim, test_X, variant="Dual-AutoIPQ", wsm=warm_start_model)
            elif(conds["method"] == "ABCSMC"):
                preds = run_ABCSMC(p, sim, data["val"]["X"][i], test_X, num_populations=p["bl_abcsmc_num_populations"])
            elif(conds["method"] == "GP"):
                preds = run_GP(p, data["val"], i, model, test_X)
            elif(conds["method"] == "RF"):
                preds = run_RF(p, data["val"], i, model, test_X)
            elif(conds["method"] == "BNN"):
                preds = run_BNN(p, data["val"], i, model, test_X)
            elif(conds["method"] == "classification"):
                model = train_CLF(p, sim, x_i, epsilon_loss, model_type="GBC")
                preds = run_CLF(p, model, test_X)
            elif(conds["method"] == "auto-sklearn"):
                model = train_CLF(p, sim, x_i, epsilon_loss, model_type="auto-sklearn")
                preds = run_CLF(p, model, test_X)
            # TODO: active learning classification
            else:
                raise NotImplementedError()
                
                
            # Compute metric
            metric = f1_score(test_Y, preds)
            pms.append(metric)
            
        else:
            print("Dataset too small or shape mismatch for instance " + str(i) + " on conditions " + str(conds) + ": " + str(test_X.shape) + ", " + str(test_Y.shape))
    
    return(1-np.median(np.array(metric))) # Because we normally minimise

 
 
#######################################
# Running experiments
#######################################


if(__name__ == '__main__'):
    p = set_default_parameters()
    p["exp_eval_sample_local"] = sample_local
    #p["exp_eval_sample_local_indirect"] = sample_local_indirect
    p["data_store_sample_test"] = False # We don't save new sample points
    #p["store_sample_path"] = p["exp_project_path"] + "data/experiments/" + conds["simulator"] + "/test/" # For evaluating on previously stored points
    load_parameters_file(p, param_paths[conds["simulator"]])
    data = load_dataset(conds["simulator"], project_path, max_instances=total_instances, noise=p["data_noise"])
    
    if(conds["method"] == "AutoIPQ"):
        cs = create_search_space_AutoIPQ_unified(p)
    elif(conds["method"] in ["Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ"]):
        cs = create_search_space_AutoIPQ(p, variant=conds["method"])
    elif(conds["method"] == "ABCSMC"):
        cs = create_search_space_ABCSMC(p)
    elif(conds["method"] == "GP"):
        cs = create_search_space_GP(p)
    elif(conds["method"] == "RF"):
        cs = create_search_space_RF(p)
    elif(conds["method"] == "BNN"):
        cs = create_search_space_BNN(p)
    elif(conds["method"] == "classification"):
        cs = create_search_space_classification(p)
    elif(conds["method"] == "auto-sklearn"):
        cs = create_search_space_autosklearn(p)
        
        
    # This would be much better if I could actually pass them as args, throws errors
    global p_g
    global conds_g
    global data_g
    global project_path_g
    global overwrite_g
    global clf_test_sample_size_g
    global val_batch_size_g
    
    p_g = p
    conds_g = conds
    data_g = data
    project_path_g = project_path
    clf_test_sample_size_g = clf_test_sample_size
    val_batch_size_g = val_batch_size
        
    scenario = Scenario(
        cs,
        name="HPO_"+conds["method"],
        output_directory=currdir,
        #n_trials=2, # TODO: remove
        walltime_limit=172800, # TODO: uncomment (48 hours)
        #walltime_limit=3600, # TODO: remove
    )
    
    smac = HyperparameterOptimizationFacade(
        scenario,
        run_experiment_smac,
        overwrite=True,
    )
    
    incumbent = smac.optimize()
    
    # Get cost of default configuration
    default_cost = smac.validate(cs.get_default_configuration())
    print("\nDefault cost:", default_cost)
    print("Default configuration:", cs.get_default_configuration())

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print("\nIncumbent cost:", incumbent_cost)
    print("Incumbent:", incumbent)
    
    print("Terminated successfully:", conds)