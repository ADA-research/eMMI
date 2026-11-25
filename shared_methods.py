import numpy as np

import time

from sklearn.ensemble import GradientBoostingClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

import pyabc

from tabpfn import TabPFNRegressor

from src.AutoIPQ import *

from shared_utils import *











def run_ConvAutoIPQ(p, data, i, sim, test_X, wsm=None):
        
    method_kwargs = {
                        "budget": p["method_budget"], 
                        "budget_split": [p["method_conv_budget_split_a"], p["method_conv_budget_split_b"], p["method_conv_budget_split_c"]],
                        "num_solutions": p["method_conv_population_size"],
                        "k": p["method_conv_k"], 
                        "approach": "iterated",
                        "learning_rate": p["method_conv_learning_rate"],
                        "constraint_penalty": p["method_conv_constraint_penalty"],
                        "warm_start_method": "hybrid",
                        "warm_start_model": wsm,
                        "return_opt": False, 
                        "store_XY": False, 
                        "store_test":p["exp_eval_sample_local"],
                        "train_clf": True,
                    }

    
    instance_x = data["X"][i]
    
    # Run AutoIPQ to train classifier(s)
    auto = AutoIPQ(sim, instance_x, p["exp_parameters"], p["method_population_size"], optimiser=p["method_optimiser_1"], optimiser_2a=p["method_optimiser_2a"], optimiser_2b=p["method_optimiser_2b"], optimiser_2c=p["method_optimiser_2c"], optimiser_2d=p["method_optimiser_2d"])

    #l_clf = auto.run(p["exp_epsilon"], instantiate_boost, **method_kwargs)
    l_ch = auto.run(p["exp_epsilon"], None, **method_kwargs)
    
    
    # Inference
    
    
    
    preds = []
    for ch in l_ch:
        pred = auto.points_in_convex_hull(ch, test_X)
        preds.append(pred)
    preds = np.array(preds)
    pred = np.max(np.array(preds), axis=0) # If point in any hulls, label should be 1
        
    return(pred)


def run_AutoIPQ_old(p, data, i, sim, test_X, variant="Dual-AutoIPQ", wsm=None):

    if(variant == "Conv-AutoIPQ"):
        p["method_step2_approach"] = "iterated"
    elif(variant == "Seq-AutoIPQ"):
        p["method_step2_approach"] = "serial"
    elif(variant == "Dual-AutoIPQ"):
        p["method_step2_approach"] = "split"
    else:
        raise NotImplementedError

    def instantiate_boost():
        return(GradientBoostingClassifier())
        
    #def instantiate_autosklearn():
    #    automl = AutoSklearnClassifier(
    #        time_left_for_this_task=120,
    #        per_run_time_limit=30,
    #        #tmp_folder="/home/laurens/tmp/autosklearn_classification_example_tmp",
    #    )
    #    return(automl)
        
        
    if(variant == "Conv-AutoIPQ"):
        raise NotImplementedError()
    elif(variant == "Seq-AutoIPQ"):
        method_kwargs = {
                            "budget": p["method_budget"], 
                            "budget_split": [p["method_seq_budget_split_a"], p["method_seq_budget_split_b"], p["method_seq_budget_split_c"]],
                            "num_solutions": p["method_seq_population_size"],
                            "k": p["method_seq_k"], 
                            "approach": "serial",
                            "epsilon2": p["method_seq_epsilon2"],
                            "learning_rate": p["method_seq_learning_rate"],
                            "init_noise": p["method_seq_init_noise"], 
                            "constraint_penalty": p["method_seq_constraint_penalty"],
                            "filter_density": p["method_seq_filter_density"],
                            "density_filter_percentile": p["method_seq_density_filter_percentile"],
                            "density_filter_same_class": p["method_seq_density_filter_same_class"],
                            "warm_start_method": "hybrid",
                            "warm_start_model": wsm,
                            "random_exploration_proportion": p["method_seq_random_exploration_proportion"],
                            "return_opt": False, 
                            "store_XY": False, 
                            #"store_test":p["exp_eval_sample_local"],
                            "train_clf": True,
                        }
    elif(variant == "Dual-AutoIPQ"):
        method_kwargs = {
                            "budget": p["method_budget"], 
                            "budget_split": [p["method_dual_budget_split_a"], p["method_dual_budget_split_b"], p["method_dual_budget_split_c"]],
                            "num_solutions": p["method_dual_population_size"],
                            "k": p["method_dual_k"], 
                            "diversity_weight": p["method_dual_diversity_weight"],
                            "approach": "split",
                            "learning_rate": p["method_dual_learning_rate"],
                            "init_noise": p["method_dual_init_noise"], 
                            "constraint_penalty": p["method_dual_constraint_penalty"],
                            "filter_density": p["method_dual_filter_density"],
                            "density_filter_percentile": p["method_dual_density_filter_percentile"],
                            "density_filter_same_class": p["method_dual_density_filter_same_class"],
                            "warm_start_method": "hybrid",
                            "warm_start_model": wsm,
                            "random_exploration_proportion": p["method_dual_random_exploration_proportion"],
                            "return_opt": False, 
                            "store_XY": False, 
                            #"store_test":p["exp_eval_sample_local"],
                            "train_clf": True,
                        }
    else:
        raise NotImplementedError()
    
    instance_x = data["X"][i]
    
    # Run AutoIPQ to train classifier(s)
    auto = AutoIPQ(sim, instance_x, p["exp_parameters"], p["method_population_size"], optimiser=p["method_optimiser_1"], optimiser_2a=p["method_optimiser_2a"], optimiser_2b=p["method_optimiser_2b"], optimiser_2c=p["method_optimiser_2c"], optimiser_2d=p["method_optimiser_2d"])

    l_clf = auto.run(p["exp_epsilon"], instantiate_boost, **method_kwargs)
    #l_clf = auto.run(p["exp_epsilon"], instantiate_autosklearn, **method_kwargs)
        
    pred = np.zeros(test_X.shape[0])
    for clf_i in range(test_X.shape[0]):
        vec = test_X[clf_i, :].reshape((1, test_X.shape[1]))
        for clf in l_clf:
            pred_local = clf.predict(vec)[0]
            if(pred_local == 1):
                pred[clf_i] = 1
                break
        
    return(pred)



def run_AutoIPQ(p, data, i, sim, test_X, variant="Dual-AutoIPQ", wsm=None, return_auto=False, return_preds=True, **kwargs):

    if(variant == "Conv-AutoIPQ"):
        p["method_step2_approach"] = "iterated"
    elif(variant == "Seq-AutoIPQ"):
        p["method_step2_approach"] = "serial"
    elif(variant == "Dual-AutoIPQ"):
        p["method_step2_approach"] = "split"
    elif(variant == "uniform"):
        p["method_step2_approach"] = "split" # Doesn't really matter, doesn't use it anyway
    else:
        raise NotImplementedError

    def instantiate_boost():
        return(GradientBoostingClassifier())
        
    #def instantiate_autosklearn():
    #    automl = AutoSklearnClassifier(
    #        time_left_for_this_task=120,
    #        per_run_time_limit=30,
    #        #tmp_folder="/home/laurens/tmp/autosklearn_classification_example_tmp",
    #    )
    #    return(automl)
        
        
    if(variant == "Conv-AutoIPQ"):
        method_kwargs = {
                            "budget": p["method_budget"], 
                            "budget_split": [p["method_conv_budget_split_a"], p["method_conv_budget_split_b"], p["method_conv_budget_split_c"]],
                            "num_solutions": p["method_conv_population_size"],
                            "k": p["method_conv_k"], 
                            "approach": "iterated",
                            "learning_rate": p["method_conv_learning_rate"],
                            "constraint_penalty": p["method_conv_constraint_penalty"],
                            "warm_start_method": "hybrid",
                            "warm_start_model": wsm,
                            "return_opt": False, 
                            "store_XY": False, 
                            "store_test":p["exp_eval_sample_local"],
                            "train_clf": True,
                        }
    elif(variant == "Seq-AutoIPQ"):
        method_kwargs = {
                            "budget": p["method_budget"], 
                            "budget_split": [p["method_seq_budget_split_a"], p["method_seq_budget_split_b"], p["method_seq_budget_split_c"]],
                            "num_solutions": p["method_seq_population_size"],
                            "k": p["method_seq_k"], 
                            "approach": "serial",
                            "epsilon2": p["method_seq_epsilon2"],
                            "learning_rate": p["method_seq_learning_rate"],
                            "init_noise": p["method_seq_init_noise"], 
                            "constraint_penalty": p["method_seq_constraint_penalty"],
                            "filter_density": p["method_seq_filter_density"],
                            "density_filter_percentile": p["method_seq_density_filter_percentile"],
                            "density_filter_same_class": p["method_seq_density_filter_same_class"],
                            "warm_start_method": "hybrid",
                            "warm_start_model": wsm,
                            "random_exploration_proportion": p["method_seq_random_exploration_proportion"],
                            "return_opt": False, 
                            "store_XY": False, 
                            #"store_test":p["exp_eval_sample_local"],
                            "train_clf": True,
                        }
    elif(variant == "Dual-AutoIPQ"):
        method_kwargs = {
                            "budget": p["method_budget"], 
                            "budget_split": [p["method_dual_budget_split_a"], p["method_dual_budget_split_b"], p["method_dual_budget_split_c"]],
                            "num_solutions": p["method_dual_population_size"],
                            "k": p["method_dual_k"], 
                            "diversity_weight": p["method_dual_diversity_weight"],
                            "approach": "split",
                            "learning_rate": p["method_dual_learning_rate"],
                            "init_noise": p["method_dual_init_noise"], 
                            "constraint_penalty": p["method_dual_constraint_penalty"],
                            "filter_density": p["method_dual_filter_density"],
                            "density_filter_percentile": p["method_dual_density_filter_percentile"],
                            "density_filter_same_class": p["method_dual_density_filter_same_class"],
                            "warm_start_method": "hybrid",
                            "warm_start_model": wsm,
                            "random_exploration_proportion": p["method_dual_random_exploration_proportion"],
                            "return_opt": False, 
                            "store_XY": False, 
                            #"store_test":p["exp_eval_sample_local"],
                            "train_clf": True,
                        }
    elif(variant == "uniform"):
        method_kwargs = {
                            "budget": p["method_budget"], 
                            "budget_split": [p["method_budget_split_a"], 1, 1],
                            "return_opt": False, 
                            "store_XY": False, 
                            #"store_test":p["exp_eval_sample_local"],
                            "train_clf": False if not(return_preds) else True
                        }
    else:
        raise NotImplementedError()

    # Overwrite method_kwargs by manually specified kwargs
    for k,v in kwargs.items():
        method_kwargs[k] = v
    
    instance_x = data["X"][i]
    
    # Run AutoIPQ to train classifier(s)
    auto = AutoIPQ(sim, instance_x, p["exp_parameters"], p["method_population_size"], optimiser=p["method_optimiser_1"], optimiser_2a=p["method_optimiser_2a"], optimiser_2b=p["method_optimiser_2b"], optimiser_2c=p["method_optimiser_2c"], optimiser_2d=p["method_optimiser_2d"])

    l_clf = auto.run(p["exp_epsilon"], instantiate_boost, **method_kwargs)
    #l_clf = auto.run(p["exp_epsilon"], instantiate_autosklearn, **method_kwargs)

    if(variant in ["uniform"]):
        epsilon_loss = auto.step1_opt_loss + p["exp_epsilon"]
        sample_size = int((p["method_budget_split_b"]+p["method_budget_split_c"]) / (p["method_budget_split_b"]+p["method_budget_split_c"] + p["method_budget_split_a"]) * p["method_budget"])
        if(method_kwargs["train_clf"]):
            clf = train_CLF(p, sim, instance_x, epsilon_loss, sample_size=sample_size, model_type="GBC")
            l_clf = [clf]
        else:
            assert return_preds == False

    if(return_preds):


        
        if(variant in ["Dual-AutoIPQ", "Seq-AutoIPQ", "uniform"]):

            pred = np.zeros(test_X.shape[0])
            for clf_i in range(test_X.shape[0]):
                vec = test_X[clf_i, :].reshape((1, test_X.shape[1]))
                for clf in l_clf:
                    pred_local = clf.predict(vec)[0]
                    if(pred_local == 1):
                        pred[clf_i] = 1
                        break

        elif( variant == "Conv-AutoIPQ"):
            preds = []
            for ch in l_clf:
                pred = auto.points_in_convex_hull(ch, test_X)
                preds.append(pred)
            preds = np.array(preds)
            pred = np.max(np.array(preds), axis=0) # If point in any hulls, label should be 1

        else:
            raise NotImplementedError()

    else:
        pred = None
        
    if(return_auto):
        return(pred, auto)
    else:
        return(pred)


def run_ABCSMC(p, sim, observation, test_X, num_populations=10):

    budget = p["method_budget"]
    population_size = int(budget / num_populations)


    def pyabc_wrapper(parameters):
        return({"data": sim.simulate(parameters)})

    def wrap_priors(p, sim):
        dist_kwargs = {}

        param_ranges = sim.get_param_ranges()

        for param in p["exp_parameters"]:
            r = param_ranges[param]
            dist_kwargs[param] = pyabc.RV("uniform", r[2], r[3]-r[2])

        prior = pyabc.Distribution(**dist_kwargs)
        return(prior)


    def distance(x, x0):
        return(np.mean(np.absolute(x["data"]-x0["data"])))


    abc = pyabc.ABCSMC(pyabc_wrapper, wrap_priors(p, sim), distance, population_size=population_size)


    db_path = "/scratch/arp/test.db"
    abc.new("sqlite:///" + db_path, {"data": observation})

    history = abc.run(minimum_epsilon=p["exp_epsilon"], max_nr_populations=num_populations)


    df, w = history.get_distribution()

    bounds = {}

    for param in p["exp_parameters"]:
        df_local = df[[param]]
        df_local['p'] = w 
        df_local = df_local.sort_values(by=[param])
        df_local['pc'] = np.cumsum(df_local['p'].values)
        low_bound = 0.025
        upper_bound = 0.975

        df_local = df_local[df_local['pc']>low_bound]
        df_local = df_local[df_local['pc']<upper_bound]

        bounds[param] = [np.min(df_local[param]), np.max(df_local[param])]

    
    pred = np.ones(test_X.shape[0])
    for i in range(test_X.shape[0]):
        c = test_X[i, :]
        for j in range(test_X.shape[1]):
            if(test_X[i, j] < bounds[p["exp_parameters"][j]][0] or test_X[i, j] > bounds[p["exp_parameters"][j]][1]):
                pred[i] = 0 

    return(pred)


def run_TabPFN(regs, observation, test_X):

    # Predict for observation
    lows = np.zeros(len(regs))
    highs = np.zeros(len(regs))

    obs = observation.reshape((1, len(observation)))

    for j in range(len(regs)):
        pred_y_low, pred_y_high = regs[j].predict(obs, quantiles=[0.025, 0.975], output_type="quantiles") # list of quantiles; array of results per quantile, use 95% CI
        lows[j] = pred_y_low 
        highs[j] = pred_y_high

    # Classify test_X based on probability

    preds = np.zeros((test_X.shape[0], len(regs)))
    for j in range(preds.shape[1]):
        vec = np.ones(test_X.shape[0])
        vec[test_X[:, j] < lows[j]] = 0
        vec[test_X[:, j] > highs[j]] = 0

    pred = np.min(preds, axis=1)

    return(pred)



def run_GP(p, data, i, model, test_X):
    # For evaluating locally on AutoIPQ test set

    pred = np.ones(test_X.shape[0])

    observation = data["X"][i]
    observation = observation.reshape((1, len(observation)))
    mu, sigma = model.predict(observation, return_std=True)
    mu = mu[0]
    sigma = sigma[0]
    
    for clf_i in range(test_X.shape[0]):
        vec = test_X[clf_i, :]
        for j in range(test_X.shape[1]):
            if(vec[j] < (mu[j] - 2*sigma[j]) or vec[j] > (mu[j] + 2*sigma[j])):
                pred[clf_i] = 0
        
    return(pred)
    
    
def run_RF(p, data, i, model, test_X):

    pred = np.ones(test_X.shape[0])

    observation = data["X"][i]
    observation = observation.reshape((1, len(observation)))
    
    num_trees = p["bl_rf_n_estimators"]
    individual_preds = np.zeros((num_trees, len(p["exp_parameters"])))
    for tree in range(num_trees):
        individual_preds[tree, :] = model.estimators_[tree].predict(observation)[0, :]
    mu = np.zeros(individual_preds.shape[1])
    sigma = np.zeros(individual_preds.shape[1])
    for j in range(individual_preds.shape[1]):
        mu[j] = np.mean(individual_preds[:,j])
        sigma[j] = np.std(individual_preds[:,j])
    
    for clf_i in range(test_X.shape[0]):
        vec = test_X[clf_i, :]
        for j in range(test_X.shape[1]):
            if(vec[j] < (mu[j] - 2*sigma[j]) or vec[j] > (mu[j] + 2*sigma[j])):
                pred[clf_i] = 0
        
    return(pred)


def run_CLF(p, model, test_X):
        
    pred = np.zeros(test_X.shape[0])
        
        
    for clf_i in range(test_X.shape[0]):
        vec = test_X[clf_i, :].reshape((1, test_X.shape[1]))
        pred[clf_i] = model.predict(vec)[0]

        
    return(pred)


def run_BNN(p, data, i, model, test_X):

    pred = np.ones(test_X.shape[0])

    observation = data["X"][i]
    observation = torch.from_numpy(observation.reshape((1, len(observation))).astype(np.float32))
    
    preds_local = np.zeros((p["bl_bnn_inference_repeats"], data["Y"].shape[1]))
    for k in range(p["bl_bnn_inference_repeats"]):
        y_pred = model(observation)[0]
        preds_local[k, :] = y_pred.data.numpy()
        
    mu = np.mean(preds_local, axis=0)
    sigma = np.std(preds_local, axis=0)
        
    for clf_i in range(test_X.shape[0]):
        vec = test_X[clf_i, :]
        for j in range(test_X.shape[1]):
            if(vec[j] < (mu[j] - 2*sigma[j]) or vec[j] > (mu[j] + 2*sigma[j])):
                pred[clf_i] = 0
        
    return(pred)



def train_TabPFN(data):


    train_X = data["train"]["X"]
    train_Y = data["train"]["Y"]

    st = time.time()

    regs = []
    for j in range(train_Y.shape[1]):
        regressor = TabPFNRegressor()
        regressor.fit(train_X, train_Y[:, j])
        regs.append(regressor)

    et = time.time()
    training_time = et - st

    return(regs, training_time)


        
def train_GP(p, data):    
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, ExpSineSquared, Matern, RBF, RationalQuadratic, WhiteKernel
    
    kernel_s = p["bl_gp_kernel"]
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
    
    model_kwargs =  {
                        "kernel": p["bl_gp_kernel"],
                        "alpha": p["bl_gp_alpha"],
                        "optimizer": p["bl_gp_optimizer"],
                        "n_restarts_optimizer": p["bl_gp_restarts"],
                        "normalize_y": p["bl_gp_normalize_y"]
                    }

    model = GaussianProcessRegressor(**model_kwargs)
    
    # Fit model and save training time
    st = time.time()
    model.fit(data["train"]["X"], data["train"]["Y"])
    et = time.time()
    training_time = et-st
    
    return(model, training_time)


def train_RF(p, data):    
    from sklearn.ensemble import RandomForestRegressor
    model_kwargs =  {
                        "n_estimators": p["bl_rf_n_estimators"],
                        #"criterion": p["bl_rf_criterion"], # wasn't used in HPO
                        "max_depth": p["bl_rf_max_depth"],
                        "min_samples_split": p["bl_rf_min_samples_split"],
                        "min_samples_leaf": p["bl_rf_min_samples_leaf"],
                        "min_weight_fraction_leaf": p["bl_rf_min_weight_fraction_leaf"],
                        "max_features": p["bl_rf_max_features"],
                        "max_leaf_nodes": p["bl_rf_max_leaf_nodes"],
                        "min_impurity_decrease": p["bl_rf_min_impurity_decrease"],
                        "bootstrap": p["bl_rf_bootstrap"],
                        "ccp_alpha": p["bl_rf_ccp_alpha"],
                    }
    model = RandomForestRegressor(**model_kwargs)
    
    # Fit model and save training time
    st = time.time()
    model.fit(data["train"]["X"], data["train"]["Y"])
    et = time.time()
    training_time = et-st
    
    return(model, training_time)
    
    
def train_CLF(p, sim, x_i, epsilon_loss, model_type="GBC", sample_size=None, **kwargs):   
    # Basic version of eMMI; instead of sampling intelligently and approximating, just sample randomly and approximate 

    if(sample_size is None):
        sample_size = p["method_budget"] # Should have same number of instances/function evals as other versions of eMMI
    train_X, train_Y = create_eval_sample(p, sim, x_i, epsilon_loss, sample_size)

    if(model_type == "GBC"):
        model = GradientBoostingClassifier(**kwargs)
    elif(model_type == "auto-sklearn"):
        model = AutoSklearnClassifier(
            time_left_for_this_task=120,
            per_run_time_limit=30,
            **kwargs
            #tmp_folder="/home/laurens/tmp/autosklearn_classification_example_tmp",
        )
    else:
        raise NotImplementedError()
        
    model.fit(train_X, train_Y)
        
    # Don't compute training time, this is done per instance so is contained in instance running time
    
    return(model)
    
    
def instantiate_BNN(p, num_in, num_out):
    # Based on https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/demos/Bayesian%20Neural%20Network%20Regression.ipynb
                    
    num_layers = p["bl_bnn_num_hidden_layers"]
    layers = []
    for i in range(num_layers):
        in_f = p["bl_bnn_num_features"]
        out_f = p["bl_bnn_num_features"]
        if(i == 0):
            in_f = num_in
        elif(i == num_layers-1):
            out_f = num_out

        layers.append(bnn.BayesLinear(prior_mu=p["bl_bnn_prior_mu"], prior_sigma=p["bl_bnn_prior_sigma"], in_features=in_f, out_features=out_f))
        if(i != num_layers-1):
            layers.append(nn.ReLU())
                    
    model = nn.Sequential(*layers)
    
    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = p["bl_bnn_kl_weight"]

    optimizer = optim.Adam(model.parameters(), lr=p["bl_bnn_learning_rate"])
    
    return(model, mse_loss, kl_loss, optimizer)
    
    
def train_BNN(p, data): 
    # Based on https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/demos/Bayesian%20Neural%20Network%20Regression.ipynb
    X = torch.from_numpy(data["train"]["X"].astype(np.float32))
    Y = torch.from_numpy(data["train"]["Y"].astype(np.float32))

    model, mse_loss, kl_loss, optimizer = instantiate_BNN(p, X.shape[1], Y.shape[1]) # X is observation, Y is parameters

    # Fit model and save training time
    st = time.time()
    for step in range(p["bl_bnn_training_epochs"]):
        pre = model(X)
        mse = mse_loss(pre, Y)
        kl = kl_loss(model)
        cost = mse + p["bl_bnn_kl_weight"]*kl
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    et = time.time()
    training_time = et-st
    
    return(model, training_time)