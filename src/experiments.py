from .utils import *
from .optimise import *
from .search_spaces import *
from .ML import *
from .AutoRTML import *

import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.svm import SVR

#import autosklearn.regression

# Use line below for SMAC >=2.0 facade, currently can't use with auto-sklearn
#from smac import HyperparameterOptimizationFacade, Scenario
# For SMAC 1.4
#from smac.facade.smac_hpo_facade import SMAC4HPO
#from smac.scenario.scenario import Scenario

#from ConfigSpace import Configuration, ConfigurationSpace

import os
import copy

hacky_global_var = {}

def run_experiment_config_wrapper(config, seed=42):
    # Currently just supporting iterated search

    # To allow it to run with a single input
    X_train = hacky_global_var['X_train']
    Y_train = hacky_global_var['Y_train']
    M_train = hacky_global_var['M_train']
    # Technically val, not test split
    X_test = hacky_global_var["X_val"]
    Y_test = hacky_global_var["Y_val"]
    M_test = hacky_global_var["M_val"]

    cfg = hacky_global_var["cfg"]

    # Update cfg with smac config
    if("budget" in config):
        cfg.PARAM.OPT.BUDGET = config['budget']
    if("prop_sd" in config):
        cfg.PARAM.OPT.PROP_SD = config['prop_sd']
    if("is_iterations" in config):
        cfg.PARAM.IS.IS_ITERATIONS = config["is_iterations"]
    if("is_weight" in config):
        cfg.PARAM.IS.WEIGHT = config["is_weight"]
    if("is_k" in config):
        cfg.PARAM.IS.K = config["is_k"]
    if("is_strata" in config):
        cfg.PARAM.IS.NUM_STRATA = config["is_strata"]
    if("invsim_noise" in config):
        cfg.PARAM.INVSIM.NOISE = config["invsim_noise"]
    if("HC_num_restarts" in config):
        cfg.PARAM.OPT.HC_NUM_RESTARTS = config["HC_num_restarts"]
    if("discriminator_target" in config):
        cfg.PARAM.DISC.TARGET_QUANTITY = config["discriminator_target"]
    if("constraint_penalty_strength" in config):
        cfg.PARAM.CONST.COMPUTE_CORR_MATRIX = config["constraint_penalty_strength"]
    

    res = run_experiment(X_train, Y_train, M_train, cfg, X_test=X_test, Y_test=Y_test, M_test=M_test, save=False, performance_only=True)

    return(res)


def optimise_hyperparameters(cfg, overwrite=False, **kwargs):
    # Give access to cfg etc even though we cannot pass them to the function directly
    for k,v in kwargs.items():
        hacky_global_var[k] = v
        hacky_global_var['cfg'] = cfg

    # Set up search space
    search_space = ConfigurationSpace()
    search_space = add_general(search_space)
    if(cfg.COND.SPATIAL_REGULARISATION):
        search_space = add_regularisation(search_space)
    if(cfg.COND.CONSTRAIN or cfg.PARAM.CONST.COMPUTE_CORR_MATRIX):
        search_space = add_constraints(search_space)
    if(cfg.COND.USE_DISCRIMINATOR):
        search_space = add_discriminator(search_space)
    if(cfg.COND.USE_ITERATED_SEARCH):
        search_space = add_iterated(search_space)
    if(cfg.COND.INVERTED_SIMULATOR):
        search_space = add_invsim(search_space)
    if(cfg.PARAM.OPT.HC_RESTART):
        search_space = add_HC_restart(search_space)

    # Run SMAC, version >= 2.0
    #scenario = Scenario(search_space, walltime_limit=cfg.HPO.WALLCLOCK_TIME, n_trials=cfg.HPO.N_TRIALS, seed=cfg.SEED)
    #smac = HyperparameterOptimizationFacade(scenario, run_experiment_config_wrapper, overwrite=overwrite)
    #incumbent = smac.optimize()

    # Run SMAC, version 1.4
    scenario = Scenario(
    {
        "run_obj": "quality",
        "runcount_limit": cfg.HPO.N_TRIALS,
        "cs": search_space,
        "wallclock-limit": cfg.HPO.WALLCLOCK_TIME,
        "output_dir": cfg.HPO.OUTPUT_PATH + "_" + cfg.EXP.EXPERIMENT_NAME,
    })
    smac = SMAC4HPO(scenario=scenario, rng=cfg.SEED, tae_runner=run_experiment_config_wrapper)
    incumbent = smac.optimize()

    # Print results
    print("Incumbent:\n", str(incumbent))

    # Save incumbent
    if(not(os.path.exists("HPO"))):
        os.mkdir("HPO")
    with open("HPO/incumbent_" + cfg.EXP.EXPERIMENT_NAME + ".txt", 'w') as fp:
        fp.write(str(incumbent))
    







def run_experiment(X_test, Y_test, M_test, cfg, X_train=None, Y_train=None, M_train=None, 
                   save=True, overwrite=True, performance_only=False, **kwargs):
    # Run experiment

    # Set up files
    if(save):
        save_path = setup_files(cfg, overwrite=overwrite)

    # Setup ranges
    if(cfg.PARAM.OPT.DIST_FROM_TRAIN):
        param_ranges = dist_from_train(Y_train, get_default_ranges())
    else:
        param_ranges = get_default_ranges()

    # Setup indices
    instance_indices_train = cfg.EXP.TO_RUN
    instance_indices_test = cfg.EXP.TO_RUN
    if(cfg.EXP.RUN_ALL):
        instance_indices_train = [i for i in range(0, len(Y_train))]
        instance_indices_test = [i for i in range(0, len(Y_test))]

    ####################
    # Training phase
    ####################

    # Correlations
    if(cfg.COND.PARAMETER_CORRELATIONS):
        corr_dict = get_corr_dict(X_train, Y_train, M_train, cfg, instance_indices_train, param_ranges=param_ranges)
        kwargs['corr_dict'] = corr_dict # Slightly hacky


    # TODO: implement AutoRTML training
    if(cfg.COND.BASE_METHOD == 'AutoRTML'):
        predictors, parameter_values = get_training_set(X_train, Y_train, M_train, cfg, instance_indices=instance_indices_train)
        model = train_AutoRTML(predictors, parameter_values, cfg)
    elif(cfg.COND.BASE_METHOD == 'optimisation'):
        pass
    elif(cfg.COND.BASE_METHOD == 'InvSim'):
        model = train_inverted_simulator(cfg)


    ####################
    # Inference phase
    ####################

    # TODO: implement AutoRTML inference, InvSim inference
    for i in instance_indices_test:
        if(cfg.COND.BASE_METHOD == 'AutoRTML'):
            pred = run_AutoRTML(X_test[i], M_test[i], cfg, model, param_ranges=param_ranges, **kwargs)
        elif(cfg.COND.BASE_METHOD == 'optimisation'):
            pred = run_optimisation(X_test[i], M_test[i], cfg, 
                                    parameters=cfg.PARAM.OPT.TO_OPTIMISE, param_ranges=param_ranges, **kwargs)
        elif(cfg.COND.BASE_METHOD == 'InvSim'):
            pred = run_inverted_simulator(X_test[i, cfg.S2_INDICES], model)

        # Save stuff
        if(save):
            save_to_file(save_path, Y_test[i], pred, M_test[i])


    return()


def get_training_set(X, Y, M, cfg, instance_indices=-1, true_value=False):
    # Get a training dataset

    if(instance_indices == -1):
        instance_indices = [i for i in range(0, len(Y))]

    # Initialise dataset

    parameters = cfg.PARAM.OPT.TO_OPTIMISE
    if(true_value):
        parameters = [p for p in parameters if not(p == cfg.PARAM.OPT.TARGET)]

    predictors = np.zeros((len(instance_indices), X.shape[1], X.shape[2], X.shape[3]))
    parameter_values = np.zeros((len(instance_indices), len(parameters)))

    # Iterate
    for i in instance_indices:
        res_params = run_optimisation(X[i], M[i], cfg, parameters=parameters)
        res_vec = np.array([res_params[p] for p in parameters])
        predictors[i, :, :, :] = X[i, :, :, :]
        parameter_values[i, :] = res_vec

    return(predictors, parameter_values)


def get_corr_dict(X, Y, M, cfg, instance_indices, target_param='lai'):

    # Setup arrays
    to_optimise_true = [p for p in cfg.PARAM.OPT.TO_OPTIMISE if not(p == target_param)]
    num_examples = len(instance_indices)
    num_params = len(cfg.PARAM.OPT.TO_OPTIMISE)

    Yd = np.zeros((num_examples, num_params))

    # Iterate over training set examples
    i = 0
    for t in instance_indices:
        true = Y[t]
        config_local = get_default_config()
        config_local[target_param] = true

        # Get global solution with true target
        solution_config = run_optimisation(X[t], M[t], cfg, parameters=to_optimise_true, initial_config=config_local)
        solution_config[target_param] = true
        param_vec = config_to_vec(solution_config, cfg.PARAM.OPT.TO_OPTIMISE)
        Yd[i, :] = param_vec        
    
    # Compute correlations
    corr_dict = {}
    for i in range(0, len(cfg.PARAM.OPT.TO_OPTIMISE)):
        p1 = cfg.PARAM.OPT.TO_OPTIMISE[i]
        for j in range(0, len(cfg.PARAM.OPT.TO_OPTIMISE)):
            p2 = cfg.PARAM.OPT.TO_OPTIMISE[j]
            key = (p1, p2)
            corr = pearsonr(Yd[:, i], Yd[:, j])
            corr_dict[key] = corr[0]

    return(corr_dict)


def preprocess_instance(x, m, cfg, fix_target={}):
    param_ranges = get_default_ranges() # Get specified ranges of parameters
    initial_config = get_default_config() # Initialise PROSAIL parameter config
    parameters = cfg.PARAM.OPT.TO_OPTIMISE.copy() # Specified free parameters for PROSAIL
    # Fix specified target parameter(s) at specific value
    if(len(fix_target) > 0):
        for p,v in fix_target.items():
            initial_config[p] = v
            parameters.remove(p)


    config = initial_config.copy() # Avoid overwriting initial config
    # Convert full X (image or pixel with bands for spectrum+extra features) into a single central
    # pixel with only spectral features. Also automatically set geometry parameters in config
    # (derived from acquisition metadata), and the SR for the satellite platform (to convert
    # hyperspectral PROSAIL output to multispectral S2 format)
    x, config, SR = initialise_instance(x, config, m, cfg, spectral_indices=cfg.S2_INDICES)

    # Prepare all setup for the optimiser configuration, e.g., algorithm, budget, etc
    optimiser_params = {"config":config} # PROSAIL config
    optimiser_params = cfg_to_params(cfg, optimiser_params) # Most parameters
    optimiser_params["distance_func"] = string_to_func(cfg.PARAM.OPT.DISTANCE_FUNC) # Function handle for obj
    optimiser_params["meta"] = m # Instance metadata, in case it is needed later
    optimiser_params["parameters"] = parameters # Free parameters in PROSAIL inputs

    return(x, m, param_ranges, config, SR, optimiser_params)


def create_optimisation_trace(x, SR, optimiser_params, discard_stagnation=True):
    # Initialise matrix: rows are iteration, columns are parameters
    mat = np.zeros((optimiser_params["budget"], len(optimiser_params["parameters"])))

    # Initialise HC optimiser with prepared parameters
    optimiser = HCOptimiser(x, SR, **optimiser_params)

    # Start optimising; store best solution at every iteration
    for i in range(0, optimiser_params['budget']):
        optimiser.step()
        sol_i = optimiser.get_solution(format="vec")
        mat[i, :] = sol_i

    if(discard_stagnation):
        mat = np.unique(mat, axis=0)

    return(mat)


def sf_to_spectral(x, i=-1, j=-1, spectral_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    # Convert instance to spectral features only (s+f) --> (s).
    # If instance is an image, take central pixel (h x w x s+f) --> (s).

    if(len(x.shape) > 1):    
        if(i == -1):
            i = int(x.shape[0] / 2) - 1 # -1 for 0 indexing
        if(j == -1):
            j = int(x.shape[1] / 2) - 1
        x = x[i, j, :]
    
    return(x[spectral_indices])


def initialise_instance(x, config, m, cfg, **kwargs):

    # Minor processing
    x = sf_to_spectral(x, **kwargs)

    # Geometry and spectral responses
    platform = m['platform']
    if(platform == 'sentinel-2a'):
        SR = load_spectral_responses(cfg.S2A_SR_PATH)
    elif(platform == 'sentinel-2b'):
        SR = load_spectral_responses(cfg.S2B_SR_PATH)
    tts, tto, psi = to_prosail_geom(m)
    config['tts'] = tts
    config['tto'] = tto
    config['psi'] = psi

    return(x, config, SR)


def train_AutoRTML(predictors, param_values, cfg):
    # Train model for AutoRTML

    return()


def run_AutoRTML(x, m, cfg, model):
    # Run AutoRTML on one instance

    lai_index = 0
    for p in cfg.PARAM.OPT.TO_OPTIMISE:
        if(p == 'lai'):
            break
        else:
            lai_index += 1

    # Setup stuff
    num_examples = len(cfg.EXP.TO_RUN) * cfg.PARAM.IS.IS_ITERATIONS
    num_params = len(cfg.PARAM.OPT.TO_OPTIMISE)
    num_features = len(cfg.PARAM.DISC.FEATURE_INDICES) + num_params

    solution_set = run_optimisation(x, m, cfg, return_set=True, discriminator_model=model)

    lowest_pred_mae = np.inf
    lowest_index = -1 # Index of predicted best solution

    # Iterate over solutions in set, predict quantities
    for r in range(0, solution_set.shape[0]):

        instance_solution = solution_set[r, :]
        instance_properties = np.zeros(num_features)
        instance_properties[:len(cfg.PARAM.DISC.FEATURE_INDICES)] = x
        instance_properties[len(cfg.PARAM.DISC.FEATURE_INDICES):] = instance_solution
        instance_properties = instance_properties.reshape((1, instance_properties.shape[0]))
        
        pred_error = model.predict(instance_properties)[0]
        if(pred_error < lowest_pred_mae):
            lowest_pred_mae = pred_error
            lowest_index = r

    # Select solution, return solution's LAI
    pred = solution_set[lowest_index, lai_index]

    return(pred)


def run_diversity_optimisation(X, M, Y, cfg):
    for i in cfg.EXP.TO_RUN:
        x = X[i][cfg.S2_INDICES]
        m = M[i]
        y = Y[i]

        optimiser_params = cfg_to_params(cfg, {})

        auto = AutoRTML(x, m, cfg.PARAM.OPT.TO_OPTIMISE, cfg.PARAM.IS.IS_ITERATIONS, optimiser="HC", fixed_params={}, **optimiser_params)
        solutions = auto.run(100)

        return(solutions)


def run_optimisation(x, m, cfg, parameters=-1, initial_config=-1, return_set=False, **kwargs):
    # Run optimisation

    # Setup
    if(parameters == -1):
        parameters = cfg.PARAM.OPT.TO_OPTIMISE
    if(initial_config == -1):
        initial_config = get_default_config()

    config = initial_config.copy()
    x, config, SR = initialise_instance(x, config, m, cfg, spectral_indices=cfg.S2_INDICES)

    # Setup arguments for optimiser class
    optimiser_params = {}
    optimiser_params["config"] = config
    optimiser_params = cfg_to_params(cfg, optimiser_params)
    optimiser_params["distance_func"] = string_to_func(cfg.PARAM.OPT.DISTANCE_FUNC)
    optimiser_params["meta"] = m
    optimiser_params["parameters"] = parameters
    # Catch-all for convenient argument passing
    for k,v in kwargs.items():
        optimiser_params[k] = v

    # Initialise optimiser
    if(cfg.COND.ALGORITHM == 'HC'):
        optimiser = HCOptimiser(x, SR, **optimiser_params)
    elif(cfg.COND.ALGORITHM == 'RS'):
        optimiser = RSOptimiser(x, SR, **optimiser_params)
    elif(cfg.COND.ALGORITHM == 'CMA-ES'):
        optimiser = CMAESOptimiser(x, SR, **optimiser_params)
    else:
        print("Invalid optimiser specified")

    # Run optimisation
    if(return_set):
        optimiser.run_iterated(optimiser_params['budget'])
    else:
        for _ in range(0, optimiser_params['budget']):
            optimiser.step()

    # Get solution
    if(return_set):
        solution = optimiser.get_solution_set()
    else:
        solution = optimiser.get_solution()

    return(solution)


def run_invsim():
    # Run inverted simulator
    return()


def dist_from_train(Y, ranges, param_name='lai'):
    param_ranges = copy.deepcopy(ranges)
    param_ranges[param_name][4] = np.mean(Y)
    param_ranges[param_name][5] = np.std(Y)
    return(param_ranges)


def setup_files(cfg, overwrite=False):
    path = cfg.EXP.BASE_PATH
    if(not(os.path.exists(path))):
        os.mkdir(path)

    path = path + cfg.EXP.RESULT_VERSION_NAME
    if(not(os.path.exists(path))):
        os.mkdir(path)

    path_meta = path + "/" + cfg.EXP.EXPERIMENT_NAME + "_META.csv"
    path = path + "/" + cfg.EXP.EXPERIMENT_NAME + ".csv"
    if(overwrite or not(os.path.exists(path))):
        header = "key,true,pred,MAE\n"
        with open(path, 'w') as fp:
            fp.write(header)

        # Save experimental configuration
        with open(path_meta, 'w') as fp:
            fp.write(str(cfg))

    else:
        print("Warning: found existing results; appending")

    return(path)


def save_to_file(path, y, pred, m):
    key = m['key']
    ae = abs(y - pred)
    s = str(key) + "," + str(y) + "," + str(pred) + "," + str(ae) + "\n"

    with open(path, 'a') as fp:
        fp.write(s)