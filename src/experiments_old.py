from utils import *
from optimise import *
from search_spaces import *
from ML import *

import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.svm import SVR

import autosklearn.regression

# Use line below for SMAC >=2.0 facade, currently can't use with auto-sklearn
#from smac import HyperparameterOptimizationFacade, Scenario
# For SMAC 1.4
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from ConfigSpace import Configuration, ConfigurationSpace

import os

hacky_global_var = {}

def run_experiment_auto(X, Y, M, c, custom_params, val_split=0.2, overwrite=True):
    # For hyperparameter tuning using SMAC

    to_run = custom_params['to_run']
    if(custom_params['run_all']):
        to_run = range(0, len(Y))

    np.random.shuffle(to_run)
    split_point = int(val_split*len(Y))

    val_instances = to_run[:split_point]
    test_instances = to_run[split_point:]

    custom_params_val = custom_params.copy()
    custom_params_val['run_all'] = False
    custom_params_val['to_run'] = val_instances

    custom_params_test = custom_params.copy()
    custom_params_test['run_all'] = False
    custom_params_test['to_run'] = test_instances

    hacky_global_var = {
        'X':X,
        'Y':Y,
        'M':M,
        'c':c,
        'custom_params':custom_params_val,
        'overwrite':overwrite,
    }

    search_space = ConfigurationSpace()
    search_space = add_general(search_space)
    if(custom_params['regularise']):
        search_space = add_regularisation(search_space)
    if(custom_params['constrain']):
        search_space = add_constraints(search_space)
    if(custom_params['use_discriminator']):
        search_space = add_discriminator(search_space)

    scenario = Scenario(search_space, n_trials=200)
    smac = HyperparameterOptimizationFacade(scenario, run_experiment_config_wrapper)
    incumbent = smac.optimize()

    # Set relevant custom_params after optimisation

    custom_params_test = process_config(incumbent, custom_params_test)

    # Run on test set
    run_experiment(X, Y, M, c, custom_params=custom_params_test, overwrite=overwrite, performance_only=False)

    return()


def run_experiment_config_wrapper_old(config):
    # To allow it to run with a single input
    X = hacky_global_var['X']
    Y = hacky_global_var['Y']
    M = hacky_global_var['M']
    c = hacky_global_var['c']
    custom_params = hacky_global_var['custom_params']
    overwrite = hacky_global_var['overwrite']

    # Process config
    custom_params = process_config(config, custom_params)
    
    # Run, use MAE as validation loss
    res = run_experiment(X, Y, M, c, custom_params=custom_params, overwrite=overwrite, performance_only=True)
    return(res)


def process_config(config, custom_params):
    # Process config
    cfg = {k: config[k] for k in config if config[k]}
    for k,v in cfg.items():
        custom_params[k] = v

    # Extra stuff
    if(custom_params['regularise']):
        custom_params['window'] = [cfg['window_size'], cfg['window_size']]
        custom_params['lambdas'] = {
                                    'lai':cfg['lambda_lai'],
                                    'cab':cfg['lambda_cab'],
                                    'cw':cfg['lambda_cw'],
                                    'cm':cfg['lambda_cm'],
                                    'lidfa2':cfg['lambda_lidfa2'],
                                    'psoil':cfg['lambda_psoil'],
                                    'rsoil':cfg['lambda_rsoil'],
                                }
        
    return(custom_params)


def load_features(feature_set, meta):
    vec = np.zeros(len(feature_set))
    for i in range(0, len(feature_set)):
        vec[i] = meta[feature_set[i]]

    return(vec)


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
    

def train_discriminator(X_train, Y_train, M_train, config, cfg):

    # Hacky because I want to use the same pipeline as discriminator
    if(cfg.COND.ORACLE):
        return(None)

    lai_index = 0
    for p in cfg.PARAM.OPT.TO_OPTIMISE:
        if(p == 'lai'):
            break
        else:
            lai_index += 1

    # Training phase

    # Setup training/results stuff
    num_examples = len(cfg.EXP.TO_RUN) * cfg.PARAM.IS.IS_ITERATIONS
    num_params = len(cfg.PARAM.OPT.TO_OPTIMISE)
    num_features = len(cfg.PARAM.DISC.FEATURE_INDICES) + num_params

    maes = np.zeros(num_examples)
    Xd = np.zeros((num_examples, num_features))
    Yd = np.zeros(num_examples)
    if(cfg.PARAM.CONST.COMPUTE_CORR_MATRIX):
        configurations = np.zeros((num_examples, num_params))
    i = 0

    # Iterate over training set examples
    for t in cfg.EXP.TO_RUN:
        true = Y_train[t]

        # Don't confuse config (for PROSAIL) with cfg (experimental setup, hyperparameters)
        solution_set = run_instance(X_train[t, :], Y_train[t], M_train[t], config, cfg, return_all=True, 
                                    plot=cfg.EXP.PLOT)
        
        diff_vec = np.absolute(solution_set[:, lai_index] - true)
        best_solution_index = np.argmin(diff_vec)

        # For computing correlations on the training set
        if(cfg.PARAM.CONST.COMPUTE_CORR_MATRIX):
            configurations[i, :] = solution_set[best_solution_index, :]

        # Iterate over solutions in the solution set
        for s_r in range(0, solution_set.shape[0]):
            pred = solution_set[s_r, lai_index]
            true = Y_train[t]

            instance_solution= solution_set[s_r, :]

            # Create feature vector for this instance (features + solution)
            instance_properties = np.zeros(num_features)
            instance_properties[:len(cfg.PARAM.DISC.FEATURE_INDICES)] = X_train[t, :]
            instance_properties[len(cfg.PARAM.DISC.FEATURE_INDICES):] = instance_solution

            # Predict if solution is best
            if(cfg.PARAM.DISC.TARGET_QUANTITY == 'is_best'):
                if(s_r == best_solution_index):
                    # A little convoluted, inverted because we want to minimise later
                    target = 0
                else:
                    target = 1
            # Probability solution is best
            elif(cfg.PARAM.DISC.TARGET_QUANTITY == 'prob_best'):
                # Again, want to minimise later, so 1 is worst, 0 is best
                max_diff = np.max(diff_vec)
                min_diff = np.min(diff_vec)
                prob = (pred - min_diff) / (max_diff - min_diff)
                target = prob

            # Predict error
            elif(cfg.PARAM.DISC.TARGET_QUANTITY == 'error'):
                target = abs(true - pred)

            else:
                print("Invalid ML target " + str(target))

            maes[i] = abs(true - pred)
            Xd[i] = instance_properties
            Yd[i] = target
            i += 1

    discriminator = instantiate_discriminator(cfg.PARAM.DISC.DISCRIMINATOR_MODEL, cfg=cfg)
    discriminator.fit(Xd, Yd)

    if(cfg.PARAM.CONST.COMPUTE_CORR_MATRIX):
        corr_dict = {}
        for i in range(0, cfg.PARAM.OPT.TO_OPTIMISE):
            p1 = cfg.PARAM.OPT.TO_OPTIMISE[i]
            for j in range(0, cfg.PARAM.OPT.TO_OPTIMISE):
                p2 = cfg.PARAM.OPT.TO_OPTIMISE[j]
                key = (p1, p2)
                corr = pearsonr(configurations[:, i], configurations[:, j])
                corr_dict[key] = corr[0]
    else:
        corr_dict = None

    return(discriminator, corr_dict)


def run_discriminator(X_test, Y_test, M_test, t, config, cfg, discriminator):

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

    solution_set = run_instance(X_test[t, :], Y_test[t], M_test[t], config, cfg, return_all=True, discriminator_model=discriminator, 
                                plot=cfg.EXP.PLOT)

    true = Y_test[t]

    lowest_pred_mae = np.inf
    lowest_index = -1 # Index of predicted best solution

    for r in range(0, solution_set.shape[0]):

        instance_solution = solution_set[r, :]
        instance_properties = np.zeros(num_features)
        instance_properties[:len(cfg.PARAM.DISC.FEATURE_INDICES)] = X_test[t, :]
        instance_properties[len(cfg.PARAM.DISC.FEATURE_INDICES):] = instance_solution
        instance_properties = instance_properties.reshape((1, instance_properties.shape[0]))

        if(cfg.COND.ORACLE):
            # Generate noise
            if(cfg.PARAM.DISC.ORACLE_NOISE_METHOD == 'proportional'):
                # Add noise proportional as N(0, sigma) with sigma proportional to true value
                noise = np.random.normal(0, (cfg.PARAM.DISC.ORACLE_NOISE/100*Y_test[t]))
                noisy_y = min(max(0, noise+Y_test[t]), 10) # TODO: don't hardcode LAI limits
                pred_error = abs(solution_set[r, lai_index] - noisy_y)
            elif(cfg.PARAM.DISC.ORACLE_NOISE_METHOD == 'weighted_average'):
                # Add noise as weighted mean of a random value and the true value
                noise = np.random.normal(3.88, 1.98)
                noisy_y = (cfg.PARAM.DISC.ORACLE_NOISE/100*noise) + ((1 - cfg.PARAM.DISC.ORACLE_NOISE/100)*Y_test[t])
                noisy_y = min(max(0, noisy_y), 10) # TODO: don't hardcode LAI limits
                pred_error = abs(solution_set[r, lai_index] - noisy_y)
        else:
            pred_error = discriminator.predict(instance_properties)[0]
        if(pred_error < lowest_pred_mae):
            lowest_pred_mae = pred_error
            lowest_index = r

    pred = solution_set[lowest_index, lai_index]

    return(pred, true)


def run_experiment(X_train, Y_train, M_train, cfg, X_test=None, Y_test=None, M_test=None, 
                   save=True, overwrite=True, performance_only=False):
    
    # Will be manually specified, still good to have a check
    if(performance_only):
        save = False

    # Check if model is being used
    if(cfg.COND.USE_DISCRIMINATOR or cfg.COND.INVERTED_SIMULATOR):
        pass
    else:
        # Not technically accurate, more like we only have a test set, this is just more convenient for args
        X = X_train
        Y = Y_train
        M = M_train

    # TODO: verify that this param_ranges is used by Optimiser objects etc
    if(cfg.PARAM.OPT.DIST_FROM_TRAIN):
        param_ranges['lai'][4] = np.mean(Y_train)
        param_ranges['lai'][5] = np.std(Y_train)


    # Adapt budget for spatial regularisation and solution sets
    # This seems redundant given cfg to params
    budget = cfg.PARAM.OPT.BUDGET
    window = cfg.PARAM.REG.WINDOW
    individual_budget = int(budget / (window[0]*window[1]-1)) if window[1] > 1 else budget
    individual_budget = int(individual_budget / cfg.PARAM.IS.IS_ITERATIONS) if cfg.COND.USE_ITERATED_SEARCH else individual_budget
    cfg.PARAM.OPT.INDIVIDUAL_BUDGET = individual_budget


    # Create folders, files

    if(save):
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


    # Run stuff

    config = default_params.copy()

    if(cfg.EXP.RUN_ALL):
        cfg.EXP.TO_RUN = [i for i in range(0, len(Y_train))] # Kinda hacky but it should work


    # Inverted simulator

    if(cfg.COND.INVERTED_SIMULATOR):
        model = train_inverted_simulator(cfg)

        # Inference, bit messier because we want to neatly save stuff
        if(cfg.EXP.RUN_ALL):
            cfg.EXP.TO_RUN = [i for i in range(0, len(Y_test))] # Because we have train/test split

        maes = np.zeros(len(cfg.EXP.TO_RUN))

        i = 0
        for t in cfg.EXP.TO_RUN:
            true = Y_test[t]
            pred = run_inverted_simulator(X_test[t, cfg.S2_INDICES], model)
            mae = abs(true - pred)
            maes[i] = mae

            s = str(M_test[t]['key']) + "," + str(true) + "," + str(pred) + "," + str(mae) + "\n"

            # Save
            if(save):
                with open(path, 'a') as fp:
                    fp.write(s)

            i += 1



    # Discriminator (Iterated Search or Constraints)

    if(cfg.COND.USE_DISCRIMINATOR):

        # Training
        discriminator, corr_dict = train_discriminator(X_train, Y_train, M_train, config, cfg)
        cfg.PARAM.CONST.CORR_DICT = corr_dict

        # Inference, bit messier because we want to neatly save stuff
        if(cfg.EXP.RUN_ALL):
            cfg.EXP.TO_RUN = [i for i in range(0, len(Y_test))] # Because we have train/test split

        maes = np.zeros(len(cfg.EXP.TO_RUN))
        i = 0
        for t in cfg.EXP.TO_RUN:
            pred, true = run_discriminator(X_test, Y_test, M_test, t, config, cfg, discriminator)
            mae = abs(true - pred)
            maes[i] = mae
            s = str(M_test[t]['key']) + "," + str(true) + "," + str(pred) + "," + str(mae) + "\n"

            # Save
            if(save):
                with open(path, 'a') as fp:
                    fp.write(s)

            i += 1


    # No discriminator
    else: 
        # Compute correlation stuff from training set if used
        if(cfg.PARAM.CONST.COMPUTE_CORR_MATRIX):
            if(cfg.EXP.RUN_ALL):
                cfg.EXP.TO_RUN = [i for i in range(0, len(Y_train))] # Because we have train/test split
            discriminator, corr_dict = train_discriminator(X_train, Y_train, M_train, config, cfg)
            cfg.PARAM.CONST.CORR_DICT = corr_dict

        if(cfg.EXP.RUN_ALL):
            cfg.EXP.TO_RUN = [i for i in range(0, len(Y_test))] # Because we have train/test split
            
        maes = np.zeros(len(cfg.EXP.TO_RUN))
        i = 0
        for t in cfg.EXP.TO_RUN:

            pred = run_instance(X_test[t, :], Y_test[t], M_test[t], config, cfg, plot=cfg.EXP.PLOT)
            true = Y[t]

            mae = np.abs(true - pred) # Everything should broadcast nicely for QD
            maes[i] = np.mean(mae) # Usually single, this way can also take population
            i += 1

            if(not(performance_only)):
                s = str(M_test[t]['key']) + "," + str(true) + "," + str(pred) + "," + str(mae) + "\n"

                # Save
                if(save):
                    with open(path, 'a') as fp:
                        fp.write(s)

    # Return MAE for HPO
    if(performance_only):
        return(np.mean(maes))
                

def run_instance(x, y, m, config_original, cfg, return_all=False, **kwargs):

    config = config_original.copy() # This will be passed to PROSAIL
    params = {} # This will later be passed to the optimiser constructor


    # Preparing instance variables and exact parameters (geometry)

    real_refl = x
    real_lai = y
    platform = m['platform']
    if(platform == 'sentinel-2a'):
        SR = load_spectral_responses(cfg.S2A_SR_PATH)
    elif(platform == 'sentinel-2b'):
        SR = load_spectral_responses(cfg.S2B_SR_PATH)
    tts, tto, psi = to_prosail_geom(m)
    config['tts'] = tts
    config['tto'] = tto
    config['psi'] = psi



    # Warm start

    if(cfg.COND.WARM_START or cfg.COND.ALGORITHM == "WSO"):
        config_local = config.copy()
        config_local['lai'] = real_lai
        to_optimise_local = [p for p in cfg.PARAM.OPT.TO_OPTIMISE if not(p == 'lai')]
        cfg_local = cfg.copy()
        cfg_local.PARAM.OPT.TO_OPTIMISE = to_optimise_local
        cfg_local.COND.WARM_START = False
        cfg_local.COND.CONSTRAIN = False
        cfg_local.COND.ALGORITHM = "HC"

        # Little recursive call, not using warm start in next recursive level
        opt_params = run_instance(x, y, m, config_local, cfg_local, return_all=True, **kwargs)

        # Store and add noise
        wsv = {}
        for k,v in opt_params.items():
            wsv[k] =  v + np.random.normal(0, cfg.COND.WARM_START_NOISE * v)
            config[k] = wsv[k]
        wsv['lai'] = config_local['lai'] + np.random.normal(0, cfg.COND.WARM_START_NOISE * config_local['lai'])
        params["warm_start_values"] = wsv

    # Set up optimiser params
    params["config"] = config
    params = cfg_to_params(cfg, params)
    params["distance_func"] = string_to_func(cfg.PARAM.OPT.DISTANCE_FUNC)
    params["meta"] = m
    # Catch-all for convenient argument passing
    for k,v in kwargs.items():
        params[k] = v

    return_set = False
    if(cfg.COND.QUALITY_DIVERSITY or cfg.COND.USE_ITERATED_SEARCH):
        return_set = True

    # Load image data

    if('instance' in cfg.EXP.PLOT):
        print("Key: " + str(m["key"]))
        coords = m['footprint']
        print("Centroid: " + str(((coords[1] + coords[3])/2, (coords[0] + coords[2]) / 2)))
        print("Datetime: " + str(m['datetime']))

    img_orig = np.load(m['image_path'])
    img_orig = img_orig[:,:,cfg.S2_INDICES]

    if('instance' in cfg.EXP.PLOT):
        normalise_and_visualise(img_orig, title=m['key'] + " RGB")


    centre_x = m['x']
    centre_y = m['y']
    v_step = int(cfg.PARAM.REG.WINDOW[0]/2)
    h_step = int(cfg.PARAM.REG.WINDOW[1]/2)
    img_t = max(0, centre_y - v_step)
    img_b = min(img_orig.shape[0], centre_y + v_step + 1)
    img_l = max(0, centre_x - h_step)
    img_r = min(img_orig.shape[1], centre_x + h_step + 1)

    img = img_orig[img_t:img_b, img_l:img_r]

    if('clipped' in cfg.EXP.PLOT):
        normalise_and_visualise(img, title=m['key'] + " RGB trimmed")



    # Create grid
    if(cfg.COND.USE_ITERATED_SEARCH):
        # Currently kind of annoying to implement regularisation with iterated search
        optimiser_grid = HCOptimiser(img_orig[centre_y, centre_x, :], SR, **params) # Not technically grid
        # Run optimisation
        if(not(params["algorithm"] == "WSO")):
            optimiser_grid.run_iterated(params["individual_budget"])
    else:
        optimiser_grid = OptimiserGrid(img, SR, **params)
        # Run optimisation
        if(not(params["algorithm"] == "WSO")):
            optimiser_grid.run(params["individual_budget"])


    if(return_set):
        return(optimiser_grid.get_solution_set()) # Will need further processing later
    
    else:
        # TODO: not always correct if using multiple pixels per tile, should keep real index at some point
        i = int(img.shape[0] / 2)
        j = int(img.shape[1] / 2)
        cell = optimiser_grid[i,j]
        return(cell.get_val('lai'))
    

    # This old stuff is a bit of a mess, above is probably better
    # Look at centre pixel in detail TODO: this breaks now
    i = centre_y
    j = centre_x
    cell = optimiser_grid[i,j]

    if('centre' in cfg.EXP.PLOT):
        cell.plot_trackers(real_lai=real_lai)

    if('exact' in cfg.EXP.PLOT):
        print("Real LAI: " + str(real_lai))
        print("Best LAI centre: " + str(cell.get_tracker('lai')[-1]))
        print("Median best LAI: " + str(optimiser_grid.extract_average_param(param='lai')))

    if('all' in cfg.EXP.PLOT):
        optimiser_grid.plot_trackers(real_lai=real_lai, real_refl=real_refl)





    # Returning stuff. This is such an absolute mess and I am embarrassed
    if(return_all):
        if(params["quality_diversity"]):
            res_param_list = []
            solution_set = cell.get_solution_set()
            for i in range(0, len(solution_set)):
                res_params = {}
                j = 0
                for p in params["parameters"]:
                    res_params[p] = solution_set[i, j]
                    j += 1
                res_param_list.append(res_params)

            return(res_param_list)

        else:
            res_params = {}
            for p in params["parameters"]:
                res_params[p] = cell.get_val(p)
            return(res_params)
    else:
        if(params["algorithm"] == "WSO"):
            return(params["warm_start_values"]['lai'])
        
        if(params["quality_diversity"]):
            res_list = []
            solution_set = cell.get_solution_set()
            j = -1
            i = 0
            for p in params["parameters"]:
                if(p == 'lai'):
                    j = i 
                    break 
                i += 1
            for i in range(0, len(solution_set)):
                res_list.append(solution_set[i, j])
            return(res_list)

        else:
            lai = cell.get_val('lai')

            return(lai)


def cfg_to_params(cfg, params):

    params["budget"] = cfg.PARAM.OPT.BUDGET
    budget = cfg.PARAM.OPT.BUDGET
    window = cfg.PARAM.REG.WINDOW
    params["individual_budget"] = int(budget / (window[0]*window[1]-1)) if window[1] > 1 else budget
    params["individual_budget"] = int(params["individual_budget"] / cfg.PARAM.IS.IS_ITERATIONS) if cfg.COND.USE_ITERATED_SEARCH else params["individual_budget"]
    params["individual_budget"] = int(params["individual_budget"] / cfg.PARAM.OPT.CMA_POPULATION_SIZE) if cfg.COND.ALGORITHM=="CMA-ES" else params["individual_budget"]

    params["parameters"] = cfg.PARAM.OPT.TO_OPTIMISE
    params["prop_sd"] = cfg.PARAM.OPT.PROP_SD
    params["mutate_all"] = cfg.PARAM.OPT.MUTATE_ALL
    params["cma_population_size"] = cfg.PARAM.OPT.CMA_POPULATION_SIZE
    params["cma_sigma"] = cfg.PARAM.OPT.CMA_SIGMA
    params["HC_restart"] = cfg.PARAM.OPT.HC_RESTART
    params["HC_num_restarts"] = cfg.PARAM.OPT.HC_NUM_RESTARTS

    params["window"] = cfg.PARAM.REG.WINDOW
    params["gamma"] = cfg.PARAM.REG.GAMMA
    lambdas = {
        'lai':cfg.PARAM.REG.LAMBDAS.LAI,
        'cab':cfg.PARAM.REG.LAMBDAS.CAB,
        'cw':cfg.PARAM.REG.LAMBDAS.CW,
        'cm':cfg.PARAM.REG.LAMBDAS.CM,
        'lidfa2':cfg.PARAM.REG.LAMBDAS.LIDFA2,
        'psoil':cfg.PARAM.REG.LAMBDAS.PSOIL,
        'rsoil':cfg.PARAM.REG.LAMBDAS.RSOIL,
    }
    params["lambdas"] = lambdas

    params["constraint_penalty_strength"] = cfg.PARAM.CONST.PENALTY_STRENGTH
    params["use_corr_dict"] = cfg.PARAM.CONST.COMPUTE_CORR_MATRIX
    params["corr_dict"] = cfg.PARAM.CONST.CORR_DICT

    params["discriminator_iterations"] = cfg.PARAM.DISC.DISCRIMINATOR_ITERATIONS
    params["discriminator_features"] = cfg.PARAM.DISC.EXTRA_FEATURES
    params["discriminator_feature_indices"] = cfg.PARAM.DISC.FEATURE_INDICES

    params["is_iterations"] = cfg.PARAM.IS.IS_ITERATIONS
    params["is_weight"] = cfg.PARAM.IS.WEIGHT
    params["is_k"] = cfg.PARAM.IS.K
    params["is_strata"] = cfg.PARAM.IS.NUM_STRATA

    params["solution_set_size"] = cfg.PARAM.QD.SOLUTION_SET_SIZE
    params["qd_weight"] = cfg.PARAM.QD.WEIGHT
    pweight = {
        'lai':cfg.PARAM.QD.PWEIGHT.LAI,
        'cab':cfg.PARAM.QD.PWEIGHT.CAB,
        'cw':cfg.PARAM.QD.PWEIGHT.CW,
        'cm':cfg.PARAM.QD.PWEIGHT.CM,
        'psoil':cfg.PARAM.QD.PWEIGHT.PSOIL,
        'rsoil':cfg.PARAM.QD.PWEIGHT.RSOIL,
    }
    params["pweight"] = pweight
           
    params["algorithm"] = cfg.COND.ALGORITHM
    params["use_spatial_regularisation"] = cfg.COND.SPATIAL_REGULARISATION
    params["use_discriminator"] = cfg.COND.USE_DISCRIMINATOR
    params["constrain"] = cfg.COND.CONSTRAIN
    params["warm_start"] = cfg.COND.WARM_START
    params["quality_diversity"] = cfg.COND.QUALITY_DIVERSITY
    params["use_iterated_search"] = cfg.COND.USE_ITERATED_SEARCH

    return(params)


def string_to_func(obj):
    if(obj == "absolute_error" or obj == "absolute_distance"):
        return(absolute_distance)
    else:
        print("WARNING: invalid objective specified")
