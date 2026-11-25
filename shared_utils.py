import numpy as np
import matplotlib.pyplot as plt

import json
import os

from src.ForwardModel import *
from src.simulate import *
from src.load_data import keys_to_numpy, get_keys_and_indices, remove_invalid
from src.AutoIPQ import AutoIPQ

# Functions


def set_default_parameters(**kwargs):

    p = {}

    ############### Experimental ###############

    p["exp_name"] = "AutoIPQ"
    p["exp_save_results"] = False
    p["exp_save_dir"] = "/scratch/arp/AutoIPQ/results/" + p["exp_name"] + "/"
    p["exp_project_path"] = "/scratch/arp/AutoIPQ/"
    p["exp_eval_sample_local"] = True

    p["exp_seed"] = 42

    p["exp_max_instances"] = 200

    p["exp_parameters"] = ["lai", "cab", "lidfa2", "hspot", "n", "cw", "cm",]
    p["exp_parameters_simulation"] = ["tts", "tto", "psi"]
    
    p["exp_epsilon"] = 0.1

    p["exp_uc_oracle_iterations"] = 5000

    ############### Data ###############

    p["project_path"] = "/scratch/arp/AutoIPQ/"
    p["realdata_S2A_SR_path"] = "src/resources/data/S2A_spectral_responses.csv"
    p["realdata_S2B_SR_path"] = "src/resources/data/S2B_spectral_responses.csv"

    p["realdata_S2_wavelengths"] = [443,490,560,665,705,740,783,842,865,1610,2190]

    p["simdata_simulator"] = "prosail"
    p["simdata_generator"] = "random"
    p["simdata_num_instances"] = 1000
    p["simdata_num_samples"] = 30
    p["simdata_dim"] = 0 # For > 0, duplicates parameters along independent dimensions
    
    p["data_noise"] = None # None or 0.0-1.0
    p["data_store_sample_test"] = False

    ############### Method ###############
   
    p["method_budget"] = 5000
    
    p["method_budget_split"] = [0.2, 0.1, 0.7]
    p["method_population_size"] = 20
    p["method_k"] = int(p["method_population_size"] / 2)
    p["method_diversity_weight"] = 0.5
    p["method_step2_approach"] = "serial"
    p["method_epsilon2"] = 0.1
    p["method_learning_rate"] = 0.1
    p["method_init_noise"] = 0.01
    p["method_constraint_penalty"] = np.inf
    p["method_filter_density"] = False
    p["method_density_filter_percentile"] = 10
    p["method_density_filter_same_class"] = False  
    p["method_optimiser"] = "HC"
    p["method_optimiser_1"] = "CMAES"
    p["method_optimiser_2a"] = "HC"
    p["method_optimiser_2b"] = "HC"
    p["method_optimiser_2c"] = "HC"
    p["method_optimiser_2d"] = "HC"
    p["method_random_exploration_proportion"] = "0.0"
    
    p["method_seq_budget_split_a"] = 0.2
    p["method_seq_budget_split_b"] = 0.1
    p["method_seq_budget_split_c"] = 0.7
    p["method_seq_budget_split"] = [0.2, 0.1, 0.7]
    p["method_seq_population_size"] = 20
    p["method_seq_k"] = int(p["method_population_size"] / 2)
    p["method_seq_diversity_weight"] = 0.5
    p["method_seq_step2_approach"] = "serial"
    p["method_seq_epsilon2"] = 0.1
    p["method_seq_learning_rate"] = 0.1
    p["method_seq_init_noise"] = 0.01
    p["method_seq_constraint_penalty"] = np.inf
    p["method_seq_filter_density"] = False
    p["method_seq_density_filter_percentile"] = 10
    p["method_seq_density_filter_same_class"] = False  
    p["method_seq_random_exploration_proportion"] = 0.0
    p["method_seq_optimiser_1"] = "CMAES"
    p["method_seq_optimiser_2a"] = "HC"
    p["method_seq_optimiser_2b"] = "HC"
    p["method_seq_optimiser_2c"] = "HC"
    p["method_seq_optimiser_2d"] = "HC"
    
    p["method_dual_budget_split_a"] = 0.2
    p["method_dual_budget_split_b"] = 0.1
    p["method_dual_budget_split_c"] = 0.7
    p["method_dual_budget_split"] = [0.2, 0.1, 0.7]
    p["method_dual_population_size"] = 20
    p["method_dual_k"] = int(p["method_population_size"] / 2)
    p["method_dual_diversity_weight"] = 0.5
    p["method_dual_step2_approach"] = "serial"
    p["method_dual_epsilon2"] = 0.1
    p["method_dual_learning_rate"] = 0.1
    p["method_dual_init_noise"] = 0.01
    p["method_dual_constraint_penalty"] = np.inf
    p["method_dual_filter_density"] = False
    p["method_dual_density_filter_percentile"] = 10
    p["method_dual_density_filter_same_class"] = False  
    p["method_dual_random_exploration_proportion"] = 0.0
    p["method_dual_optimiser_1"] = "CMAES"
    p["method_dual_optimiser_2a"] = "HC"
    p["method_dual_optimiser_2b"] = "HC"
    p["method_dual_optimiser_2c"] = "HC"
    p["method_dual_optimiser_2d"] = "HC"
    
    p["method_conv_budget_split_a"] = 0.2
    p["method_conv_budget_split_b"] = 0.1
    p["method_conv_budget_split_c"] = 0.7
    p["method_conv_budget_split"] = [0.2, 0.1, 0.7]
    p["method_conv_population_size"] = 20
    p["method_conv_k"] = int(p["method_population_size"] / 2)
    p["method_conv_diversity_weight"] = 0.5
    p["method_conv_step2_approach"] = "serial"
    p["method_conv_epsilon2"] = 0.1
    p["method_conv_learning_rate"] = 0.1
    p["method_conv_init_noise"] = 0.01
    p["method_conv_constraint_penalty"] = np.inf
    p["method_conv_filter_density"] = False
    p["method_conv_density_filter_percentile"] = 10
    p["method_conv_density_filter_same_class"] = False  
    p["method_conv_optimiser_1"] = "CMAES"
    p["method_conv_optimiser_2a"] = "HC"
    p["method_conv_optimiser_2b"] = "HC"
    p["method_conv_optimiser_2c"] = "HC"
    p["method_conv_optimiser_2d"] = "HC"
    
    ############### Baselines ###############

    p["bl_abcsmc_num_populations"] = 10
    
    p["bl_gp_kernel"] = None
    p["bl_gp_alpha"] = 1e-10
    p["bl_gp_optimizer"] = "fmin_l_bfgs_b"
    p["bl_gp_restarts"] = 0
    p["bl_gp_normalize_y"] = False
    
    p["bl_rf_n_estimators"] = 100
    p["bl_rf_criterion"] = "gini"
    p["bl_rf_max_depth"] = None
    p["bl_rf_min_samples_split"] = 2
    p["bl_rf_min_samples_leaf"] = 1
    p["bl_rf_min_weight_fraction_leaf"] = 0.0
    p["bl_rf_max_features"] = 1.0
    p["bl_rf_max_leaf_nodes"] = None
    p["bl_rf_min_impurity_decrease"] = 0.0
    p["bl_rf_bootstrap"] = True
    p["bl_rf_ccp_alpha"] = 0.0
    
    p["bl_bnn_prior_mu"] = 0
    p["bl_bnn_prior_sigma"] = 0.1
    p["bl_bnn_num_hidden_layers"] = 2
    p["bl_bnn_num_features"] = 100
    p["bl_bnn_kl_weight"] = 0.1
    p["bl_bnn_learning_rate"] = 0.01
    p["bl_bnn_training_epochs"] = 3000
    p["bl_bnn_inference_repeats"] = 30


    for k,v in kwargs.items():
        p[k] = v

    return(p)
    

def load_parameters_file(p, path):

    with open(path, 'r') as fp:
        d = json.load(fp)

    for k,v in d.items():
        # Workaround for some datatypes not being supported by json
        v2 = v
        if(v == "None"):
            v2 = None
        elif(v == "True"):
            v2 = True
        elif(v == "False"):
            v2 = False
        elif(v == "np.inf"):
            v2 = np.inf
        p[k] = v2

    return(p)
    

def load_parameters_from_smac(p, sim, method, path="smac/"):
    smac_path = path + sim + "/" + method + "/HPO_" + method + "/0/"
    int_path = smac_path + "intensifier.json"
    hist_path = smac_path + "runhistory.json"

    if(os.path.exists(path + sim)):
        if(os.path.exists(path + sim + "/" + method)):

            with open(int_path, 'r') as fp:
                int_data = json.load(fp)
                incumbent = int_data["incumbent_ids"][0]
            with open(hist_path, 'r') as fp:
                hist_data = json.load(fp)
                config = hist_data["configs"][str(incumbent)]

            if("constraint_penalty_inf" in config):
                if(config["constraint_penalty_inf"] == "True"):
                    try:
                        spl = list(config.keys())[0].split("_")
                        key = spl[0] + "_" + spl[1] + "_"
                        config[key + "constraint_penalty"] = np.inf
                    except:
                        print("problem for ", smac_path)
                        print("keys:", config.keys())
                        print("\n")

            for k, v in config.items():
                p[k] = v

    return(p)
    
def instantiate_sim(model, num_samples=1, **kwargs):
    model = model.lower()
    
    # Hacky
    if(num_samples > 1):
        kwargs["num_samples"] = num_samples
    
    if(model == "prosail"):
        return(Prosail(**kwargs))
    
    elif(model == "mdgs"):
        return(MultiDimGaussianSample(**kwargs))
    elif(model == "lr2d"):
        return(LinearRegressor2D(**kwargs))
    elif(model == "lr5d"):
        return(LinearRegressor5D(**kwargs))
    elif(model == "lr10d"):
        return(LinearRegressor10D(**kwargs))
    elif(model == "lr20d"):
        return(LinearRegressor20D(**kwargs))
    elif(model == "lr50d"):
        return(LinearRegressor50D(**kwargs))
        
    elif(model == "dpr_tp"):
        return(DPR_TwoPendulums(**kwargs))
    elif(model == "dpr_lorenz63"):
        return(DPR_Lorenz63(**kwargs))
        
    elif(model == "sbi_gm"):
        return(SBI_GM(**kwargs))
    elif(model == "sbi_slcp"):
        return(SBI_SLCP(**kwargs))
    elif(model == "sbi_bglm"):
        return(SBI_BGLM(**kwargs))
    elif(model == "sbi_glu"):
        return(SBI_GLU(**kwargs))
    elif(model == "sbi_tm"):
        return(SBI_TM(**kwargs))
    elif(model == "sbi_gl"):
        return(SBI_GL(**kwargs))
    elif(model == "sbi_lv"): # NOTE: ranges not ready
        return(SBI_LV(**kwargs))
    elif(model == "sbi_sir"): # NOTE: ranges not ready
        return(SBI_SIR(**kwargs))
    elif(model == "sbi_slcpd"):
        return(SBI_SLCPD(**kwargs))
    elif(model == "sbi_bglmr"):
        return(SBI_BGLMR(**kwargs))
    
    elif(model == "toyexample"):
        return(ToyExample(**kwargs))
    
    elif(model == "ackley"):
        return(AckleyFunction(**kwargs))
    elif(model == "rastrigin"):
        return(RastriginFunction(**kwargs))
    elif(model == "rosenbrock"):
        return(RosenbrockFunction(**kwargs))
    elif(model == "mccormick"):
        return(McCormickFunction(**kwargs))
    elif(model == "threehumpcamel"):
        return(ThreeHumpCamelFunction(**kwargs))
    elif(model == "himmelblau"):
        return(HimmelblauFunction(**kwargs))
    else:
        raise NotImplementedError
    

def load_dataset(simulator, project_path, splits=["train", "val", "test"], max_instances=None, noise=None):
    data = {}
    
    data_dir = project_path + "data/experiments/" + simulator + "/"
    #splits = ["train", "val", "test"]
    suffix = ""
    if(noise is not None):
        suffix = "_noise" + str(noise)
    for split in splits:
        X = pd.read_csv(data_dir + split + "/instances_X" + suffix + ".csv").values[:,1:] # First column is id
        Y = pd.read_csv(data_dir + split + "/instances_Y.csv").values[:,1:]
        M = pd.read_csv(data_dir + split + "/instances_M.csv").values[:,1:]
        if(split in ["val", "test"] and suffix != ""):
            opt = pd.read_csv(data_dir + split + "/instances_opt" + suffix + ".csv").sort_values(by='instance')
            opt_loss = opt.values[:,1]
            opt = opt.values[:,2:]
        else:
            opt = None
            opt_loss = None
        
        if(max_instances is not None):
            X = X[:max_instances]
            Y = Y[:max_instances]
            M = M[:max_instances]
            if(split in ["val", "test"] and suffix != ""):
                opt = opt[:max_instances]
                opt_loss = opt_loss[:max_instances]
        
        data_local =    {
                            "X":X, 
                            "Y":Y,
                            "M":M,
                            "opt":opt,
                            "opt_loss":opt_loss,
                        }
                        
        data[split] = data_local
        
    return(data)


def create_eval_sample(p, sim, x_i, epsilon_loss, clf_test_sample_size):
    # Easier to perform simulation and compute loss using implementation bundled in AutoIPQ (note: not dependent on actual method)
    auto = AutoIPQ(sim, x_i, p["exp_parameters"], 10)
    
    # Sample, combine into matrix
    column_vals = []
    for param in p["exp_parameters"]:
        range_min = sim.param_ranges[param][2]
        range_max = sim.param_ranges[param][3]
        param_vals = np.random.uniform(low=range_min, high=range_max, size=clf_test_sample_size)
        column_vals.append(param_vals.reshape((len(param_vals), 1)))
    
    test_X = np.concatenate(column_vals, axis=1) # NOTE: X for clf is Y for problem (X is config, Y is valid/invalid)

    
    # Compute test_Y   
    test_Y = np.zeros(test_X.shape[0])
    for sub_i in range(test_X.shape[0]):
        x_dict = {p["exp_parameters"][j]:test_X[sub_i][j] for j in range(test_X.shape[1])}
        sim_output = sim.simulate(x_dict)
        loss = auto.optimiser.f1(sim_output, x_i) # x_i is 'real' instance observations
        if(loss <= epsilon_loss):
            test_Y[sub_i] = 1
            
    return(test_X, test_Y)
    
    
def load_local_eval_sample(p, conds, i, epsilon_loss, version="local"):
    path = p["exp_project_path"] + "data/experiments/" + conds["simulator"] + "/test/"
    if(version == "indirect"):
        path = path + "clf_samples_indirect/"
    elif(version == "uniform"):
        path = path + "clf_samples_uniform/"
    else:
        path = path + "clf_samples/"
                
    X = pd.read_csv(path + "p" + str(i) + "_X.csv").values
    L = pd.read_csv(path + "p" + str(i) + "_Y.csv").values[:,0]
    Y = np.zeros_like(L)
    Y[L < epsilon_loss] = 1
    
    return(X, Y)
   
   
def estimate_epsilon(p, X, noise_level, max_instances=-1, method="percentile", percentile=95, return_epsilons=False, verbose=False):

    if(max_instances == -1):
        max_instances = X.shape[0]

    num_instances = min(max_instances, X.shape[0])
    sim = instantiate_sim(p["simdata_simulator"])
    epsilons = np.zeros(num_instances)


    X_noisy = X.copy()
    for j in range(X.shape[1]):
        X_noisy[:, j] = X_noisy[:, j] + np.random.normal(loc=0, scale=np.mean(X[:, j])*noise_level)

    for i in range(num_instances):
        auto = AutoIPQ(sim, X[i, :], p["exp_parameters"], 10)
        _, opt_loss = auto.local_search(auto.optimiser, p["method_budget"])
        #print(opt_loss)
        auto = AutoIPQ(sim, X_noisy[i, :], p["exp_parameters"], 10)
        _, opt_loss2 = auto.local_search(auto.optimiser, p["method_budget"])
        #print(opt_loss2)
        epsilon_local = abs(opt_loss2-opt_loss)

        epsilons[i] = epsilon_local


    if(verbose):
        plt.hist(epsilons, bins=50)
        plt.show()

        print("Mean:", np.mean(epsilons))
        print("Std:", np.std(epsilons))
        print("Min:", np.min(epsilons))
        print("Max:", np.max(epsilons))
        print("95 percentile:", np.percentile(epsilons, 95))

    if(method == "std"):
        epsilon = np.std(epsilons)
    elif(method == "max"):
        epsilon = np.max(epsilons)
    elif(method == "percentile"):
        epsilon = np.percentile(epsilons, percentile)
    else:
        raise NotImplementedError

    if(return_epsilons):
        return(epsilon, epsilons)
    
    else:
        return(epsilon)
        
        
def generate_lr_data(num_samples=100, dim=5):
    d = np.random.uniform(low=0, high=1, size=(num_samples, dim))
    pd.DataFrame(d).to_csv("src/resources/data/lr"+str(dim)+"d.csv", index=False, header=False)


def load_lr_data(dim=5):
    d = pd.read_csv("src/resources/data/lr"+str(dim)+"d.csv").values
    return(d)


def add_gaussian_noise_X(p, X, intensity=None):
    if(intensity is None):
        noise_factor = p["data_noise_intensity"]
    else:
        noise_factor = intensity
    X_local = X.copy()
    for j in range(0, X_local.shape[1]):
        X_local[:, j] = X_local[:, j] + np.random.normal(0, scale=abs(np.mean(X_local[:, j]))*noise_factor, size=X_local.shape[0])

    return(X_local)


def load_data_synth(p):
    if(p["simdata_num_samples"] > 1):
        sim = instantiate_sim(p["simdata_simulator"], num_samples=p["simdata_num_samples"])
    elif(p["simdata_dim"] > 0):
        sim = instantiate_sim(p["simdata_simulator"], dim=p["simdata_dim"])
    else:
        sim = instantiate_sim(p["simdata_simulator"])
    if(p["simdata_generator"] == "random"):
        data_generator = RandomLUTGenerator(sim, p["exp_parameters"], seed=p["exp_seed"], sim_parameters=p["exp_parameters_simulation"])
    elif(p["simdata_generator"] == "LHS"):
        data_generator = LHSLUTGenerator(sim, p["exp_parameters"], seed=p["exp_seed"], sim_parameters=p["exp_parameters_simulation"])
    else:
        data_generator = UniformLUTGenerator(sim, p["exp_parameters"], seed=p["exp_seed"], sim_parameters=p["exp_parameters_simulation"])

    X, Y, M_table = data_generator.generate(p["simdata_num_instances"])

    M = []


    for i in range(0, M_table.shape[0]):
        d = {p["exp_parameters_simulation"][j]:M_table[i, j] for j in range(len(p["exp_parameters_simulation"]))}
        d["platform"] = "sentinel-2a"
        M.append(d)

    data = {
        "X": X,
        "Y": Y,
        "M_table": M_table,
        "M": M,
    }
    
    return(data)
        

def load_data_real(p):
    # Get keys and indices
    keys, train_indices, val_indices, test_indices = get_keys_and_indices(p["realdata_neon_path"], p["realdata_train_val_test"], 
                                                                num_tiles=p["realdata_num_tiles"])


    X_test_real, Y_test_real, M_test_real = keys_to_numpy(keys, test_indices, p["realdata_neon_path"], include_meta=True)
    X_test_real, Y_test_real, M_test_real, num_removed = remove_invalid(X_test_real, Y_test_real, M_test_real)

    data = {
        "X_train": None, # Will need to change this if actually loading train data
        "X_test": X_test_real,
        "Y_train": None,
        "Y_test": Y_test_real,
        "M_train": None,
        "M_test": M_test_real,
    }
    
    return(data)