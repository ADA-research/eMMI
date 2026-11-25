import numpy as np

import os
import sys

from src.AutoIPQ import *
from src.ForwardModel import *
from shared_utils import set_default_parameters, load_data_synth, instantiate_sim

from sklearn.ensemble import GradientBoostingClassifier


#######################################
# Setup
#######################################

project_dir = "/scratch/arp/AutoIPQ/"
num_instances = {"train":20000, "val":1000, "test":1000}
overwrite = False

max_instances = 200 # 1000
opt_budget = 50000
#opt_budget = 100
opt_loss_repeats = 1


sim_params = {
    "prosail":  {
                        "parameters":['lai', 'cab', 'lidfa2', 'cw'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
                    },
    "prosail_2d":  {
                        "parameters":['lai', 'cab'],
                        "sim_parameters":['tts', 'tto', 'psi'],
                        "simdata_num_samples":1,
                    },
    "prosail_sim":  {
                        "parameters":['lai', 'cab', 'hspot', 'n', 'lidfa2', 'cw', 'cm'],
                        "sim_parameters":['tts', 'tto', 'psi'],
                        "simdata_num_samples":1,
                    },
    "prosail_real": {
                        "parameters":['lai', 'cab', 'hspot', 'n', 'lidfa2', 'cw', 'cm'],
                        "sim_parameters":[],
                        "simdata_num_samples":1,
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
                        "simdata_num_samples":30, 
                    },
    "sbi_slcp":     {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_bglm":     {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_glu":      {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_tm":       {
                        "parameters":['x1', 'x2'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_gl":       {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_slcpd":      {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
    "sbi_bglmr":    {
                        "parameters":['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
                        "sim_parameters":[],
                        "simdata_num_samples":30,
                    },
}

sim_outs = {
    "prosail": 11,
    "prosail_2d": 11,
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



datasets = ["train", "val", "test"]
#datasets = ["train", "test"]
#datasets = ["val"] # TODO: remove
noise_levels = [0.1] # consider adding more; probably one noise setting is enough
simulators = [
        #"prosail",
        #"prosail_2d",
        #"prosail_sim",
        #"prosail_real",
        #"mdgs1",
        #"mdgs2",
        #"mdgs5",
        #"mdgs10",
        #"mdgs20",
        #"mdgs50",
        "lr2d",
        #"lr5d",
        "lr10d",
        "lr20d",
        "lr50d",
        #"dpr_tp",
        #"dpr_lorenz63",
        #"sbi_gm",
        #"sbi_slcp",
        #"sbi_bglm",
        #"sbi_glu",
        #"sbi_tm",
        #"sbi_gl",
        #"sbi_slcpd",
        #"sbi_bglmr",
    ]


all_conds = {}
i = 1
for ds in datasets:
    for sim in simulators:
        for nl in noise_levels:
            all_conds[i] = {"dataset":ds, "simulator":sim, "noise_level":nl}
            i += 1
        
conds = all_conds[int(sys.argv[1])]


currdir = project_dir + "data/"  
currdir = currdir + "experiments/"    
currdir = currdir + conds["simulator"] + "/"
currdir = currdir + conds["dataset"] + "/"


        
        


#######################################
# Main noisy instances generation
#######################################

# Instantiate/set all relevant stuff
p = set_default_parameters()
p["simdata_simulator"] = conds["simulator"]
if(conds["simulator"] in ["prosail", "prosail_2d", "prosail_sim", "prosail_real"]):
    p["simdata_simulator"] = "prosail"
elif(conds["simulator"].startswith("mdgs")):
    p["simdata_simulator"] = "mdgs"
    p["simdata_dim"] = int(conds["simulator"][4:])
p["exp_parameters"] = sim_params[conds["simulator"]]["parameters"]
p["exp_parameters_simulation"] = sim_params[conds["simulator"]]["sim_parameters"]
p["simdata_num_samples"] = sim_params[conds["simulator"]]["simdata_num_samples"]

p["exp_max_instances"] = num_instances[conds["dataset"]]
p["simdata_num_instances"] = num_instances[conds["dataset"]]
p["method_budget"] = opt_budget # For pre-computing global optimum
p["method_budget_split"] = [1.0, 0.0, 0.0]

# TODO: set seed and pass to all relevant stuff
# This is just to make sure train, val and test use different seeds
#i = 0
#for ds in datasets:
#    if(ds == conds["dataset"]):
#        p["exp_seed"] = p["exp_seed"] + i
#    i += 1
#np.random.seed(p["exp_seed"])


# Load existing instances
existing_path = currdir + "instances_X.csv"
# Only keep first few lines because messed up in data genetion, kept adding more copies to csv files
existing_X = pd.read_csv(existing_path).values[:num_instances[conds["dataset"]], :] # 
new_X = existing_X.copy()

# Pre-compute stds
stds = {}
for j in range(1, existing_X.shape[1]):
    std = np.std(existing_X[:, j])
    new_X[:, j] = new_X[:, j] + np.random.normal(loc=0, scale=conds["noise_level"]*std)
    
# Write noisy instances
path_X = currdir + "instances_X_noise" + str(conds["noise_level"]) + ".csv"
if(overwrite or not(os.path.exists(path_X))):
    # header
    s = "instance"
    num_outputs = sim_outs[conds["simulator"]]
    outputs = ["o" + str(j) for j in range(num_outputs)]
    for output in outputs:
        s = s + "," + output
    s = s + "\n"
    # rows
    for i in range(new_X.shape[0]):
        s_local = ""
        for j in range(new_X.shape[1]):
            val = int(new_X[i,j]) if j == 0 else new_X[i,j]
            s_local = s_local + str(val) + ","
        s = s + s_local[:-1] + "\n"
    
    with open(path_X, 'w') as fp:
        fp.write(s)
        
else:
    # If noisy stuff already exists, make sure to load its values for parallel opt computation
    new_X = pd.read_csv(path_X).values

# Iterate over noisy test/val instances, compute optimum (for now just test)

if(conds["dataset"] in ["test", "val"]):


    path_opt = currdir + "instances_opt_noise" + str(conds["noise_level"]) + ".csv"

    if(overwrite or not(os.path.exists(path_opt))):
        s = "instance,opt_loss"
        params = sim_params[conds["simulator"]]["parameters"]
        for param in params:
            s = s + "," + param
        s = s + "\n"
        with open(path_opt, 'w') as fp:
            fp.write(s)
            
    order = np.arange(min(new_X.shape[0], max_instances))
    np.random.shuffle(order)
    
    existing_runs = pd.read_csv(path_opt).values[:, 0]

    for i in order:
        if(i not in existing_runs):

            # Set priors, instantiate sim
            
            m = pd.read_csv(currdir + "instances_M.csv").values[i, 1:]
            
            priors = {}
            if(len(m) > 1):
                params_sim = sim_params[conds["simulator"]]["sim_parameters"]
                priors = {params_sim[j]:m[j] for j in range(len(params_sim))}
            sim = instantiate_sim(p["simdata_simulator"], num_samples=p["simdata_num_samples"], priors=priors, dim=p["simdata_dim"])
            
            # Set up AutoIPQ; only use step 1 result (optimum)
            
            def instantiate_boost():
                return(GradientBoostingClassifier())

            method_kwargs = {
                                "budget": p["method_budget"], 
                                "budget_split": p["method_budget_split"],
                                "num_solutions": 2,
                                "return_opt": False, 
                                "store_XY": False, 
                                "store_test":False,
                                "train_clf": False,
                            }
            instance_x = new_X[i, 1:]
            
            # Run AutoIPQ to train classifier(s)
            auto = AutoIPQ(sim, instance_x, p["exp_parameters"], 2, optimiser="RS", optimiser_2a="RS", optimiser_2b="RS", optimiser_2c="RS", optimiser_2d="RS")

            #l_clf = auto.run(p["exp_epsilon"], instantiate_boost, **method_kwargs)
            l_clf = auto.run(0.1, instantiate_boost, **method_kwargs)
                
            opt = auto.optima[0]
            opt_vec = auto.optimiser.solution_to_coords(opt)
            
            opt_loss_total = np.zeros(opt_loss_repeats)
            for k in range(opt_loss_repeats):
                opt_sim = auto.optimiser.forward_sim(opt_vec, format="vec")
                opt_loss = auto.optimiser.f1(opt_sim, instance_x)
                opt_loss_total[k] = opt_loss
                
            opt_loss = np.median(opt_loss_total)
          

            existing_runs = pd.read_csv(path_opt).values[:, 0]
            if(i not in existing_runs):
            
                with open(path_opt, 'a') as fp:
                    s = str(i) + "," + str(opt_loss)
                    for scalar in opt_vec:
                        s = s + "," + str(scalar)
                    s = s + "\n"
                    fp.write(s)
                
            existing_runs = pd.read_csv(path_opt).values[:, 0]
            
            

# Done
print("Terminated successfully:", conds)