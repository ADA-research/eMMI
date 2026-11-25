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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.feature_selection import mutual_info_regression

from scipy.stats import multivariate_normal, spearmanr, kendalltau
from scipy.spatial.distance import correlation

from matplotlib.colors import LogNorm



plt.rcParams.update({'font.size': 22})

#######################################
# Setup
#######################################

project_path = "/scratch/arp/AutoIPQ/"
total_instances = 50 # 1000 = 2000
num_samples = 1000 # 1000
overwrite = True
save_plot_instances = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#save_plot_instances = [0, 1]
i = int(sys.argv[1])-1 # Run in parallel

num_bins = 20
num_bins_asymmetric = [20, 20]
budget = 2500 # 2500

cmap = "Blues"

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


# Hardcoded for prosail 2d atm
nice_names_parameters = {
    "lai": "LAI", 
    "cab": r"$C_{ab}$",
    "x1": "x1",
    "x2": "x2",
}


simulators = ["prosail_2d", "sbi_gm", "sbi_tm"] # TODO: PCA for other ones? Arbitrary dims without plots?
#simulators = ["prosail_2d"] # TODO: remove
#conds_all = {}
#i = 1
#for simulator in simulators:
#    conds_all[i] = {"simulator":simulator}
#    i += 1        
#conds = conds_all[int(sys.argv[1])]
conds = {"simulator": "prosail_2d"} # Sacrificing ease of using other sims easily for speed, can be combined if needed
instances = np.arange(total_instances)

conds_all = {}
i = 1
for sim in simulators:
    for inst in instances:
        conds_all[i] = {"simulator":sim, "i":inst}
        #print(str(i) + ": " + str(conds_all[i]))
        i += 1
        
conds = conds_all[int(sys.argv[1])]
i = conds["i"]


# Some more specific setup

p = set_default_parameters()
load_parameters_file(p, param_paths[conds["simulator"]])
sim = instantiate_sim(p["simdata_simulator"])

deltas = {}
for param in p["exp_parameters"]:
    deltas[param] = (sim.param_ranges[param][3] - sim.param_ranges[param][2]) / num_bins


# Setup dirs


currdir = project_path + "shift_loss_correlation/"
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


plot_dir = currdir + "plots/"
if(not(os.path.exists(plot_dir))):
    try:
        os.mkdir(plot_dir)
    except:
        pass


# Load clean test data, noisy training data
data = load_dataset(conds["simulator"], project_path, max_instances=total_instances, splits=["train", "test"])
data_noise = load_dataset(conds["simulator"], project_path, max_instances=total_instances, splits=["train"], noise=p["data_noise"])

mean_vec = np.absolute(np.mean(data['test']['X'], axis=0))


# Train UQ model
uq = GaussianProcessRegressor()
#uq.fit(data_noise['train']['X'], data_noise['train']['Y'])
uq.fit(data['train']['X'], data['train']['Y'])


# Initialise correlation coefficient arrays
#cc_eM = np.zeros(total_instances)
#cc_uq = np.zeros(total_instances)


# Iterate over instances
#for i in range(total_instances):

res_store_path = currdir + str(i) + ".csv"
if(overwrite or not(os.path.exists(res_store_path))):
    with open(res_store_path, 'w') as fp:
        s = "metric,loss_landscape,uq\n"
        fp.write(s)

    x_clean = data['test']['X'][i]






    #####################################################
    # Compute loss landscape (explanatory variable 1)
    #####################################################


    dims = []
    for param in p["exp_parameters"]:
        dims.append(np.arange(sim.param_ranges[param][2], sim.param_ranges[param][3], deltas[param]))
    
    #x1_points = np.linspace(sim.param_ranges[p["exp_parameters"][0]][2], sim.param_ranges[p["exp_parameters"][0]][3], num_bins)
    #x2_points = np.linspace(sim.param_ranges[p["exp_parameters"][1]][2], sim.param_ranges[p["exp_parameters"][1]][3], num_bins)
    x1_points = np.linspace(sim.param_ranges[p["exp_parameters"][0]][2], sim.param_ranges[p["exp_parameters"][0]][3], num_bins_asymmetric[0])
    x2_points = np.linspace(sim.param_ranges[p["exp_parameters"][1]][2], sim.param_ranges[p["exp_parameters"][1]][3], num_bins_asymmetric[1])

    X1, X2 = np.meshgrid(x1_points, x2_points) # TODO: make this possible for arbitrary dim

    auto = AutoIPQ(sim, x_clean, p["exp_parameters"], 2, optimiser=p["method_optimiser"])
    Z = np.zeros_like(X1)
    for i2 in range(Z.shape[0]):
        for j2 in range(Z.shape[1]):
            config = {p["exp_parameters"][0]: X1[i2, j2], p["exp_parameters"][1]: X2[i2, j2]}
            sim_out = auto.optimiser.sim.simulate(config)
            Z[i2, j2] = auto.optimiser.f1(sim_out, x_clean)

    if(i in save_plot_instances):
        plt.pcolormesh(X1, X2, Z, norm=LogNorm(vmin=np.min(Z), vmax=np.max(Z)), rasterized=True, cmap=cmap, shading='gouraud')
        plt.title("Loss landscape")
        plt.colorbar()
        plt.xlabel(nice_names_parameters[p["exp_parameters"][0]])
        plt.ylabel(nice_names_parameters[p["exp_parameters"][1]])
        plt.savefig(plot_dir + str(i) + "_loss.pdf", bbox_inches='tight') # TODO: axis ticks
        plt.show()
        plt.clf()



    #####################################################
    # Compute shift distribution (target variable)
    #####################################################

    l_optim = np.zeros((num_samples, len(p["exp_parameters"])))

    # Create noisy datapoints
    noise_mat = np.zeros((num_samples, len(x_clean)))
    for j in range(noise_mat.shape[1]):
        noise_mat[:, j] = np.random.normal(loc=0, scale=p["data_noise"]*mean_vec[j], size=num_samples)

    noisy_samples = noise_mat + x_clean # TODO: verify that broadcasting is all correct

    # Iterate over noise repetitions
    for r in range(num_samples):
        
        x = noisy_samples[r, :]
        
        # Compute optimum
        method_kwargs = {
                            "budget": budget, 
                            "budget_split": [1.0, 0.0, 0.0],
                            "num_solutions": 2,
                            "return_opt": False, 
                            "store_XY": False, 
                            "store_test":False,
                            "train_clf": False,
                        }
                        
        auto = AutoIPQ(sim, x, p["exp_parameters"], 2, optimiser=p["method_optimiser"])
        l_clf = auto.run(0.1, None, **method_kwargs)
        opt = auto.optima[0]
        opt_vec = auto.optimiser.solution_to_coords(opt)

        # Add optimum to list of optima
        l_optim[r, :] = opt_vec


    # Compute 2D histogram
    #range_mat = np.stack([[sim.param_ranges[param][2], sim.param_ranges[param][3]] for param in p["exp_parameters"]], axis=0)
    range_mat = [ [sim.param_ranges[p["exp_parameters"][0]][2], sim.param_ranges[p["exp_parameters"][0]][3]],
                    [sim.param_ranges[p["exp_parameters"][1]][2], sim.param_ranges[p["exp_parameters"][1]][3]] ]
    
    #h, x1edges, x2edges, _ = plt.hist2d(l_optim[:, 0], l_optim[:, 1], bins=[num_bins, num_bins], density=False, range=range_mat, cmap=cmap)
    h, x1edges, x2edges = np.histogram2d(l_optim[:, 0], l_optim[:, 1], bins=[num_bins_asymmetric[0], num_bins_asymmetric[1]], 
                                            density=False, range=range_mat)
    h = h.T # histogram2 forces y first (normally matplotlib does x, y, not y, x)


    if(i in save_plot_instances):
        #plt.xlim(range_mat[0, 0], range_mat[0, 1])
        #plt.ylim(range_mat[1, 0], range_mat[1, 1])
        X1_hist, X2_hist = np.meshgrid(x1edges, x2edges)
        #plt.pcolormesh(X1_hist, X2_hist, h, norm=LogNorm(vmin=np.min(Z), vmax=np.max(Z)), rasterized=True, cmap=cmap, shading='gouraud')
        plt.pcolormesh(X1, X2, h, rasterized=True, cmap=cmap)
        plt.title("Sample density")
        plt.xlabel(nice_names_parameters[p["exp_parameters"][0]])
        plt.ylabel(nice_names_parameters[p["exp_parameters"][1]])
        plt.colorbar()
        plt.savefig(plot_dir + str(i) + "_density.pdf", bbox_inches='tight')
        plt.show()
        plt.clf()



    ##### Compare loss landscape and shift distribution ##################

    # Image
    rgb = np.zeros((num_bins_asymmetric[1], num_bins_asymmetric[0], 3)) # loss blue, frequency red, joint purple
    norm = LogNorm(vmin=np.min(Z), vmax=np.max(Z))
    Z_log = norm(Z)
    rgb[:, :, 2] = (Z_log - np.min(Z_log)) / (np.max(Z_log) - np.abs(np.min(Z_log)))  # Invert and normalise to 0-1, min/max also inv
    rgb[:,:,2] = -rgb[:,:,2] + 1 # This should invert
    rgb[:, :, 0] = (h - np.min(h)) / (np.max(h) - np.min(h))


    if(i in save_plot_instances):

        #plt.xlim(range_mat[0, 0], range_mat[0, 1])
        #plt.ylim(range_mat[1, 0], range_mat[1, 1])
        extent = [sim.param_ranges[p["exp_parameters"][0]][2], sim.param_ranges[p["exp_parameters"][0]][3],
                                                sim.param_ranges[p["exp_parameters"][1]][2], sim.param_ranges[p["exp_parameters"][1]][3]]
        #print("extent:", extent)
        
        #plt.imshow(rgb, origin='lower', interpolation='catrom', aspect='equal', extent=extent)
        plt.imshow(rgb, origin='lower', interpolation='catrom')
        plt.axis('off')
        plt.title("RGB composite")
        #plt.xlabel(nice_names_parameters[p["exp_parameters"][0]])
        #plt.ylabel(nice_names_parameters[p["exp_parameters"][1]])
        plt.savefig(plot_dir + str(i) + "_loss_frequency_joint.pdf", bbox_inches='tight')
        plt.show()
        plt.clf()



    # Add correlation coefficient to list
    v1_eM = h.flatten()
    v2_eM = Z.flatten()
    #v1 = rgb[:, :, 0].flatten()
    #v2 = rgb[:, :, 2].flatten()

    v2_eM = v2_eM[v1_eM>0] # Only run for cells with a frequency
    v1_eM = v1_eM[v1_eM>0]



    # Scatterplot
    if(i in save_plot_instances):
        #plt.xlim(range_mat[0, 0], range_mat[0, 1])
        #plt.ylim(range_mat[1, 0], range_mat[1, 1])
        plt.scatter(v1_eM, v2_eM)
        plt.title("Shift frequency VS loss")
        plt.xlabel("Shift frequency")
        plt.ylabel("Loss")
        plt.savefig(plot_dir + str(i) + "_scatter_loss.pdf", bbox_inches='tight')
        plt.show()
        plt.clf()





    #####################################################
    # Compute UQ probability density (explanatory variable 2)
    #####################################################


    # Compute UQ probability density
    observation = x_clean.reshape((1, len(x_clean)))
    mu = np.mean(l_optim, axis=0)
    sigma = np.std(l_optim, axis=0)
    cov_mat = np.diag(sigma)

    x1_points = np.linspace(sim.param_ranges[p["exp_parameters"][0]][2], sim.param_ranges[p["exp_parameters"][0]][3], num_bins_asymmetric[0])
    x2_points = np.linspace(sim.param_ranges[p["exp_parameters"][1]][2], sim.param_ranges[p["exp_parameters"][1]][3], num_bins_asymmetric[1])

    X1, X2 = np.meshgrid(x1_points, x2_points) # TODO: make this possible for arbitrary dim

    rv = multivariate_normal(mean=mu, cov=cov_mat)


    pd = np.zeros((num_bins_asymmetric[0], num_bins_asymmetric[1]))
    for i2 in range(pd.shape[0]):
        for j2 in range(pd.shape[1]):
            pd[i2, j2] = rv.pdf([X1[i2, j2], X2[i2, j2]]) # TODO: arbitrary dim



    Z = np.zeros_like(X1)
    for i2 in range(Z.shape[0]):
        for j2 in range(Z.shape[1]):
            point = [X1[i2, j2], X2[i2, j2]]
            pd[i2, j2] = rv.pdf(point)



    if(i in save_plot_instances):
        #plt.imshow(pd, origin='lower', extent=[sim.param_ranges[p["exp_parameters"][0]][2], 
        #                                                sim.param_ranges[p["exp_parameters"][0]][3], 
        #                                                sim.param_ranges[p["exp_parameters"][1]][2], 
        #                                                sim.param_ranges[p["exp_parameters"][1]][3]])
        #plt.imshow(pd, origin='lower', cmap=cmap)
        plt.pcolormesh(X1, X2, pd, rasterized=True, cmap=cmap, shading='gouraud')

        plt.title("UQ probability density")
        plt.xlabel(nice_names_parameters[p["exp_parameters"][0]])
        plt.ylabel(nice_names_parameters[p["exp_parameters"][1]])
        plt.colorbar()
        plt.savefig(plot_dir + str(i) + "_pd.pdf", bbox_inches='tight')
        plt.show()
        plt.clf()


    ##### Compare UQ distribution and shift distribution ##################


    rgb = np.zeros((num_bins_asymmetric[1], num_bins_asymmetric[0], 3)) # loss blue, frequency red, joint purple
    rgb[:, :, 2] = (pd - np.min(pd)) / (np.max(pd) - np.abs(np.min(pd)))  # Invert and normalise to 0-1, min/max also inv
    rgb[:, :, 0] = (h - np.min(h)) / (np.max(h) - np.min(h))

    # Image
    if(i in save_plot_instances):

        #plt.xlim(range_mat[0, 0], range_mat[0, 1])
        #plt.ylim(range_mat[1, 0], range_mat[1, 1])
        plt.imshow(rgb, origin='lower', interpolation='catrom')#, aspect='equal', extent=[sim.param_ranges[p["exp_parameters"][0]][2], sim.param_ranges[p["exp_parameters"][0]][3],
                                                #sim.param_ranges[p["exp_parameters"][1]][2], sim.param_ranges[p["exp_parameters"][1]][3]])
        #plt.title("Probability density and optimum shift frequency")
        plt.axis('off')
        plt.title("RGB composite")
        #plt.xlabel(nice_names_parameters[p["exp_parameters"][0]])
        #plt.ylabel(nice_names_parameters[p["exp_parameters"][1]])
        plt.savefig(plot_dir + str(i) + "_pd_frequency_joint.pdf", bbox_inches='tight')
        plt.show()
        plt.clf()

    if(i in save_plot_instances):
        #plt.xlim(range_mat[0, 0], range_mat[0, 1])
        #plt.ylim(range_mat[1, 0], range_mat[1, 1])
        plt.scatter(h.flatten(), pd.flatten())
        plt.title("Shift frequency VS UQ Probability density")
        plt.xlabel("Shift frequency")
        plt.ylabel("Probability density")
        plt.savefig(plot_dir + str(i) + "_scatter_uq.pdf", bbox_inches='tight')
        plt.show()
        plt.clf()

    # Add correlation coefficient to list
    v1_uq = h.flatten()
    v2_uq = pd.flatten()

    #v1 = rgb[:, :, 0].flatten()
    #v2 = rgb[:, :, 2].flatten()

    v2_uq = v2_uq[v1_uq>0] # Only run for cells with a frequency
    v1_uq = v1_uq[v1_uq>0]


    # Correlation coefficient
    cc_uq = np.corrcoef(v1_uq, v2_uq)[0, 1]
    cc_eM = np.corrcoef(v1_eM, v2_eM)[0, 1]

    # Mutual information
    try:
        mi_mat = mutual_info_regression(np.stack([v2_eM, v2_uq], axis=1), v2_uq)
        mi_eM = mi_mat[0]
        mi_uq = mi_mat[1]
    except:
        mi_eM = np.nan
        mi_uq = np.nan # shouldn't really happen with enough samples, but just in case

    # Rank correlation
    rc_eM = spearmanr(v2_eM, v1_eM)[0]
    rc_uq = spearmanr(v2_uq, v1_uq)[0]

    # Kendall's tau
    kt_eM = kendalltau(v2_eM, v1_eM)[0]
    kt_uq = kendalltau(v2_uq, v1_uq)[0]

    # Distance correlation
    dc_eM = correlation(v2_eM, v1_eM)
    dc_uq = correlation(v2_uq, v1_uq)



    if(overwrite or not(os.path.exists(res_store_path))):
        with open(res_store_path, 'a') as fp:
            s = "correlation_coefficient" + "," + str(cc_eM) + "," + str(cc_uq) + "\n"
            s = s + "mutual_information" + "," + str(mi_eM) + "," + str(mi_uq) + "\n"
            s = s + "rank_correlation" + "," + str(rc_eM) + "," + str(rc_uq) + "\n"
            s = s + "kendall_tau" + "," + str(kt_eM) + "," + str(kt_uq) + "\n"
            s = s + "distance_correlation" + "," + str(dc_eM) + "," + str(dc_uq) + "\n"

            fp.write(s)



    
print("Terminated successfully:", conds)