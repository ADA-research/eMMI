import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wilcoxon, ttest_ind



base_path = "/scratch/arp/AutoIPQ/shift_loss_correlation/"
simulators = ["prosail_2d", "sbi_gm", "sbi_tm"]
methods = ["loss_landscape", "uq"]

metric = "rank_correlation" # correlation_coefficient, mutual_information, rank_correlation, kendall_tau, distance_correlation


significance_level = 0.05
table_decimals = 2



# Create LaTeX table of results

#metric_names = ["Accuracy", "F1-measure"] # For now just do F1 score


table_decimals = 2
nice_names_sim = {
    "prosail": "PROSAIL", 
    "prosail_2d": "PROSAIL 2D", 
    "dpr_tp": "Two Pendulums",
    "dpr_lorenz63": "Lorenz 63",
    "mdgs2": r"MDGS ($d=2$)",
    "mdgs5": r"MDGS ($d=5$)",
    "mdgs10": r"MDGS ($d=10$)",
    "mdgs20": r"MDGS ($d=20$)",
    "mdgs50": r"MDGS ($d=50$)",
    "sbi_bglm": "BGLM", 
    "sbi_gm": "GM", 
    "sbi_tm": "TM",
    "sbi_bglm": "BGLM",
    "sbi_gl": "GL",
    "sbi_glu": "GLU",
}

nice_names_methods = {
    "loss_landscape": r"Loss landscape",
    "uq": "Probability density",
}



mat = np.zeros((len(simulators), len(methods)))
std_mat = np.zeros((len(simulators), len(methods)))
sig_mat = np.zeros((len(simulators), len(methods))) # signficance, 1 means significantly better

i = 0
for sim in simulators:
    result_path = base_path + sim + "/"

    val_mat = np.zeros((50, len(methods))) # Hardcoded to 50 instances now

    for d in range(val_mat.shape[0]):

        try:
        #for d in os.listdir(result_path):
            # saved a little awkwardly
            res_df = pd.read_csv(result_path + str(d) + ".csv")
            res_df = res_df[res_df['metric'] == metric]

            j = 0
            for m in methods:

                #result_path = base_path + sim + "/" + m + "/true_point_evals.csv"
                #res_df = pd.read_csv(result_path)
                #print("length " + str(len(res_df['true'].values)) + " for simulator " + str(sim) + " and method " + str(m))
                #f1 = f1_score(res_df['true'].values, res_df['result'].values)

                val_mat[d,j] = res_df[m].values[0]
                j += 1
        except:
            print("problem for path " + result_path + str(d) + ".csv")
            print(res_df[m].head())
            print("\n")



    j = 0
    for m in methods:
        mat[i,j] = np.round(np.nanmean(np.absolute(val_mat[:, j])), decimals=table_decimals) # nanmean for failed mutual information; shouldn't happen with enough samples (small test problem)
        std_mat[i,j] = np.round(np.absolute(np.nanstd(val_mat[:, j])), decimals=table_decimals)
        j += 1

    print(val_mat)

    # Significance test; hardcoded to 2, unlikely to need more for now
    j = 0
    for m in range(len(methods)):
        for m2 in range(len(methods)):
            if(not(m == m2)):
                t = ttest_ind(np.absolute(mat[:, m]), np.absolute(mat[:, m2]), alternative="greater")
                if(t[1] < significance_level):
                    sig_mat[i, j] = sig_mat[i, j] + 1
        j += 1


    i += 1



# CSV table
s = "Dataset," + str(nice_names_methods[methods[0]]) + "," + str(nice_names_methods[methods[1]]) + "\n"
for i in range(len(simulators)):
    c1 = str(nice_names_sim[simulators[i]])
    if(sig_mat[i,0] == 1):
        c2 = "$\mathbf{" + str(mat[i, 0]) + " \pm " + str(std_mat[i, 0]) + "}$"
    else:
        c2 = "$" + str(mat[i, 0]) + " \pm " + str(std_mat[i, 0]) + "$"
    if(sig_mat[i,1] == 1):
        c3 = "$\mathbf{" + str(mat[i, 1]) + " \pm " + str(std_mat[i, 1]) + "}$"
    else:
        c3 = "$" + str(mat[i, 1]) + " \pm " + str(std_mat[i, 0]) + "$"
    s = s + c1 + "," + c2 + "," + c3 + "\n"

with open("results/result_table_slc.csv", 'w') as fp:
    fp.write(s)





print("Terminated successfully.")