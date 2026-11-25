import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wilcoxon



base_path = "/scratch/arp/AutoIPQ/manifold_validation/"
simulators = ["prosail", "prosail_2d", "dpr_tp", "dpr_lorenz63", "sbi_gm", "sbi_tm", "lr5d"]
#simulators = ["prosail", "prosail_2d", "dpr_tp", "dpr_lorenz63", "sbi_gm", "sbi_tm"] # TODO: remove
methods = ["eM", "UC"]

significance_level = 0.05
table_decimals = 2


metrics = ["accuracy", "precision", "recall"]



# Create LaTeX table of results

#metric_names = ["Accuracy", "F1-measure"] # For now just do F1 score


table_decimals = 2
nice_names_sim = {
    "prosail": "PROSAIL", 
    "prosail_2d": "PROSAIL 2D", 
    "dpr_tp": "TP",
    "dpr_lorenz63": "Lorenz63",
    "mdgs2": r"MDGS ($d=2$)",
    "mdgs5": r"MDGS ($d=5$)",
    "mdgs10": r"MDGS ($d=10$)",
    "mdgs20": r"MDGS ($d=20$)",
    "mdgs50": r"MDGS ($d=50$)",
    "lr5d": "LR",
    "sbi_bglm": "BGLM", 
    "sbi_gm": "GM", 
    "sbi_tm": "TM",
    "sbi_bglm": "BGLM",
    "sbi_gl": "GL",
    "sbi_glu": "GLU",
}

nice_names_methods = {
    "eM": r"$\epsilon$-manifold",
    "UC": "Uncertainty Quantification",
}



mat = np.zeros((len(simulators), len(methods)*len(metrics)))
std_mat = np.zeros((len(simulators), len(methods)*len(metrics)))
sig_mat = np.zeros((len(simulators), len(methods)*len(metrics))) # signficance, 1 means significantly better



metric_offset = 0

for metric in metrics:
    if(metric == "accuracy"):
        metric_func = accuracy_score
    elif(metric == "precision"):
        metric_func = precision_score
    elif(metric == "recall"):
        metric_func = recall_score
    else:
        raise NotImplementedError

    



    i = 0
    for sim in simulators:
        j = 0
        for m in methods:

            result_path = base_path + sim + "/" + m + "/true_point_evals.csv"
            res_df = pd.read_csv(result_path)
            #print("length " + str(len(res_df['true'].values)) + " for simulator " + str(sim) + " and method " + str(m))
            #f1 = f1_score(res_df['true'].values, res_df['result'].values)
            scores = 1 - np.absolute(res_df['true'].values - res_df['result'].values) # 1 if correct, 0 if incorrect
            score = metric_func(res_df['true'].values, res_df['result'].values)


            col_index = j*len(metrics) + metric_offset 


            print("sim:", sim)
            print("method:", m)
            print("metric:", metric)
            print("row index i:", i)
            print("method index j:", j)
            print("metric index offset:", metric_offset)
            print("column index:", col_index)

            print("\n")




            mat[i,col_index] = np.round(score, decimals=table_decimals)
            try:
                std_mat[i,col_index] = np.round(np.std(scores), decimals=table_decimals)

            except:
                print("problem for " + result_path)

            j += 1

        i += 1


    # Significance test; hardcoded to 2, unlikely to need more for now
    i = 0
    for sim in simulators:

        result_path = base_path + sim + "/" + methods[0] + "/true_point_evals.csv"
        res_df = pd.read_csv(result_path)
        result_path = base_path + sim + "/" + methods[1] + "/true_point_evals.csv"
        res_df_bl = pd.read_csv(result_path)

        scores = 1 - np.absolute(res_df['true'].values - res_df['result'].values) # 1 if correct, 0 if incorrect
        scores_bl = 1 - np.absolute(res_df_bl['true'].values - res_df_bl['result'].values) # 1 if correct, 0 if incorrect

        score = metric_func(res_df['true'].values, res_df['result'].values)
        score_bl = metric_func(res_df_bl['true'].values, res_df_bl['result'].values)

        #print(sim)
        #print(scores.shape)
        #print(scores_bl.shape)



        pvalue_m1_better = wilcoxon(scores, scores_bl, alternative="greater").pvalue
        pvalue_m2_better = wilcoxon(scores, scores_bl, alternative="less").pvalue

        if(pvalue_m1_better < significance_level):
            sig_mat[i, 0:3] = 1
        elif(pvalue_m1_better < significance_level):
            sig_mat[i, 3:6] = 1
        elif(pvalue_m1_better > significance_level and pvalue_m2_better > significance_level):
            sig_mat[i, 0:3] = 1
            sig_mat[i, 3:6] = 1
        else:
            print("shouldn't happen")
            #raise NotImplementedError

        i += 1

    metric_offset += 1



# CSV table accuracy
s = "Dataset,"
for m in methods:
    s = s + nice_names_methods[m] + ","
s = s[:-1] + "\n"

for i in range(len(simulators)):
    r = str(nice_names_sim[simulators[i]]) + "," # dataset name

    for j in range(len(methods)):

        for metric_offset in range(len(metrics)):

            if(metrics[metric_offset] == 'accuracy'):

                col_index = j*len(metrics) + metric_offset 

                if(sig_mat[i,col_index] == 1):
                    c2 = "$\mathbf{" + str(mat[i, col_index]) + " \pm " + str(std_mat[i, col_index]) + "}$"
                else:
                    c2 = "$" + str(mat[i, col_index]) + " \pm " + str(std_mat[i, col_index]) + "$"

                r = r + c2 + ","

    r = r[:-1] + "\n"
    s = s + r

with open("results/result_table_manifold.csv", 'w') as fp:
    fp.write(s)



# CSV table precision/recall


s = "Dataset,"
for m in methods:
    for metric in metrics:
        if(metric != 'accuracy'):
            s = s + nice_names_methods[m] + "_" + metric + ","
s = s[:-1] + "\n"

for i in range(len(simulators)):
    r = str(nice_names_sim[simulators[i]]) + "," # dataset name

    for j in range(len(methods)):

        for metric_offset in range(len(metrics)):

            if(metrics[metric_offset] != 'accuracy'):

                col_index = j*len(metrics) + metric_offset 
                c2 = "$" + str(mat[i, col_index]) + "$"

                r = r + c2 + ","

    r = r[:-1] + "\n"
    s = s + r

with open("results/result_table_manifold_extra.csv", 'w') as fp:
    fp.write(s)


# Direct LaTeX table export
#df = pd.DataFrame(mat, columns=[nice_names_methods[m] for m in methods], index=[nice_names_sim[s] for s in simulators])


#df.to_latex("results/result_table_manifold.tex", 
#                  caption=r"Numerical results for $\epsilon$-manifold validation.", 
#                  label="tab:results_manifold")


print("Terminated successfully.")