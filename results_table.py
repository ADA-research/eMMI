import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scipy.stats import wilcoxon



metric = "recall" # accuracy, f1

if(metric == "accuracy"):
    metric_func = accuracy_score # accuracy_score, f1_score
    metric_name = "Accuracy"
    suffix2 = ""
elif(metric == "f1"):
    metric_func = f1_score
    metric_name = "F1-measure"
    suffix2 = "f1_measure"
elif(metric == "precision"):
    metric_func = precision_score
    metric_name = "Precision"
    suffix2 = "precision"
elif(metric == "recall"):
    metric_func = recall_score
    metric_name = "Recall"
    suffix2 = "recall"


sim_sets = ["all", "sbi", "physical", "lr", "mdgs", "all"] # all, physical, sbi, mdgs
sim_sets = ["all"] # all, physical, sbi, mdgs
table_formats = ["csv"] # csv, latex

combine_emmi = True


base_path = "/scratch/arp/AutoIPQ/results/"
#simulators_all = ["prosail", "prosail_2d", "mdgs2", "dpr_tp", "dpr_lorenz63", "sbi_gm", "sbi_bglm", "sbi_gl"]
simulators_all = ["prosail", "prosail_2d", "dpr_tp", "dpr_lorenz63", "sbi_gm", "sbi_tm", "lr5d"]
simulators_physical = ["prosail", "prosail_2d", "dpr_tp", "dpr_lorenz63"]
#simulators_sbi = ["sbi_gm", "sbi_bglm", "sbi_gl"]
simulators_sbi = ["sbi_gm", "sbi_tm", "lr5d"]
simulators_mdgs = ["mdgs2", "mdgs5", "mdgs10", "mdgs20", "mdgs50"]
simulators_lr = ["lr2d", "lr5d", "lr10d", "lr20d", "lr50d"]

methods_separate = ["RF", "GP", "BNN", "classification", "Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ", ]
methods_separate = ["classification", "Conv-AutoIPQ", "Seq-AutoIPQ", "Dual-AutoIPQ"] # TODO: remove
methods_combined = ["RF", "GP", "BNN", "ABCSMC", "TabPFN", "AutoIPQ", ]
sample_types = ["uniform"]

if(combine_emmi):
    methods = methods_combined
    table_name = "result_table_combined"
else:
    methods = methods_separate
    table_name = "result_table"



# Parameters

budget = 20000
sample_type = "uniform"
significance_level = 0.05

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
    "lr2d": r"LR ($d=2$)",
    "lr5d": r"LR ($d=5$)",
    "lr10d": r"LR ($d=10$)",
    "lr20d": r"LR ($d=20$)",
    "lr50d": r"LR ($d=50$)",
    "sbi_bglm": "BGLM", 
    "sbi_gm": "GM", 
    "sbi_tm": "TM",
    "sbi_bglm": "BGLM",
    "sbi_gl": "GL",
    "sbi_glu": "GLU",
}

nice_names_methods = {
    "RF": "RF",
    "GP": "GP",
    "BNN": "BNN", 
    "ABCSMC": "ABCSMC",
    "TabPFN": "TabPFN",
    "classification": "U-eMMI", 
    "Conv-AutoIPQ": "Conv-eMMI",
    "Seq-AutoIPQ": "Seq-eMMI", 
    "Dual-AutoIPQ": "Dual-eMMI",
    "AutoIPQ": "AutoIPQ",
}



for sim_set in sim_sets:

    suffix = ""
    if(sim_set == "all"):
        simulators = simulators_all
        suffix = "_all"
    elif(sim_set == "physical"):
        simulators = simulators_physical
        suffix = "_physical"
    elif(sim_set == "sbi"):
        simulators = simulators_sbi
        suffix = "_sbi"
    elif(sim_set == "mdgs"):
        simulators = simulators_mdgs
        suffix = "_mdgs"
    elif(sim_set == "lr"):
        simulators = simulators_lr
        suffix = "_lr"
    else:
        raise NotImplementedError








    # CSV version

    if("csv" in table_formats):

        acc_mat = np.zeros((len(methods), len(simulators)))
        acc_std_mat = np.zeros((len(methods), len(simulators)))
        acc_sig_mat = np.zeros((len(methods), len(simulators)))
        acc_sig_full = np.zeros((len(methods), len(simulators), 200)) # Hardcoded for now

        f1_mat = np.zeros((len(methods), len(simulators)))
        f1_std_mat = np.zeros((len(methods), len(simulators)))
        f1_sig_mat = np.zeros((len(methods), len(simulators)))
        f1_sig_full = np.zeros((len(methods), len(simulators), 200))


        # Swapped i and j because results are stored per simulator first, but table should have methods as rows
        j = 0
        for simulator in simulators:
            sim_path = base_path + simulator + "/"

            i = 0
            # Method level; rows
            for method in methods:
                

                method_path = sim_path + method + "/"
                budget_path = method_path + str(budget) + "/"
                sample_path = budget_path + sample_type + "/"

                # Dataset property, but this is easiest way of doing it
                num_in = 0
                num_out = 0

                acc = np.zeros(len(os.listdir(sample_path)))
                #prec = []
                #rec = []
                f1 = np.zeros(len(os.listdir(sample_path)))
                inst_i = 0

                for d in os.listdir(sample_path):
                    if(not(d.endswith("times.csv")) and not(d.endswith("time.csv"))): # Don't try to read running/training time files
                        try:
                            df = pd.read_csv(sample_path + d)
                            true = df["true_class"].values
                            pred = df["pred_class"].values

                            acc[inst_i] = metric_func(true, pred)
                            f1[inst_i] = f1_score(true, pred) # TODO: now not combining so just using acc for all; could delete f1 but not a big deal

                            inst_i += 1
                        except:
                            print("Warning: failed to load stuff for", sample_path+d)
                            pass

                
                acc = acc[:inst_i] # For failed stuff
                f1 = f1[:inst_i]

                acc_sig_full[i, j, :len(acc)] = acc
                f1_sig_full[i, j, :len(acc)] = f1

                acc_mean = np.round(np.mean(acc), decimals=table_decimals)
                acc_std = np.round(np.std(acc), decimals=table_decimals)
                acc_mat[i,j] = acc_mean
                acc_std_mat[i,j] = acc_std

                f1_mean = np.round(np.mean(f1), decimals=table_decimals)
                f1_std = np.round(np.std(f1), decimals=table_decimals)
                f1_mat[i,j] = f1_mean
                f1_std_mat[i,j] = f1_std

                i += 1

            j += 1


        # Significance testing; need access to other methods' results, so inefficient (probably good way to be efficient, but not worth)
        j = 0
        for simulator in simulators:
            sim_path = base_path + simulator + "/"

            i = 0
            # Method level; rows
            for method in methods:
                
                # Compare to all other methods
                num_outperformed_acc = 0
                num_outperformed_f1 = 0

                i2 = 0
                for method2 in methods:
                    if(not(i == i2)):
                        pval = wilcoxon(acc_sig_full[i, j, :], acc_sig_full[i2, j, :], alternative="greater").pvalue
                        if(pval < significance_level):
                            num_outperformed_acc += 1

                        pval = wilcoxon(f1_sig_full[i, j, :], f1_sig_full[i2, j, :], alternative="greater").pvalue
                        if(pval < significance_level):
                            num_outperformed_f1 += 1

                    i2 += 1
                
                acc_sig_mat[i,j] = num_outperformed_acc
                f1_sig_mat[i,j] = num_outperformed_f1


                i += 1

            j += 1


        # Convoluted to deal with ties

        best_method_acc = []
        for d_i in range(acc_sig_mat.shape[1]):
            vec = acc_sig_mat[:, d_i]
            best_list = np.argwhere(vec == np.amax(vec)).flatten().tolist()
            best_method_acc.append(best_list)

        best_method_f1 = []
        for d_i in range(f1_sig_mat.shape[1]):
            vec = f1_sig_mat[:, d_i]
            best_list = np.argwhere(vec == np.amax(vec)).flatten().tolist()
            best_method_f1.append(best_list)

        #best_method_acc = np.argmax(acc_sig_mat, axis=0)
        #best_method_f1 = np.argmax(f1_sig_mat, axis=0)


        # Creating file
        s = "method,"
        for sim in simulators:
            #s = s + "acc_" + sim + "," + "f1_" + sim + ","
            s = s + sim + ","
        s = s[:-1] + "\n"

        i = 0

        for method in methods:
            j = 0
            row = nice_names_methods[method] + ","
            for sim in simulators:
                if(i in best_method_acc[j]):
                    row = row + "$\mathbf{" + str(acc_mat[i,j]) + " \pm " + str(acc_std_mat[i,j]) + "}$," # accuracy
                else:
                    row = row + "$" + str(acc_mat[i,j]) + " \pm " + str(acc_std_mat[i,j]) + "$," # accuracy

                #if(i in best_method_f1[j]):
                #    row = row + "$\mathbf{" + str(f1_mat[i,j]) + " \pm " + str(f1_std_mat[i,j]) + "}$," # f1
                #else:
                #    row = row + "$" + str(f1_mat[i,j]) + " \pm " + str(f1_std_mat[i,j]) + "$," # f1

                j += 1
            row = row[:-1] + "\n"
            s = s + row

            i += 1

        with open("results/" + table_name + suffix + "_" + suffix2 + ".csv", 'w') as fp:
            fp.write(s)



    # Direct LaTeX version




    if("latex" in table_formats):



        tup = []
        for sim in simulators:
            for met in metric_names:
                tup.append((nice_names_sim[sim], met))

        cols_index = pd.MultiIndex.from_tuples(tup)
        df_table = pd.DataFrame(columns=cols_index, index=nice_names_methods.keys())


        # Simulator level; super column
        for simulator in simulators:
            sim_path = base_path + simulator + "/"


            # Measure level; sub column
            for met in metric_names:


                method_val_list = []
                # Method level; rows
                for method in methods:
                    

                    method_path = sim_path + method + "/"
                    budget_path = method_path + str(budget) + "/"
                    sample_path = budget_path + sample_type + "/"

                    # Dataset property, but this is easiest way of doing it
                    num_in = 0
                    num_out = 0

                    acc = []
                    prec = []
                    rec = []
                    f1 = []

                    for d in os.listdir(sample_path):
                        try:
                            df = pd.read_csv(sample_path + d)
                            true = df["true_class"].values
                            pred = df["pred_class"].values

                            num_in += len(true[true == 1])
                            num_out += len(true[true == 0])

                            acc.append(accuracy_score(true, pred))
                            prec.append(precision_score(true, pred, zero_division=0))
                            rec.append(recall_score(true, pred))
                            f1.append(f1_score(true, pred)) 
                        except:
                            pass

                    acc_mean = np.round(np.mean(acc), decimals=table_decimals)
                    acc_std = np.round(np.std(acc), decimals=table_decimals)
                    prec_mean = np.round(np.mean(prec), decimals=table_decimals)
                    prec_std = np.round(np.std(prec), decimals=table_decimals)
                    rec_mean = np.round(np.mean(rec), decimals=table_decimals)
                    rec_std = np.round(np.std(rec), decimals=table_decimals)
                    f1_mean = np.round(np.mean(f1), decimals=table_decimals)
                    f1_std = np.round(np.std(f1), decimals=table_decimals)
                    

                    if(met == "Accuracy"):
                        cell = "$" + str(acc_mean) + "\pm" + str(acc_std) + "$"
                    elif(met in ["F1", "F1-measure"]):
                        cell = "$" + str(f1_mean) + "\pm" + str(f1_std) + "$"
                    method_val_list.append(cell)

                id = (nice_names_sim[simulator], met)
                df_table[id] = method_val_list






        df_table.to_latex("results/result_table" + suffix + ".tex", 
                        caption="Numerical results for a budget of $20\,000$ iterations.", 
                        label="tab:results")
        

print("Terminated successfully.")