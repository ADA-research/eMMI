import numpy as np
import pandas as pd


#simulators = ["lr5d", "lr2d", "lr10d", "lr20d", "lr50d", "prosail", "prosail_2d", "dpr_tp", "dpr_lorenz63", "sbi_gm",
#              "sbi_tm", "mdgs2", "mdgs10", "mdgs20", "mdgs50"]

simulators = ["lr5d", "lr2d", "lr10d", "lr20d", "lr50d"]

splits = ["val", "test"]
noise_level = 0.1

base_path = "/scratch/arp/AutoIPQ/data/experiments/"


overwrite_optima = False

# Optima

if(overwrite_optima):

    for sim in simulators:
        currdir = base_path + sim + "/"
        for split in splits:
            res_path = currdir + split + "/instances_opt_noise" + str(noise_level) + ".csv"
            backup_path = currdir + split + "/instances_opt_noise" + str(noise_level) + "_backup.csv"

            
            # Load csv and save backup
            df = pd.read_csv(res_path)
            df.to_csv(backup_path, index=False)

            # Remove duplicate instances, remove malformed rows, and sort by instances
            df = df.drop_duplicates(subset=["instance"], keep="last")
            df = df.replace('', np.nan)
            df = df.dropna()
            df = df.sort_values(by=["instance"])

            # Save new
            df.to_csv(res_path, index=False)


# Validation points

for sim in simulators:
    currdir = base_path + sim + "/"
    for split in splits:
        new_path = currdir + split + "/clf_samples_uniform/"
        for i in range(200):
            try:
                df_X = pd.read_csv(new_path + "p" + str(i) + "_X.csv")
                df_Y = pd.read_csv(new_path + "p" + str(i) + "_Y.csv")
                if(len(df_X) != len(df_Y)):
                    print("Inconsistent lengths for X and Y in instance" + new_path + str(i) + "_*.csv")
            except:
                print("Couldn't open " + str(new_path + str(i) + "_X.csv") + ", probably missing")

        print("\n")

