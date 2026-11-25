import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import datetime
import json
import pickle
from random import shuffle
import copy

from sklearn.model_selection import train_test_split

from .utils import normalise_and_visualise



def load_gbov(gbov_path, margin=0.001, oldest_acceptable_year=2017, gbov_aux_path="", verbose=False):

    # Process GBOV into neat dataset

    if(gbov_aux_path != ""):
        aux_data = pd.read_csv(gbov_aux_path)
        aux_data = aux_data[['plotID', 'latitude', 'longitude']]
        aux_plots = {}
        for i,r in aux_data.iterrows():
            aux_plots[r['plotID']] = (r['longitude'], r['latitude'])

    station_polys = {}

    data = pd.DataFrame([], columns=['station', 'plotID', 'year', 'month', 'day', 'IGBP_class',
                            'LAI_miller_over', 'LAI_miller_under', 'LAI_warren_over', 'LAI_warren_under'])

    corrupt_files = []

    for f in os.listdir(gbov_path):
        if(f.split(".")[-1] == "csv"):
            fp = gbov_path + f
            df = pd.read_csv(fp,sep=';')
            for i,r in df.iterrows():
                try:
                    lat = float(r['Lat_IS'])
                    lon = float(r['Lon_IS'])
                    time = r['TIME_IS']
                    year = int(time[0:4])
                    month = int(time[4:6])
                    day = int(time[6:8])

                    station = r['Site']
                    plot_id = r['PLOT_ID']

                    IGBP_class = r['IGBP_class'] # Type of vegetation, useful for ecosystem filtering

                    # Hacky way of saving coordinates. Labeling is inconsistent; try to
                    # load from external file if invalid (-998, -998).
                    if(plot_id not in station_polys):
                        if(lat < 500 or lon < 500):
                            if(plot_id in aux_plots):
                                lon = aux_plots[plot_id][0]
                                lat = aux_plots[plot_id][1]
                                poly = [lon-margin, lat+margin, lon+margin, lat-margin]
                                station_polys[plot_id] = poly
                            else:
                                if(verbose):
                                    print("Unable to find coordinates for plot " + plot_id)
                        else:
                            poly = [lon-margin, lat+margin, lon+margin, lat-margin]
                            station_polys[plot_id] = poly
                    
                    # Up is "overstory", ie canopy
                    # Down is "understory", which is under canopy but over floor
                    LAI_miller_up = float(r['LAI_Miller_up'])
                    LAI_miller_down = float(r['LAI_Miller_down'])
                    LAI_warren_up = float(r['LAI_Warren_up'])
                    LAI_warren_down = float(r['LAI_Warren_down'])

                    entry = [station, plot_id, year, month, day, IGBP_class,
                            LAI_miller_up, LAI_miller_down, LAI_warren_up, LAI_warren_down]
                    entry = [np.nan if e == -999.00 else e for e in entry] # nans > infeasible value

                    data.loc[len(data.index)] = entry

                except KeyError:
                    if(f not in corrupt_files):
                        if(verbose):
                            print("Malformed file: " + f)
                        corrupt_files.append(f)

    data_recent = data[data['year'] >= oldest_acceptable_year]

    return(data_recent, station_polys)


def gbov_lon_split(data, station_polys, lon_split=37, verbose=False):

    north_stations = [k for k,v in station_polys.items() if v[1] > lon_split]
    south_stations = [k for k,v in station_polys.items() if v[1] < lon_split]

    data_north = data.loc[data['plotID'].isin(north_stations)]
    data_south = data.loc[data['plotID'].isin(south_stations)]

    if(verbose):
        print("Number of north instances: " + str(len(data_north)))
        print("Number of south instances: " + str(len(data_south)))
        print("\n\n")

    if(verbose):
        print("Number of north stations: " + str(len(north_stations)))
        print("Number of south stations: " + str(len(south_stations)))
        print("\n\n")

    return(data_north, data_south)


def gbov_eco_split(data, eco_a, eco_b, verbose=False):

    data_eco_a = data.loc[data['IGBP_class'].isin(eco_a)]
    data_eco_b = data.loc[data['IGBP_class'].isin(eco_b)]

    if(verbose):
        print("Number of ecosystem A instances: " + str(len(data_eco_a)))
        print("Number of ecosystem B instances: " + str(len(data_eco_b)))
        print("\n\n")

    return(data_eco_a, data_eco_b)


def gbov_to_matrices(data, save_path="", ecosystem="",
                        LAI_version="LAI_miller_under", include_cloudy=False, verbose=False):


    # Y1: input parameters
    # X1: simulated reflectance (converted to MS)
    # X2: real reflectance (MS)
    # Y2: predicted parameters

    # Set up true matrices

    X2 = np.zeros((len(data), 11+1))

    Y2_true = np.zeros((len(data), 1))

    meta = {}

    # Load cloud removed list, keep empty if not filtered (not technically correct but convenient)

    cloudy_list = []
    # Commented because we don't DL cloudy stuff in the first place
    #if(not(include_cloudy)):
    #    with open(save_path + "cloud_removed_instances.txt", 'r') as fp:
    #        cloudy_list = fp.read().split("\n")

    # Fill matrices

    num_nan = 0
    num_cloud = 0
    num_path = 0

    i = 0
    for row_num, r in data.iterrows():
        lai = r[LAI_version]
        key = r['plotID'] + "_" + str(r['year']) + "_" + str(r['month']) + "_" + str(r['day'])
        if(not(np.isnan(lai)) and not(key in cloudy_list) and
                os.path.exists(save_path + key + ".npy")): 
            refl = np.load(save_path + key + ".npy")
            # Use centre pixel here, but not strictly necessary, later will load full image anyway
            centre_i = int(refl.shape[0] / 2)
            centre_j = int(refl.shape[1] / 2)
            refl = refl[centre_i, centre_j, :]
            refl = np.insert(refl, 0, i) # Insert index to later be able to filter meta

            with open(save_path + key + "_meta.json", 'r') as fp:
                meta_local = json.load(fp)
            #meta_local = json.load(save_path + key + "_meta.json")
            meta_local['index'] = i
            meta_local['key'] = key
            meta_local['image_path'] = save_path + key + ".npy"
            meta[i] = meta_local

            X2[i,:] = refl
            Y2_true[i,0] = lai

            i += 1
        else:
            if(np.isnan(lai)):
                num_nan += 1
            if(key in cloudy_list):
                num_cloud += 1
            if(not(os.path.exists(save_path + key + ".npy"))):
                num_path += 1

    if(verbose):    
        print("Number of NaN LAI labels: " + str(num_nan))
        print("Number of cloudy instances: " + str(num_cloud))
        print("Number of GBOV observations without satellite data: " + str(num_path))

    # Drop unused rows
    if(verbose):
        print("Rows before cleanup: " + str(len(X2)))
    X2 = X2[:i,:]
    Y2_true = Y2_true[:i,0]
    if(verbose):
        print("Rows after cleanup: " + str(len(X2)))
        print("\n")

    return(X2, Y2_true, meta)


def meta_train_test_split(X, Y, meta, test_size=0.2, random_state=None):
    # Split data into train/test sets, also split meta dict indexed by the row index of the new train/test matrices

    if(len(Y) == 0):
        print("Warning: empty dataset in split")
        return(None, None, None, None, None, None)

    if(test_size > 0.0):
        if(not(random_state)):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    else:
        X_train = X
        Y_train = Y
        meta_train = meta
        X_test = None
        Y_test = None
        meta_test = None

    meta_train = {}
    meta_test = {}
    train_indices = X_train[:,0].tolist()
    if(test_size > 0.0):
        test_indices = X_test[:,0].tolist()
    else:
        test_indices = []

    for k,v in meta.items():
        if(k in train_indices):
            new_index = train_indices.index(k)
            meta_train[new_index] = v
        elif(k in test_indices):
            new_index = test_indices.index(k)
            meta_test[new_index] = v
        else:
            print("Should not happen")

    # Remove index column that is now no longer needed
    X_train = X_train[:, 1:]
    if(test_size > 0.0):
        X_test = X_test[:, 1:]

    return(X_train, X_test, Y_train, Y_test, meta_train, meta_test)


def keys_to_numpy(keys, indices, path, max_img_size=(55, 55), num_features=27, default_window=(5, 5),
                  include_meta=False, s2_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], visualise_imgs=False, force_vis=False):
    
    if(visualise_imgs):
        if(len(indices) > 10):
            if(not(force_vis)):
                visualise_imgs = False
                print("WARNING: too many indices for visualisation, defaulting to disabling visualisation. To prevent this behaviour, please set force_vis=True.")

    # Convert specification from keys + indices for this set (train/val/test) into actual loaded data

    meta = {}
    max_instances = np.product(max_img_size) * len(indices)

        
    # Either load every pixel as an instance, or just the centre pixel from one image as one instance
    sat = np.zeros((max_instances, len(s2_indices)))
    lai = np.zeros((max_instances))


    # Iterate over tiles
    total_instances = 0
    for ind in indices:
        key = keys[ind]
        img = np.load(path + key + "_sat.npy")
        # Remove extra feature bands for sat, for sat_extra, just keep as needed
        img = img[:, :, s2_indices]
        mask = np.load(path + key + "_mask.npy")
        img_lai = np.load(path + key + "_lai.npy")

        # Visualisation
        if(visualise_imgs):
            normalise_and_visualise(img, title="Satellite image")
            plt.imshow(np.clip(img_lai[:,:], a_min=0, a_max=10))
            plt.title("LAI measurements")
            plt.colorbar()
            plt.show()
            plt.imshow(mask)
            plt.title("Mask")
            plt.colorbar()
            plt.show()

        # TODO: hacky workaround, not flipped (checked) but sizes are, should ideally be fixed properly
        # Doing this because the mask (valid LAI measurement mask) and LAI/S2 images are not perfectly aligned
        #min_width = min([48, mask.shape[1], img.shape[1]])
        #min_height = min([48, mask.shape[0], img.shape[0]])
        #img = img[:min_height, :min_width]
        #mask = mask[:min_height, :min_width]
        #img_lai = img_lai[:min_height, :min_width]

        if(include_meta):
            with open(path + key + "_meta.json", 'r') as fp:
                m = json.load(fp)
                m['image_path'] = path + key + "_sat.npy"
                m['key_tile'] = key
                m['tts'] = m['sun_zenith']
                m['tto'] = m['view_zenith']
                m['psi'] = abs(m['sun_azimuth'] - m['view_azimuth'])

        # Making this inefficient now, to support loading image subsets

        # Just take the centre pixel

        i = int(img.shape[0] / 2)
        j = int(img.shape[1] / 2)
        i2 = int(mask.shape[0] / 2) # Mask/LAI and sat shapes not always aligned
        j2 = int(mask.shape[1] / 2)

        # Only add valid pixels
        if(mask[i2,j2] == 0.0): # Values of 1 means there is a problem (values of, e.g., -9998.0 for LAI)
            if(include_meta):
                m['x'] = j
                m['y'] = i
                m['key'] = key + "_" + str(i) + "_" + str(j)
                meta[total_instances] = m
            sat[total_instances, :] = img[i, j]
            lai[total_instances] = img_lai[i2, j2]
            total_instances += 1



    sat = sat[:total_instances]
    lai = lai[:total_instances]

    if(include_meta):
        return(sat, lai, meta)
    else:
        return(sat, lai)


def keys_to_numpy_old(keys, indices, path, file_type, max_img_size=(55, 55), num_features=27, default_window=(5, 5),
                  include_meta=False, centre_only=False, s2_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):

    meta = {}
    max_instances = np.product(max_img_size) * len(indices)

    if(file_type == "lai"):
        extension = "_lai.npy"
        y_dim = 1
    elif(file_type == "sat" or file_type == "sat_only"):
        extension = "_sat.npy"
        y_dim = num_features
    else:
        print("Invalid file type")

    if(centre_only):
        res = np.zeros((max_instances, y_dim))
    else:
        res = np.zeros((max_instances, default_window[0]*2+1, default_window[1]*2+1, y_dim))

    # Iterate over tiles
    total_instances = 0
    for ind in indices:
        key = keys[ind]
        img = np.load(path + key + extension)
        mask = np.load(path + key + "_mask.npy")

        # TODO: hacky workaround, not flipped (checked) but sizes are, should ideally be fixed properly
        min_width = min([48, mask.shape[1], img.shape[1]])
        min_height = min([48, mask.shape[0], img.shape[0]])
        img = img[:min_height, :min_width]
        mask = mask[:min_height, :min_width]

        if(include_meta):
            with open(path + key + "_meta.json", 'r') as fp:
                m = json.load(fp)
                m['image_path'] = path + key + "_sat.npy"
                m['key_tile'] = key

        # Making this inefficient now, to support loading image subsets

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if(mask[i, j] < 1):
                    if(include_meta):
                        m_local = copy.deepcopy(m)
                        m_local['x'] = j
                        m_local['y'] = i
                        m_local['key'] = key + "_" + str(i) + "_" + str(j)
                        meta[total_instances] = m_local
                    if(centre_only):
                        res[total_instances, :] = img[i, j]
                        total_instances += 1
                    else:
                        left = j - default_window[1]
                        right = j + default_window[1]
                        top = i - default_window[0]
                        bottom = i + default_window[0]

                        if(left >= 0 and right <= (img.shape[1]-1) and top >= 0 and bottom <= (img.shape[0]-1)):
                            if(file_type == 'lai'):
                                res[total_instances, :, :, 0] = img[top:bottom+1, left:right+1]
                            else:
                                res[total_instances, :, :, :] = img[top:bottom+1, left:right+1, :]
                            total_instances += 1


    res = res[:total_instances]

    if(file_type == "lai"):
        res = res[:, 0]

    if(include_meta):
        return(res, meta)
    
    return(res)


def get_keys_and_indices(dataset_path, train_val_test, num_tiles=-1, save=False, load=False, experiments_path="experiments/data_split_indices/"):
    # Return a list of keys (basically filenames), a list of train indices (for key list), val and test indices

    # This is to use the same instances every time
    if(save):
        path_sep = experiments_path.split("/")
        s = ""
        for i in range(0, len(path_sep)-1):
            s = s + path_sep[i] + "/"
            if(not(os.path.exists(s))):
                os.mkdir(s)


    # Paths for the stored variables for loading
    p1 = experiments_path + str(num_tiles) + "_keys.pkl"
    p2 = experiments_path + str(num_tiles) + "_train.pkl"
    p3 = experiments_path + str(num_tiles) + "_val.pkl"
    p4 = experiments_path + str(num_tiles) + "_test.pkl"

    # To ensure all runs use the same instances
    if(load):
        if(os.path.exists(p1) and os.path.exists(p2) and os.path.exists(p3) and os.path.exists(p4)):
            # Load saved stuff
            with open(p1, 'rb') as fp:
                keys = pickle.load(fp)
            with open(p2, 'rb') as fp:
                train_indices = pickle.load(fp)
            with open(p3, 'rb') as fp:
                val_indices = pickle.load(fp)
            with open(p4, 'rb') as fp:
                test_indices = pickle.load(fp)

            return(keys, train_indices, val_indices, test_indices)


    
    # Get a list of keys

    # TODO: efficient implementation using downloads.csv and pandas downloaded column
    keys = []
    for d in os.listdir(dataset_path):
        if(d.endswith("_lai.npy")):
            keys.append(d[:-8])

    # To avoid biases
    shuffle(keys)
            
    # For loading a smaller subset of the dataset (one key is one image with multiple versions)
    if(num_tiles > 0):
        keys = keys[:num_tiles]

    # Randomly assign keys to train/val/test sets using vector of 0-1 probabilities
    probs = np.random.random(size=len(keys))

    train_indices = np.where(probs < train_val_test[0])[0]
    val_indices = np.where((probs > train_val_test[0]) & (probs < train_val_test[1]))[0]
    test_indices = np.where((probs > train_val_test[1]) & (probs < train_val_test[2]))[0]

    # Storing indices for later use if saving
    if(save and not(os.path.exists(p1) and os.path.exists(p2) and os.path.exists(p3) and os.path.exists(p4))):
        with open(p1, 'wb') as fp:
            pickle.dump(keys, fp)
        with open(p2, 'wb') as fp:
            pickle.dump(train_indices, fp)
        with open(p3, 'wb') as fp:
            pickle.dump(val_indices, fp)
        with open(p4, 'wb') as fp:
            pickle.dump(test_indices, fp)

    return(keys, train_indices, val_indices, test_indices)


def remove_invalid(X, Y, M, y_upper_threshold=10, y_lower_threshold=0, 
                   x_feature_index=11, nonzero_threshold=1.0, num_nonzero_threshold=1):
    oldlen = len(Y)

    # Filter by Y values
    f = Y <= y_upper_threshold
    X = X[f]
    Y = Y[f]

    f = Y >= y_lower_threshold
    X = X[f]
    Y = Y[f]

    # Filter by X values
    # Don't do this for now, mainly for extra features
    #bool_matrix = X > nonzero_threshold
    #f = np.sum(bool_matrix[:, x_feature_index:], axis=1) >= num_nonzero_threshold
    #X = X[f]
    #Y = Y[f]

    num_removed = oldlen - len(Y)

    # TODO: verify that it's OK to not remove instances from M
    # It's probably not, because indices get shuffled around

    return(X, Y, M, num_removed)



