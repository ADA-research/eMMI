import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import copy

#import prosail


# Some defaults that are nice to have here

default_params = {
    'cab':32.81,
    'cw':0.0129,
    'cm':0.008,
    'lai':3.88,
    'n':1,
    'car':8.51,
    'cbrown':0.18,
    'typelidf':2,
    'lidfa':-1,
    'lidfa2':45,
    'lidfb':-0.15,
    'hspot':0.01,
    'tts':30,
    'tto':10,
    'psi':0,
    'prospect_version':"D",
    'factor':'SDR', 
    'rsoil':1,
    'psoil':0.5, 
    'ant':0.0,
    'prot':0.0,
    'cbc':0.0,
    'alpha':40.0,
}


default_param_ranges = {
    'cab':['normal', 'float', 0.3, 106.72, 32.81, 18.87],
    'cw':['normal', 'float', 0.0043, 0.07, 0.0129, 0.0073],
    'cm':['normal', 'float', 0.0008, 0.0331, 0.0077, 0.0035],
    'lai':['normal', 'float', 0.001, 10, 3.88, 1.98],
    'n':['uniform', 'float', 0.8, 3.2],
    'car':['normal', 'float', 0.04, 25.30, 8.51, 3.92],
    'cbrown':['normal', 'float', 0, 1, 0.18, 0.37],
    'typelidf':['selection', 'categorical', [1, 2]],
    'lidfa':['uniform', 'float', -1, 1],
    'lidfa2':['uniform', 'int', 0, 90], # For typelidf=2
    'lidfb':['uniform', 'float', -1, 1],
    'hspot':['uniform', 'float', -1, 1],
    'tts':['uniform', 'int', 0, 90],
    'tto':['uniform', 'int', 0, 90],
    'psi':['uniform', 'int', 0, 360],
    'prospect_version':['selection', 'categorical', ["5", "D"]],
    'factor':['selection', 'categorical', ['SDR', 'BHR', 'DHR', 'HDR']], # Should probably just be set correctly, not optimised
    'rsoil':['uniform', 'float', 0.5, 1.5],
    'psoil':['uniform', 'float', 0, 1],
    'ant':['uniform', 'float', 0, 1],
    'prot':['uniform', 'float', 0, 1],
    'cbc':['uniform', 'float', 0, 1],
    'alpha':['uniform', 'float', 0, 180],
}

def get_default_config():
    return(copy.deepcopy(default_params))


def get_default_ranges():
    return(copy.deepcopy(default_param_ranges))


def config_to_vec(config, parameters):
    # Convert prosail config dictionary to a numerical vector
    res = np.zeros(len(parameters))
    i = 0
    for p in parameters:
        res[i] = config[p]
        i += 1

    return(res)


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

    #params["window"] = cfg.PARAM.REG.WINDOW
    #params["gamma"] = cfg.PARAM.REG.GAMMA
    #lambdas = {
    #    'lai':cfg.PARAM.REG.LAMBDAS.LAI,
    #    'cab':cfg.PARAM.REG.LAMBDAS.CAB,
    #    'cw':cfg.PARAM.REG.LAMBDAS.CW,
    #    'cm':cfg.PARAM.REG.LAMBDAS.CM,
    #    'lidfa2':cfg.PARAM.REG.LAMBDAS.LIDFA2,
    #    'psoil':cfg.PARAM.REG.LAMBDAS.PSOIL,
    #    'rsoil':cfg.PARAM.REG.LAMBDAS.RSOIL,
    #}
    #params["lambdas"] = lambdas

    #params["constraint_penalty_strength"] = cfg.PARAM.CONST.PENALTY_STRENGTH
    #params["use_corr_dict"] = cfg.PARAM.CONST.COMPUTE_CORR_MATRIX
    #params["corr_dict"] = cfg.PARAM.CONST.CORR_DICT

    #params["discriminator_iterations"] = cfg.PARAM.DISC.DISCRIMINATOR_ITERATIONS
    #params["discriminator_features"] = cfg.PARAM.DISC.EXTRA_FEATURES
    #params["discriminator_feature_indices"] = cfg.PARAM.DISC.FEATURE_INDICES

    #params["is_iterations"] = cfg.PARAM.IS.IS_ITERATIONS
    params["is_weight"] = cfg.PARAM.IS.WEIGHT
    params["is_k"] = cfg.PARAM.IS.K
    #params["is_strata"] = cfg.PARAM.IS.NUM_STRATA

    #params["solution_set_size"] = cfg.PARAM.QD.SOLUTION_SET_SIZE
    #params["qd_weight"] = cfg.PARAM.QD.WEIGHT
    #pweight = {
    #    'lai':cfg.PARAM.QD.PWEIGHT.LAI,
    #    'cab':cfg.PARAM.QD.PWEIGHT.CAB,
    #    'cw':cfg.PARAM.QD.PWEIGHT.CW,
    #    'cm':cfg.PARAM.QD.PWEIGHT.CM,
    #    'psoil':cfg.PARAM.QD.PWEIGHT.PSOIL,
    #    'rsoil':cfg.PARAM.QD.PWEIGHT.RSOIL,
    #}
    #params["pweight"] = pweight
           
    params["algorithm"] = cfg.COND.ALGORITHM
    #params["use_spatial_regularisation"] = cfg.COND.SPATIAL_REGULARISATION
    #params["use_discriminator"] = cfg.COND.USE_DISCRIMINATOR
    #params["constrain"] = cfg.COND.CONSTRAIN
    #params["warm_start"] = cfg.COND.WARM_START
    #params["quality_diversity"] = cfg.COND.QUALITY_DIVERSITY
    params["use_iterated_search"] = cfg.COND.USE_ITERATED_SEARCH

    return(params)


def string_to_func(obj):
    if(obj == "absolute_error" or obj == "absolute_distance"):
        return(absolute_distance)
    else:
        print("WARNING: invalid objective specified")



# Some useful functions

def proportional_distance(true, pred):
    """Return the proportional distance between matched elements of the true and predicted vectors.

    Args:
        true (1D numpy array): the observed true spectrum
        pred (1D numpy array): the predicted spectrum
    """
    return(np.sum(np.absolute((true - pred) / true)))

def absolute_distance(true, pred):
    return(np.sum(np.absolute(true - pred)))

def weighted_distance(true, pred, weights=[41, 12, 10, 20, 2.5, 1.5, 20, 21, 20, 30, 46]):
    weights = weights / np.sum(weights)
    diff = np.absolute(true-pred)
    weighted = np.dot(diff, weights)
    return(weighted)

def SAM(s1, s2):
    # Using code from https://pysptools.sourceforge.io/_modules/pysptools/distance/dist.html#SAM
    try:
        s1_norm = math.sqrt(np.dot(s1, s1))
        s2_norm = math.sqrt(np.dot(s2, s2))
        sum_s1_s2 = np.dot(s1, s2)
        angle = math.acos(sum_s1_s2 / (s1_norm * s2_norm))
    except ValueError:
        return(0.0)
    return(angle)

def normalise_and_visualise(img,title="",rgb=[3,2,1],percentile_clip=True,save_fig=False,**kwargs):
    """Retrieve the RGB bands from an image, normalise using percentile clipping, and visualise the result.

    Args:
        img (3D numpy array): the input image to visualise
        title (str, optional): A title to pass to the visualisation. Defaults to "".
        rgb (list, optional): Band indices corresponding to the red, green and blue bands, respectively. Defaults to [3,2,1] (Sentinel-2).
        percentile_clip (bool, optional): Use percentile clipping to normalise values. Defaults to True.
        save_fig (bool, optional): if True, save the figure (must specify path in kwargs). Defaults to False.
    """
    new_img = np.zeros((img.shape[0],img.shape[1],3))
    new_img[:,:,0] = img[:,:,rgb[0]]
    new_img[:,:,1] = img[:,:,rgb[1]]
    new_img[:,:,2] = img[:,:,rgb[2]]
    
    if(percentile_clip):
        min_val = np.nanpercentile(new_img,1)
        max_val = np.nanpercentile(new_img,99)

        new_img = np.clip((new_img-min_val)/(max_val-min_val),0,1)
    
    plt.imshow(new_img,interpolation="nearest")
    plt.title(title)
    plt.axis('off')
    if(save_fig):
        plt.savefig(kwargs['path'],bbox_inches='tight')
    plt.show()


def remove_bands(img, bands):
    res = img.copy()
    res = np.delete(res, bands, axis=2)
    return(res)


def load_features(feature_set, meta):
    vec = np.zeros(len(feature_set))
    for i in range(0, len(feature_set)):
        vec[i] = meta[feature_set[i]]

    return(vec)


def load_spectral_responses(path, level="L2A", platform="S2A"):
    """Load spectral responses for a satellite. Used to weight hyperspectral bands when converting to multispectral data.

    Args:
        path (str): path to the spectral response file (can be downloaded online for the relevant satellite platform)
        level (str, optional): data processing level of the data product. Defaults to "L2A".
        platform (str, optional): the platform name of the satellite. Defaults to "S2A".
    """

    col_names = ["SR_WL", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

    df = pd.read_csv(path)
    df.columns = col_names

    if(level == "L2A"):
        bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    elif(level == "L1C"):
        bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    else:
        print("Invalid Sentinel-2 processing level: " + str(level))

    col_names = ["SR_WL"]
    for b in bands:
        col_names.append(b)
    
    df = df[col_names]

    return(df)

def df_index_func(df, i, col="SR_WL"):
    # A better programmer than me would somehow index by SR_WL to make it much easier
    return(df.index[df[col]==i].tolist()[0])


def hyper_to_S2(hyper, SR, hyper_wavelengths=-1, s2_bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]):
    """Use spectral response DataFrame to weight hyperspectral bands when converting to multispectral data.

    Args:
        hyper (1D numpy array): a high-dimensional 1D array containing a hyperspectral spectrum.
        SR (DataFrame): spectral response DataFrame
        hyper_wavelengths (list, optional): wavelengths of the hyper vector indices. Defaults to -1 (indices offset by 400).
        s2_bands (list, optional): names of the bands in the target satellite platform. Defaults to ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"].
    """

    if(hyper_wavelengths == -1):
        hyper_wavelengths = [b+400 for b in range(0,len(hyper))]

    S2 = np.zeros(len(s2_bands))

    SR_low_i = df_index_func(SR, hyper_wavelengths[0])
    SR_high_i = df_index_func(SR, hyper_wavelengths[-1]+1)

    SR_matrix = SR[SR_low_i:SR_high_i]
    SR_matrix = SR_matrix[s2_bands].values
    
    # Iterate over S2 bands
    for b in range(0,len(s2_bands)):

        # Subset hyper bands covered by the S2 band's range
        weights = SR_matrix[:,b]

        # Compute avg S2 band value as sum(subset * distance) / sum(distance)
        S2[b] = np.dot(weights, np.array(hyper)) / np.sum(weights)

    return(S2)


def hyper_to_S2_range(hyper, hyper_wavelengths=-1, s2_bands=[443,490,560,665,705,740,783,842,865,1610,2190], 
            s2_bandwidths=[21,66,36,31,15,15,20,106,21,91,175], aggr='mean'):

    if(hyper_wavelengths == -1):
        hyper_wavelengths = [b+400 for b in range(0,len(hyper))]

    S2 = np.zeros(len(s2_bands))
    
    # Iterate over S2 bands
    band_ranges = []
    for b in range(0,len(s2_bands)):
        # Compute wavelength ranges covered by the S2 bands

        band_min = s2_bands[b] - s2_bandwidths[b]
        band_max = s2_bands[b] + s2_bandwidths[b]

        # Subset hyper bands covered by the S2 band's range
        ind_min = hyper_wavelengths.index(band_min)
        ind_max = hyper_wavelengths.index(band_max)
        hyper_sub = hyper[ind_min:ind_max]

        if(aggr == "weighted"):
            # Compute abs distance to core wavelength to weight contribution
            dist_vec = np.abs(np.array(hyper_wavelengths[ind_min:ind_max]) - s2_bands[b])
            dist_multiplier_vec = (np.max(dist_vec) - dist_vec) / np.sum(dist_vec)

            # Compute avg S2 band value as sum(subset * distance) / sum(distance)
            S2[b] = np.dot(dist_multiplier_vec, np.array(hyper_sub))

        elif(aggr == "mean"):
            S2[b] = np.mean(hyper_sub)

        else:
            print("WARNING: invalid aggregation method: " + str(aggr))
            return(False)

    return(S2)

def hyper_to_S2_full(hyper, hyper_wavelengths=-1, s2_bands=[443,490,560,665,705,740,783,842,865,1610,2190]):

    if(hyper_wavelengths == -1):
        hyper_wavelengths = [b+400 for b in range(0,len(hyper))]

    S2 = np.zeros(len(s2_bands))
    
    # Iterate over S2 bands
    band_ranges = []
    for b in range(0,len(s2_bands)):
        # Compute wavelength ranges covered by the S2 bands

        if(b>0):
            band_min = int((s2_bands[b]+s2_bands[b-1])/2)
        else:
            band_min = s2_bands[b]
        if(b<len(s2_bands)-1):
            band_max = int((s2_bands[b]+s2_bands[b+1])/2)
        else:
            band_max = s2_bands[b]


        # Subset hyper bands covered by the S2 band's range
        ind_min = hyper_wavelengths.index(band_min)
        ind_max = hyper_wavelengths.index(band_max)
        hyper_sub = hyper[ind_min:ind_max]

        # Compute abs distance to core wavelength to weight contribution
        dist_vec = np.abs(np.array(hyper_wavelengths[ind_min:ind_max]) - s2_bands[b])
        dist_multiplier_vec = (np.max(dist_vec) - dist_vec) / np.sum(dist_vec)

        # Compute avg S2 band value as sum(subset * distance) / sum(distance)
        S2[b] = np.dot(dist_multiplier_vec, np.array(hyper_sub))

    return(S2)

def MS_to_selected_bands(X, selected_bands="S2"):
    if(selected_bands=="S2" or selected_bands=="Sentinel-2"):
        selected_bands = [0, 1, 2, 3, 7, 9, 10]
    elif(selected_bands=="LS" or selected_bands=="Landsat"):
        selected_bands = [0, 1, 2, 3, 4, 5, 6]

    # To keep dimensionality general
    if(len(X.shape) == 1):
        X = X.reshape((1, len(X)))
    res = X[:, [selected_bands]]
    return(res)

def mutate_random(param, param_ranges):
    """Generate a random new setting for a parameter, within pre-defined parameter ranges.

    Args:
        param (str): parameter to be updated
        param_ranges (dict): dictionary of {str:tuple} pairs, where the string is the name of the parameter, and the tuple contains the type of distribution, its ranges, and its parameterisation.
    """
    if(param_ranges[param][0] == 'normal'):
        val = np.random.normal(loc=param_ranges[param][4], scale=param_ranges[param][5])
        val = max(param_ranges[param][2], val)
        val = min(param_ranges[param][3], val)
    elif(param_ranges[param][0] == 'uniform'):
        if(param_ranges[param][1] == 'int'):
            val = np.random.randint(low=param_ranges[param][2], high=param_ranges[param][3])
        elif(param_ranges[param][1] == 'float'):
            val = np.random.uniform(low=param_ranges[param][2], high=param_ranges[param][3])
    elif(param_ranges[param][0] == 'selection'):
        # So sorry for this line of 'code'
        val = param_ranges[param][2][np.random.randint(low=0, high=len(param_ranges[param][2]))]

    return(val)


def mutate_proportional(param, param_val, param_ranges, prop_sd=0.05):
    """Mutate an existing parameter value with an intensity proportional to the current value.

    Args:
        param (str): parameter to mutate
        param_val (float/int): current value of the parameter
        param_ranges (dict): dictionary of {str:tuple} pairs, where the string is the name of the parameter, and the tuple contains the type of distribution, its ranges, and its parameterisation.
        prop_sd (float, optional): proportion of the parameter value to use as standard deviation for generating the next data point. Defaults to 0.05.
    """
    if(param_ranges[param][1] == 'float'):
        sd = prop_sd * ((param_ranges[param][3] - param_ranges[param][2]) / 2)
        val = np.random.normal(loc=param_val, scale=sd)
        val = max(param_ranges[param][2], val)
        val = min(param_ranges[param][3], val)
    elif(param_ranges[param][1] == 'int'):
        sd = prop_sd * (param_ranges[param][3] - param_ranges[param][2])
        sd = max(sd, 1) # Ugly but necessary
        val = int(np.random.normal(loc=param_val, scale=sd))
        val = max(param_ranges[param][2], val)
        val = min(param_ranges[param][3], val)
    elif(param_ranges[param][1] == 'categorical'):
        # So sorry for this line of 'code'
        val = param_ranges[param][2][np.random.randint(low=0, high=len(param_ranges[param][2]))]

    return(val)


def map_value(val, min_old, max_old, min_new, max_new):

    # Based on https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
    width_old = max_old - min_old
    width_new = max_new - min_new

    new_val = val - min_old / width_old
    new_val = min_new + (new_val * width_new)

    return(new_val)


def reduce_ranges(param_ranges, refl, lenience={'cw':0.2, 'cm':0.2, 'psoil':0.2}):
    """Use reflectance spectrum and domain knowledge to reduce the search space ranges.

    Args:
        param_ranges (dict): Dictionary containing the ranges of parameters and distribution information (e.g., ('normal', 'float', 0, 1, 0.5, 0.1))
        refl (numpy vector): Reflectance vector
        lenience (dict, optional): Proportion of predicted value the new ranges can be removed fromt he predicted value. Defaults to {'cw':0.2, 'cm':0.2, 'psoil':0.2}.
    """

    new_ranges = param_ranges.copy()

    # Precompute some indices
    # ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    ndvi = (refl[7] - refl[3]) / (refl[7] + refl[3])

    ndmi = (refl[7] - refl[9]) / (refl[7] + refl[9])
    ndwi = (refl[2] - refl[7]) / (refl[2] + refl[7])
    ndwi1 = (refl[2] - refl[9]) / (refl[2] + refl[9])
    ndwi2 = (refl[2] - refl[10]) / (refl[2] + refl[10])

    msi1 = refl[9] / refl[7]

    evi = 2.5 * (refl[7] - refl[3]) / ((refl[7] + 6 * refl[3] - 7.5 * refl[1]) + 1)
    slavi = refl[7] / (refl[3] + refl[10])
    srvi = refl[7] / refl[3] # ?
    vari = (refl[2] - refl[3]) / (refl[2] + refl[3] - refl[1])

    # Probably not the fastest, but we shouldn't iterate this function anyway
    for k,v in param_ranges.items():
        if(k == 'cw'):
            # https://www.mdpi.com/2072-4292/12/17/2708
            # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EA002063
            #ndmi_projected = map_value(ndmi, -1, 1, param_ranges['cw'][2], param_ranges['cw'][3])
            #range_new = (ndmi_projected - (lenience['cw']*ndmi_projected), ndmi_projected + (lenience['cw']*ndmi_projected))
            t1 = 0.0361 * math.exp(5.8654*ndvi)
            t2 = 0.2114 * math.exp(6.5283*ndwi1)
            t3 = 0.1243 * math.exp(5.2937*ndwi2)
            t4 = 18.796 * math.exp(-4.519*msi1)
            avg = (t1+t2+t3+t4)/4
            avg = t1
            range_new = (max(avg - lenience['cw']*avg, param_ranges['cw'][2]), min(avg + lenience['cw']*avg, param_ranges['cw'][3]))
            if(param_ranges['cw'][0] == 'normal'):
                new_entry = (param_ranges['cw'][0], param_ranges['cw'][1], range_new[0], range_new[1], param_ranges['cw'][4], param_ranges['cw'][5])
            else:
                new_entry = (param_ranges['cw'][0], param_ranges['cw'][1], range_new[0], range_new[1])
            new_ranges[k] = new_entry

        elif(k == 'cm'):
            # https://www.mdpi.com/2072-4292/11/16/1936
            t1 = -4.98*evi - 1.995
            t2 = -1.8506*slavi + 0.7080
            t3 = -0.9253*srvi + 0.7080
            t4 = -0.9253*vari - 0.2173
            avg = (t1+t2+t3+t4) / 4
            range_new = (max(avg - lenience['cm']*avg, param_ranges['cm'][2]), min(avg + lenience['cm']*avg, param_ranges['cm'][3]))
            if(param_ranges['cm'][0] == 'normal'):
                new_entry = (param_ranges['cm'][0], param_ranges['cm'][1], range_new[0], range_new[1], param_ranges['cm'][4], param_ranges['cm'][5])
            else:
                new_entry = (param_ranges['cm'][0], param_ranges['cm'][1], range_new[0], range_new[1])
            new_ranges[k] = new_entry

        elif(k == 'psoil'):
            # https://eos.com/make-an-analysis/ndwi/ , https://www.researchgate.net/figure/Sensitivity-of-NDWI-and-NDII-to-soil-moisture-a-and-b-and-leaf-water-content_fig5_233783774
            # for ranges
            avg = map_value(ndwi, -0.2, 0.3, param_ranges['psoil'][2], param_ranges['psoil'][3])
            range_new = (max(avg - lenience['psoil']*avg, param_ranges['psoil'][2]), min(avg + lenience['psoil']*avg, param_ranges['psoil'][3]))
            if(param_ranges['psoil'][0] == 'normal'):
                new_entry = (param_ranges['psoil'][0], param_ranges['psoil'][1], range_new[0], range_new[1], param_ranges['psoil'][4], param_ranges['psoil'][5])
            else:
                new_entry = (param_ranges['psoil'][0], param_ranges['psoil'][1], range_new[0], range_new[1])
            new_ranges[k] = new_entry        
            
        else:
            new_ranges[k] = v

    return(new_ranges)


def refl_to_indices(refl):
    indices = {}

    indices['ndvi'] = (refl[7] - refl[3]) / (refl[7] + refl[3])

    indices['ndmi'] = (refl[7] - refl[9]) / (refl[7] + refl[9])
    indices['ndwi'] = (refl[2] - refl[7]) / (refl[2] + refl[7])
    indices['ndwi1'] = (refl[2] - refl[9]) / (refl[2] + refl[9])
    indices['ndwi2'] = (refl[2] - refl[10]) / (refl[2] + refl[10])

    indices['msi1'] = refl[9] / refl[7]

    # Based on paper
    indices['evi'] = 2.5 * (refl[2] - refl[9]) / ((refl[2] + 6 * refl[9] - 7.5 * refl[9]) + 1)
    indices['slavi'] = refl[2] / (refl[9] + refl[9])
    indices['srvi'] = refl[9] / refl[2]
    indices['vari'] = (refl[2] - refl[9]) / (refl[2] + refl[9] - refl[2])

    return(indices)


def get_expected_values(refl, file="constraints/constraints_indices.csv"):

    constraints = pd.read_csv(file)

    ev = {}
    ev_counts = {}

    for v in constraints['target_variable'].values:
        ev[v] = 0
        ev_counts[v] = 0

    # Compute quantities (e.g., NDVI)
    indices = refl_to_indices(refl)

    for _, c in constraints.iterrows():

        # Per constraint, compute expected value target variable given source variable (cfg or index) and relationship
        target_variable = c['target_variable']
        source_variable = c['source_variable']
        formula = c['relationship']
        formula = formula.replace("x", "indices['" + str(source_variable) + "']").replace("y=", "")

        exp = eval(formula) # TODO: maths stuff without resorting to eval

        ev[target_variable] = ev[target_variable] + exp
        ev_counts[target_variable] = ev_counts[target_variable] + 1
        

    # Average 
    for k,v in ev.items():
        ev[k] = ev[k] / ev_counts[k]

    return(ev)


def custom_prosail(config):
    """Run prosail from the prosail library using the supplied configuration of parameters.

    Args:
        config (dict): Dictionary containing parameter_name:value pairs.
    """

    # Based on code from https://github.com/nunocesarsa/RTM_Inversion/blob/main/Jupyter/RTM%20Inversion/RTM_Noise_inversion_Optimized.ipynb

    #default parameters
    cab = config['cab']
    cw = config['cw']
    cm = config['cm']
    lai = config['lai']
    n = config['n']
    car = config['car']
    cbrown = config['cbrown']
    typelidf = config['typelidf']
    lidfa = config['lidfa']
    lidfb = config['lidfb']
    hspot = config['hspot']
    tts = config['tts']
    tto = config['tto']
    psi = config['psi']
    prospect_version = config['prospect_version']
    factor = config['factor']
    rsoil = config['rsoil']
    psoil = config['psoil']

    if(config['typelidf'] == 2):
        # If typelidf=2, uses only lidfa which has different meaning (angle of ellipsoid) and range (0-90)
        lidfa = config['lidfa2']

    rho_out = prosail.run_prosail(n=n,
                                    cab=cab, # Chlorophyll
                                    car=car, # Caroten
                                    cbrown=cbrown, # Brown pigment
                                    cw=cw, # Equivalent water thickness
                                    cm=cm, # Dry matter content
                                    lai=lai, # LAI
                                    lidfa=lidfa, # Leaf angle parameter (angle if typelidf=2, no idea but low if typelidf=1)
                                    hspot=hspot, # Hotspots
                                    tts=tts, # Solar zenith
                                    tto=tto, # Observer zenith
                                    psi=psi, # Relative azimuth
                                    typelidf=typelidf, # Method of computing leaf angle distribution
                                    lidfb=lidfb, # Extra leaf angle parameter if typelidf=1
                                    prospect_version=prospect_version, # Version of prospect, PRO seems broken
                                    factor=factor, # Reflectance factor, 1 of 4, directional or hemisphere for 2 directions
                                    rsoil=rsoil, # Brightness multiplier soil spectrum
                                    psoil=psoil, # Moisture (0=dry, 1=wet) parameter for soil
                                    ant=config['ant'], # Anthocyanin content (D/PRO only)
                                    #prot=config['prot'], # Protein content, PRO only
                                    #cbc=config['cbc'], # Carbon based constituents, PRO only
                                    alpha=config['alpha'] # Alpha angle for surface scattering
                                )
    return(rho_out)


def to_prosail_geom(meta):
    # Derive tts, tto, msi from sun_azimuth, observer_zenith etc

    tts = meta['sun_zenith'] # Solar zenith
    tto = meta['view_zenith'] # Solar zenith
    psi = abs(meta['sun_azimuth'] - meta['view_azimuth']) # Relative azimuth
    return(tts, tto, psi)




