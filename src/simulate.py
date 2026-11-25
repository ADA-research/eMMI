import numpy as np
import pandas as pd
import scipy.stats

#import prosail

from .utils import *
#from AutoRTML import AutoRTMLError



class LUTGenerator():
    def __init__(self, sim, free_parameters, out_parameters=None, sim_parameters=None, seed=None):
        # free_parameters: parameters that can vary for simulation
        # out_parameters: parameters returned as Y data
        # sim_parameters: parameters that are varied for simulation, but always known in practice and added as metadata (returned as M)
        self.sim = sim
        self.free_parameters = free_parameters

        if(seed is not None):
            np.random.seed(seed)


        if(out_parameters is not None):
            self.out_parameters = out_parameters
        else:
            self.out_parameters = self.free_parameters

        if(sim_parameters is not None):
            self.sim_parameters = sim_parameters
            self.free_parameters = free_parameters + sim_parameters
        else:
            self.sim_parameters = []

        sim_params = self.sim.get_config().keys()
        for p in self.free_parameters:
            assert p in sim_params
        for p in self.out_parameters:
            assert p in sim_params
        for p in self.sim_parameters:
            assert p in sim_params


    def generate(self, num_instances, out_parameter=None):

        # Initialise data
        if(out_parameter is None):
            # Default to returning all FREE parameters (but not all in config)
            Y = np.zeros((num_instances, len(self.out_parameters)))
        else:
            # Only output specified parameter
            assert out_parameter in self.sim.get_config().keys
            Y = np.zeros((num_instances))
        X = np.zeros((num_instances, self.sim.output_dim))
        if(len(self.sim_parameters) > 0):
            M = np.zeros((num_instances, len(self.sim_parameters)))
        else:
            M = np.ones((num_instances, 1)) * np.nan


        config = self.sim.get_config()
        param_ranges = self.sim.get_param_ranges()

        for i in range(0, num_instances):
            config_local = self.sample(config, param_ranges)

            sim_output = self.sim.simulate(config_local)
            X[i,:] = sim_output

            if(out_parameter is None):
                vec = np.zeros(len(self.out_parameters))
                for j in range(0, len(vec)):
                    vec[j] = config_local[self.out_parameters[j]]
                Y[i,:] = vec
            else:
                Y[i] = config_local[out_parameter]

            if(len(self.sim_parameters) > 0):
                for j in range(0, len(self.sim_parameters)):
                    M[i, j] = config_local[self.sim_parameters[j]]

        return(X, Y, M)


class RandomLUTGenerator(LUTGenerator):
    def __init__(self, sim, free_parameters, out_parameters=None, sim_parameters=None, **kwargs):
        super().__init__(sim, free_parameters, out_parameters=out_parameters, sim_parameters=sim_parameters, **kwargs)


    def sample(self, config, param_ranges):

        config_local = config.copy()

        for param in self.free_parameters:
            if(param_ranges[param][0] == 'normal'):
                val = min(max(param_ranges[param][2], np.random.normal(loc=param_ranges[param][4], scale=param_ranges[param][5])), param_ranges[param][3])
            elif(param_ranges[param][0] == 'uniform'):
                if(param_ranges[param][1] == 'int'):
                    val = np.random.randint(low=param_ranges[param][2], high=param_ranges[param][3])
                elif(param_ranges[param][1] == 'float'):
                    val = np.random.uniform(low=param_ranges[param][2], high=param_ranges[param][3])
            elif(param_ranges[param][0] == 'selection'):
                # So sorry for this line of 'code'
                val = param_ranges[param][2][np.random.randint(low=0, high=len(param_ranges[param][2]))] 
            config_local[param] = val

        return(config_local)



class LHSLUTGenerator(LUTGenerator):
    # TODO: this needs updating for M
    def __init__(self, sim, free_parameters, out_parameters=None, sim_parameters=None, **kwargs):
        super().__init__(sim, free_parameters, out_parameters=out_parameters, sim_parameters=sim_parameters, **kwargs)


    def generate(self, num_instances, out_parameter=None):

        # Initialise data
        if(out_parameter is None):
            # Default to returning all FREE parameters (but not all in config)
            Y = np.zeros((num_instances, len(self.out_parameters)))
        else:
            # Only output specified parameter
            assert out_parameter in self.sim.get_config().keys
            Y = np.zeros((num_instances))
        X = np.zeros((num_instances, self.sim.output_dim))


        config_local = self.sim.get_config()
        param_ranges = self.sim.get_param_ranges()

        sampler = scipy.stats.qmc.LatinHypercube(d=len(self.free_parameters))
        sample = sampler.random(n=num_instances)

        for i in range(0, num_instances):
            for j in range(0, len(self.free_parameters)):
                param = self.free_parameters[j]
                # Scale from 0-1 to min-max range, e.g., 0.5 * (10-8=2) + 8 = 9 for range 8-10
                val = sample[i,j] * (param_ranges[param][3] - param_ranges[param][2]) + param_ranges[param][2]
                config_local[param] = val

            sim_output = self.sim.simulate(config_local)
            X[i,:] = sim_output

            if(out_parameter is None):
                vec = np.zeros(len(self.out_parameters))
                for j in range(0, len(vec)):
                    vec[j] = config_local[self.out_parameters[j]]
                Y[i,:] = vec
            else:
                Y[i] = config_local[out_parameter]


        return(X, Y)




class UniformLUTGenerator(LUTGenerator):
    def __init__(self, sim, free_parameters, out_parameters=None, sim_parameters=None, **kwargs):
        super().__init__(sim, free_parameters, out_parameters=out_parameters, sim_parameters=sim_parameters, **kwargs)


    def sample(self, config, param_ranges):

        config_local = config.copy()

        for param in self.free_parameters:
            if(param_ranges[param][0] == 'uniform' or param_ranges[param][0] == 'normal'):
                if(param_ranges[param][1] == 'int'):
                    val = np.random.randint(low=param_ranges[param][2], high=param_ranges[param][3])
                elif(param_ranges[param][1] == 'float'):
                    val = np.random.uniform(low=param_ranges[param][2], high=param_ranges[param][3])
            elif(param_ranges[param][0] == 'selection'):
                # So sorry for this line of 'code'
                val = param_ranges[param][2][np.random.randint(low=0, high=len(param_ranges[param][2]))] 
            config_local[param] = val

        return(config_local)





def generate_data_LHS(budget, S2A_SR_path, config=default_params, param_ranges=default_param_ranges, parameters=['cab', 'cw', 'cm', 'lai'], y_param='lai', noise=0.0):
    Y1 = np.zeros(budget)
    X1 = np.zeros((budget, 11))

    S2A_SR = load_spectral_responses(S2A_SR_path)

    config_local = config.copy()

    sampler = scipy.stats.qmc.LatinHypercube(d=len(parameters))
    sample = sampler.random(n=budget)

    for i in range(0, budget):
        for j in range(0, len(parameters)):
            param = parameters[j]
            # Scale from 0-1 to min-max range, e.g., 0.5 * (10-8=2) + 8 = 9 for range 8-10
            val = sample[i,j] * (param_ranges[param][3] - param_ranges[param][2]) + param_ranges[param][2]
            config_local[param] = val

        x1_HS = custom_prosail(config_local)
        x1_MS = hyper_to_S2(x1_HS, S2A_SR)
        # Existing work found Gaussian noise to be beneficial
        if(noise > 0.0):
            x1_MS = x1_MS + np.random.normal(0, noise, size=len(x1_MS)) * x1_MS
        X1[i,:] = x1_MS
        Y1[i] = config_local[y_param]

    return(X1, Y1)


def generate_data_LHS_MO(budget, S2A_SR_path, config=default_params, param_ranges=default_param_ranges, noise=0.0, parameters=['cab', 'cw', 'cm', 'lai'], out_parameters=['lai']):
    Y1 = np.zeros((budget, len(out_parameters)))
    X1 = np.zeros((budget, 11))

    S2A_SR = load_spectral_responses(S2A_SR_path)

    config_local = config.copy()

    sampler = scipy.stats.qmc.LatinHypercube(d=len(parameters))
    sample = sampler.random(n=budget)

    for i in range(0, budget):
        for j in range(0, len(parameters)):
            param = parameters[j]
            # Scale from 0-1 to min-max range, e.g., 0.5 * (10-8=2) + 8 = 9 for range 8-10
            val = sample[i,j] * (param_ranges[param][3] - param_ranges[param][2]) + param_ranges[param][2]
            config_local[param] = val

        x1_HS = custom_prosail(config_local)
        x1_MS = hyper_to_S2(x1_HS, S2A_SR)
        # Existing work found Gaussian noise to be beneficial
        if(noise > 0.0):
            x1_MS = x1_MS + np.random.normal(0, noise, size=len(x1_MS)) * x1_MS
        X1[i,:] = x1_MS
        vec = np.zeros(len(out_parameters))
        for j in range(0, len(vec)):
            vec[j] = config_local[out_parameters[j]]
        Y1[i,:] = vec


    return(X1, Y1)


# Random search

def generate_data_random(budget, S2A_SR_path, config=default_params, param_ranges=default_param_ranges, parameters=['cab', 'cw', 'cm', 'lai'], y_param='lai'):
    Y1 = np.zeros(budget)
    X1 = np.zeros((budget, 11))

    S2A_SR = load_spectral_responses(S2A_SR_path)

    config_local = config.copy()

    for i in range(0, budget):
        for param in parameters:
            if(param_ranges[param][0] == 'normal'):
                val = min(max(param_ranges[param][2], np.random.normal(loc=param_ranges[param][4], scale=param_ranges[param][5])), param_ranges[param][3])
            elif(param_ranges[param][0] == 'uniform'):
                if(param_ranges[param][1] == 'int'):
                    val = np.random.randint(low=param_ranges[param][2], high=param_ranges[param][3])
                elif(param_ranges[param][1] == 'float'):
                    val = np.random.uniform(low=param_ranges[param][2], high=param_ranges[param][3])
            elif(param_ranges[param][0] == 'selection'):
                # So sorry for this line of 'code'
                val = param_ranges[param][2][np.random.randint(low=0, high=len(param_ranges[param][2]))] 
            config_local[param] = val

        x1_HS = custom_prosail(config_local)
        X1[i,:] = hyper_to_S2(x1_HS, S2A_SR)
        Y1[i] = config_local[y_param]

    return(X1, Y1)


def generate_data_random_MO(budget, S2A_SR_path, config=default_params, param_ranges=default_param_ranges, parameters=['cab', 'cw', 'cm', 'lai'], out_parameters=['lai']):
    Y1 = np.zeros((budget, len(out_parameters)))
    X1 = np.zeros((budget, 11))

    S2A_SR = load_spectral_responses(S2A_SR_path)

    config_local = config.copy()

    for i in range(0, budget):
        for param in parameters:
            if(param_ranges[param][0] == 'normal'):
                val = min(max(param_ranges[param][2], np.random.normal(loc=param_ranges[param][4], scale=param_ranges[param][5])), param_ranges[param][3])
            elif(param_ranges[param][0] == 'uniform'):
                if(param_ranges[param][1] == 'int'):
                    val = np.random.randint(low=param_ranges[param][2], high=param_ranges[param][3])
                elif(param_ranges[param][1] == 'float'):
                    val = np.random.uniform(low=param_ranges[param][2], high=param_ranges[param][3])
            elif(param_ranges[param][0] == 'selection'):
                # So sorry for this line of 'code'
                val = param_ranges[param][2][np.random.randint(low=0, high=len(param_ranges[param][2]))] 
            config_local[param] = val

        x1_HS = custom_prosail(config_local)
        X1[i,:] = hyper_to_S2(x1_HS, S2A_SR)
        vec = np.zeros(len(out_parameters))
        for j in range(0, len(vec)):
            vec[j] = config_local[out_parameters[j]]
        Y1[i,:] = vec

    return(X1, Y1)


def add_soft_constraints(refl, config, file='constraints_forest.csv'):
    df = pd.read_csv(file)

    err = 0

    indices = refl_to_indices(refl)
    var_dict = config.copy()
    for k,v in indices.items():
        var_dict[k] = v

    # Iterate over constraints (rows)
    for _, r in df.iterrows():
        var1 = r['target_variable']
        #var1 = str(r[0])
        var2 = r['source_variable']
        #var2 = str(r[1])
        corr = r['correlation']
        #corr = float(r[2])
        conf = r['confidence']
        #conf = float(r[3])

        var1_est = corr * var_dict[var2]
        err += (abs(var1_est - var_dict[var1]) / var1_est) * conf

    return(err)


def add_proportional_constraints(refl, config, base_loss, meta={}, penalty_strength=1.0, use_R_squared=False, average_per_var=True, file='constraints_forest.csv'):
    """Add penalty based on original loss. Compute values 0-1 per constraint; take average value as penalty. Loss becomes base_loss + penalty_strength*penalty*base_loss.

    Args:
        refl (numpy vector): true reflectance for this instance
        config (dict): parameter configuration for PROSAIL
        base_loss (float): _description_
        meta (dict, optional): metadata of instance; could be used for constraints. Defaults to {}.
        penalty_strength (float, optional): Scales relative importance of constraints (<1: less important, >1: more important). Defaults to 1.
        use_R_squared (bool, optional): determines whether a specified R^2 value is used to weight penalties. Defaults to False.
        average_per_var (bool, optional): if True, averages penalties per target parameter if multiple rules are given. Defaults to False.
        file (str, optional): file containing user-defined constraints. Defaults to 'constraints_forest.csv'.
    """
    total_error = 0
    constraints = pd.read_csv(file)

    # Compute quantities (e.g., NDVI)
    indices = refl_to_indices(refl)
    config_local = config.copy()
    for k,v in indices.items():
        config_local[k] = v


    if(average_per_var):
        penalties = {}
        penalty_counts = {}
        for v in constraints['target_variable'].values:
            penalties[v] = 0
            penalty_counts[v] = 0
    else:
        penalty = 0

    for _, c in constraints.iterrows():

        # Per constraint, compute expected value target variable given source variable (cfg or index) and relationship
        target_variable = c['target_variable']
        source_variable = c['source_variable']
        formula = c['relationship']
        formula = formula.replace("x", "config_local['" + str(source_variable) + "']").replace("y=", "")

        exp = eval(formula) # TODO: maths stuff without resorting to eval

        # Per constraint, compute proportional absolute error
        obs = config_local[target_variable]
        prop_error = abs(obs - exp) / exp
        if(use_R_squared):
            prop_error = prop_error * c['r_squared'] # Not technically proportional error anymore

        # (Use R squared somehow???)
        local_penalty = prop_error

        # Increment
        if(average_per_var):
            penalties[target_variable] = penalties[target_variable] + local_penalty
            penalty_counts[target_variable] = penalty_counts[target_variable] + 1
        else:
            penalty += local_penalty

    # Average proportional errors
    if(average_per_var):
        penalty = 0
        num_vars = 0
        for k,v in penalties.items():
            penalties[k] = penalties[k] / penalty_counts[k]
            penalty += penalties[k]
            num_vars += 1
        penalty = penalty / num_vars
    else:
        penalty = penalty / len(constraints)

    # Compute final error
    total_error = base_loss + penalty_strength*penalty*base_loss

    # Return
    return(total_error)


# MC

def generate_data_MC(X2, S2A_SR_path, num_instances=100, MC_epochs=5000, config=default_params, param_ranges=default_param_ranges, remove_seen=False, parameters=['cab', 'cw', 'cm', 'lai']):
    to_remove_rows = np.zeros(num_instances, dtype=np.int8)

    Y1 = np.zeros((num_instances, 1))
    X1 = np.zeros((num_instances, 11))

    S2A_SR = load_spectral_responses(S2A_SR_path)

    for i in range(0, num_instances):
        ind = np.random.randint(low=0, high=len(X2)-1)
        to_remove_rows[i] = ind
        real_refl = X2[ind,:]

        closest_refl, closest_lai = run_MC(real_refl, S2A_SR, epochs=MC_epochs, config=config, param_ranges=param_ranges, parameters=parameters)

        X1[i,:] = closest_refl
        Y1[i,0] = closest_lai


    if(remove_seen):
        X2_unseen = np.delete(X2, to_remove_rows, axis=0)
        return(X1, Y1, X2_unseen)
    else:
        return(X1, Y1)


def generate_data_naive(X2, S2A_SR_path, num_instances=100, epochs=2000, config=default_params, param_ranges=default_param_ranges, remove_seen=False, parameters=['cab', 'cw', 'cm', 'lai']):
    to_remove_rows = np.zeros(num_instances, dtype=np.int8)

    Y1 = np.zeros((num_instances, 1))
    X1 = np.zeros((num_instances, 11))

    S2A_SR = load_spectral_responses(S2A_SR_path)

    for i in range(0, num_instances):
        ind = np.random.randint(low=0, high=len(X2)-1)
        to_remove_rows[i] = ind
        real_refl = X2[ind,:]

        closest_refl, closest_lai, _ = run_naive_independent(real_refl, S2A_SR, epochs=epochs, config=config, param_ranges=param_ranges, parameters=parameters)

        X1[i,:] = closest_refl
        Y1[i,0] = closest_lai


    if(remove_seen):
        X2_unseen = np.delete(X2, to_remove_rows, axis=0)
        return(X1, Y1, X2_unseen)
    else:
        return(X1, Y1)


def sim_and_compare(real_refl, config, SR, distance_func=None, distance_func_weights=False, **kwargs):
    """Simulate using supplied config, compare output to true reflectance

    Args:
        real_refl (numpy vector): reflectance spectrum of real instance
        config (dict): dictionary of PROSAIL input parameters
        SR (numpy vector): spectral weights to transform hyperspectral outputs to sentinel-2 spectra
        distance_func (function pointer, optional): user-supplied objective function to optimise (must take only two spectra as input). Defaults to None.
        distance_func_weights (numpy vector or list, optional): Weights for different dist_funcs if list of function is provided. Defaults to False (equal weights).
    """

    x1_HS = custom_prosail(config)
    x1_MS = hyper_to_S2(x1_HS, SR)
    # Check if distance function is supplied; use MAE otherwise
    if(distance_func):
        # Check if distance function is a single function or a list of objectives
        if(type(distance_func) == type([])):
            # Check if distance function weights are supplied; default to equal (ones) otherwise
            if(type(distance_func_weights) == type(False)):
                if(not(distance_func_weights)):
                    distance_func_weights = np.ones(len(distance_func))
            total_dist = 0
            df_i = 0
            for df in distance_func:
                total_dist += distance_func_weights[df_i] * df(real_refl, x1_MS)
                df_i += 1
            dist = total_dist
        else:
            dist = distance_func(real_refl, x1_MS)
    else:
        dist = np.sum(np.absolute(real_refl - x1_MS)) # Absolute error

    constrain = False
    if('constrain' in kwargs):
        if(kwargs['constrain']):
            constrain = True

    if(constrain):
        file = 'constraints/constraints_forest.csv'
        if('file' in kwargs):
            file = kwargs['file']

        #penalty = add_soft_constraints(real_refl, config)
        #dist += constraint_weight * 
        penalty_strength = 1.0
        if('penalty_strength' in kwargs):
            penalty_strength = kwargs['penalty_strength']

        use_R_squared = False
        if('use_R_squared' in kwargs):
            use_R_squared = kwargs['use_R_squared']

        average_per_var = False
        if('average_per_var' in kwargs):
            penalty_strength = kwargs['average_per_var']

        dist = add_proportional_constraints(real_refl, config, dist, penalty_strength=penalty_strength, file=file, use_R_squared=use_R_squared, average_per_var=average_per_var)

    return(dist, x1_MS)


def sim_and_compare_window(real_refls, config, SR, distance_func=None, distance_func_weights=False, window=3):
    """Simulate using supplied config, compare output to true reflectance

    Args:
        real_refl (numpy vector): reflectance spectrum of real instance
        config (dict): dictionary of PROSAIL input parameters
        SR (numpy vector): spectral weights to transform hyperspectral outputs to sentinel-2 spectra
        distance_func (function pointer, optional): user-supplied objective function to optimise (must take only two spectra as input). Defaults to None.
        distance_func_weights (numpy vector or list, optional): Weights for different dist_funcs if list of function is provided. Defaults to False (equal weights).
    """

    x1_HS = custom_prosail(config)
    x1_MS = hyper_to_S2(x1_HS, SR)

    all_pixel_dists = 0
    centre_i = int(real_refls.shape[0] / 2)
    centre_j = int(real_refls.shape[1] / 2)
    radius = int((window-1) / 2)

    for i in range(centre_i - radius, centre_i + radius + 1):
        for j in range(centre_j - radius, centre_j + radius + 1):
            real_refl = real_refls[i,j,:]
            # Check if distance function is supplied; use MAE otherwise
            if(distance_func):
                # Check if distance function is a single function or a list of objectives
                if(type(distance_func) == type([])):
                    # Check if distance function weights are supplied; default to equal (ones) otherwise
                    if(type(distance_func_weights) == type(False)):
                        if(not(distance_func_weights)):
                            distance_func_weights = np.ones(len(distance_func))
                    total_dist = 0
                    df_i = 0
                    for df in distance_func:
                        total_dist += distance_func_weights[df_i] * df(real_refl, x1_MS)
                        df_i += 1
                    dist = total_dist
                else:
                    dist = distance_func(real_refl, x1_MS)
            else:
                dist = np.sum(np.absolute(real_refl - x1_MS)) # Absolute error
            all_pixel_dists += dist 

    num_pixels = window * window
    dist - all_pixel_dists / num_pixels

    return(dist, x1_MS)


def run_HC(real_refl, SR, epochs=5000, config=default_params, param_ranges=default_param_ranges, parameters=['cab', 'cw', 'cm', 'lai'], init='default', distance_func=None, 
            track_performance=False, track_lai=False, track_params=False, track_best_params=False, **kwargs):
    """Run a hill climbing algorithm to find the best parameters that can simulate the user-supplied real reflectance.

    Args:
        real_refl (_type_): _description_
        S2A_SR (_type_): _description_
        epochs (int, optional): _description_. Defaults to 5000.
        config (_type_, optional): _description_. Defaults to default_params.
        param_ranges (_type_, optional): _description_. Defaults to param_ranges.
        parameters (list, optional): _description_. Defaults to ['cab', 'cw', 'cm', 'lai'].
        distance_func (_type_, optional): _description_. Defaults to None.
        track_performance (bool, optional): _description_. Defaults to False.
        track_params (bool, optional): _description_. Defaults to False.
        track_best_params (bool, optional): _description_. Defaults to False.
    """

    trackers = {}
    best_config = config.copy()
    closest_dist = np.inf
    closest_refl = np.zeros(len(real_refl))
    best_lai = -1

    return_trackers = False

    if(track_performance):
        dists = np.zeros(epochs)
        return_trackers = True
    if(track_params):
        param_tracker = []
        for _ in parameters:
            param_tracker.append(np.zeros(epochs))
        return_trackers = True
    if(track_lai):
        lais = np.zeros(epochs)

    for e in range(0, epochs):
        config_local = best_config.copy()

        if(e == 0):
            if(init == 'random'):
                # Initialise parameters randomly
                for param in parameters:
                    config_local[param] = mutate_random(param, param_ranges)

        for param in parameters:
            val = mutate_proportional(param, config_local[param], param_ranges, prop_sd=kwargs['prop_sd'])
            config_local[param] = val 

        dist, refl = sim_and_compare(real_refl, config_local, SR, distance_func, **kwargs)

        if(dist < closest_dist):
            best_config = config_local.copy()
            closest_dist = dist
            closest_refl = refl
            best_lai = config_local['lai']

        if(track_performance):
            dists[e] = dist
        if(track_params):
            for i in range(0, len(parameters)):
                param_tracker[i][e] = config_local[parameters[i]]
        if(track_lai):
            lais[e] = best_lai

    if(track_performance):
        trackers['performance'] = dists
    if(track_lai):
        trackers['lai'] = lais
    if(track_params):
        param_tracker2 = {}
        i = 0
        for p in parameters:
            param_tracker2[p] = param_tracker[i]
            i += 1
        trackers['params'] = param_tracker2
    if(track_best_params):
        return_trackers = True
        trackers['best_params'] = best_config

    if(return_trackers):
        return(closest_refl, best_config['lai'], trackers)
    else:
        return(closest_refl, best_config['lai'])


def run_HC_window(real_refls, SR, epochs=5000, config=default_params, param_ranges=default_param_ranges, parameters=['cab', 'cw', 'cm', 'lai'], init_randomly=True, distance_func=None, 
            track_performance=False, track_lai=False, track_params=False, track_best_params=False, **kwargs):
    """Run a hill climbing algorithm to find the best parameters that can simulate the user-supplied real reflectance.

    Args:
        real_refl (_type_): _description_
        S2A_SR (_type_): _description_
        epochs (int, optional): _description_. Defaults to 5000.
        config (_type_, optional): _description_. Defaults to default_params.
        param_ranges (_type_, optional): _description_. Defaults to param_ranges.
        parameters (list, optional): _description_. Defaults to ['cab', 'cw', 'cm', 'lai'].
        distance_func (_type_, optional): _description_. Defaults to None.
        track_performance (bool, optional): _description_. Defaults to False.
        track_params (bool, optional): _description_. Defaults to False.
        track_best_params (bool, optional): _description_. Defaults to False.
    """

    trackers = {}
    best_config = config.copy()
    closest_dist = np.inf
    closest_refl = np.zeros(real_refls.shape[2])
    best_lai = -1

    return_trackers = False

    if(track_performance):
        dists = np.zeros(epochs)
        return_trackers = True
    if(track_params):
        param_tracker = []
        for _ in parameters:
            param_tracker.append(np.zeros(epochs))
        return_trackers = True
    if(track_lai):
        lais = np.zeros(epochs)

    for e in range(0, epochs):
        config_local = best_config.copy()

        if(init_randomly and e==0):
            # Initialise parameters randomly
            for param in parameters:
                config_local[param] = mutate_random(param, param_ranges)

        for param in parameters:
            val = mutate_proportional(param, config_local[param], param_ranges, prop_sd=kwargs['prop_sd'])
            config_local[param] = val 

        dist, refl = sim_and_compare_window(real_refls, config_local, SR, distance_func)

        if(dist < closest_dist):
            best_config = config_local.copy()
            closest_dist = dist
            closest_refl = refl
            best_lai = config_local['lai']

        if(track_performance):
            dists[e] = dist
        if(track_params):
            for i in range(0, len(parameters)):
                param_tracker[i][e] = config_local[parameters[i]]
        if(track_lai):
            lais[e] = best_lai

    if(track_performance):
        trackers['performance'] = dists
    if(track_lai):
        trackers['lai'] = lais
    if(track_params):
        param_tracker2 = {}
        i = 0
        for p in parameters:
            param_tracker2[p] = param_tracker[i]
            i += 1
        trackers['params'] = param_tracker2
    if(track_best_params):
        return_trackers = True
        trackers['best_params'] = best_config

    if(return_trackers):
        return(closest_refl, best_config['lai'], trackers)
    else:
        return(closest_refl, best_config['lai'])
            


def run_MC(real_refl, S2A_SR, epochs=5000, config=default_params, param_ranges=default_param_ranges, parameters=['cab', 'cw', 'cm', 'lai'], distance_func=None, distance_func_weights=False,
                track_performance=False, track_lai=False, track_refl=False, track_params=False):

    if(track_performance):
        tot_perf = np.zeros(epochs)
    if(track_refl):
        tot_refl = np.zeros((epochs, 11))
    if(track_lai):
        tot_lai = np.zeros(epochs)
    if(track_params):
        tot_params = np.zeros((epochs, len(parameters)))

    trackers = {}

    closest_dist = np.inf
    closest_refl = np.zeros(11)
    closest_lai = -1
    best_config = {}

    for e in range(0, epochs):

        config_local = config.copy()

        for param in parameters:
            val = mutate_random(param, param_ranges)
            config_local[param] = val

        dist, x1_MS = sim_and_compare(real_refl, config_local, S2A_SR, distance_func=distance_func, distance_func_weights=distance_func_weights)

        if(dist < closest_dist):
            closest_dist = dist
            closest_refl = x1_MS
            closest_lai = config_local['lai']
            best_config = config_local

        if(track_performance):
            tot_perf[e] = dist
        if(track_refl):
            tot_refl[e, :] = closest_refl
        if(track_lai):
            tot_lai[e] = closest_lai
        if(track_params):
            tot_params[e, :] = np.array([config_local[param] if param_ranges[param][1] in ["int", "float"] else 0.0 for param in parameters])

    if(track_performance):
        trackers['performance'] = tot_perf
    if(track_refl):
        trackers['refl'] = tot_refl
    if(track_lai):
        trackers['lai'] = tot_lai
    if(track_params):
        trackers['parameters'] = tot_params
        trackers['best_params'] = best_config

    if(track_performance or track_refl or track_lai or track_params):
        return(closest_refl, closest_lai, trackers)
    else:
        return(closest_refl, closest_lai)


def check_constraints(solution, constraints=None):
    # For proof of concept add constraints as list of var1, var2, inequality: [('cab', 'cw', '>'), ('lai', 'cm', '<')]

    num_violations = 0
    for c in constraints:
        if(c[2] == ">"):
            if(solution[c[0]] < solution[c[1]]):
                num_violations += 1
        elif(c[2] == "<"):
            if(solution[c[0]] > solution[c[1]]):
                num_violations += 1
        else:
            print("Unsupported operand " + str(c[2]))

    return(num_violations)



def run_naive_independent(real_refl, S2A_SR, epochs=100, config=default_params, param_ranges=default_param_ranges, parameters=['cab', 'cw', 'cm', 'lai'], distance_func=None, include_lai=False,
                num_repeats=1, track_performance=False, track_lai=False, track_refl=False, track_params=False):

    #if(include_lai and 'lai' in parameters):
    #    parameters.remove('lai')

    config_best = config.copy()

    for _ in range(0, num_repeats):

        best_params = {}
        for param in parameters:

            config_local = config_best.copy()

            closest_dist = np.inf
            best_param = None

            for e in range(0, epochs):
                if(param_ranges[param][0] == 'normal'):
                    val = np.random.normal(loc=param_ranges[param][2], scale=param_ranges[param][3])
                elif(param_ranges[param][0] == 'uniform'):
                    if(param_ranges[param][1] == 'int'):
                        val = np.random.randint(low=param_ranges[param][2], high=param_ranges[param][3])
                    elif(param_ranges[param][1] == 'float'):
                        val = np.random.uniform(low=param_ranges[param][2], high=param_ranges[param][3])
                elif(param_ranges[param][0] == 'selection'):
                    # So sorry for this line of 'code'
                    val = param_ranges[param][2][np.random.randint(low=0, high=len(param_ranges[param][2]))] 

                config_local[param] = val

                x1_HS = custom_prosail(config_local)
                x1_MS = hyper_to_S2(x1_HS, S2A_SR)
                if(distance_func):
                    dist = distance_func(real_refl, x1_MS)
                else:
                    dist = np.sum(np.absolute(real_refl - x1_MS)) # Absolute error

                if(dist < closest_dist):
                    closest_dist = dist
                    best_param = config_local[param]

            best_params[param] = best_param

        for p,v in best_params.items():
            config_best[p] = v

    # Run with best individual

    if(track_performance):
        tot_perf = np.zeros(epochs)
    if(track_refl):
        tot_refl = np.zeros((epochs, 11))
    if(track_lai):
        tot_lai = np.zeros(epochs)
    if(track_params):
        tot_params = np.zeros((epochs, len(parameters)))

    trackers = {}

    config_local = config_best.copy()
    for p,v in best_params.items():
        config_local[p] = v 

    if(include_lai):

        param = 'lai'
        best_lai = None
        closest_dist = np.inf

        for e in range(0, epochs):
            if(param_ranges[param][0] == 'normal'):
                val = np.random.normal(loc=param_ranges[param][2], scale=param_ranges[param][3])
            elif(param_ranges[param][0] == 'uniform'):
                if(param_ranges[param][1] == 'int'):
                    val = np.random.randint(low=param_ranges[param][2], high=param_ranges[param][3])
                elif(param_ranges[param][1] == 'float'):
                    val = np.random.uniform(low=param_ranges[param][2], high=param_ranges[param][3])
            elif(param_ranges[param][0] == 'selection'):
                # So sorry for this line of 'code'
                val = param_ranges[param][2][np.random.randint(low=0, high=len(param_ranges[param][2]))] 

            config_local[param] = val

            x1_HS = custom_prosail(config_local)
            x1_MS = hyper_to_S2(x1_HS, S2A_SR)
            if(distance_func):
                dist = distance_func(real_refl, x1_MS)
            else:
                dist = np.sum(np.absolute(real_refl - x1_MS)) # Absolute error

            if(dist < closest_dist):
                closest_dist = dist
                best_lai = config_local[param]

            if(track_performance):
                tot_perf[e] = dist
            if(track_lai):
                tot_lai[e] = best_lai
            if(track_params):
                tot_params[e, :] = np.array([config_local[param] for param in parameters])

        best_params['lai'] = best_lai
        config_local['lai'] = best_lai
    else:
        best_lai = config['lai']
        tot_lai = np.array([best_lai for _ in range(0, len(tot_lai))])
        
    x1_HS = custom_prosail(config_local)
    x1_MS = hyper_to_S2(x1_HS, S2A_SR)

    if(track_performance):
        trackers['performance'] = tot_perf
    if(track_refl):
        trackers['refl'] = tot_refl
    if(track_lai):
        trackers['lai'] = tot_lai
    if(track_params):
        trackers['parameters'] = tot_params

    if(track_performance or track_refl or track_lai or track_params):
        return(x1_MS, best_lai, best_params, trackers)
    else:
        return(x1_MS, best_lai, best_params)