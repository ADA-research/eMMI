#from ConfigSpace import Configuration, ConfigurationSpace
#from ConfigSpace.conditions import InCondition
#from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter


def add_general(search_space):
    # Algorithm, budget. No algorithm because we may just want to see the effects.
    #smac_alg = CategoricalHyperparameter("algorithm", ["RS", "HC"], default_value="HC")
    #smac_budget = UniformIntegerHyperparameter("budget", 100, 12000, default_value=8000)
    smac_prop_sd = UniformFloatHyperparameter("prop_sd", 0, 0.1, default_value=0.02) # TODO: make log somehow

    search_space.add_hyperparameters([smac_prop_sd])
    return(search_space)

def add_regularisation(search_space):
    smac_window_size = UniformIntegerHyperparameter("window_size", 1, 12, default_value=3)
    smac_lambda_lai = UniformFloatHyperparameter("lambda_lai", 0, 1, default_value=0.5)
    smac_lambda_cab = UniformFloatHyperparameter("lambda_cab", 0, 1, default_value=0.5)
    smac_lambda_cw = UniformFloatHyperparameter("lambda_cw", 0, 1, default_value=0.5)
    smac_lambda_cm = UniformFloatHyperparameter("lambda_cm", 0, 1, default_value=0.5)
    smac_lambda_psoil = UniformFloatHyperparameter("lambda_psoil", 0, 1, default_value=0.5)
    smac_lambda_rsoil = UniformFloatHyperparameter("lambda_rsoil", 0, 1, default_value=0.5)
    smac_gamma = UniformFloatHyperparameter("gamma", 0, 1, default_value=0.1)

    search_space.add_hyperparameters([smac_window_size, smac_lambda_lai, smac_lambda_cab, smac_lambda_cm, 
                                      smac_lambda_cw, smac_lambda_psoil, smac_lambda_rsoil, smac_gamma])
    return(search_space)

def add_constraints(search_space):
    smac_penalty_strength = UniformFloatHyperparameter("constraint_penalty_strength", 0, 5, default_value=1.0)

    search_space.add_hyperparameters([smac_penalty_strength])
    return(search_space)

def add_discriminator(search_space):
    #smac_discriminator_target = CategoricalHyperparameter("discriminator_target", ['error', 'is_best', 'prob_best'], default_value='error')

    #search_space.add_hyperparameters([smac_discriminator_target])
    return(search_space)

def add_iterated(search_space):
    smac_is_iterations = UniformIntegerHyperparameter("is_iterations", 1, 30, default_value=5)
    smac_is_weight = UniformFloatHyperparameter("is_weight", 0, 1, default_value=0.1)
    smac_is_k = UniformIntegerHyperparameter("is_k", 1, 30, default_value=1)
    smac_is_num_strata = UniformIntegerHyperparameter("is_strata", 1, 30, default_value=5)

    search_space.add_hyperparameters([smac_is_iterations, smac_is_weight, smac_is_k, smac_is_num_strata])
    return(search_space)

def add_invsim(search_space):
    smac_invsim_noise = UniformFloatHyperparameter("invsim_noise", 0, 1, default_value=0.2)

    search_space.add_hyperparameters([smac_invsim_noise])
    return(search_space)

def add_HC_restart(search_space):
    smac_HC_num_restarts = UniformIntegerHyperparameter("HC_num_restarts", 1, 30, default_value=5)

    search_space.add_hyperparameters([smac_HC_num_restarts])
    return(search_space)