import numpy as np

from .simulate import *

from sklearn.svm import SVR
#import autosklearn.regression


def instantiate_discriminator(model_def, cfg=None):
    if(model_def == "SVR"):
        return(SVR())
    elif(model_def == "auto-sklearn"):
        reg = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=cfg.PARAM.INVSIM.AUTOML_WALLCLOCK_TIME,
            per_run_time_limit=cfg.PARAM.DISC.AUTOML_RUN_TIME_LIMIT,
            tmp_folder=cfg.PARAM.DISC.AUTOML_PATH + "_tmp_" + cfg.EXP.EXPERIMENT_NAME,
            #output_directory=cfg.PARAM.DISC.AUTOML_PATH + "_out_" + cfg.EXP.EXPERIMENT_NAME,
            #delete_tmp_folder_after_terminate=True,
            #delete_output_folder_after_terminate=True,
            seed=cfg.SEED,
        )
        return(reg)
    else: 
        print("Invalid discriminator model specified: " + str(model_def))


def train_inverted_simulator(cfg):
    num_instances = cfg.PARAM.INVSIM.NUM_INSTANCES

    #X, Y = generate_data_random(num_instances, cfg.S2A_SR_PATH, parameters=cfg.PARAM.OPT.TO_OPTIMISE)
    X, Y = generate_data_LHS(num_instances, cfg.S2A_SR_PATH, parameters=cfg.PARAM.OPT.TO_OPTIMISE)

    model = instantiate_discriminator(cfg.PARAM.INVSIM.MODEL, cfg=cfg)
    model.fit(X, Y)
    
    return(model)


def run_inverted_simulator(x_test, model):
    x = x_test.reshape((1, len(x_test)))
    pred = model.predict(x)[0]
    return(pred)
