import numpy as np
import pandas as pd

from .utils import *

from abc import ABC, abstractmethod
import json
import math

# Would be nicer if this was only imported in the prosail class
#import prosail

# Would be nicer if these are only imported in the SBI class
#import torch 
#import sbibm
#import dapper as dpr
#import dapper.mods as modelling
from scipy.stats import multivariate_normal






class SimulationError(Exception):
    pass


class SimulationModel(ABC):
    """Abstract class for simulation models to be used for inverse problems, with some actual inherited methods"""

    def __init__(self, defaults_path, priors={}):

        # Load default values and ranges from json file (based on domain knowledge/empirical evidence; can be customised by user)
        with open(defaults_path, 'r') as fp:
            defaults = json.load(fp)
            self.default_params = defaults["default_params"]
            self.default_param_ranges = defaults["default_param_ranges"]
            self.output_dim = defaults["output_dim"]

        # Initialise config, ranges to copy of defaults
        self.reset()

        # Initialise priors, if passed, to appropriate values
        self.set_config(priors)


    def set_config(self, parameters):
        for p, v in parameters.items():
            if(not(p in self.default_params)):
                raise SimulationError("Supplied parameters contain invalid parameter: " + str(p))

            self.config[p] = v


    def set_ranges(self, ranges):
        for k,v in ranges.items():
            assert len(v) >= 3
            assert v[0] in ['normal', 'uniform', 'selection']
            if(not(k in self.param_ranges)):
                raise SimulationError("Supplied ranges contain invalid parameter: " + str(k))
            self.param_ranges[k] = v


    def reset(self):
        self.config = self.default_params.copy()
        self.param_ranges = self.default_param_ranges.copy()


    def get_config(self):
        return(self.config.copy())
    

    def get_param_ranges(self):
        return(self.param_ranges.copy())


    def multi_dim_params(self, num_dims):
        # For arbitrary dimensionality with identical ranges/defaults, generate numbered entries automatically
        new_defaults = {}
        new_ranges = {}
        new_config = {}
        for k, v in self.default_params.items():
            # Using 0 indexing
            for j in range(num_dims):
                new_defaults[k + str(j)] = v
                new_ranges[k + str(j)] = self.param_ranges[k]
                new_config[k + str(j)] = self.config[k]
        self.default_params = new_defaults
        self.param_ranges = new_ranges
        self.config = new_config
        self.output_dim = self.output_dim * num_dims


    def print_parameters(self):
        print("Possible free parameters to modify:", list(self.param_ranges.keys()))


    @abstractmethod
    def simulate(self, parameters=None):
        """Simulate with current set of parameters, combination of defaults and priors defined at initialisation. 
        Can run on temporary set of alternative parameters defined by dict parameters (unspecified is unchanged)."""
        pass



class ToyExample(SimulationModel):
    def __init__(self, priors={}, defaults_path="src/resources/sim/toyexample_default.json"):

        # Init super
        super().__init__(defaults_path, priors=priors)

    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        out = config["a"] + config["b"] + config["c"]
        out = np.ones(self.output_dim) * out
        
        return(out)


class MultiDimGaussianSample(SimulationModel):
    # A multidimensional Gaussian distribution parameterised by mean and standard deviation on every dimension.
    # The model receives as input a mean and standard deviation, and returns the sample mean and standard deviation for the specified distribution. For our use cases, noise will be added to these sample means and standard deviations.
    # The SBI task is to infer the true means and standard deviations for a given instance, given noise.
    # The AutoIPQ task is to identify the manifold of possible means and standard deviations for a given instance, under an assumed intensity of noise.
    # For convenience and ease of arbitrary-dim parameterisation, all dimensions are independent (no covariance).

    def __init__(self, priors={}, defaults_path="src/resources/sim/mdgs_default.json", dim=2, sample_size=15, **kwargs):

        #global multivariate_normal

        #stats = __import__('scipy.stats', globals(), locals())
        #multivariate_normal = stats.multivariate_normal

        # Init super
        super().__init__(defaults_path, priors=priors)
        self.dim = dim
        self.sample_size = sample_size
        # Copy default settings for arbitrary dimensionality
        self.multi_dim_params(dim)


    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        # Initialise array of means and covariance matrix
        mean_arr = np.zeros(self.dim)
        cov = np.identity(self.dim)

        # Assign config values to mean array and covariance matrix
        for k in range(self.dim):
            mean_arr[k] = config["mu" + str(k)]
            cov[k, k] = config["sigma" + str(k)]

        # Sample from specified distribution
        dist = multivariate_normal(mean_arr, cov)
        sample = dist.rvs(size=self.sample_size)
        if(self.sample_size == 1):
            sample = sample.reshape((len(sample), 1))

        # Return sample means and stds per dim ([mu1, sigma1, mu2, sigma2, ..., muN, sigmaN])
        out = np.zeros(self.dim*2)
        i = 0
        for k in range(self.dim):
            out[i] = np.mean(sample[:, k])
            i += 1
            out[i] = np.std(sample[:, k])
            i += 1
                
        return(out)


class LinearRegressor(SimulationModel):
    # A simple linear regression model, y = a1x1 + a2x2 + ... + amxm + b
    # For a model specification and data setting, simulates outputs, to be compared to ground truth from data setting
    # Goal is to parameterise model; noise represents noise on ground truth
    # Data is hard-coded for now, makes things easier

    def __init__(self, priors={}, defaults_path="src/resources/sim/lr5d_default.json", data_path="src/resources/data/lr5d.csv", **kwargs):
        
        # Init super
        super().__init__(defaults_path, priors=priors)
        # Copy default settings for arbitrary dimensionality
        #self.multi_dim_params(dim) # TODO: multi_dim params doesn't work with bias, for now hardcode
        if(data_path is None):
            feature_data = np.arange(100) / 100
            self.feature_data = np.tile(feature_data.reshape((100, 1)), (1, 5))
        else:
            self.feature_data = pd.read_csv(data_path, header=None).values

    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        w = np.array([v for k,v in config.items() if not(k == 'b')]).reshape((self.dim, 1))
        out = np.dot(self.feature_data, w) + config['b']
                
        return(out[:, 0])


class LinearRegressor2D(LinearRegressor):
    def __init__(self, defaults_path="src/resources/sim/lr2d_default.json", data_path="src/resources/data/lr2d.csv", **kwargs):
        self.dim = 2
        # Init super
        super().__init__(defaults_path=defaults_path, data_path=data_path, **kwargs)

class LinearRegressor5D(LinearRegressor):
    def __init__(self, defaults_path="src/resources/sim/lr5d_default.json", data_path="src/resources/data/lr5d.csv", **kwargs):
        self.dim = 5
        # Init super
        super().__init__(defaults_path=defaults_path, data_path=data_path, **kwargs)

class LinearRegressor10D(LinearRegressor):
    def __init__(self, defaults_path="src/resources/sim/lr10d_default.json", data_path="src/resources/data/lr10d.csv", **kwargs):
        self.dim = 10
        # Init super
        super().__init__(defaults_path=defaults_path, data_path=data_path, **kwargs)

class LinearRegressor20D(LinearRegressor):
    def __init__(self, defaults_path="src/resources/sim/lr20d_default.json", data_path="src/resources/data/lr20d.csv", **kwargs):
        self.dim = 20
        # Init super
        super().__init__(defaults_path=defaults_path, data_path=data_path, **kwargs)

class LinearRegressor50D(LinearRegressor):
    def __init__(self, defaults_path="src/resources/sim/lr50d_default.json", data_path="src/resources/data/lr50d.csv", **kwargs):
        self.dim = 50
        # Init super
        super().__init__(defaults_path=defaults_path, data_path=data_path, **kwargs)






class MultiDimGaussianPDF(SimulationModel):
    # Idea: a multidimensional Gaussian distribution with arbitrary dimensionality from the model's perspective (but all dims share parameterisation for convenience). The output should be 1 - the probability density at a given point, so the optimum is the mean (x1 x2 etc are mean for every dim). Epsilon can then easily correspond to, e.g., 1 std (pdf(mean) - (pdf(mean-1*std))). Goal would be to find the manifold where the pdf is at most epsilon from the pdf at the mean.

    def __init__(self, task, priors={}, defaults_path="src/resources/sim/mdg_default.json", dim=2, mean=0, std=1):

        global multivariate_normal

        multivariate_normal = __import__('scipy.stats.multivariate_normal', globals(), locals())
        
        # Init super
        super().__init__(defaults_path, priors=priors)
        # TODO: experimental
        #for k, v in self.default_params.items():
        #    self.default_params[k] = float(v)
        self.dim = dim
        self.mean = mean
        self.std = std
        mean_arr = np.array([mean for _ in range(dim)])
        cov = np.identity(dim)*std
        self.dist = multivariate_normal(mean_arr, cov)
        self.mean_pdf = dist.pdf(mean_arr)
        # TODO: something allowing variable default ranges, for now just hardcode 2 to test idea

    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        mean_vec = np.array(config.values())
        out = np.array([self.dist.pdf(mean_vec, self.cov)])
                
        return(out)




# RTMs
class Prosail(SimulationModel):
    def __init__(self, priors={}, defaults_path="src/resources/sim/prosail_default.json", SR_path="src/resources/data/S2A_spectral_responses.csv", **kwargs):

        global prosail

        prosail = __import__('prosail', globals(), locals())

        # Init super
        super().__init__(defaults_path, priors=priors)

        # Load spectral response file for converting HS output to S2-like MS
        if(SR_path is not None):
            self.SR = self.load_spectral_responses(SR_path)
        else:
            self.SR = None
        


    def load_spectral_responses(self, path, level="L2A", platform="S2A"):
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
    

    def hs_to_ms(self, hyperspectral, SR_path=None, SR=None, hyper_wavelengths=-1, s2_bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]):
        """Use spectral response DataFrame to weight hyperspectral bands when converting to multispectral data.

        Args:
            hyper (1D numpy array): a high-dimensional 1D array containing a hyperspectral spectrum.
            SR (DataFrame): spectral response DataFrame
            hyper_wavelengths (list, optional): wavelengths of the hyper vector indices. Defaults to -1 (indices offset by 400).
            s2_bands (list, optional): names of the bands in the target satellite platform. Defaults to ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"].
        """

        if(SR_path is not None):
            self.SR = load_spectral_responses(SR_path)
        if(SR is not None):
            self.SR = SR
        if(SR_path is None and SR is None and self.SR is None):
            raise SimulationError("At least one of SR or SR_path must be specified, or SR must be loaded at initialisation.")

        if(hyper_wavelengths == -1):
            hyper_wavelengths = [b+400 for b in range(0,len(hyperspectral))]

        S2 = np.zeros(len(s2_bands))

        SR_low_i = df_index_func(self.SR, hyper_wavelengths[0])
        SR_high_i = df_index_func(self.SR, hyper_wavelengths[-1]+1)

        SR_matrix = self.SR[SR_low_i:SR_high_i]
        SR_matrix = SR_matrix[s2_bands].values
        
        # Iterate over S2 bands
        for b in range(0,len(s2_bands)):

            # Subset hyper bands covered by the S2 band's range
            weights = SR_matrix[:,b]

            # Compute avg S2 band value as sum(subset * distance) / sum(distance)
            S2[b] = np.dot(weights, np.array(hyperspectral)) / np.sum(weights)

        return(S2)


    def simulate(self, parameters=None, convert_to_ms=True, **kwargs):
        """Run prosail from the prosail library using the supplied configuration of parameters.
        """

        # Based on code from https://github.com/nunocesarsa/RTM_Inversion/blob/main/Jupyter/RTM%20Inversion/RTM_Noise_inversion_Optimized.ipynb

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        # parameters
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
        
        output = rho_out
        self.output_hs = output
        if(convert_to_ms):
            output = self.hs_to_ms(rho_out)
            self.output_ms = output


        return(output)
  


# Dynamic systems from DAPPER DA benchmark
class DPR_TwoPendulums(SimulationModel):
    def __init__(self, priors={}, defaults_path="src/resources/sim/dpr_two_pendulums_default.json", dt=0.1, seed=42, **kwargs):

        global dpr
        global modelling

        dpr = __import__('dapper', globals(), locals())
        modelling = __import__('dapper.mods', globals(), locals())
        
        # Init super
        super().__init__(defaults_path, priors=priors)

        dpr.set_seed(seed)
        self.dt = dt 


    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        # Slightly hacky way to run with own initial conditions
        from dapper.mods.DoublePendulum.settings101 import Dyn, Obs, jj
        from dapper.mods.DoublePendulum import LP_setup, x0
        from dapper.tools.chronos import Chronology

        t = Chronology(self.dt, dko=100, T=20, BurnIn=1)

        x0 = np.array([config[p] for p in self.default_params.keys()])

        X0 = dpr.mods.GaussRV(mu=x0, C=0.00**2)
        HMM = dpr.mods.HiddenMarkovModel(Dyn, Obs, t, X0, LP=LP_setup(jj))

        HMM.tseq.T = 1 # used to be 30 for 300 ime steps
        trues, _ = HMM.simulate(desc="") # Not very efficient because it always seems to simulate at least 100 time steps

        out = trues[1, :] # Keep final conditions as observations
        
        return(out)
    

class DPR_Lorenz63(SimulationModel):
    def __init__(self, priors={}, defaults_path="src/resources/sim/dpr_lorenz63_default.json", dt=0.1, seed=42, **kwargs):

        global dpr
        global modelling

        dpr = __import__('dapper', globals(), locals())
        modelling = __import__('dapper.mods', globals(), locals())

        # Init super
        super().__init__(defaults_path, priors=priors)

        dpr.set_seed(seed)
        self.dt = dt


    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        # Slightly hacky way to run with own initial conditions
        from dapper.mods.Lorenz63.sakov2012 import Dyn, Obs, tseq, jj
        from dapper.mods.Lorenz63 import x0, Tplot
        from dapper.tools.chronos import Chronology

        tseq = Chronology(self.dt, dko=25, Ko=1000, Tplot=Tplot, BurnIn=1 * Tplot)

        x0 = np.array([config[p] for p in self.default_params.keys()])

        X0 = dpr.mods.GaussRV(mu=x0, C=0.00**2)
        HMM = dpr.mods.HiddenMarkovModel(Dyn, Obs, tseq, X0)

        HMM.tseq.T = 1
        trues, _ = HMM.simulate(desc="")

        out = trues[-1, :] # Keep final conditions as observations
        
        return(out)


# SBI benchmark suite
class SBI(SimulationModel):


    def __init__(self, task, priors={}, defaults_path="src/resources/sim/sbi_slcp_default.json", num_samples=30, **kwargs):

        global torch
        global sbibm

        torch = __import__('torch', globals(), locals())
        sbibm = __import__('sbibm', globals(), locals())
        
        # Init super
        super().__init__(defaults_path, priors=priors)
        # TODO: experimental
        #for k, v in self.default_params.items():
        #    self.default_params[k] = float(v)
        task = sbibm.get_task(task)
        simulator = task.get_simulator()
        self.simulator = simulator
        self.num_samples = num_samples

    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        out_mat = np.zeros((self.num_samples, self.output_dim))

        param_vec = torch.from_numpy(np.array([v for _,v in config.items()]).reshape(1, len(config)).astype(np.float32))
        for i in range(self.num_samples):
            out_mat[i, :] = self.simulator(param_vec).numpy()[0, :]

        
        out = np.mean(out_mat, axis=0)
        
        return(out)
    

class SBI_GM(SBI):
    def __init__(self, defaults_path="src/resources/sim/sbi_gm_default.json", **kwargs):
        super().__init__("gaussian_mixture", defaults_path=defaults_path, **kwargs)


class SBI_SLCP(SBI):
    def __init__(self, defaults_path="src/resources/sim/sbi_slcp_default.json", **kwargs):
        super().__init__("slcp", defaults_path=defaults_path, **kwargs)


class SBI_BGLM(SBI):
    def __init__(self, defaults_path="src/resources/sim/sbi_bglm_default.json", **kwargs):
        super().__init__("bernoulli_glm", defaults_path=defaults_path, **kwargs)


class SBI_GLU(SBI):
    def __init__(self, defaults_path="src/resources/sim/sbi_glu_default.json", **kwargs):
        super().__init__("gaussian_linear_uniform", defaults_path=defaults_path, **kwargs)


class SBI_TM(SBI):
    def __init__(self, defaults_path="src/resources/sim/sbi_tm_default.json", **kwargs):
        super().__init__("two_moons", defaults_path=defaults_path, **kwargs)


class SBI_GL(SBI):
    def __init__(self, defaults_path="src/resources/sim/sbi_gl_default.json", **kwargs):
        super().__init__("gaussian_linear", defaults_path=defaults_path, **kwargs)


class SBI_LV(SBI):
    def __init__(self, defaults_path="src/resources/sim/sbi_lv_default.json", **kwargs):
        super().__init__("lotka_volterra", defaults_path=defaults_path, **kwargs)


class SBI_SIR(SBI):
    def __init__(self, defaults_path="src/resources/sim/sbi_sir_default.json", **kwargs):
        super().__init__("sir", defaults_path=defaults_path, **kwargs)


class SBI_SLCPD(SBI):
    def __init__(self, defaults_path="src/resources/sim/sbi_slcpd_default.json", **kwargs):
        super().__init__("slcp_distractors", defaults_path=defaults_path, **kwargs)


class SBI_BGLMR(SBI):
    def __init__(self, defaults_path="src/resources/sim/sbi_bglmr_default.json", **kwargs):
        super().__init__("bernoulli_glm_raw", defaults_path=defaults_path, **kwargs)




# Optimisation benchmark functions
class AckleyFunction(SimulationModel):
    def __init__(self, priors={}, defaults_path="src/resources/sim/ackley_default.json"):

        # Init super
        super().__init__(defaults_path, priors=priors)

    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        a = math.exp(-0.2 * math.sqrt(0.5 * (config['x']**2 + config['y']**2)))
        b = -math.exp(0.5 * (math.cos(2 * math.pi * config['x']) + math.cos(2 * math.pi * config['y'])))
        out = -20 * a - b + math.e + 20
        out = np.ones(self.output_dim) * out
        
        return(out)
    

class RastriginFunction(SimulationModel):
    def __init__(self, priors={}, defaults_path="src/resources/sim/rastrigin_default.json"):

        # Init super
        super().__init__(defaults_path, priors=priors)

    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        t1 = 10 * len(config)
        t2 = 0
        for k,v in config.items():
            t2 += v**2 - 10*math.cos(2*math.pi*v)

        out = t1 + t2
        out = np.ones(self.output_dim) * out
        
        return(out)
    

class RosenbrockFunction(SimulationModel):
    def __init__(self, priors={}, defaults_path="src/resources/sim/rosenbrock_default.json"):

        # Init super
        super().__init__(defaults_path, priors=priors)

    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        t1 = (config['a'] - config['x'])**2
        t2 = config['b'] * (config['y'] - config['x']**2)**2
        out = t1 + t2
        out = np.ones(self.output_dim) * out
        
        return(out)
    

class McCormickFunction(SimulationModel):
    def __init__(self, priors={}, defaults_path="src/resources/sim/mccormick_default.json"):

        # Init super
        super().__init__(defaults_path, priors=priors)

    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        t1 = math.sin(config['x'] + config['y'])
        t2 = (config['x'] - config['y'])**2
        out = t1 + t2 - 1.5*config['x'] + 2.5*config['y'] + 1
        out = np.ones(self.output_dim) * out
        
        return(out)
    

class ThreeHumpCamelFunction(SimulationModel):
    def __init__(self, priors={}, defaults_path="src/resources/sim/camel_default.json"):

        # Init super
        super().__init__(defaults_path, priors=priors)

    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        a = 2 * config['x']**2
        b = 1.05 * config['x']**4
        c = config['x']**6 / 6
        d = config['x'] * config['y']
        e = config['y']**2
        out = a - b + c + d + e
        out = np.ones(self.output_dim) * out
        
        return(out)


class HimmelblauFunction(SimulationModel):
    def __init__(self, priors={}, defaults_path="src/resources/sim/himmelblau_default.json"):

        # Init super
        super().__init__(defaults_path, priors=priors)

    
    def simulate(self, parameters=None):

        # self.config contains default values + user-supplied custom priors/init values from object initialisation
        config = self.config.copy()
        # Allow user to run new simulation, optionally with user-supplied new custom parameter values
        if(parameters is not None):
            for k,v in parameters.items():
                config[k] = v

        a = config['x']**2 + config['y'] - 11
        b = config['x'] + config['y']**2 - 7
        out = a**2 + b**2
        out = np.ones(self.output_dim) * out
        
        return(out)