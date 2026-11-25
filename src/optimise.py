import numpy as np
import scipy.stats

from shapely.geometry import Point

import random

from .simulate import *
from .utils import *

from cmaes import CMA

#import warnings
#warnings.filterwarnings("error")


class Optimiser():
    def __init__(self, sim, real_output, free_parameters, stratified_parameters=[], init='default', init_config=None, init_noise=0.0,
            loss_func=None, solution_set_size=-1, seed=None, mode='default', population=[], constraint_penalty=np.inf, classifiers=[],
            store_evals=False, num_evals=5000, verbose=False,
            use_iterated_search=False, is_iterations=10, is_weight=0, is_k=1, is_strata=10, **kwargs):
        self.sim = sim
        self.real_output = real_output
        self.free_parameters = free_parameters
        self.stratified_parameters = stratified_parameters
        self.solution_set_size = solution_set_size
        self.use_iterated_search = use_iterated_search
        self.is_iterations = is_iterations
        self.is_weight = is_weight
        self.is_k = is_k
        self.is_strata = is_strata
        self.current_solutions = []
        self.reverse = False
        self.mode = mode
        self.population = population
        self.constraints = []
        self.constraint_penalty = constraint_penalty
        self.l_clf = classifiers
        self.store_evals = store_evals
        self.verbose = verbose

        self.i = 0

        self.config = sim.get_config()
        self.param_ranges = sim.get_param_ranges()

        # Initialise parameters
        self.init_params(init, init_config, init_noise=init_noise)


        # Some conditional stuff
        if(use_iterated_search):
            self.solution_set = np.zeros((is_iterations, len(free_parameters)))
            self.is_iteration = 0
        if(seed is not None):
            pass#np.random.seed(seed)

        self.best_performance = np.inf

        # Use user-supplied functions if passed; default to built-in otherwise
        self.loss_func = None
        self.set_loss(loss_func)

        # Setup evals stuff
        if(self.store_evals):
            self.init_evals(num_evals)

        
        self.sim_refl = None
        self.best_sim_refl = None


    def init_evals(self, num_evals):
        self.evals_X = np.zeros((num_evals, len(self.free_parameters)))
        self.evals_X2 = np.zeros((num_evals, len(self.free_parameters)))
        self.evals_Y = np.zeros(num_evals)
        self.evals_L = np.zeros(num_evals)


# MARK
    def run_iterated(self, epochs, param_ranges_gridded, current_solutions=[], re_init_config=True, re_init_method='uniform', **kwargs):

        self.current_solutions = current_solutions
        self.param_ranges_gridded = param_ranges_gridded
        self.stratified_parameters = list(self.param_ranges_gridded.keys())

        self.reset() # probably was weird because it kept best performance from previous interations

        if(re_init_config):
            # Reinitialise solution
            for p in self.free_parameters:
                if(re_init_method == 'uniform'):
                    self.best_params[p] = self.mutate_uniform(p, self.param_ranges)
                elif(re_init_method == 'normal'):
                    self.best_params[p] = self.mutate_normal(p, self.param_ranges)
                else:
                    raise NotImplementedError
                self.current_params = self.best_params.copy()


        # Run optimisation
        for _ in range(0, epochs):
            self.step()

        # Get solution, assign to solution set, set trace
        solution = self.get_solution(**kwargs)

        return(solution)
    

# MARK
    def run_exp(self, budget):
        if(self.store_evals):
            evals = np.zeros((budget, len(self.free_parameters)))

        for i in range(budget):
            self.step()

            if(self.store_evals):
                evals[i, :] = self.get_current_solution(format="vec")

        if(self.store_evals):
            self.evals = evals

        return(self.get_solution(format="dict"))

# MARK
    def run(self, epochs, re_init_config=True, re_init_method='uniform', **kwargs):
        self.stratified_parameters = list(self.param_ranges_gridded.keys())

        self.reset()

        if(re_init_config):
            # Reinitialise solution
            for p in self.free_parameters:
                if(re_init_method == 'uniform'):
                    self.best_params[p] = self.mutate_uniform(p, self.param_ranges)
                elif(re_init_method == 'normal'):
                    self.best_params[p] = self.mutate_normal(p, self.param_ranges)
                else:
                    raise NotImplementedError
                self.current_params = self.best_params.copy()


        # Run optimisation
        for _ in range(0, epochs):
            self.step()

        # Get solution, assign to solution set, set trace
        solution = self.get_solution(**kwargs)


        return(solution)


    def set_loss(self, loss_func, mode_only=False):
        if(not(mode_only)):
            if(loss_func is None or loss_func == "proportional_distance"):
                self.f1 = self.prop_distance
            elif(loss_func == "mae"):
                self.f1 = self.mae
            elif(loss_func == "minmax"):
                self.f1 = self.minmax_loss
            elif(loss_func == "identity"):
                self.f1 = self.identity_loss
            elif(loss_func == "sam"):
                self.f1 = self.sam_loss
            else:
                self.f1 = loss_func

        if(self.mode in ["default", "1"]):
            self.loss_func = self.f1
            self.clear_constraints()
            self.add_constraint(self.constraint_1)
        elif(self.mode == "2a"):
            self.loss_func = self.f2a
        elif(self.mode == "2b"):
            self.loss_func = self.f2b
        elif(self.mode == "2c"):
            self.loss_func = self.f2c
        elif(self.mode == "2d"):
            self.loss_func = self.f2d
        elif(self.mode == "2e"):
            self.loss_func = self.f2d # Same function (pure diversity, rely on constraints+init for different behaviour)
        else:
            raise NotImplementedError()


    def add_constraint(self, constraint_func):
        """Add a constraint function to the optimiser. If violated, the objective function value will always be np.inf.

        Args:
            constraint_func (function): constraint function handle. Must take a configuration as input and return True/False.
        """
        self.constraints.append(constraint_func)


    def clear_constraints(self):
        """Clear list of existing constraints.
        """
        self.constraints = []


    def add_clf(self, clf):
        self.l_clf.append(clf)


    def clear_clf(self):
        self.l_clf = []


    def constraint_2a(self, config: dict):
        """Pre-made constraint for step 2a (l1 loss must be smaller than self.epsilon_loss)

        Args:
            config (_type_): parameter configuration
        """
        active = False
        sim_x = self.forward_sim(config, format="dict")
        f1_loss = self.f1(sim_x, self.real_output)

        if(f1_loss > self.epsilon_loss):
            active = True

        return(active)
    

    def constraint_2b(self, config: dict):
        """Pre-made constraint for step 2b (l1 loss must be greater than self.epsilon_loss)

        Args:
            config (_type_): parameter configuration
        """
        active = False
        sim_x = self.forward_sim(config, format="dict")
        f1_loss = self.f1(sim_x, self.real_output)

        if(f1_loss <= self.epsilon_loss):
            active = True

        return(active)
    

    def constraint_2d(self, config: dict):
        """Pre-made constraint for step 2d (l1 loss difference with epsilon loss must be lower than self.epsilon2)

        Args:
            config (_type_): parameter configuration
        """
        active = False
        sim_x = self.forward_sim(config, format="dict")
        f1_loss = self.f1(sim_x, self.real_output)
        diff = abs(f1_loss - self.epsilon_loss) / self.epsilon_loss # normalised for easy parameterisation, may not be ideal
        #print("sim_x:", sim_x)
        #print("real x:", self.real_output)
        #print("epsilon loss:", self.epsilon_loss)
        #print("f1 loss:", f1_loss)
        #print("diff:", diff)
        #print("constraint:", diff>self.epsilon2)

        if(diff > self.epsilon2):
            active = True

        return(active)


    def constraint_1(self, config: dict):
        """Pre-made constraint for step 1. Does nothing for unimodal problems (len(l_clf) == 0), but activates if a solution is in the region that is part of
        an existing optimum's epsilon-neighbourhood, approximated by the decision boundary of a classifier.

        Args:
            config (dict): parameter configuration
        """
        active = False
        config_vec = self.solution_to_coords(config)
        for clf in self.l_clf:
            pred = clf.predict(config_vec.reshape((1, len(config_vec))))[0]
            if(pred == 1):
                active = True
        return(active)


    def init_params(self, init, init_config=None, init_noise=0.0):
        if(init == 'default'):
            self.best_params = {}
            for p in self.free_parameters:
                self.best_params[p] = self.config[p]
        elif(init == 'random'):
            self.best_params = {}
            for p in self.free_parameters:
                self.best_params[p] = self.mutate_random(p, self.param_ranges)
        elif(init == 'manual'):
            assert init_config is not None
            self.best_params = {}
            for k,v in init_config.items():
                manual_val = v
                if(init_noise > 0.0):
                    manual_val = manual_val + np.random.normal(loc=0, scale=init_noise*abs(manual_val))
                self.best_params[k] = manual_val
        else:
            raise NotImplementedError("Invalid initialisation strategy: " + str(init))
        self.best_params = {k:v for k,v in self.best_params.items() if k in self.free_parameters}
        self.current_params = self.best_params.copy()
        self.new_params = self.current_params.copy()


    def setup_step2_exploration(self, opt_loss, epsilon, num_evals=5000):
        # Set GO loss and epsilon, used for f2 (diversity objective)
        # Assign list of Optimiser objects to allow for diversity optimisation
        self.epsilon_loss = opt_loss + epsilon
        self.store_evals = True
        self.init_evals(num_evals)
    

    def setup_step2(self, population, opt_config, opt_loss, epsilon, alpha=0.1, k=1, num_evals=5000):
        # Set GO loss and epsilon, used for f2 (diversity objective)
        # Assign list of Optimiser objects to allow for diversity optimisation
        self.population = population
        self.opt_loss = opt_loss
        self.opt_config = opt_config # pre-normalised would be more efficient, but let's keep modular and consistent for now
        self.epsilon = epsilon
        self.alpha = alpha
        self.epsilon_loss = opt_loss + epsilon
        self.k = min(k, len(population))
        self.set_loss(None, mode_only=True)
        self.store_evals = True
        self.init_evals(num_evals)
        

    def setup_step2a(self, population, opt_config, opt_loss, epsilon, **kwargs):
        self.mode = '2a'
        self.current_params = self.coords_to_solution(opt_config) # warm start at optimum to move away from there
        self.best_params = self.coords_to_solution(opt_config)
        self.setup_step2(population, opt_config, opt_loss, epsilon, **kwargs)
        self.add_constraint(self.constraint_2a)


    def setup_step2b(self, population, opt_config, opt_loss, epsilon, **kwargs):
        self.mode = '2b'
        # Would be nice to guarantee initialisation outside region
        self.setup_step2(population, opt_config, opt_loss, epsilon, **kwargs)
        self.add_constraint(self.constraint_2b)


    def setup_step2c(self, opt_loss, epsilon, **kwargs):
        self.mode = '2c'
        self.epsilon = epsilon
        self.epsilon_loss = opt_loss + epsilon
        self.set_loss(None, mode_only=True)


    def setup_step2d(self, population, opt_config, opt_loss, epsilon, epsilon2, **kwargs):
        #self.verbose = True # TODO: remove
        self.mode = '2d'
        self.epsilon2 = epsilon2 # maybe not the best name, is range from epsilon-boundary
        self.setup_step2(population, opt_config, opt_loss, epsilon, **kwargs)
        self.add_constraint(self.constraint_2d) # TODO: uncomment, testing if constraint is blocking all the movement, if commented should be sampled all over the place


    def setup_step2e(self, population, opt_config, opt_loss, epsilon, **kwargs):
        self.mode = '2e'
        self.setup_step2(population, opt_config, opt_loss, epsilon, **kwargs)
        self.add_constraint(self.constraint_2a) # Same as 2A (but loss is only diversity)


    def f2a(self, x_new, x_real, y_config=None):
        # Second objective function for constrained diversity optimisation (constraint: epsilon-neighbourhood of best loss)

        # This is just to allow f2 to be computed for previous generations
        if(y_config is None):
            config = self.solution_to_coords(self.new_params)
        else:
            config = self.solution_to_coords(y_config)

        # Term 1: diversity (mean of normalised knn distances with population)
        #config = self.normalise_y(self.get_current_solution(format='vec'))
        config = self.normalise_y(config)
        pop_configs = np.array([self.normalise_y(individual.get_current_solution(format='vec')) for individual in self.population if not individual is self])
        pop_diffs = np.sum(np.absolute(pop_configs - self.normalise_y(config)), axis=1)
        knn_dist = np.mean(np.sort(pop_diffs)[:self.k])

        # Term 2: distance to GO
        opt_dist = np.sum(np.absolute(self.normalise_y(self.opt_config) - config))

        # Combine into single loss
        #loss = 0 - (1-self.alpha)*knn_dist - self.alpha*np.std(pop_diffs[:self.k])
        loss = 0 - (1-self.alpha)*opt_dist - self.alpha*knn_dist

        return(loss)
    

    def f2b(self, x_new, x_real, y_config=None):
        # Second objective function for constrained diversity optimisation (constraint: epsilon-neighbourhood of best loss)

        # This is just to allow f2 to be computed for previous generations
        if(y_config is None):
            config = self.solution_to_coords(self.new_params)
        else:
            config = self.solution_to_coords(y_config)

        # Term 1: diversity (mean of normalised knn distances with population)
        #config = self.normalise_y(self.get_current_solution(format='vec'))
        #config = self.normalise_y(config)
        pop_configs = np.array([self.normalise_y(individual.get_current_solution(format='vec')) for individual in self.population if not individual is self])
        pop_diffs = np.sum(np.absolute(pop_configs - self.normalise_y(config)), axis=1)
        knn_dist = np.mean(np.sort(pop_diffs)[:self.k])

        # Term 2: distance to GO
        opt_dist = np.sum(np.absolute(self.normalise_y(self.opt_config) - config))

        # Combine into single loss
        #loss = 0 - (1-self.alpha)*knn_dist - self.alpha*np.std(pop_diffs[:self.k])
        loss = (1-self.alpha)*opt_dist - self.alpha*knn_dist

        loss = np.sum(np.absolute(self.normalise_y(config) - self.normalise_y(self.opt_config)))

        return(loss)
    

    def f2c(self, x_new, x_real, y_config=None):
        # Third objective function for step 2, alternative to f2a+f2b. Find epsilon-boundary ASAP, then return and terminate
        f1_loss = self.f1(x_new, x_real)
        loss = abs(f1_loss - self.epsilon_loss)
        return(loss)


    def f2d(self, x_new, x_real, y_config=None):
        # Fourth objective function for step 2, alternative to f2a and f2b, executed after f2c (constrained diversity after init to epsilon boundary)

        # This is just to allow f2 to be computed for previous generations
        if(y_config is None):
            config = self.solution_to_coords(self.new_params)
        else:
            config = self.solution_to_coords(y_config)

        # Term 1: diveristy
        pop_configs = np.array([self.normalise_y(individual.get_current_solution(format='vec')) for individual in self.population if not individual is self])
        pop_diffs = np.sum(np.absolute(pop_configs - self.normalise_y(config)), axis=1)
        knn_dist = np.mean(np.sort(pop_diffs)[:self.k])


        # Could add term 2 for quality to take inspiration from QD work; for now, use constraint instead

        # Return

        loss = 0 - knn_dist
        return(loss)
    

    def f2d_experimental(self, x_new, x_real, y_config=None):
        # Experimental version. Remove and remove _old from above if not working
        # Fourth objective function for step 2, alternative to f2a and f2b, executed after f2c (constrained diversity after init to epsilon boundary)

        # This is just to allow f2 to be computed for previous generations
        if(y_config is None):
            config = self.solution_to_coords(self.new_params)
        else:
            config = self.solution_to_coords(y_config)


        traces = []
        for opt in self.population:
            traces.append(opt.get_evals("X2"))

        X2 = np.concatenate(traces)
        if(len(X2) > 0):
            X2_norm = self.normalise_Y(X2)

            # Compute diversity to existing trace
            pop_diffs = np.sum(np.absolute(X2_norm - self.normalise_y(config)), axis=1)
            knn_dist = np.mean(np.sort(pop_diffs))#[:self.k])

            # Return

            loss = 0 - knn_dist

        else:
            loss = 0

        return(loss)



    def sim_and_loss(self, y_config=None, update=True, **kwargs):
        # If update is False, just used to compute loss from config (e.g. for gradient approximation)
        if(y_config is None):
            config = self.new_params
        else:
            config = self.coords_to_solution(y_config)
        # Check and correct ranges (could happen with some optimisers)
        for k in self.free_parameters:
            v = config[k]
        #for k,v in config.items():
            # TODO: somewhere, new_config adds all parameters instead of only free ones
            # should find out where this happens; for now kind of hacky solution

            if(v < self.param_ranges[k][2]):
                config[k] = self.param_ranges[k][2]
            elif(v > self.param_ranges[k][3]):
                config[k] = self.param_ranges[k][3]

            

        if(self.verbose):
            print("\nSim and loss config:", config)

        # Simulate
        sim_output = self.sim.simulate(parameters=config, **kwargs)
        self.sim_output = sim_output

        # Compute loss and constraints (if any)

        # For steps 2a, 2b or 2d, sim_and_loss is called with previous config as y_config arg. If config is not None, explicilty pass y_config to loss func.
        if(y_config is not None):
            loss = self.loss_func(sim_output, self.real_output, y_config=config)
        else:
            loss = self.loss_func(sim_output, self.real_output)


        for cf in self.constraints:
            cf_active = cf(config)
            if(cf_active):
                #try:
                loss = loss + max(abs(loss), 0.00000000001) * self.constraint_penalty # changing to * instead of ** because ** may be weird for 0-1
                # Smooth to prevent warning/nan when computing abs(0)*inf
                #except RuntimeWarning:
                #    print("Invalid stuff warning.")
                #    print("loss type:", type(loss))
                #    print("constr penalty type:", self.constraint_penalty)
                #    print("loss val:", loss)
                #    print("constr penalty val:", self.constraint_penalty)
                #    loss = loss + abs(loss)*self.constraint_penalty
                #    print("\n")
                if(self.reverse):
                    loss = -loss # TODO: not entirely correct but not reversing anyway

        # Update solution
        if(update):
            self.update_solution(loss, config, sim_output)

        # Add evals to matrices if storing
        if(self.store_evals and update):
            self.save_evals(sim_output, self.new_params, loss)
        

        return(loss)


    def forward_sim(self, y, format="vec"):
        config_local = self.config.copy()
        if(format == "vec"):
            i = 0
            for p in self.free_parameters:
                config_local[p] = y[i]
                i += 1             
        elif(format == "dict"):
            for k,v in y.items():
                config_local[k] = v


        # Generate simulation

        sim_output = self.sim.simulate(parameters=config_local)
        return(sim_output)


    def save_evals(self, sim_output, config, loss):
        # Add evals to matrices if storing
        self.evals_X[self.i, :] = self.solution_to_coords(config)
        self.evals_X2[self.i, :] = self.solution_to_coords(self.current_params)
        if(self.epsilon_loss is not None):
            if(self.f1(sim_output, self.real_output) < self.epsilon_loss):
                self.evals_Y[self.i] = 1 # initialised as 0, so no update if outside of neighbourhood
        else:
            self.evals_Y[self.i] = np.nan
        self.evals_L[self.i] = loss

        self.i = self.i + 1


    def get_evals(self, eval_type="both"):
        """Return the data points evaluated in step 2a or 2b, for use in training a classifier to approximate the epsilon-neighbourhood of an optimum.

        Args:
            eval_type (str, optional): Type of matrix to return. Can be "X" (sampled configurations), "X2" (current configurations), "Y" (labels), "L" (loss) or "all". Defaults to "all".
        """
        if(eval_type == "X" or eval_type == "x"):
            return(self.evals_X[:self.i, :].copy())
        elif(eval_type == "X2" or eval_type == "x2"):
            return(self.evals_X2[:self.i].copy())
        elif(eval_type == "Y" or eval_type == "y"):
            return(self.evals_Y[:self.i].copy())
        elif(eval_type == "L" or eval_type == "l"):
            return(self.evals_L[:self.i].copy())
        elif(eval_type in ["both", "all"]):
            return(self.evals_X[:self.i, :].copy(), self.evals_X2[:self.i, :].copy(), self.evals_Y[:self.i].copy(), self.evals_L[:self.i].copy())


    def normalise_y(self, y):
        y_new = np.zeros_like(y)

        i = 0
        for param in self.free_parameters:
            y_new[i] = (y[i] - self.param_ranges[param][2]) / self.param_ranges[param][3] # divided by max, offset by min, for 0-1 norm
            i += 1

        return(y_new)
    

    def normalise_Y(self, Y):
        # Like above but with matrix
        Y_new = np.zeros_like(Y)

        j = 0
        for param in self.free_parameters:
            Y_new[:, j] = (Y[:, j] - self.param_ranges[param][2]) / self.param_ranges[param][3] # divided by max, offset by min, for 0-1 norm
            j += 1

        return(Y_new)


    def identify_range_cell(self, solution):
        # For every dimension (parameter) in the solution (for which we want to compute stratified distance), compute index
        # in the stratified parameter range and add to vector. Later used to compare existing solutions (encoded this way) with 
        # current solution (encoded).

        # Find which cell a solution belongs to
        index_vec = np.zeros(len(self.stratified_parameters))

        i = 0
        for p in self.stratified_parameters:
            p_val = solution[p]
            r = np.array(self.param_ranges_gridded[p])
            index_vec[i] = len(r[r<p_val]) - 1
            i += 1

        return(index_vec)


    def update_solution(self, loss, config, sim_x):

        if(self.verbose):
            print("Optimiser:", self)
            print("New params update func:", self.new_params)
            print("prev params update func:", self.current_params)

        if(self.mode in ["default", "1", "2c", "2e"]):
            best_performance = self.best_performance
        elif(self.mode in ["diversity", "2a", "2b", "2d"]):
            # Can't use previously stored fitness values because population will have shifted, needs to be recomputed on the fly
            prev_gen_config = self.current_params
            #previous_performance = self.loss_func(self.forward_sim(prev_gen_config, format='dict'), self.real_output, y_config=prev_gen_config)
            previous_performance = self.sim_and_loss(y_config=self.solution_to_coords(prev_gen_config), update=False)
            best_performance = previous_performance
            if(self.verbose):
                print("prev performance:", previous_performance)
                print("new performance:", loss)

        if((loss <= best_performance and not(self.reverse)) or (loss >= best_performance and self.reverse)):
            #if(self.verbose):
                #if(self.mode in ["2a", "2b", "2d"]):
                    #print("doing an update: " + str(best_performance) + " to " + str(loss))
            self.best_performance = loss
            self.best_params = config.copy()
            self.current_params = config.copy()
            self.best_sim_x = sim_x


    def get_solution(self, format="vec", **kwargs):
        if(format == "dict"):
            res = {}
            for p in self.free_parameters:
                res[p] = self.best_params[p]
            return(res)
        elif(format == "vec"):
            res = np.array([self.best_params[p] for p in self.free_parameters])
            return(res)


    def get_current_solution(self, format="vec", **kwargs):
        if(format == "dict"):
            res = {}
            for p in self.free_parameters:
                res[p] = self.current_params[p]
            return(res)
        elif(format == "vec"):
            res = np.array([self.current_params[p] for p in self.free_parameters])
            return(res)


    def get_best_loss(self):
        return(self.best_performance)


    def mae(self, v1, v2, **kwargs):
        loss = np.mean(np.absolute(v1-v2))
        self.current_loss = loss
        return(loss)


    def prop_distance(self, v1, v2, **kwargs):
        # v1 is sim, v2 is real
        loss = np.mean(np.absolute(v1-v2)/np.absolute(v2))
        self.current_loss = loss
        return(loss)


    def minmax_loss(self, v1, v2, **kwargs):
        core_loss = self.old_loss(v1, v2)
        target_loss_diff = abs(core_loss - self.target_loss)
        self.current_loss = target_loss_diff
        return(target_loss_diff)
    

    def identity_loss(self, v1, v2, **kwargs):
        # used for benchmarking functions where there is no retrieval/inversion, just a loss landscape
        loss = v1[0]
        self.current_loss = loss
        return(loss)
        
        
    def sam_loss(self, v1, v2, **kwargs):
        # Based on https://github.com/PatrickTUM/UnCRtainTS/blob/main/model/src/learning/metrics.py
        # and own
        v2 = v2.reshape((1, 1, v2.shape[0]))
        v1 = v1.reshape((1, 1, v1.shape[0]))
        mat = v2 * v1
        mat = np.sum(mat, axis=2) # Use axis=2 to operate on bands
        mat = mat / np.sqrt(np.sum(v2 * v2, axis=2))
        mat = mat / np.sqrt(np.sum(v1 * v1, axis=2))
        sam = np.mean(np.arccos(np.clip(mat, a_min=-1, a_max=1))*180/np.pi)
        return(sam)
    

    def coords_to_solution(self, coords):
        # Convert list of coordinates to list of solutions (dictionary)
        solution = {self.free_parameters[j]:coords[j] for j in range(0, len(self.free_parameters))}
        return(solution)
    

    def solution_to_coords(self, s):
        # Convert solution (dictionary) to coordinates for geometric operations
        coords = np.array([s[p] for p in self.free_parameters])
        return(coords)

# MARK
    def point_is_in_hull(self, point, points_array, buffer=0.0):
        # Check if point is in convex hull defined by point cloud
        for d in range(0, len(points_array.shape)):
            max_point = np.max(points_array[:, d]) + buffer
            min_point = np.min(points_array[:, d]) - buffer
            
            if(point[d] > max_point+buffer or point[d] < min_point):
                return(False)
            
        return(True)

# MARK
    def set_prop_sd(self, prop_sd):
        self.prop_sd = prop_sd 


    def set_gridded_ranges(self, param_ranges_gridded):
        self.param_ranges_gridded = param_ranges_gridded


    def mutate_random(self, param, param_ranges):
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
    

    def mutate_uniform(self, param, param_ranges):
        """Generate a random new setting for a parameter, within pre-defined parameter ranges.

        Args:
            param (str): parameter to be updated
            param_ranges (dict): dictionary of {str:tuple} pairs, where the string is the name of the parameter, and the tuple contains the type of distribution, its ranges, and its parameterisation.
        """

        if(param_ranges[param][0] == 'uniform' or param_ranges[param][0] == 'normal'):
            if(param_ranges[param][1] == 'int'):
                val = np.random.randint(low=param_ranges[param][2], high=param_ranges[param][3])
            elif(param_ranges[param][1] == 'float'):
                val = np.random.uniform(low=param_ranges[param][2], high=param_ranges[param][3])
        elif(param_ranges[param][0] == 'selection'):
            # So sorry for this line of 'code'
            val = param_ranges[param][2][np.random.randint(low=0, high=len(param_ranges[param][2]))]

        return(val)


    def reset(self):
        self.best_performance = np.inf
        self.init_params("random")
        self.clear_constraints()
        self.clear_clf



class HCOptimiser(Optimiser):
    def __init__(self, sim, real_output, free_parameters, prop_sd=0.05, **kwargs):
        
        super().__init__(sim, real_output, free_parameters, **kwargs)
        self.prop_sd = prop_sd


    def step(self, num_steps=1, **kwargs):
        # Run (typically) 1 iteration of the hill climbing

        for _ in range(0, num_steps):

            # Compute new values
            self.new_params = self.current_params.copy()
            for p in self.free_parameters:
                self.new_params[p] = self.mutate_proportional(p, self.current_params[p], self.param_ranges, prop_sd=self.prop_sd)

            # Compute loss with current parameters
            self.sim_and_loss(**kwargs)

        return()


    def mutate_proportional(self, param, param_val, param_ranges, prop_sd=0.05):
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



class RSOptimiser(Optimiser):
    def __init__(self, sim, real_output, free_parameters, **kwargs):
        
        super().__init__(sim, real_output, free_parameters, **kwargs)


    def step(self, num_steps=1, **kwargs):
        # Run (typically) 1 iteration of the hill climbing

        for _ in range(0, num_steps):

            # Compute new values
            for p in self.free_parameters:
                self.new_params[p] = self.mutate_random(p, self.param_ranges)
           

            # Compute loss with current parameters
            self.sim_and_loss(**kwargs)

        return()
    

class GDOptimiser(Optimiser):
    def __init__(self, sim, real_output, free_parameters, learning_rate=1.0, delta=np.sqrt(np.finfo(float).eps), **kwargs):

        self.delta = delta # aka epsilon, perturbation for local gradient approximation, renamed to avoid confusion with our epsilon
        self.learning_rate = learning_rate
        super().__init__(sim, real_output, free_parameters, **kwargs)


    def step(self, num_steps=1, **kwargs):
        # Run (typically) 1 iteration of gradient descent
        from scipy.optimize import approx_fprime # gradient descent optimiser has scipy as extra dependency

        for _ in range(0, num_steps):
            # Approximate gradient
            old_config = self.solution_to_coords(self.current_params)
            grad = approx_fprime(old_config, self.sim_and_loss, self.delta, False)
            new_config = old_config - self.learning_rate*grad
            # Account for invalid values from the gradient
            for j in range(len(new_config)):
                k = self.free_parameters[j]
                v = new_config[j]
                if(v < self.param_ranges[k][2]):
                    new_config[j] = self.param_ranges[k][2]
                if(v > self.param_ranges[k][3]):
                    new_config[j] = self.param_ranges[k][3]

            # Compute new values
            self.new_params = self.coords_to_solution(new_config)
            self.sim_and_loss(**kwargs)
            

        return()


class CMAESOptimiser(Optimiser):
    def __init__(self, sim, real_output, free_parameters, elitist=False, **kwargs):
        from modcma import AskTellCMAES

        super().__init__(sim, real_output, free_parameters, **kwargs)

        x0 = self.solution_to_coords(self.current_params)
        cma = AskTellCMAES(d=len(x0), x0=x0, active=True, elitist=elitist)
        self.opt = cma


    def step(self, num_steps=1, **kwargs):
        # Run (typically) 1 iteration

        for it in range(0, num_steps):

            # Iterate over population
            valid = True

            # Compute new values
            x_full = self.opt.ask()
            x = x_full[:,0] # put in [d, 1] array for some reason
            config_local = self.config.copy()
            #config_local_small = {}
            i = 0
            for p in self.free_parameters:
                config_local[p] = x[i]
                # Check validity
                if(x[i] < self.param_ranges[p][2] or x[i] > self.param_ranges[p][3]):
                    self.opt.tell(x_full, np.inf)
                    valid = False
                    break
                i += 1             
            self.new_params = config_local

            if(valid):
                # Compute loss with current parameters
                loss = self.sim_and_loss(**kwargs)

                # Update population
                self.opt.tell(x_full, loss)

        return()

