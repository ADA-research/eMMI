import numpy as np
import scipy.stats

from scipy.spatial import ConvexHull, Delaunay

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import math

from .optimise import *
from .utils import *
from .simulate import *



class AutoIPQError(Exception):
    pass


class AutoIPQ():
    def __init__(
            self, 
            sim, 
            x: np.ndarray, 
            free_parameters: list, 
            solution_set_size: int, 
            num_strata_per_dim=None, 
            stratified_parameters=None, 
            optimiser: str ="HC", 
            optimiser_2c=None, 
            optimiser_2abd=None, 
            model=None, 
            seed: int=None, 
        **kwargs):
        # Set all the relevant parameters like spectrum, optimiser, SR etc

        # Set general attributes
        self.sim = sim
        self.x = x # Real 'output' for which we must solve the inverse problem
        self.free_parameters = free_parameters
        self.solution_set_size = solution_set_size
        self.solution_set_size_original = solution_set_size # Hacky
        self.solutions = np.zeros((solution_set_size, len(free_parameters)))
        self.model = model
        self._run_state = False # Useful to keep track of
        if(seed is not None):
            self.seed = seed
        else:
            self.seed = np.random.randint(low=0, high=9999999) # kind of hacky, should only be used when instantiating optimisers

        # Set valid ranges of parameters and compute strata
        self.param_ranges = sim.get_param_ranges()
        if(stratified_parameters is None):
            stratified_parameters = free_parameters
        self.stratified_parameters = stratified_parameters
        for p in self.stratified_parameters:
            assert p in self.free_parameters
        if(num_strata_per_dim is None):
            self.num_strata_per_dim = solution_set_size
        else:
            self.num_strata_per_dim = num_strata_per_dim
        assert type(self.num_strata_per_dim) == type(2)
        #self.param_ranges_gridded = self.compute_strata(self.param_ranges, stratified_parameters)
        # No longer really using the above, can remove all if no problems come up

        # Instantiate optimiser
        self.optimiser_2c = optimiser if optimiser_2c is None else optimiser_2c
        if(optimiser_2abd is None and ("optimiser_2a" in kwargs or "optimiser_2b" in kwargs or "optimiser_2d" in kwargs)):
            optimiser_2abd = kwargs["optimiser_2d"] # TODO: not completely finished, can now only specify 2d
        self.optimiser_2abd = optimiser if optimiser_2abd is None else optimiser_2abd
        self.opt_params = {"optimiser":optimiser, "sim": sim, "real_output": x, "free_parameters":self.free_parameters, "stratified_parameters":stratified_parameters, "seed":seed} # not capturing kwargs now
        for k,v in kwargs.items():
            self.opt_params[k] = v
        opt = self.instantiate_optimiser(**kwargs)
        self.optimiser = opt
        #self.optimiser.param_ranges_gridded = self.param_ranges_gridded


    def instantiate_optimiser(self, mode='default', optimiser=None, **kwargs):
        optimiser = self.opt_params["optimiser"] if optimiser is None else optimiser

        # To resolve conflicts; prioritise kwargs over existing properties
        opt_params = {k:v for k, v in self.opt_params.items() if not(k in kwargs)}
        #opt_params = {k:v for k, v in self.opt_params.items() if not(k == "seed")} # TODO: remove

        if(optimiser == "HC"):
            opt = HCOptimiser(mode=mode, **opt_params, **kwargs)
        elif(optimiser == "RS"):
            opt = RSOptimiser(mode=mode, **opt_params, **kwargs)
        elif(optimiser in ["GD", "gradient_descent"]):
            opt = GDOptimiser(mode=mode, **opt_params, **kwargs)
        elif(optimiser == "CMAES"):
            opt = CMAESOptimiser(mode=mode, **opt_params, **kwargs)
        else:
            raise AutoIPQError("Non-supported optimiser: " + str(optimiser))
        
        return(opt)


    def run_step1(self, budget_step1, warm_start_method, warm_start_model, classifiers, **kwargs):
        # Find optimum

        if(warm_start_method == 'optimise'):

            opt_local = self.instantiate_optimiser(mode="default", classifiers=classifiers)
            y_opt, opt_loss = self.local_search(opt_local, budget_step1, **kwargs)

        elif(warm_start_method == 'predict'):
            assert warm_start_model is not None
            y_opt = warm_start_model.predict(self.x.reshape(1, len(self.x)))[0, :]
            opt_local = self.instantiate_optimiser(mode="default", init='manual', init_config=y_opt)
            opt_loss = opt_local.f1(opt_local.forward_sim(y_opt, format='vec'), self.x)

        elif(warm_start_method == 'hybrid'):
            assert warm_start_model is not None
            ws_opt = warm_start_model.predict(self.x.reshape(1, len(self.x)))[0, :]
            init_config = self.optimiser.coords_to_solution(ws_opt)
            opt_local = self.instantiate_optimiser(mode='default', init='manual', init_config=init_config)
            y_opt, opt_loss = self.local_search(opt_local, budget_step1, **kwargs)
        else:
            raise NotImplementedError
        
        y_opt_dict = opt_local.coords_to_solution(y_opt)
        
        return(y_opt, opt_loss, y_opt_dict)
        

    def run_step2(self, epsilon, epsilon2, approach, num_solutions, k, num_exploration_opts, individual_budget, budget_step2_1, alpha, y_opt, opt_loss, y_opt_dict, **kwargs):
        if(approach in ["dual", "split"]):
            # Initialise population

            init_config_in = y_opt_dict # For in-samples

            half_point = int(num_solutions-num_exploration_opts/2)
            opts_a = [self.instantiate_optimiser(optimiser=self.optimiser_2abd, mode='2a', init='manual', init_config=init_config_in, seed=self.seed+seed_offset, **kwargs) for seed_offset in range(half_point)]
            opts_b = [self.instantiate_optimiser(optimiser=self.optimiser_2abd, mode='2b', seed=self.seed+half_point+seed_offset, **kwargs) for seed_offset in range(half_point)]
            opts = opts_a + opts_b
            for opt in opts[:half_point]:
                opt.setup_step2a(opts, y_opt, opt_loss, epsilon, alpha=alpha, k=k, num_evals=individual_budget)
            for opt in opts[half_point:]:
                opt.setup_step2b(opts, y_opt, opt_loss, epsilon, alpha=alpha, k=k, num_evals=individual_budget)

            opts_c = [self.instantiate_optimiser(optimiser="RS", mode='default', seed=self.seed+seed_offset, **kwargs) for seed_offset in range(num_exploration_opts)]
            for opt in opts_c:
                opt.setup_step2_exploration(opt_loss, epsilon, num_evals=individual_budget)
            opts = opts + opts_c
            

        elif(approach in ["sequential", "seq", "serial"]):

            # Find any point on epsilon boundary
            opt = self.instantiate_optimiser(optimiser=self.optimiser_2c, mode='2c', seed=self.seed)
            opt.setup_step2c(opt_loss, epsilon)
            for _ in range(budget_step2_1):
                opt.step()
            init_config = opt.get_solution(format="dict")
            #init_loss = opt.f1(opt.sim.simulate(init_config), self.x)

            # Initialise population
            opts = [self.instantiate_optimiser(optimiser=self.optimiser_2abd, mode='2d', init='manual', init_config=init_config, seed=self.seed+seed_offset, **kwargs) for seed_offset in range(num_solutions-num_exploration_opts)]
            for opt in opts:
                opt.setup_step2d(opts, y_opt, opt_loss, epsilon, epsilon2, alpha=alpha, k=k, num_evals=individual_budget)
            opts2 = [self.instantiate_optimiser(optimiser="RS", mode='default', seed=self.seed+seed_offset, **kwargs) for seed_offset in range(num_exploration_opts)]
            for opt in opts2:
                opt.setup_step2_exploration(opt_loss, epsilon, num_evals=individual_budget)
            opts = opts + opts2


        elif(approach in ["conv", "convex", "iterated"]):

            # Initialise all optimisers on optimum
            init_config = y_opt_dict

            # Initialise population
            opts = [self.instantiate_optimiser(optimiser=self.optimiser_2abd, mode='2e', init='manual', init_config=init_config, seed=self.seed+seed_offset, **kwargs) for seed_offset in range(num_solutions)]
            for opt in opts:
                opt.setup_step2e(opts, y_opt, opt_loss, epsilon, alpha=alpha, k=k, num_evals=individual_budget)

        else:
            raise AutoIPQError("Invalid step 2 approach:", approach)
        
        return(opts)



    def run(self, epsilon: float, clf_init_func, budget: int=1000, budget_split: list=[0.33, 0.33, 0.34], num_solutions: int=10, k: int=1, alpha: float=0.1, return_opt: bool=False, max_optima :int=1, 
            test_prop: float=0.2, verbose: bool=False, store_XY: bool=False, approach="serial", epsilon2: float=0.1, train_clf=True, store_test=False, 
            filter_density=False, density_filter_percentile=5, density_filter_same_class=False, classification_method=None, warm_start_method='optimise', warm_start_model=None,
            random_exploration_proportion=0, debug=False, **kwargs):
        """Perform constrained diversity optimisation to find an epsilon-neighbourhood around a global optimum.
        First performs regular optimisation to find GO. Next, initialises a population of optimisers with access to 
        the rest of the population, which optimise for distance to the GO and distance to each other, while constrained
        by the f1 loss value not exceeding the global optimum loss + epsilon.

        Args:
            epsilon (float): maximal distance to the global optimum objective function (f1) value
            clf_init_func (function): function to instantiate sklearn-like classifier to learn decision boundary for epsilon-neighbourhood. Function to instantiate new clf for every optimum.
            budget (int, optional): number of iterations available to the optimiser. Defaults to 1000.
            budget_split (list, optional): proportion of function evaluations allotted per step (1+2a+2b or 1+2c+2d). Defaults to equal (one third each).
            num_solutions (int, optional): number of solutions (population size) to perform diversity optimisation. Defaults to 10.
            k (int, optional): number nearest neighbours to consider when optimising for diversity. Defaults to 1.
            alpha (float, optional): weight of the diversity term relative to the distance to the global optimum (now experimenting with avg distance VS prop distance std). Defaults to 0.1.
            return_opt (boolean, optional): if True, also returns globally optimal solution. Defaults to False.
            max_optima (int, optional): maximal amount of times the function is iterated for multimodal problems (one iteration per optimum).
            test_prop (float, optional): proportion of data points to use for testing (used for sklearn's train_test_split).
            verbose (boolean, optional): if True, use verbose output (mainly test accuracy classifiers)
            store_XY (boolean, optional): if True, stores function evaluations from all optimisers
            approach (string, optional): should be "serial" or "split". If "split", half of the population optimises for distance from f1 optimum with constraint, other half optimises for proximity with inverse constraint. If "serial", first find a location on the epsilon-boundary, then optimise population at this location and optimise for diversity (constrained by distance to epsilon boundary).
            epsilon2 (float, optional): maximal proportional difference of f1 loss to epsilon-loss for f2d constraint
            train_clf (boolean, optional): if True (default), trains classifier on sampled points
            filter_density (boolean, optional): if True (default), removes points from sample with few samples around them
            density_filter_percentile (int, optional): percentile threshold to filter points from sample by density (e.g., if 5, drop 5% lowest density points)
            density_filter_same_class (boolean, optional): if True, only check density from the same class
        """

        # TODO: make this horrible mess of a function much prettier, split into sub-functions

        assert len(budget_split) == 3
        budget_split = list(np.array(budget_split) / np.sum(budget_split))
        # All budgets should have at least 1 function evaluation at all steps to avoid weird stuff
        budget_step1 = max(int(budget*budget_split[0]), 1)
        budget_step2_1 = max(int(budget*budget_split[1]), 1)
        budget_step2_2 = max(int(budget*budget_split[2]), 1)
        individual_budget = max(int(budget_step2_2 / num_solutions), 1) # For 2a+2b, just use 2b budget for both, not great solution but works for now
        num_exploration_opts = int(num_solutions * random_exploration_proportion)

        if(classification_method is None):
            classification_method = "convex_hull" if approach == "iterated" else "classifier"
        
        classifiers = []
        optima = []
        Xs = []
        X2s = []
        Ys = []
        Ls = []
        best_epsilon_loss = np.inf

        for i_optimum in range(max_optima):

            # Step 1: find optimum

            y_opt, opt_loss, y_opt_dict = self.run_step1(budget_step1, warm_start_method, warm_start_model, classifiers, **kwargs)
            if(opt_loss > best_epsilon_loss):
                # TODO: may not be good if using local search, better optima than local optimum may exist but would terminate
                break
            else:
                epsilon_loss = opt_loss + epsilon
                if(epsilon_loss < best_epsilon_loss):
                    best_epsilon_loss = epsilon_loss

            #optima.append(opt_local.get_solution(format='dict'))
            optima.append(y_opt_dict)
            self.step1_optimum = y_opt_dict
            self.step1_opt_loss = opt_loss
            

            # Step 2: diversity optimisation

            # Setting up optimiser populations
            opts = self.run_step2(epsilon, epsilon2, approach, num_solutions, k, num_exploration_opts, individual_budget, budget_step2_1, alpha, y_opt, opt_loss, y_opt_dict, **kwargs)

            # Main step 2 loop, can be population, iterated or uniform
            if(approach in ["split", "dual", "sequential", "seq", "serial"]):
                for b_i in range(individual_budget):
                    # Current limitations: awkward with inherently population-based methods (e.g., CMAES), commits to asynchronous updates
                    o_i = 0
                    for opt in opts:
                        opt.step()
                        o_i += 1
            elif(approach in ["iterated", "uniform"]):
                for opt in opts:
                    for b_i in range(individual_budget):
                        opt.step()
            else:
                raise NotImplementedError
            
            solutions = []
            for opt in opts:
                solutions.append(opt.get_solution(format='dict'))
            self.solution_set = solutions
            solutions = []
            for opt in opts:
                solutions.append(opt.get_solution(format='vec'))
            self.solution_matrix = solutions


            if(debug):
                print("\nStarting clf stuff")


            # Step 3: represent boundary
            # TODO: clean all this up into separate functions too
            if(classification_method == "classifier"):

                # Train classifier
                if(train_clf):
                    clf = clf_init_func()
                else:
                    clf = None

                # Only process stuff needed for training clf first
                X_list = [opt.get_evals(eval_type='X') for opt in opts]
                Y_list = [opt.get_evals(eval_type='Y') for opt in opts]

                X = np.concatenate(X_list, axis=0)
                Y = np.concatenate(Y_list, axis=0)

                if(store_XY):
                    # Other opt evals that are useful
                    X2_list = [opt.get_evals(eval_type='X2') for opt in opts]
                    L_list = [opt.get_evals(eval_type='L') for opt in opts]
                    # Append to lists for multiple optima
                    Xs_pop = np.array(X_list)
                    Xs.append(Xs_pop)
                    X2s_pop = np.array(X2_list)
                    X2s.append(X2s_pop)
                    Ys_pop = np.array(Y_list)
                    Ys.append(Ys_pop)
                    Ls_pop = np.array(L_list)
                    Ls.append(Ls_pop)

                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_prop)
                if(store_test):
                    self.sample_X = X_test # These are parameters (Y overall), X for classifier
                    self.sample_Y = Y_test
                    self.sampling_opt = opts[0]
                        

                if(filter_density):
                    try:
                        from scipy.stats import gaussian_kde
                        if(density_filter_same_class):
                            # Avoid training on outliers and contradicting samples
                            X_trues = X_train[Y_train == 1]
                            X_falses = X_train[Y_train == 0]
                            X_trues_flipped = np.moveaxis(X_trues, 0, -1)
                            X_falses_flipped = np.moveaxis(X_falses, 0, -1)
                            kernel_true = gaussian_kde(X_trues_flipped)
                            kernel_false = gaussian_kde(X_falses_flipped)

                            dens = kernel_true(X_trues_flipped)
                            dens_threshold = np.percentile(dens, int(density_filter_percentile/2))
                            X_trues = X_trues[dens >= dens_threshold]
                            Y_trues = np.ones(X_trues.shape[0])

                            dens = kernel_false(X_falses_flipped)
                            dens_threshold = np.percentile(dens, int(density_filter_percentile/2))
                            X_falses = X_falses[dens >= dens_threshold]
                            Y_falses = np.zeros(X_falses.shape[0])

                            X_train = np.concatenate([X_trues, X_falses], axis=0)
                            Y_train = np.concatenate([Y_trues, Y_falses], axis=0)

                            # Shuffle in case the order biases some model the user might use
                            inds = np.arange(X_train.shape[0])
                            np.random.shuffle(inds)
                            X_train = X_train[inds]
                            Y_train = Y_train[inds]

                        else:
                            # Avoid training on outliers
                            # Flip dims for kde
                            X_train_flipped = np.moveaxis(X_train, 0, -1)
                            # Fit KDE, compute densities and threshold
                            kernel = gaussian_kde(X_train_flipped)
                            dens = kernel(X_train_flipped)
                            dens_threshold = np.percentile(dens, density_filter_percentile)
                            # Drop points with lower density than threshold
                            X_train = X_train[dens >= dens_threshold]
                            Y_train = Y_train[dens >= dens_threshold]
                    
                    except Exception as e:
                        print("Error when trying to filter outliers:")
                        print(e)

                try:
                    if(train_clf):
                        clf.fit(X_train, Y_train)
                    classifiers.append(clf)
                    if(verbose):
                        pred = clf.predict(X_test)
                        acc = accuracy_score(Y_test, pred)
                        prec = precision_score(Y_test, pred)
                        rec = recall_score(Y_test, pred)
                        print("Classifier " + str(i_optimum) + " validation accuracy: " + str(acc) + " with precision " + str(prec) + " and recall " + str(rec))
                except:
                    print("WARNING: failed to train classifier. Did not add classifier to list.")
                    print("#positive samples:" + str(len(Y_train[Y_train == 1])))
                    print("#negative samples:" + str(len(Y_train[Y_train == 0])))

            elif(classification_method == "convex_hull"):
                if(train_clf):
                    convex_hull = Delaunay(self.solution_matrix)
                    self.convex_hull = convex_hull
                    classifiers.append(convex_hull) # A little awkward, but it works

            
        # Mainly for debugging purposes
        if(store_XY):
            self.Xs = Xs
            self.X2s = X2s
            self.Ys = Ys
            self.Ls = Ls
            self.opts = opts
            
        self.optima = optima

        # Return classifiers

        #solutions = [opt.get_solution(format='dict') for opt in opts]

        if(return_opt):
            return(classifiers, optima)
        else:
            return(classifiers)


    def quantify_ill_posedness(self, l_clf):
        # Specific to SVMs

        # TODO: check for overlap when projected
        ips = {}
        i = 0
        for param in self.free_parameters:
            covered = 0
            for clf in l_clf:
                svs = clf.support_vectors_[:, i]
                svs_min = np.min(svs)
                svs_max = np.max(svs)
                norm_diff = (svs_max - svs_min) / (self.param_ranges[param][3] - self.param_ranges[param][2])
                covered += norm_diff
            ips[param] = covered
            i += 1
        return(ips)
            



    def compute_strata(self, param_ranges, stratified_parameters):
        # Compute cell sizes based on probabilities of values (equal probability cells)
        param_ranges_gridded = {}
        assert len(stratified_parameters) > 0
        for p in stratified_parameters:
            p_dist_type = param_ranges[p][0]
            p_min = param_ranges[p][2]
            p_max = param_ranges[p][3]
            p_list = [p_min]
            prob_step = 1.0 / self.num_strata_per_dim
            prob = 0

            if(p_dist_type == 'normal'):
                for i in range(0, self.num_strata_per_dim-1):
                    prob = prob + prob_step
                    val = scipy.stats.norm.ppf(prob, loc=param_ranges[p][4], scale=param_ranges[p][5])
                    p_list.append(val)
                p_list.append(param_ranges[p][3]) # Do max manually
                param_ranges_gridded[p] = p_list

            elif(p_dist_type == 'uniform'):
                for i in range(0, self.num_strata_per_dim):
                    prob = prob + prob_step
                    val = p_min + (i+1)*prob_step*(p_max-p_min)
                    p_list.append(val)
                param_ranges_gridded[p] = p_list

        # Save for later use in reward
        return(param_ranges_gridded)
    

# MARK
    def compute_minmax(self, budget, proportion=0.2, **kwargs):

        # Result: identify loss value for 'circle' based on user-supplied proportion, set min+max+loss value of self.optimiser

        # Compute best solution

        self.optimiser.set_gridded_ranges(self.param_ranges_gridded)
        solution = self.optimiser.run(budget, self.param_ranges_gridded, **kwargs)
        best_loss = self.optimiser.best_performance
        self.optimiser.reset()


        # Compute worst solution
        # Set reverse, run again, reset reverse
        self.optimiser.reverse = True
        solution = self.optimiser.run(budget, self.param_ranges_gridded, **kwargs)
        worst_loss = self.optimiser.best_performance
        self.optimiser.reverse = False


        # Compute target loss, set relevant params
        target_loss = best_loss + abs(best_loss - worst_loss) * proportion 
        target_loss = best_loss + 0.05 # TODO: possibly remove, need to check if difference or this is better, either way this is hardcoded so bad
        self.optimiser.best_loss = best_loss
        self.optimiser.worst_loss = worst_loss
        self.optimiser.target_loss = target_loss

        # Set loss func to minmax
        self.optimiser.set_loss("minmax")

        # Return min and max
        return(best_loss, worst_loss)
    

# MARK
    def find_solution_set_geometry(self, epsilon, max_solutions=100, merge_polys=True, **kwargs):
        GOs = []
        solution_sets = []
        losses = []
        for _ in range(0, max_solutions):
            # Find current GO and its loss; constrain by polys of solutions already found
            GO, l = self.find_GO(solution_sets, **kwargs)

            if(l > epsilon):
                # Once next global optimum is higher than epsilon, no need to search anymore
                if(len(losses) == 0):
                    # But we do want to return something if there are no solutions lower than epsilon
                    # In this case ill-posedness would be everything? Create artificial solution set covering everything in range
                    solution_set = []
                    p_i = 0
                    for param in self.free_parameters:
                        point = {param2:(self.param_ranges[param2][3]-self.param_ranges[param2][2])/2 for param2 in self.free_parameters}
                        point[param] = self.param_ranges[param][2]
                        solution_set.append(point)
                        point = {param2:(self.param_ranges[param2][3]-self.param_ranges[param2][2])/2 for param2 in self.free_parameters}
                        point[param] = self.param_ranges[param][3]
                        solution_set.append(point)
                        p_i += 1
                    GOs.append(GO)
                    losses.append(l)
                    solution_sets.append(solution_set)
                return(GOs, solution_sets, losses)

            GOs.append(GO)
            losses.append(l)

            # Compute solution set + polygon for new solution
            solution_set = self.estimate_hyperplane(GOs, GO, losses[0], epsilon=epsilon, **kwargs) # Use original GO for loss+epsilon computation
            #print([l for d,l in solution_set])
            solution_set = [d for d,l in solution_set if not(np.isinf(l))] # Don't keep loss element, but don't keep inf losses either (no solution not violating constraints)

            if(len(solution_set) > 2):
                # Only do this if we have solutions in the set that are not inf (see above), and enough to compute hulls
                # For polys, merge overlapping region, add non-overlapping
                if(merge_polys):
                    unique = True
                    solution_set_points = self.solutions_to_coords(solution_set)
                    for i in range(0, len(solution_sets)):
                        existing_sset_points = self.solutions_to_coords(solution_sets[i])
                        if(self.poly_overlaps(existing_sset_points, solution_set_points)):
                            solution_sets[i] = self.coords_to_solutions(self.merge_polys(existing_sset_points, solution_set_points))
                            unique = False
                    if(unique):
                        solution_sets.append(solution_set)


        # Return optima and associated solution sets
        return(GOs, solution_sets, losses)


    def local_search(self, opt, budget, **kwargs):
        for _ in range(budget):
            opt.step()
        y_opt = opt.get_solution(format="vec")
        loss_opt = opt.get_best_loss()
        return(y_opt, loss_opt)


# MARK
    def find_GO(self, polys, existing_polys_buffer=0.0, **kwargs):
        # Optimise for GO, subject to GO not being in existing polys
        self.reset()
        self.optimiser.mode = "optima"
        self.optimiser.existing_polys = [self.solutions_to_coords(poly) for poly in polys]
        self.optimiser.existing_polys_buffer = existing_polys_buffer
        opt_result = self.run(solution_set_size=1, **kwargs)
        return(opt_result[0][0], opt_result[0][1])


# MARK
    def estimate_hyperplane(self, GOs, GO, GO_loss, epsilon=0.1, method="population", **kwargs):
        # Method can be "population" or "iterated"
        self.reset()
        if(method == "population"):
            self.div_optimiser = HCDiversityOptimiser(self.sim, self.x, population_size=20, prop_sd=0.05, loss_func=self.optimiser.loss_func,
                                                    free_parameters=self.free_parameters, stratified_parameters=self.stratified_parameters, seed=self.seed, 
                                                    init_strategy="point", init_value=GO, **kwargs)
        elif(method == "iterated"):
            self.div_optimiser = self.optimiser
        else:
            raise NotImplementedError
        self.div_optimiser.target_loss = GO_loss + epsilon
        self.div_optimiser.mode = "hyperplane"
        self.div_optimiser.GOs = GOs #[d for d,_ in GOs] # Get rid of losses
        self.div_optimiser.GO = GO
        self.div_optimiser.param_ranges_gridded = self.optimiser.param_ranges_gridded
        for k,v in GO.items():
            self.div_optimiser.best_params[k] = v # For local search, initialise to GO to search in neighbourhood

        if(method == "population"):
            solutions = self.div_optimiser.optimise_diversity(**kwargs)
        elif(method == "iterated"):
            solutions = self.run(solution_set_size=self.solution_set_size_original, **kwargs)

        return(solutions)
    

# MARK
    def compute_uncertainty(self, solution_sets,  prior_prob=True):
        # This needs to keep using solution sets instead of geometry, because Shapely is just 2d...
        uncs = {}

        for solutions in solution_sets:
            # [({a:1, b:2}, 0.5), ({a:2, b:4}, 1.5)]
            maxs = {}
            mins = {}

            for s in solutions:
                # {a:1, b:2}
                for p, v in s.items():
                    # a, 1
                    # Check if record exists, if not initialise
                    if(p not in uncs):
                        # This will be done once per solution_set
                        uncs[p] = 0
                    if(p not in maxs):
                        # This will be done once per solution
                        maxs[p] = v
                        mins[p] = v
                    # Update dicts with min/max vals
                    if(v > maxs[p]):
                        maxs[p] = v
                    if(v < mins[p]):
                        mins[p] = v

            for p in maxs.keys():
                if(prior_prob and self.param_ranges[p][0] == 'normal'):
                    # Uncertainty based on cumulative probabilities min/max, for uniform is just as if no distribution
                    cdf_max = scipy.stats.norm(self.param_ranges[p][4], self.param_ranges[p][5]).cdf(maxs[p])
                    cdf_min = scipy.stats.norm(self.param_ranges[p][4], self.param_ranges[p][5]).cdf(mins[p])
                    unc = 1 - (cdf_max - cdf_min)
                else:
                    # Uncertainty based on proportion of possible space covered (assume uniform within bounds)
                    min_range = self.param_ranges[p][2]
                    max_range = self.param_ranges[p][3]
                    unc = (maxs[p] - mins[p]) / (max_range - min_range)
                uncs[p] = uncs[p] + unc # We add because every solution in solution_sets should correspond to a local minimum

        return(uncs)


# MARK
    def run_old(self, budget=1000, solution_set_size="auto", solution_format="dict", return_full_config=False, solution_set=[], epsilon=1.0, auto_solution_set_size=100, **kwargs):
        # Run optimisation and return a solution set
        if(type(solution_set_size) == type(1) and solution_set_size > 0):
            # Use specified solution set size
            self.solution_set_size = solution_set_size
        elif(type(solution_set_size) == type("") and solution_set_size == "auto"):
            # Keep running until no more optima within epsilon, use max of 100 unless otherwise specified
            # Essentially allows algorithm to run for multimodal cases
            self.solution_set_size = auto_solution_set_size
        else:
            pass # default to just using existing solution set size

        run_budget = max(1, int(budget / self.solution_set_size))
       
        self.optimiser.set_gridded_ranges(self.param_ranges_gridded)
        current_solutions = []
        current_configs = []
        current_losses = []

        # Iterate
        for i in range(0, self.solution_set_size):
            solution = self.optimiser.run_iterated(run_budget, self.param_ranges_gridded, current_solutions=current_solutions, **kwargs)

            # Automatically stop searching if using "auto" and the latest global optimum is larger than epsilon
            if(type(solution_set_size) == type("") and solution_set_size == "auto"):
                if(self.optimiser.best_performance > epsilon):
                    break

            # Append current solution
            if(solution_format == "dict"):
                d = {}
                j = 0
                for p in self.free_parameters:
                    d[p] = solution[j]
                    j += 1
                current_solutions.append(d)
            elif(solution_format == "vec"):
                current_solutions.append(solution)
            else:
                raise AutoIPQError("Unsupported solution format: " + str(solution_format))
            current_losses.append(self.optimiser.best_performance)
            
            if(return_full_config):
                current_configs.append(self.optimiser.best_params)



        # Combine solutions with their losses
        res = []
        for i in range(0, len(current_solutions)):
            s = current_solutions[i]
            l = current_losses[i]
            res.append((s, l))


        self.solutions = current_solutions
        self._run_state = True
        if(return_full_config):
            return(current_configs)
        return(res)


    def solutions_to_coords(self, solutions):
        # Convert list of solutions (dictionary) to list of coordinates for geometric operations
        points = np.zeros((len(solutions), len(self.free_parameters)))
        i = 0
        for s in solutions:
            coords = [s[p] for p in self.free_parameters]
            points[i, :] = coords
            i += 1
        return(points)
    
    
    def coords_to_solutions(self, coords):
        # Convert list of coordinates to list of solutions (dictionary)
        # Inefficient to keep changing back and forth, but easier to work with
        solutions = []
        for i in range(0, coords.shape[0]):
            s = {self.free_parameters[j]:coords[i, j] for j in range(0, len(self.free_parameters))}
            solutions.append(s)

        return(solutions)


    def points_in_convex_hull(self, convex_hull, points):
        s_list = np.array(convex_hull.find_simplex(points))
        s_list[s_list >= 0] = 1.0
        s_list[s_list < 0] = 0.0
        s_list = s_list.astype(int)
        return(s_list)

# MARK
    def poly_overlaps(self, poly1, poly2):
        # Check if any part of two polygons (defined by sets of points) overlap
        assert len(poly1.shape) == len(poly2.shape)

        for d in range(0, poly1.shape[1]):
            max_point1 = np.max(poly1[:, d])
            min_point1 = np.min(poly1[:, d])

            max_point2 = np.max(poly2[:, d])
            min_point2 = np.min(poly2[:, d])
            
            if(not((min_point1 >= min_point2 and min_point1 <= max_point2) or (max_point1 >= min_point2 and max_point1 <= max_point2)) or
                not((min_point2 >= min_point1 and min_point2 <= max_point1) or (max_point2 >= min_point1 and max_point2 <= max_point1))):
                return(False)
            
        return(True)

# MARK
    def merge_polys(self, poly1, poly2):
        # Return new array containing all points from both polys
        return(np.concatenate((poly1, poly2), axis=0))

# MARK
    def solution_sets_to_hulls(self, solution_sets):
        hulls = []
        for sset in solution_sets:
            points = self.solutions_to_coords(sset)
            hull = ConvexHull(points)
            hulls.append(hull)

        return(hulls)


# MARK
    def compute_ill_posed(self, epsilon=1.5, relative_loss=False, loss_func=None, force_run=False, method="sample", num_runs=1, return_CI=False, confidence_interval=0.95, **kwargs):

        # Compute number of ill-posed solutions for this instance (for example: 5 possible diverse solutions within
        # threshold of spectral loss function)

        # Possible methods: "sample" (simple fraction), "estimate" (estimate #ip for all strata), "norm" (estimate, then return to normal range using power root)

        assert num_runs > 0
        assert confidence_interval > 0.0
        assert confidence_interval < 1.0
        assert method in ["sample", "estimate", "norm"]

        if(return_CI and num_runs < 2):
            raise AutoIPQError("Must perform multiple runs in order to return a confidence interval.")
        
        nums_viable = np.zeros(num_runs)

        for r in range(0, num_runs):
            if(force_run or not(self._run_state)):
                # (re-)run optimisation to generate solutions
                self.run(**kwargs)
            solutions = self.solutions

            losses = np.zeros(len(solutions))

            i = 0
            for s in solutions:
                sim_output = self.sim.simulate(parameters=s)

                loss = self.optimiser.loss_func(sim_output, self.x)

                losses[i] = loss
                i += 1

            if(relative_loss):
                # Recompute losses as proportion of best loss
                best_loss = np.min(losses)
                losses = np.absolute(losses - best_loss) / best_loss

            num_viable = len(losses[losses < epsilon])
            if(method == 'estimate' or method == 'norm'):
                # Account for number of strata explored compared to the number of solutions, to avoid getting more ill-posed with more priors (because
                # more solutions in set will be of higher quality). Does get rid of natural maximum.
                num_strata =  self.num_strata_per_dim ** len(self.stratified_parameters)
                # Number of possible maxima divided by sample size in strata, multiplied by num in sample to estimate number in all strata.
                num_viable = num_strata / len(solutions) * num_viable 
                if(method == 'norm'):
                    num_viable = num_viable ** (1/len(self.stratified_parameters))

            force_run = True # After first run, if we are doing more, we have to re-run every time

            nums_viable[r] = num_viable


        if(num_runs > 1):
            # Assume normality
            point_estimate = np.mean(nums_viable)
            std = np.std(nums_viable)
            ci = scipy.stats.norm.interval(confidence_interval, loc=point_estimate, scale=std/math.sqrt(len(nums_viable)))
            return(point_estimate, ci)
        else:
            return(nums_viable[0])

    

    def visualise_loss_landscape_multi(self, param1, param2, num_solutions, relative_loss=False, loss_func=None, num_repeats=5, **kwargs):

        # Very hardcoded right now just to check principle

        stratified_parameters = [param1, param2]
        #self.optimiser.free_parameters = stratified_parameters

        loss_matrix = np.zeros((self.num_strata_per_dim, self.num_strata_per_dim, num_solutions))

        current_solutions = [] # Add GO to this for every iteration

        for t in range(0, num_solutions):
            print("Iteration", t)

            p1s = np.ones((self.num_strata_per_dim, self.num_strata_per_dim)) * np.inf
            p2s = np.ones((self.num_strata_per_dim, self.num_strata_per_dim)) * np.inf

            for j1 in range(0, len(self.param_ranges_gridded[param1])-1):
                minval1 = self.param_ranges_gridded[param1][j1]
                maxval1 = self.param_ranges_gridded[param1][j1+1]

                self.optimiser.param_ranges[param1] = ["uniform", "float", minval1, maxval1]

                for j2 in range(0, len(self.param_ranges_gridded[param2])-1):
                    minval2 = self.param_ranges_gridded[param2][j2]
                    maxval2 = self.param_ranges_gridded[param2][j2+1]

                    self.optimiser.param_ranges[param2] = ["uniform", "float", minval2, maxval2]

                    total_loss = 0
                    if(stratified_parameters == self.optimiser.free_parameters):
                        num_repeats = 1

                    for _ in range(0, num_repeats):
                        local_optimum = self.optimiser.run_iterated(param_ranges_gridded=self.optimiser.param_ranges_gridded, current_solutions=current_solutions, format='dict', **kwargs)
                        p1s[j1, j2] = local_optimum[param1]
                        p2s[j1, j2] = local_optimum[param2]
                        #sim_output = self.sim.simulate(local_optimum)
                        #loss = self.optimiser.loss_func(sim_output, self.x)
                        loss = self.optimiser.best_performance
                        total_loss += loss

                    total_loss = total_loss / num_repeats
                    loss_matrix[j1, j2, t] = total_loss

            best_p1 = -1
            best_p2 = -1
            min_loss = np.min(loss_matrix[:, :, t])
            for j1 in range(0, loss_matrix.shape[0]):
                for j2 in range(0, loss_matrix.shape[1]):
                    if(loss_matrix[j1, j2, t] == min_loss):
                        best_p1 = p1s[j1, j2]
                        best_p2 = p2s[j1, j2]


            current_solutions.append({param1:best_p1, param2:best_p2})


        return(loss_matrix, current_solutions)



    def plot_landscape(self, param1, param2, num_repeats=1, real_values=False, res=10, **kwargs):

        # Very hardcoded right now just to check principle

        stratified_parameters = [param1, param2]
        #self.optimiser.free_parameters = stratified_parameters

        loss_matrix = np.zeros((res, res))
        # Real x/y as opposed to stratum id
        real_x = np.zeros_like(loss_matrix)
        real_y = np.zeros_like(loss_matrix)

        step_x = (self.param_ranges[param1][3] - self.param_ranges[param1][2]) / res
        step_y = (self.param_ranges[param2][3] - self.param_ranges[param2][2]) / res

        for j1 in range(res):
            # Forcing borders to take most extreme values for prettier plotting
            if(j1 == res-2):
                x_val = self.param_ranges[param1][3]
            else:
                x_val = self.param_ranges[param1][2] + j1*step_x

            for j2 in range(0, res):
                if(j2 == res-2):
                    y_val = self.param_ranges[param2][3]
                else:
                    y_val = self.param_ranges[param2][2] + j2*step_y


                losses = np.zeros(num_repeats)

                for k in range(0, num_repeats):


                    local_vals = {param1: x_val, param2: y_val}
                    out = self.optimiser.sim.simulate(local_vals)
                    loss = self.optimiser.f1(out, self.optimiser.real_output)
                    losses[k] = loss

                loss_matrix[j1, j2] = np.median(losses)
                real_x[j1, j2] = x_val
                real_y[j1, j2] = y_val

        if(real_values):
            res = {"x": real_x, "y": real_y, "loss": loss_matrix}
        else:
            res = loss_matrix

        return(res)


    def estimate_hyperplane_bruteforce(self, param1, param2, num_repeats=1, real_values=False, optimise_local=False, **kwargs):

        # Very hardcoded right now just to check principle

        stratified_parameters = [param1, param2]
        #self.optimiser.free_parameters = stratified_parameters

        loss_matrix = np.zeros((self.num_strata_per_dim, self.num_strata_per_dim))
        # Real x/y as opposed to stratum id
        real_x = np.zeros_like(loss_matrix)
        real_y = np.zeros_like(loss_matrix)

        for j1 in range(0, len(self.param_ranges_gridded[param1])-1):
            # Forcing borders to take most extreme values for prettier plotting
            if(j1 == len(self.param_ranges_gridded[param1])-2):
                minval1 = self.param_ranges[param1][3]
            else:
                minval1 = self.param_ranges_gridded[param1][j1]
            if(j1 == 0):
                maxval1 = self.param_ranges[param1][2]
            else:
                maxval1 = self.param_ranges_gridded[param1][j1+1]

            self.optimiser.param_ranges[param1] = ["uniform", "float", minval1, maxval1]

            for j2 in range(0, len(self.param_ranges_gridded[param2])-1):

                # Forcing borders to take most extreme values for prettier plotting
                if(j2 == len(self.param_ranges_gridded[param2])-2):
                    minval2 = self.param_ranges[param2][3]
                else:
                    minval2 = self.param_ranges_gridded[param2][j2]
                if(j2 == 0):
                    maxval2 = self.param_ranges[param2][2]
                else:
                    maxval2 = self.param_ranges_gridded[param2][j2+1]

                self.optimiser.param_ranges[param2] = ["uniform", "float", minval2, maxval2]

                total_loss = 0
                if(stratified_parameters == self.optimiser.free_parameters):
                    num_repeats = 1

                for _ in range(0, num_repeats):
                    if(optimise_local):
                        # Find the local optimum within this stratum and save it
                        local_vals = self.optimiser.run(format='dict', **kwargs)
                        loss = self.optimiser.best_performance
                    else:
                        # Just run for centre of stratum
                        avg_val_p1 = (maxval1 + minval1) / 2
                        avg_val_p2 = (maxval2 + minval2) / 2
                        local_vals = {param1: avg_val_p1, param2: avg_val_p2}
                        out = self.optimiser.sim.simulate(local_vals)
                        loss = self.optimiser.loss_func(out, self.optimiser.real_output)
                    #sim_output = self.sim.simulate(local_optimum)
                    #loss = self.optimiser.loss_func(sim_output, self.x)
                    total_loss += loss
                    if(real_values):
                        real_x[j1, j2] = real_x[j1, j2] + local_vals[param1]
                        real_y[j1, j2] = real_y[j1, j2] + local_vals[param2]
                total_loss = total_loss / num_repeats
                loss_matrix[j1, j2] = total_loss

                if(real_values):
                    # Not sure if it makes sense to average them but we don't iterate in experiments anymore anyway
                    real_x[j1, j2] = real_x[j1, j2] / num_repeats
                    real_y[j1, j2] = real_y[j1, j2] / num_repeats

        if(real_values):
            res = {"x": real_x, "y": real_y, "loss": loss_matrix}
        else:
            res = loss_matrix

        return(res)


    
# MARK
    def compute_uncertainty_old(self, solutions=None, method='fraction', correct_zero=True, invert=False, force_run=False, **kwargs):
        if(solutions is None):
            # Use object's own stored solutions if none explicitly specified
            if(force_run or not(self._run_state)):
                # (re-)run optimisation to generate solutions
                # Do not pass to compute_ill_posed since we don't want to run twice
                self.run(**kwargs)
            solutions = self.solutions

        num_viable = self.compute_ill_posed(solutions=solutions, **kwargs)
        # Make no solution highest uncertainty
        if(correct_zero):
            num_viable = num_viable - 1
            if(num_viable < 0):
                num_viable = len(solutions)

        uncertainty = -1
        if(method == 'fraction'):
            uncertainty = num_viable / len(solutions)
        else:
            raise AutoIPQError("Invalid method specified: " + str(method))

        if(invert):
            uncertainty = 1 - uncertainty

        return(uncertainty)


    def reset(self):
        self._run_state = False
        self.solutions = np.zeros((self.solution_set_size, len(self.free_parameters)))
        self.optimiser.reset()