import random
import numpy as np
from deap import base, creator
from deap import tools
import fitness_function as ff


class FeatureSelectionGA:

    def __init__(self,model,x,y,cv_split=5,verbose=0):

        self.model =  model
        self.n_features = x.shape[1]
        self.toolbox = None
        self.creator = self._create()
        self.cv_split = cv_split
        self.x = x
        self.y = y
        self.verbose = verbose
        if self.verbose==1:
            print("Model {} will select best features among {} features using cv_split :{}.".format(model,x.shape[1],cv_split))
            print("Shape of train_x: {} and target: {}".format(x.shape,y.shape))
        self.final_fitness = []
        self.fitness_in_generation = {}
        self.best_ind = None
    
    def evaluate(self,individual):
        fit_obj = ff.FitenessFunction(self.cv_split)
        np_ind = np.asarray(individual)
        if np.sum(np_ind) == 0:
            fitness = 0.0
        else:
            feature_idx = np.where(np_ind==1)[0]
            fitness = fit_obj.calculate_fitness(self.model,self.x[:,feature_idx],self.y)
        
        if self.verbose == 1:
            print("Individual: {}  Fitness_score: {} ".format(individual,fitness))
            
        return fitness,
    
    
    def _create(self):
        creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FeatureSelect)
        return creator

        
    def register_toolbox(self,toolbox):
        toolbox.register("evaluate", self.evaluate)
        self.toolbox = toolbox
     
    
    def _init_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox
        
        
    def _default_toolbox(self):
        toolbox = self._init_toolbox()
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        return toolbox
    
    def get_final_scores(self,pop,fits):
        self.final_fitness = list(zip(pop,fits))
        
    
        
    def generate(self,n_pop,cxpb = 0.9,mutxpb = 0.01,ngen=5,set_toolbox = False):
        
        if self.verbose==1:
            print("Population: {}, crossover_probablity: {}, mutation_probablity: {}, total generations: {}".format(n_pop,cxpb,mutxpb,ngen))

        self.toolbox = self._default_toolbox()
        pop = self.toolbox.population(n_pop)
        CXPB, MUTPB, NGEN = cxpb,mutxpb,ngen

        # Evaluate the entire population
        print("EVOLVING.......")
        fitnesses = list(map(self.toolbox.evaluate, pop))
        
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            print("-- GENERATION {} --".format(g+1))
            offspring = self.toolbox.select(pop, len(pop))
            self.fitness_in_generation[str(g+1)] = max([ind.fitness.values[0] for ind in pop])
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            weak_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, weak_ind))
            for ind, fit in zip(weak_ind, fitnesses):
                ind.fitness.values = fit
            print("Evaluated %i individuals" % len(weak_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring
            
                    # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        if self.verbose==1:
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)


        self.best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values))
        self.get_final_scores(pop,fits)
        
        return pop,self.best_ind,self.best_ind.fitness.values


# model = sklearn.linear_model.LogisticRegression(solver='liblinear')
# model = svm.LinearSVC()
# iris = load_iris()
# x_train = iris.data
# y_train = iris.target
# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')
#     fsga = FeatureSelectionGA(model,x_train,y_train)
#     pop = fsga.generate(100)
#     print(pop)
    
    
