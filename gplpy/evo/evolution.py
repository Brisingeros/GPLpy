"""Evolutionary Daemon and Executor with logging capacities"""

import math
from collections import namedtuple
from threading import Thread, Semaphore, Lock, Event
from enum import Enum
import numpy as np
from scipy.special import softmax
from gplpy.gggp.metaderivation import MetaDerivation, EDA
from gplpy.gggp.grammar import ProbabilisticModel
from gplpy.gggp.derivation import Derivation, OnePointMutation, WX

import pandas as pd
import os

import datetime


class Optimization(Enum):
    min = False
    max = True

class Problem(Thread):
    optimization = Optimization.min

    def __init(self):
        super().__init__()


class Individual:
    def __init__(self, derivation_init, problem, converged, fitness_args=(), max_recursions=100, derivation=None, 
                 async_learning=False, learning_tolerance_step=50, learning_tolerance=0.02, maturity_tolerance_factor=10):
        if derivation:
            self.derivation = derivation
        else:
            self.derivation = derivation_init(max_recursions)

        self._fitness = None
        self.learning_iterations = 0

        # Asyncronous learning atributes
        self._join = True
        self.async_learning = async_learning
        self.alive = Event()
        self.learn = Semaphore(1)
        self.mature = Event()
        self.no_learning = Event()
        self.converged = converged
        self.learning_tolerance_step = learning_tolerance_step
        self.learning_tolerance = learning_tolerance
        self.maturity_tolerance = learning_tolerance * maturity_tolerance_factor

        self.ecco_factor = 0
        self.problem = problem(self, fitness_args)
        
        self.no_learning.set()

    @property
    def fitness(self):
        # Wait until individual is mature to continue evolution
        self.mature.wait()
        self.no_learning.wait()
        # Start a cycle of asyncronous learning
        if self.async_learning:
            self.learn.release()
        return self._fitness

    def join(self):
        if self._join:
            self.problem.join()

    def start(self):
        self.alive.set()
        self.problem.start()
        if not self.async_learning:
            self._join = False
            del self.problem


class Evolution(object):
    def __init__(self, logger, grammar, setup, problem, fitness_args=()):

        # Grammar
        self.max_recursions = setup.max_recursions

        # Population
        self.population = []
        self.population_size = setup.population_size
       
        # Optimization
        self.generation = 0
        self.learning_iterations = 0
        self.optimization = problem.optimization
        self.tolerance_step = setup.tolerance_step
        self.tolerance = setup.tolerance

        # Local search
        self.async_learning = setup.async_learning
        self.learning_tolerance_step = setup.learning_tolerance_step
        self.learning_tolerance = setup.learning_tolerance
        self.maturity_tolerance_factor = setup.maturity_tolerance_factor
        self.converged = Event()

        # Fitness
        self.problem = problem
        self.fitness_args = fitness_args
        self.last_average_fitness = float("inf") if self.optimization == Optimization.min else float("-inf")
        self.last_tolerance_average_fitness = float("inf") if self.optimization == Optimization.min else float("-inf")

        # Crossover function
        self.x = setup.crossover.crossover
        if isinstance(setup.selection_rate,float):
            self.parents_pool_size = int(round(setup.population_size * setup.selection_rate))
        elif isinstance(setup.selection_rate, int):
            self.parents_pool_size = setup.selection_rate
        self.population_regeneration = None
        self.immigration_size = int(round(setup.population_size * setup.immigration_rate))
        self.logger = logger


   
    def population_control(self):
        # If population size below specified, insertion of new individuals
        while len(self.population) < self.population_size:
            individual = Individual(derivation_init=self.derivation_init,
                                    problem=self.problem,
                                    fitness_args=self.fitness_args,
                                    max_recursions=self.max_recursions,
                                    learning_tolerance_step=self.learning_tolerance_step,
                                    learning_tolerance=self.learning_tolerance,
                                    async_learning=self.async_learning,
                                    converged=self.converged,
                                    maturity_tolerance_factor=self.maturity_tolerance_factor)
       
            individual.start()
            self.population.append(individual)

        # Sorting population
        if self.optimization == Optimization.max:
            self.population.sort(key=lambda i: (-i.fitness, i.learning_iterations))
        else:
            self.population.sort(key=lambda i: (i.fitness, i.learning_iterations))

    def elitist_selection(self):
        return self.population[0:self.parents_pool_size]

    def tournament_selection(self, battle_size=2):
        # Random selection of 4 individuals of the population
        tournament = np.random.choice(self.population, size=self.parents_pool_size * battle_size, replace=False)
        # Tournament: 2 individuals battle
        if self.optimization == Optimization.min:
            return [min(tournament[i:i + battle_size], key=lambda x: x._fitness)
                    for i in range(0, len(tournament), battle_size)]
        else:
            return [max(tournament[i:i + battle_size], key=lambda x: x._fitness)
                    for i in range(0, len(tournament), battle_size)]

    def softmax_selection(self):
        fitness = np.array([i._fitness for i in self.population])
        fitness_std_1 = fitness/np.std(fitness)
        if self.optimization == Optimization.min:
            return np.random.choice(self.population,
                                    size=self.parents_pool_size,
                                    replace=False,
                                    p=softmax(np.abs(fitness_std_1-np.max(fitness_std_1))))
        else:
            return np.random.choice(self.population,
                                    size=self.parents_pool_size,
                                    replace=False,
                                    p=softmax(fitness_std_1-np.min(fitness_std_1)))

    def replacement(self, offspring):
        # Deletion of worst individuals
        if self.population_regeneration is None:
            self.population_regeneration = self.population_size * -1 if len(offspring) >= self.population_size else (self.immigration_size+len(offspring)) * -1
             
        for i in self.population[self.population_regeneration:]:
            # End learning
            i.alive.clear()
            self.learning_iterations += i.learning_iterations
            # Stop learning, let it die
            i.learn.release()
            i.join()
        del self.population[self.population_regeneration:]
        # Insertion of offspring
        self.population += offspring

    def converge(self):
        average_fitness = sum(i._fitness for i in self.population) / len(self.population)
        #Obtain improvement
        if self.optimization == Optimization.min:
            improvement = 1 - average_fitness / self.last_average_fitness
        else:
            if math.isinf(self.last_average_fitness):
                improvement = 1.0
            else:
                improvement = (average_fitness - self.last_average_fitness) / abs(self.last_average_fitness)
        self.last_average_fitness = average_fitness

        if self.logger:
            self.logger.log_evolution(best_fitness=float(self.population[0]._fitness),
                                      average_fitness=float(average_fitness),
                                      improvement=float(improvement))
        else:
            print(datetime.datetime.utcnow().strftime("%H:%M:%S"))
            print('Best individual fitness: %.2f, Average fitness: %.2f, Improvement: %.2f' % (self.population[0]._fitness, average_fitness, improvement))

        if self.optimization == Optimization.min:
            if self.population[0]._fitness == 0:
                return True

        if not self.generation % self.tolerance_step:
            if self.optimization == Optimization.min:
                tolerance_improvement = 1 - average_fitness / self.last_tolerance_average_fitness
            else:
                if math.isinf(self.last_tolerance_average_fitness):
                    tolerance_improvement = 1.0
                else:
                    tolerance_improvement = (average_fitness - self.last_tolerance_average_fitness) / abs(self.last_tolerance_average_fitness)
            if tolerance_improvement < self.tolerance:
                return True
            self.last_tolerance_average_fitness = average_fitness
        return False

    def finish(self):
        self.converged.set()

        for i in self.population:
            i.alive.clear()
            #Avoid individual getting stuck
            i.learn.release()
            i.join()
            self.learning_iterations += i.learning_iterations


        return self.population[0]._fitness, sum(i._fitness for i in self.population) / len(self.population), self.generation, self.learning_iterations, self.population[0].finalDerivation


class Evolution_EDA(Evolution):
    def __init__(self, logger, grammar, setup, problem, fitness_args=()):
        super().__init__(logger, grammar, setup, problem, fitness_args)
        
        self.meta_derivation = MetaDerivation(grammar=grammar,
                                              max_recursions=setup.max_recursions,
                                              probabilistic_model=setup.probabilistic_model,
                                              model_update_rate = setup.model_update_rate)
        
        self.exploration_rate = setup.exploration_rate
        self.offspring_size = int(round(setup.population_size * setup.offspring_rate))
        self.derivation_init = self.meta_derivation.new_derivation

    def evolve(self):
        self.population_control()

        while not self.converge():
            self.generation += 1
            # Evolution
            self.replacement(self.crossover(self.tournament_selection()))
            # Filling up population (population initialization or immigrant population) and sorting
            self.population_control()

        return self.finish()

    def crossover(self, parents):
        offspring_derivations = self.x(meta_derivation=self.meta_derivation,
                                       derivation_trees=[p.derivation.tree for p in parents],
                                       offspring_size=self.offspring_size,
                                       max_recursions=self.max_recursions,
                                       exploration_rate=self.exploration_rate)

        offspring = []
        for d in offspring_derivations:
            individual = Individual(derivation_init=self.derivation_init,
                                    problem=self.problem,
                                    fitness_args=self.fitness_args,
                                    max_recursions=self.max_recursions,
                                    derivation=d,
                                    learning_tolerance_step=self.learning_tolerance_step,
                                    learning_tolerance=self.learning_tolerance,
                                    maturity_tolerance_factor=self.maturity_tolerance_factor,
                                    async_learning=self.async_learning,
                                    converged=self.converged)
            individual.start()
            offspring.append(individual)
        #print(len(offspring[0].word), len(offspring[1].word))
        #print("Target:  " + str(len(self.fitness_args)))

        return offspring


class Evolution_EDA_ECO(Evolution_EDA):
    def __init__(self, logger, grammar, setup, problem, fitness_args=()):
        super().__init__(logger, grammar, setup, problem, fitness_args)

        self.dataframe = []

    def converge(self):
        average_fitness = sum(i._fitness for i in self.population[:(len(self.population)//3)]) / (len(self.population)//3)
        #Obtain improvement
        if self.optimization == Optimization.min:
            improvement = 1 - average_fitness / self.last_average_fitness
        else:
            if math.isinf(self.last_average_fitness):
                improvement = 1.0
            else:
                improvement = (average_fitness - self.last_average_fitness) / abs(self.last_average_fitness)
        self.last_average_fitness = average_fitness

        if self.logger:
            self.logger.log_evolution(best_fitness=float(self.population[0]._fitness),
                                      average_fitness=float(average_fitness),
                                      improvement=float(improvement))
        else:
            self.dataframe.append([self.population[0]._fitness, average_fitness, improvement, 0.0])
            print(datetime.datetime.utcnow().strftime("%H:%M:%S"))
            print('%d, Best individual fitness: %.2f, Average 1/3 fitness: %.2f, Improvement: %.2f' % (len(self.dataframe)-1, self.population[0]._fitness, average_fitness, improvement))

        if self.optimization == Optimization.min:
            if self.population[0]._fitness == 0:
                return True

        if not self.generation % self.tolerance_step:
            if self.optimization == Optimization.min:
                average_fitness = min([i[1] for i in self.dataframe])
                tolerance_improvement = 1 - average_fitness / self.last_tolerance_average_fitness
            else:
                average_fitness = max([i[1] for i in self.dataframe])
                if math.isinf(self.last_tolerance_average_fitness):
                    tolerance_improvement = 1.0
                else:
                    tolerance_improvement = (average_fitness - self.last_tolerance_average_fitness) / abs(self.last_tolerance_average_fitness)
            if tolerance_improvement < self.tolerance:
                return True
            self.last_tolerance_average_fitness = average_fitness
        return False

    def finish(self):
        self.converged.set()

        for i in self.population:
            i.alive.clear()
            #Avoid individual getting stuck
            i.learn.release()
            i.join()
            self.learning_iterations += i.learning_iterations

        
        average_fitness = sum(i._fitness for i in self.population[:(len(self.population)//3)]) / (len(self.population)//3)
        self.dataframe.append([self.population[0]._fitness, average_fitness, 0.0, self.learning_iterations])

        return self.population[0]._fitness, sum(i._fitness for i in self.population) / len(self.population), self.generation, self.learning_iterations, self.population[0].finalDerivation, self.dataframe


class Evolution_EDA_LPV_ECO(Evolution):
    def __init__(self, logger, grammar, setup, problem, fitness_args=()):
        super().__init__(logger, grammar, setup, problem, fitness_args)
        
        self.meta_derivation = MetaDerivation(grammar=grammar,
                                              max_recursions=setup.max_recursions,
                                              probabilistic_model=setup.probabilistic_model,
                                              model_update_rate = setup.model_update_rate)
        
        self.exploration_rate = setup.exploration_rate
        self.offspring_size = int(round(setup.population_size * setup.offspring_rate))
        self.derivation_init = self.meta_derivation.new_derivation
        
        self.dataframe = []

    def population_control(self):
        super().population_control()
        best = self.population[0]

        ######################
        def ecco_calc(indiv, best_indiv):
            if hasattr(indiv, 'finalDerivation') or (indiv.learning_iterations >= best_indiv.learning_iterations):
                indiv.ecco_factor = 0
                return indiv

            fitnesses = [indiv._fitness, best_indiv._fitness]
            fitnesses.sort(reverse=True)

            indiv.ecco_factor = min([((indiv.learning_iterations - best_indiv.learning_iterations) / (best_indiv.learning_iterations)) * ((fitnesses[0] - fitnesses[1])), 0])
            
            return indiv

        self.population = [ecco_calc(i, best) for i in self.population]

    def tournament_selection(self, battle_size=2):
        # Random selection of 4 individuals of the population
        tournament = np.random.choice(self.population, size=self.parents_pool_size * battle_size, replace=False)
        # Tournament: 2 individuals battle
        if self.optimization == Optimization.min:
            return [min(tournament[i:i + battle_size], key=lambda x: (x._fitness - x.ecco_factor))
                    for i in range(0, len(tournament), battle_size)]
        else:
            return [max(tournament[i:i + battle_size], key=lambda x: (x._fitness + x.ecco_factor))
                    for i in range(0, len(tournament), battle_size)]

    def replacement(self, offspring):
        if self.optimization == Optimization.max:
            self.population.sort(key=lambda i: (-(i._fitness - i.ecco_factor), i.learning_iterations))
        else:
            self.population.sort(key=lambda i: ((i._fitness + i.ecco_factor), i.learning_iterations))

        return super().replacement(offspring)

    def evolve(self):
        self.population_control()

        while not self.converge():
            self.generation += 1
            # Evolution
            self.replacement(self.crossover(self.tournament_selection()))
            # Filling up population (population initialization or immigrant population) and sorting
            self.population_control()

        return self.finish()

    def crossover(self, parents):
        offspring_derivations = self.x(meta_derivation=self.meta_derivation,
                                       derivation_trees=[p.derivation.tree for p in parents],
                                       offspring_size=self.offspring_size,
                                       max_recursions=self.max_recursions,
                                       exploration_rate=self.exploration_rate)

        offspring = []
        for d in offspring_derivations:
            individual = Individual(derivation_init=self.derivation_init,
                                    problem=self.problem,
                                    fitness_args=self.fitness_args,
                                    max_recursions=self.max_recursions,
                                    derivation=d,
                                    learning_tolerance_step=self.learning_tolerance_step,
                                    learning_tolerance=self.learning_tolerance,
                                    maturity_tolerance_factor=self.maturity_tolerance_factor,
                                    async_learning=self.async_learning,
                                    converged=self.converged)
            individual.start()
            offspring.append(individual)
        #print(len(offspring[0].word), len(offspring[1].word))
        #print("Target:  " + str(len(self.fitness_args)))

        return offspring

    def converge(self):
        average_fitness = sum(i._fitness for i in self.population[:(len(self.population)//3)]) / (len(self.population)//3)
        #Obtain improvement
        if self.optimization == Optimization.min:
            improvement = 1 - average_fitness / self.last_average_fitness
        else:
            if math.isinf(self.last_average_fitness):
                improvement = 1.0
            else:
                improvement = (average_fitness - self.last_average_fitness) / abs(self.last_average_fitness)
        self.last_average_fitness = average_fitness

        if self.logger:
            self.logger.log_evolution(best_fitness=float(self.population[0]._fitness),
                                      average_fitness=float(average_fitness),
                                      improvement=float(improvement))
        else:
            self.dataframe.append([self.population[0]._fitness, average_fitness, improvement, 0.0])
            print(datetime.datetime.utcnow().strftime("%H:%M:%S"))
            print('%d, Best individual fitness: %.2f, Average 1/3 fitness: %.2f, Improvement: %.2f' % (len(self.dataframe)-1, self.population[0]._fitness, average_fitness, improvement))

        if self.optimization == Optimization.min:
            if self.population[0]._fitness == 0:
                return True

        if not self.generation % self.tolerance_step:
            if self.optimization == Optimization.min:
                average_fitness = min([i[1] for i in self.dataframe])
                tolerance_improvement = 1 - average_fitness / self.last_tolerance_average_fitness
            else:
                average_fitness = max([i[1] for i in self.dataframe])
                if math.isinf(self.last_tolerance_average_fitness):
                    tolerance_improvement = 1.0
                else:
                    tolerance_improvement = (average_fitness - self.last_tolerance_average_fitness) / abs(self.last_tolerance_average_fitness)
            if tolerance_improvement < self.tolerance:
                return True
            self.last_tolerance_average_fitness = average_fitness
        return False


    def finish(self):
        self.converged.set()

        for i in self.population:
            i.alive.clear()
            #Avoid individual getting stuck
            i.learn.release()
            i.join()
            self.learning_iterations += i.learning_iterations

        
        average_fitness = sum(i._fitness for i in self.population[:(len(self.population)//3)]) / (len(self.population)//3)
        self.dataframe.append([self.population[0]._fitness, average_fitness, 0.0, self.learning_iterations])
        
        return self.population[0]._fitness, sum(i._fitness for i in self.population) / len(self.population), self.generation, self.learning_iterations, self.population[0].finalDerivation, self.dataframe


class Evolution_EDA_LP_ECO(Evolution):
    def __init__(self, logger, grammar, setup, problem, fitness_args=()):
        super().__init__(logger, grammar, setup, problem, fitness_args)
        
        self.meta_derivation = MetaDerivation(grammar=grammar,
                                              max_recursions=setup.max_recursions,
                                              probabilistic_model=setup.probabilistic_model,
                                              model_update_rate = setup.model_update_rate)
        
        self.exploration_rate = setup.exploration_rate
        self.offspring_size = int(round(setup.population_size * setup.offspring_rate))
        self.derivation_init = self.meta_derivation.new_derivation
        
        self.dataframe = []

    def population_control(self):
        super().population_control()
        best = self.population[0]

        ######################
        def ecco_calc(indiv, best_indiv):
            if hasattr(indiv, 'finalDerivation') or (indiv.learning_iterations >= best_indiv.learning_iterations):
                indiv.ecco_factor = 0
                return indiv

            fitnesses = [indiv._fitness, best_indiv._fitness]
            fitnesses.sort(reverse=True)

            indiv.ecco_factor = min([((indiv.learning_iterations - best_indiv.learning_iterations) / (best_indiv.learning_iterations)) * ((fitnesses[0] - fitnesses[1])), 0])
            
            return indiv

        self.population = [ecco_calc(i, best) for i in self.population]

    def tournament_selection(self, battle_size=2):
        # Random selection of 4 individuals of the population
        tournament = np.random.choice(self.population, size=self.parents_pool_size * battle_size, replace=False)
        # Tournament: 2 individuals battle
        if self.optimization == Optimization.min:
            return [min(tournament[i:i + battle_size], key=lambda x: (x._fitness))
                    for i in range(0, len(tournament), battle_size)]
        else:
            return [max(tournament[i:i + battle_size], key=lambda x: (x._fitness))
                    for i in range(0, len(tournament), battle_size)]

    def replacement(self, offspring):
        if self.optimization == Optimization.max:
            self.population.sort(key=lambda i: (-(i._fitness - i.ecco_factor), i.learning_iterations))
        else:
            self.population.sort(key=lambda i: ((i._fitness + i.ecco_factor), i.learning_iterations))

        return super().replacement(offspring)

    def evolve(self):
        self.population_control()

        while not self.converge():
            self.generation += 1
            # Evolution
            self.replacement(self.crossover(self.tournament_selection()))
            # Filling up population (population initialization or immigrant population) and sorting
            self.population_control()

        return self.finish()

    def crossover(self, parents):
        offspring_derivations = self.x(meta_derivation=self.meta_derivation,
                                       derivation_trees=[p.derivation.tree for p in parents],
                                       offspring_size=self.offspring_size,
                                       max_recursions=self.max_recursions,
                                       exploration_rate=self.exploration_rate)

        offspring = []
        for d in offspring_derivations:
            individual = Individual(derivation_init=self.derivation_init,
                                    problem=self.problem,
                                    fitness_args=self.fitness_args,
                                    max_recursions=self.max_recursions,
                                    derivation=d,
                                    learning_tolerance_step=self.learning_tolerance_step,
                                    learning_tolerance=self.learning_tolerance,
                                    maturity_tolerance_factor=self.maturity_tolerance_factor,
                                    async_learning=self.async_learning,
                                    converged=self.converged)
            individual.start()
            offspring.append(individual)
        #print(len(offspring[0].word), len(offspring[1].word))
        #print("Target:  " + str(len(self.fitness_args)))

        return offspring

    def converge(self):
        average_fitness = sum(i._fitness for i in self.population[:(len(self.population)//3)]) / (len(self.population)//3)
        #Obtain improvement
        if self.optimization == Optimization.min:
            improvement = 1 - average_fitness / self.last_average_fitness
        else:
            if math.isinf(self.last_average_fitness):
                improvement = 1.0
            else:
                improvement = (average_fitness - self.last_average_fitness) / abs(self.last_average_fitness)
        self.last_average_fitness = average_fitness

        if self.logger:
            self.logger.log_evolution(best_fitness=float(self.population[0]._fitness),
                                      average_fitness=float(average_fitness),
                                      improvement=float(improvement))
        else:
            self.dataframe.append([self.population[0]._fitness, average_fitness, improvement, 0.0])
            print(datetime.datetime.utcnow().strftime("%H:%M:%S"))
            print('%d, Best individual fitness: %.2f, Average 1/3 fitness: %.2f, Improvement: %.2f' % (len(self.dataframe)-1, self.population[0]._fitness, average_fitness, improvement))

        if self.optimization == Optimization.min:
            if self.population[0]._fitness == 0:
                return True

        if not self.generation % self.tolerance_step:
            if self.optimization == Optimization.min:
                average_fitness = min([i[1] for i in self.dataframe])
                tolerance_improvement = 1 - average_fitness / self.last_tolerance_average_fitness
            else:
                average_fitness = max([i[1] for i in self.dataframe])
                if math.isinf(self.last_tolerance_average_fitness):
                    tolerance_improvement = 1.0
                else:
                    tolerance_improvement = (average_fitness - self.last_tolerance_average_fitness) / abs(self.last_tolerance_average_fitness)
            if tolerance_improvement < self.tolerance:
                return True
            self.last_tolerance_average_fitness = average_fitness
        return False


    def finish(self):
        self.converged.set()

        for i in self.population:
            i.alive.clear()
            #Avoid individual getting stuck
            i.learn.release()
            i.join()
            self.learning_iterations += i.learning_iterations

        
        average_fitness = sum(i._fitness for i in self.population[:(len(self.population)//3)]) / (len(self.population)//3)
        self.dataframe.append([self.population[0]._fitness, average_fitness, 0.0, self.learning_iterations])
        
        return self.population[0]._fitness, sum(i._fitness for i in self.population) / len(self.population), self.generation, self.learning_iterations, self.population[0].finalDerivation, self.dataframe


class Evolution_EDA_FP_ECO(Evolution):
    def __init__(self, logger, grammar, setup, problem, fitness_args=()):
        super().__init__(logger, grammar, setup, problem, fitness_args)
        
        self.meta_derivation = MetaDerivation(grammar=grammar,
                                              max_recursions=setup.max_recursions,
                                              probabilistic_model=setup.probabilistic_model,
                                              model_update_rate = setup.model_update_rate)
        
        self.exploration_rate = setup.exploration_rate
        self.offspring_size = int(round(setup.population_size * setup.offspring_rate))
        self.derivation_init = self.meta_derivation.new_derivation
        
        self.dataframe = []

    def population_control(self):
        super().population_control()

    def tournament_selection(self, battle_size=2):
        # Random selection of 4 individuals of the population
        tournament = np.random.choice(self.population, size=self.parents_pool_size * battle_size, replace=False)
        # Tournament: 2 individuals battle
        if self.optimization == Optimization.min:
            return [min(tournament[i:i + battle_size], key=lambda x: (x._fitness))
                    for i in range(0, len(tournament), battle_size)]
        else:
            return [max(tournament[i:i + battle_size], key=lambda x: (x._fitness))
                    for i in range(0, len(tournament), battle_size)]

    def replacement(self, offspring):
        if self.optimization == Optimization.max:
            self.population.sort(key=lambda i: (-(i._fitness + ( (1 / ((0.05*i.learning_iterations)+1.0) ) * (i._fitness - best._fitness))), i.learning_iterations))
        else:
            best = self.population[0]
            self.population.sort(key=lambda i: ((i._fitness - ( (1 / ((0.05*i.learning_iterations)+1.0) ) * (i._fitness - best._fitness))), i.learning_iterations))

        return super().replacement(offspring)

    def evolve(self):
        self.population_control()

        while not self.converge():
            self.generation += 1
            # Evolution
            self.replacement(self.crossover(self.tournament_selection()))
            # Filling up population (population initialization or immigrant population) and sorting
            self.population_control()

        return self.finish()

    def crossover(self, parents):
        offspring_derivations = self.x(meta_derivation=self.meta_derivation,
                                       derivation_trees=[p.derivation.tree for p in parents],
                                       offspring_size=self.offspring_size,
                                       max_recursions=self.max_recursions,
                                       exploration_rate=self.exploration_rate)

        offspring = []
        for d in offspring_derivations:
            individual = Individual(derivation_init=self.derivation_init,
                                    problem=self.problem,
                                    fitness_args=self.fitness_args,
                                    max_recursions=self.max_recursions,
                                    derivation=d,
                                    learning_tolerance_step=self.learning_tolerance_step,
                                    learning_tolerance=self.learning_tolerance,
                                    maturity_tolerance_factor=self.maturity_tolerance_factor,
                                    async_learning=self.async_learning,
                                    converged=self.converged)
            individual.start()
            offspring.append(individual)
        #print(len(offspring[0].word), len(offspring[1].word))
        #print("Target:  " + str(len(self.fitness_args)))

        return offspring

    def converge(self):
        average_fitness = sum(i._fitness for i in self.population[:(len(self.population)//3)]) / (len(self.population)//3)
        #Obtain improvement
        if self.optimization == Optimization.min:
            improvement = 1 - average_fitness / self.last_average_fitness
        else:
            if math.isinf(self.last_average_fitness):
                improvement = 1.0
            else:
                improvement = (average_fitness - self.last_average_fitness) / abs(self.last_average_fitness)
        self.last_average_fitness = average_fitness

        if self.logger:
            self.logger.log_evolution(best_fitness=float(self.population[0]._fitness),
                                      average_fitness=float(average_fitness),
                                      improvement=float(improvement))
        else:
            self.dataframe.append([self.population[0]._fitness, average_fitness, improvement, 0.0])
            print(datetime.datetime.utcnow().strftime("%H:%M:%S"))
            print('%d, Best individual fitness: %.2f, Average 1/3 fitness: %.2f, Improvement: %.2f' % (len(self.dataframe)-1, self.population[0]._fitness, average_fitness, improvement))

        if self.optimization == Optimization.min:
            if self.population[0]._fitness == 0:
                return True

        if not self.generation % self.tolerance_step:
            if self.optimization == Optimization.min:
                average_fitness = min([i[1] for i in self.dataframe])
                tolerance_improvement = 1 - average_fitness / self.last_tolerance_average_fitness
            else:
                average_fitness = max([i[1] for i in self.dataframe])
                if math.isinf(self.last_tolerance_average_fitness):
                    tolerance_improvement = 1.0
                else:
                    tolerance_improvement = (average_fitness - self.last_tolerance_average_fitness) / abs(self.last_tolerance_average_fitness)
            if tolerance_improvement < self.tolerance:
                return True
            self.last_tolerance_average_fitness = average_fitness
        return False


    def finish(self):
        self.converged.set()

        for i in self.population:
            i.alive.clear()
            #Avoid individual getting stuck
            i.learn.release()
            i.join()
            self.learning_iterations += i.learning_iterations

        
        average_fitness = sum(i._fitness for i in self.population[:(len(self.population)//3)]) / (len(self.population)//3)
        self.dataframe.append([self.population[0]._fitness, average_fitness, 0.0, self.learning_iterations])
        
        return self.population[0]._fitness, sum(i._fitness for i in self.population) / len(self.population), self.generation, self.learning_iterations, self.population[0].finalDerivation, self.dataframe


class Evolution_WX(Evolution):
    def __init__(self, logger, grammar, setup, problem, fitness_args=()):
        super().__init__(logger, grammar, setup, problem, fitness_args)
        
        self.derivation_init = Derivation
        Derivation.grammar = grammar
        Derivation.probabilistic_model = setup.probabilistic_model
        self.mutate = setup.mutation.mutate
        self.mutation_rate = setup.mutation_rate
        self.parents_pool_size = 2

    def evolve(self):
        self.population_control()

        while not self.converge():
            self.generation += 1
            # Evolution
            self.replacement(self.mutate(self.crossover(self.tournament_selection()), self.mutation_rate))
            # Filling up population (population initialization or immigrant population) and sorting
            self.population_control()
        
        return self.finish()

    def crossover(self, parents):
        offspring = [Individual(derivation_init=self.derivation_init,
                                    problem=self.problem,
                                    fitness_args=self.fitness_args,
                                    max_recursions=self.max_recursions,
                                    derivation=d,
                                    learning_tolerance_step=self.learning_tolerance_step,
                                    learning_tolerance=self.learning_tolerance,
                                    maturity_tolerance_factor=self.maturity_tolerance_factor,
                                    async_learning=self.async_learning,
                                    converged=self.converged)
                     for d in self.x(derivations=[x.derivation for x in parents],
                                     max_recursions=self.max_recursions)]
        for o in offspring:
            o.start()
        for o in offspring:
            o.join()

        return offspring

Setup = namedtuple('Setup', 'name evolution max_recursions probabilistic_model crossover mutation mutation_rate exploration_rate model_update_rate population_size selection_rate offspring_rate immigration_rate tolerance_step tolerance async_learning learning_tolerance_step learning_tolerance maturity_tolerance_factor')
Setup.__new__.__defaults__ = ('GUPI+ED', Evolution_EDA, 100, ProbabilisticModel.uniform, EDA, OnePointMutation, .05, 0.001, .5, 100, 0.5, 0.25, 0.0, 10, .02, False, 10, 0.02, 10)


class Experiment:
    def __init__(self, study, experiment, grammar, problem, fitness_args=(), logger=None, setups=Setup(), samples=100):
        self.study = study
        self.experiment = experiment
        self.grammar = grammar

        self.logger = logger
        self.problem = problem
        self.fitness_args = fitness_args
        self.setups = setups
        self.samples = samples

    def run(self):
        experiments_id = []
        for s in self.setups:
            if self.logger:
                e_id = self.logger.new_experiment(name=self.experiment,
                                               setup_name=s.name,
                                               initialization="GUPI" if s.probabilistic_model is ProbabilisticModel.uniform else "Regular",
                                               crossover=s.crossover.__name__)
                experiments_id.append(e_id)
            else:
                print("Experiment: " + str(self.experiment))
                print("setup_name: " + s.name)
                print("initialization: " + s.probabilistic_model.name)
                print("crossover: " + s.crossover.__name__)

            for x in range(0, self.samples):
                if self.logger:
                    self.logger.new_sample(x)
                e = s.evolution(grammar=self.grammar,
                                logger=self.logger,
                                problem=self.problem,
                                fitness_args=self.fitness_args,
                                setup = s, 
                               )

                fit, avg_git, it, l_it, final_derivation, df = e.evolve()

                # #######################
                # f = open(f'{str(self.experiment)}_{s.name}_{s.maturity_tolerance_factor}_{x}.py', "w")
                # f.write(final_derivation)
                # f.close()
                # #######################

                (pd.DataFrame(df, columns=['best_fitness', 'avg_3_fitness', 'improvement', 'learning_iterations'])).to_csv(f'{os.getcwd()}\EXP\{s.maturity_tolerance_factor}\{s.name}_{x}.csv')

                #######################

                if self.logger:
                    self.logger.log_experiment(float(fit), it, l_it)
                else:
                    print("Sample: %d - Best fitness %.2f - Avg. fitness %.2f - Iterations %d - Learning iterations %d" % (x, fit, avg_git, it, l_it))
        if self.logger:
            self.logger.obtain_statistics(self.experiment)
        return experiments_id


__author__ = "aturing"
__license__ = "Apache License 2.0"
__version__ = "1.1.0"
__maintainer__ = "Pablo Ramos"
__email__ = "pablo.ramos@aturing.com"
__status__ = "Production"
