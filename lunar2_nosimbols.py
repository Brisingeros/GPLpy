""" TODO """
import math
import sys
import os
from gplpy.gggp.grammar import CFG
from gplpy.evo.evolution import Experiment, Optimization, Setup, Problem, Evolution_EDA_LP_ECO
from gplpy.gggp.metaderivation import EDA

import gym

from GA import GA

randomSeed = 9756745635

# SETUP EXAMPLES ###########################################################
EDX_setup = Setup(name='LP_ECO', evolution=Evolution_EDA_LP_ECO, max_recursions = 20, crossover=EDA, selection_rate=0.5, exploration_rate=0.001, model_update_rate=.5, offspring_rate=.25, immigration_rate=.15, population_size=20, tolerance_step=10, learning_tolerance_step=10, learning_tolerance=0.005, maturity_tolerance_factor=50, async_learning=True, tolerance=0.02)
# EDX_setup = Setup(name='LPV_ECO', evolution=Evolution_EDA_LPV_ECO, max_recursions = 20, crossover=EDA, selection_rate=0.5, exploration_rate=0.001, model_update_rate=.5, offspring_rate=.25, immigration_rate=.15, population_size=20, tolerance_step=10, learning_tolerance_step=10, learning_tolerance=0.005, maturity_tolerance_factor=50, async_learning=True, tolerance=0.02)
# EDX_setup = Setup(name='FP_ECO', evolution=Evolution_EDA_FP_ECO, max_recursions = 20, crossover=EDA, selection_rate=0.5, exploration_rate=0.001, model_update_rate=.5, offspring_rate=.25, immigration_rate=.15, population_size=20, tolerance_step=10, learning_tolerance_step=10, learning_tolerance=0.005, maturity_tolerance_factor=50, async_learning=True, tolerance=0.02)
# EDX_setup = Setup(name='ECO', evolution=Evolution_EDA_ECO, max_recursions = 20, crossover=EDA, selection_rate=0.5, exploration_rate=0.001, model_update_rate=.5, offspring_rate=.25, immigration_rate=.15, population_size=20, tolerance_step=10, learning_tolerance_step=10, learning_tolerance=0.005, maturity_tolerance_factor=50, async_learning=True, tolerance=0.02)

def ft(phenotype, auxData):
    derivation, env = auxData
    env.seed(randomSeed)

    #################################

    fitness = 0
    for _ in range(5):
        observation = env.reset()
        done = False
        while not done:
            inLocals = {
                'observation': observation,
                'input': [0., 0.],
                'gene': phenotype
            }
            exec(derivation, globals(), inLocals)

            observation, reward, done, info = env.step(inLocals['input'])
            fitness += reward

    fitness = (3000 - fitness) / 10.0

    return fitness


class LUNAR(Problem):
    optimization = Optimization.min

    def __init__(self, individual, args):
        super().__init__()
        self.individual = individual
        self.derivation = str(individual.derivation)
        self.env = gym.make('LunarLanderContinuous-v2')

        #Aquí realizamos las transformaciones necesarias al individuo (derivation)
        self.geneRange = []

        auxder = self.derivation.split('FLOAT')
        self.I = (len(auxder) - 1)
        derivationChanges = [f'{a}gene[{i}]' for i, a in enumerate(auxder[:-1])]
        derivationChanges.append(auxder[-1])
        derivationChanges = "".join(derivationChanges)

        auxder = derivationChanges.split('INDEX')
        self.N = (len(auxder) - 1)
        derivationChanges = [f'{a}gene[{i+self.I}]' for i, a in enumerate(auxder[:-1])]
        derivationChanges.append(auxder[-1])
        derivationChanges = "".join(derivationChanges)

        derivationChanges = f'input = [max(min(i,1.0),-1.0) for i in {derivationChanges}]'

        self.geneRange = ([[-100.0,100.0]] * self.I) + ([[0,7]] * self.N)

        self.GA = GA(popSize=30, geneRange=self.geneRange, replacementDegree=0.5, stepSize=50, fitnessFunc= ft, auxData=(derivationChanges, self.env), minimize=True, debug=False)

        ################################

    def run(self):
        self.individual.learn.acquire()
        self.individual.no_learning.clear()

        improvement = float("inf")
        while self.individual.alive.is_set() and improvement > self.individual.learning_tolerance:
            # Check maturity
            if self.individual.async_learning:

                if not self.individual.mature.is_set():
                    if improvement < self.individual.maturity_tolerance:
                        self.individual.mature.set()
                elif not self.individual.converged.is_set():
                    # Wait until individuals release to learn
                    self.individual.no_learning.set()
                    self.individual.learn.acquire()
                    self.individual.no_learning.clear()

            #Learn and obtain score
            bestIndiv, fitness_hist, fitness_mean_hist = self.GA.execute(self.individual.learning_tolerance_step)
            self.individual._fitness = bestIndiv.fitness
            self.individual.learning_iterations += self.individual.learning_tolerance_step

            if math.isinf(improvement):
                improvement = 1.0
            else:
                improvement = 1 - fitness_mean_hist[-1] / fitness_mean_hist[-(self.individual.learning_tolerance_step + 1)]

        self.env.close()

        ################################
        
        self.individual.mature.set()
        self.individual.no_learning.set()

        ################################
        
        # phenotype = bestIndiv.phenotype(self.geneRange)

        # derivation = self.derivation.split('FLOAT')
        # derivationChanges = [f'{a}{str(phenotype[i])}' for i, a in enumerate(derivation[:-1])]
        # derivationChanges.append(derivation[-1])
        # derivation = "".join(derivationChanges)

        # derivation = derivation.split('INDEX')
        # derivationChanges = [f'{a}{str(phenotype[i+self.I])}' for i, a in enumerate(derivation[:-1])]
        # derivationChanges.append(derivation[-1])
        # derivation = "".join(derivationChanges)

        # self.individual.finalDerivation = derivation

        self.individual.finalDerivation = "EXP"

        ################################

        #individual.raised.set()
        return

if __name__ == "__main__":
    sys.setrecursionlimit(10000)

    ## IS 
    study = "lunar2 controller"
    study_id = None

    # Grammar initialization
    grammar_file = os.getcwd() + '/gr/' + study.replace(' ', '_') + '.gr'
    gr = CFG(grammar_file)

    # logger initialization
    logger = False

    # Setup problem

    exp_name = "Lunar2"
    args = ()

    # Run
    samples = 1
    #, EDA_setup, WX_setup
    ids =Experiment(study=study, experiment=exp_name, grammar=gr, problem=LUNAR, fitness_args=args,
                    setups=[EDX_setup], logger=logger, samples=samples).run()