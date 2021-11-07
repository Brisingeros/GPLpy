""" TODO """
import math
import sys
import os
from gplpy.gggp.grammar import CFG
from gplpy.evo.evolution import Experiment, Optimization, Setup, Problem, Evolution_EDA_LP_ECO
from gplpy.evo.log import DBLogger
from gplpy.gggp.metaderivation import EDA


import gym

from GA import GA

from dynamic_bht import creation

import random

randomSeed = random.randrange(sys.maxsize)

# SETUP EXAMPLES ###########################################################
EDX_setup = Setup(name='LP_ECO', evolution=Evolution_EDA_LP_ECO, max_recursions = 55, crossover=EDA, selection_rate=0.5, exploration_rate=0.001, model_update_rate=.5, offspring_rate=.25, immigration_rate=.15, population_size=100, learning_tolerance_step=20)
# EDX_setup = Setup(name='LPV_ECO', evolution=Evolution_EDA_LPV_ECO, max_recursions = 55, crossover=EDA, selection_rate=0.5, exploration_rate=0.001, model_update_rate=.5, offspring_rate=.25, immigration_rate=.15, population_size=100, learning_tolerance_step=20)
# EDX_setup = Setup(name='FP_ECO', evolution=Evolution_EDA_FP_ECO, max_recursions = 55, crossover=EDA, selection_rate=0.5, exploration_rate=0.001, model_update_rate=.5, offspring_rate=.25, immigration_rate=.15, population_size=100, learning_tolerance_step=20)
# EDX_setup = Setup(name='ECO', evolution=Evolution_EDA_ECO, max_recursions = 55, crossover=EDA, selection_rate=0.5, exploration_rate=0.001, model_update_rate=.5, offspring_rate=.25, immigration_rate=.15, population_size=100, learning_tolerance_step=20)

def ft(phenotype, auxData):
    derivation, game = auxData

    #################################

    game.gene = phenotype
    fitness = 0

    for _ in range(5):
        game.reset()
        while not(game.done):
            inLocals = {
                'game': game
            }

            derivation.tick(inLocals)

        fitness += game.fitness

    return fitness

class Game:
    def __init__(self, env):
        self.env = env
        self.observation = None
        self.fitness = 0
        self.gene = []

    def reset(self):
        self.observation = self.env.reset()
        self.fitness = 0
        self.done = False

    def play(self, input):
        if self.done:
            return

        #self.env.render()
        self.observation, reward, self.done, info = self.env.step(input)
        self.fitness += reward

    def passCondition(self, code):
        inLocals = {
            'x': self.observation[0],
            'y': self.observation[1],
            'vel_x': self.observation[2],
            'vel_y': self.observation[3],
            'ang': self.observation[4],
            'vel_ang': self.observation[5],
            'gene': self.gene,
            'passing': False
        }
        exec(f'passing = {code}', globals(), inLocals)
        return inLocals['passing']


class auxIndiv:
    def __init__(self):
        self.fitness = None

    def phenotype(self, geneRange):
        return []
class auxGA:
    def __init__(self, bht, game):
        self.bht = bht
        self.game = game
        self.indiv = auxIndiv()

    def execute(self, learning_tolerance_step):
        self.indiv.fitness = ft(phenotype=[], auxData=(self.bht, self.game))

        return self.indiv, None, None

class LUNAR(Problem):
    optimization = Optimization.max

    def __init__(self, individual, args):
        super().__init__()
        self.individual = individual
        self.derivation = "".join(str(individual.derivation).split())
        self.env = gym.make('LunarLander-v2')
        self.env.seed(randomSeed)
        self.game = Game(self.env)

        self.geneRange = []

        auxder = self.derivation.split('FLOAT')
        derivationChanges = []

        self.I = (len(auxder) - 1)
        for i in range(self.I):
            derivationChanges.append(f'{auxder[i]}gene[{i}]')
            self.geneRange.append((-1.0,1.0))

        derivationChanges.append(auxder[-1])
        derivationChanges = "".join(derivationChanges)

        # auxder = derivationChanges.split('INTEGER')
        # derivationChanges = []

        # self.N = (len(auxder) - 1)
        # for i in range(self.N):
        #     derivationChanges.append(f'{auxder[i]}gene[{i+self.I}]')
        #     self.geneRange.append((0,23))

        # derivationChanges.append(auxder[-1])
        # derivationChanges = "".join(derivationChanges)

        # #######################
        # f = open('mariolike_simple.txt', "w")
        # f.write(derivationChanges)
        # f.close()
        # #######################

        self.bht = creation.string_bht_train(derivationChanges)

        if len(self.geneRange) > 0:
            self.GA = GA(popSize=50, geneRange=self.geneRange, replacementDegree=0.3, stepSize=20, fitnessFunc= ft, auxData=(self.bht, self.game), debug=False)
        else:
            self.GA = auxGA(bht=self.bht, game=self.game)

    def run(self):
        improvement = float("inf")
        previous_score = float("-inf")
        while self.individual.alive.is_set() and improvement > self.individual.learning_tolerance:
            # Check maturity
            if self.individual.async_learning:
                if not self.individual.mature.is_set():
                    if improvement < self.individual.maturity_tolerance:
                        self.individual.mature.set()
                elif not self.individual.converged.is_set():
                    # Wait until individuals realase to learn
                    self.individual.learn.acquire()

            #Learn and obtain score
            bestIndiv, fitness_hist, fitness_mean_hist = self.GA.execute(self.individual.learning_tolerance_step)
            self.individual._fitness = bestIndiv.fitness
            self.individual.learning_iterations += self.individual.learning_tolerance_step

            if math.isinf(previous_score):
                improvement = 1.0
            else:
                improvement = (self.individual._fitness - previous_score) / abs(previous_score)

            previous_score = self.individual._fitness

        self.env.close()

        ################################

        derivation = self.derivation.split('FLOAT')
        phenotype = bestIndiv.phenotype(self.geneRange)

        derivationChanges = []
        for i in range(self.I):
            derivationChanges.append(derivation[i])
            derivationChanges.append(str(phenotype[i]))

        derivationChanges.append(derivation[-1])
        derivation = "".join(derivationChanges)

        # derivation = derivation.split('INTEGER')
        # derivationChanges = []
        # for i in range(self.N):
        #     derivationChanges.append(derivation[i])
        #     derivationChanges.append(str(phenotype[i+self.I]))

        # derivationChanges.append(derivation[-1])
        # derivation = "".join(derivationChanges)
        self.individual.finalDerivation = derivation

        ################################

        self.individual.mature.set()
        return

if __name__ == "__main__":
    sys.setrecursionlimit(10000)

    ## IS 
    study = "lunar bht mariolike 3"
    study_id = None
    #study_id = ObjectId("590b99f0d140a535c9dfbe12")

    # Grammar initialization
    grammar_file = os.getcwd() + '/gr/' + study.replace(' ', '_') + '.gr'
    gr = CFG(grammar_file)

    # logger initialization
    # Set to True to log into mongodb
    logger = False
    if logger:
        logger = DBLogger(server='cluster0-21cbd.gcp.mongodb.net', user='gplpy_logger', password='q1e3w2r4', cluster=True)
        if study_id:
            logger.resume_study(study_id=study_id, grammar=grammar_file[5:])
        else:
            logger.new_study(study=study, grammar=grammar_file[5:])
        logger.createDeleteStudyRoutine()

    # Setup problem

    exp_name = "Lunar BHT mariolike simple pop100_50 recur55"
    args = ()

    # Run
    samples = 1
    #, EDA_setup, WX_setup
    ids =Experiment(study=study, experiment=exp_name, grammar=gr, problem=LUNAR, fitness_args=args,
                    setups=[EDX_setup], logger=logger, samples=samples).run()
    
    if logger and logger.server == 'localhost':
        logger.plot_experiments_evolution()
        logger.plot_range_study()
        logger.report_statistics()