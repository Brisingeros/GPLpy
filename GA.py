import math
import random
import matplotlib as plt
import numpy as np

def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value

def bottom_clip(value, lower):
    return lower if value < lower else value

#--------------------------------------------------------------------------------------------------------------------

class Indiv:
    def __init__(self, data):
      self.data = data
      self.fitness = None
      return

    def phenotype(self, geneRange):
      auxrange = np.array(geneRange, dtype=np.float64)

      denormalized = (self.data * (auxrange[:,1] - auxrange[:,0])) + auxrange[:,0]
      denormalized = denormalized.tolist()

      for index, data in enumerate(zip(denormalized, geneRange)):
          if isinstance(data[1][0], int):
            denormalized[index] = int(round(data[0]))

      return denormalized

#--------------------------------------------------------------------------------------------------------------------

class GA():
    def __init__(self, popSize, geneRange, replacementDegree, stepSize, fitnessFunc, auxData, powerMutation = 0.25, minimize = False, debug = False):
      self.debug = debug
      self.powerMutation = powerMutation
      self.popSize = bottom_clip(popSize, 2)
      self.geneRange = geneRange
      mutationProb = 1 / ((popSize ** 0.9318) * (len(geneRange) ** 0.4535))
      self.mutationProb = clip(mutationProb, 0., 1.)
      self.stepSize = bottom_clip(stepSize, 1)

      self.fitnessFunc = fitnessFunc
      self.auxData = auxData

      replacementDegree = clip(replacementDegree, 0., 1.)

      self.changesPop = math.ceil(popSize * replacementDegree)

      self.population = []
      self.fitness_hist = []
      self.fitness_mean_hist = []
      self.bestIndiv = Indiv(None)

      if minimize:
        self.ordenationFunction = (lambda x: x.sort(key=lambda y: y.fitness, reverse=False))
        self.bestIndiv.fitness = float('inf')
        self.comparationFunction = (lambda x, y: x < y)
        self.worstFunction = max
        self.bestFunction = min
        self.normalizationFunction = (lambda fitness: (-fitness - np.min(-fitness) + 1)) #Normalize fitness inputs for Roulette Selection on minimizing problems. MAX - X + 1. [1,...]
      else:
        self.ordenationFunction = (lambda x: x.sort(key=lambda y: y.fitness, reverse=True))
        self.bestIndiv.fitness = float('-inf')
        self.comparationFunction = (lambda x, y: x > y)
        self.worstFunction = min
        self.bestFunction = max
        self.normalizationFunction = (lambda fitness: (fitness - np.min(fitness) + 1)) #Normalize fitness inputs for Roulette Selection on maximizing problems. X - MIN + 1. [1,...]

      #####################################

      self.population_generation()
      self.population = self.fitness(self.population)
      self.ordenationFunction(self.population)
      self.bestIndiv = self.population[0]

      self.fitness_calc()

      if self.debug:
        plt.plot(self.fitness_hist, color='green', label="best")
        plt.plot(self.fitness_mean_hist, color='orange', label="mean")
        plt.ylabel('fitness')
        plt.xlabel('iteration')
        plt.legend(loc="upper left")
        plt.gcf()
        print(self.bestIndiv)


    #############################

    def population_generation(self):
      for _ in range(self.popSize):
          data = np.array([random.uniform(0., 1.) for _ in range(len(self.geneRange))], dtype=np.float64)
          self.population.append(Indiv(data))

      return

    ###

    def fitness(self, indivs):
      def fitness_op(indiv):
        indiv.fitness = self.fitnessFunc(indiv.phenotype(self.geneRange), self.auxData)
        return indiv


      indivs = [fitness_op(i) for i in indivs]

      return indivs

    ###

    #Roulette
    def selection(self):
      pop = self.population.copy()
      random.shuffle(pop)
      matingPool = np.zeros((5, len(self.geneRange)), dtype=np.float64)

      #Already explained norrmalization
      transformedFitness = np.array([i.fitness for i in pop], dtype=np.float64)
      transformedFitness = self.normalizationFunction(transformedFitness)

      ####################################

      totalFitness = sum(transformedFitness)

      probCumulative = transformedFitness / totalFitness
      auxProb = np.cumsum(probCumulative)
      auxProb[-1] = 0.0
      auxProb = np.roll(auxProb, 1)
      probCumulative = probCumulative + auxProb

      rnd = random.random()
      pointer = 0.0
      for index in range(5):
        pointer = (pointer + rnd) % 1
        indexAct = next(i for i,v in enumerate(probCumulative) if v > pointer)
        matingPool[index] = pop[indexAct].data

      return matingPool

    ####

    #MMX Crossover
    #https://www.researchgate.net/profile/D_Barrios_Rolania/publication/220661867_Optimisation_With_Real-Coded_Genetic_Algorithms_Based_On_Mathematical_Morphology/links/546dbc0d0cf26e95bc3cd66e.pdf
    def crossover(self, parents):
      columns = parents.shape[1]

      children = []
      intervals = np.zeros((columns, 2), dtype=np.float64)
      maxs = np.zeros(columns, dtype=np.float64)
      mins = np.zeros(columns, dtype=np.float64)

      for i in range(columns):
        col = parents[:,i]

        maxs[i] = np.max(col)
        mins[i] = np.min(col)

        g = maxs[i] - mins[i]

        if g <= 0.54:
          inter = -(0.25 * g) - (0.001 * maxs[i])
        else:
          inter = (0.5 * g) - (0.265 * maxs[i])

        intervals[i] = [clip((maxs[i] - inter), 0., 1.), clip((mins[i] + inter), 0., 1.)]

      for _ in range(self.changesPop//2):
        o = np.zeros(columns, dtype=np.float64)
        o_p = np.zeros(columns, dtype=np.float64)

        for i in range(columns):
          aux = random.uniform(intervals[i,0], intervals[i,1])
          o[i] = aux
          o_p[i] = clip(((maxs[i] + mins[i]) - aux), 0., 1.)

        children.append(Indiv(o))
        children.append(Indiv(o_p))

      return children

    ###

    #Power Mutation
    #https://www.sciencedirect.com/science/article/abs/pii/S0096300307003918
    # def mutation(self, children):
    #   for c in children:
    #       pm = random.uniform(0., 1.)

    #       if (pm >= self.mutationProb):
    #         continue

    #       r = random.uniform(0., 1.)
    #       t = random.uniform(0., 1.)
    #       s_ = random.uniform(0., 1.)
    #       s = s_ ** (1. / self.powerMutation) #Inverse of F

    #       if t < r:
    #           c.data = [(x - s * x) for x in c.data]
    #       else:
    #           c.data = [(x + s * (1. - x)) for x in c.data]


    #   return children

    def mutation(self, children):
      for c in children:
        c.data = np.array([clip((d + np.random.normal(0, 0.1, 1)[0]), 0., 1.) if (random.uniform(0., 1.) < self.mutationProb) else d for d in c.data], dtype=np.float64)

      return children

    ###

    #fitness

    ###
      
    def replacement(self, children):
      self.population = self.population + children
      self.ordenationFunction(self.population)
      self.population = self.population[:self.popSize]

      self.bestIndiv = self.population[0]

      return

    ###
      
    def fitness_calc(self):
      total = sum([i.fitness for i in self.population])
      
      self.fitness_hist.append(self.bestIndiv.fitness)
      self.fitness_mean_hist.append((total / self.popSize))

      return

    #############################

    def execute(self, iterations = 20):
      for _ in range(iterations):
        #auxPop ------- Output as mating pool in selection, input as mating pool output as children in crossover, input as children in mutation
        auxPop = self.selection()
        auxPop = self.crossover(auxPop)
        auxPop = self.mutation(auxPop)
        auxPop = self.fitness(auxPop)

        self.replacement(auxPop)
        self.bestIndiv = self.population[0]

        #########################

        self.fitness_calc()

        #########################

        if self.debug:
          plt.plot(self.fitness_hist, color='green')
          plt.plot(self.fitness_mean_hist, color='orange')
          plt.gcf()
          print(self.bestIndiv)

      return self.bestIndiv, self.fitness_hist, self.fitness_mean_hist