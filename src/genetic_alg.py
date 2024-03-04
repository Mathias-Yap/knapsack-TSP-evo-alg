import numpy as np
from problems import *
from individuals import *
import pandas as pd
import math
class genetic_alg:
    """Genetic algorithm class
    """
    
    def __init__(self,pop_size : int, genome_size: int,crossover_type: str, mutation_method: str, mutation_probability: float, problem: problem,genome_type = "bitstring"):
        """Constructor for the genetic algorithm. Give it parameters and a problem to solve

        Args:
            pop_size (int): Size of the population
            genome_size (int): The amount of genes in an individual's genome
            crossover_type (str): The method used for crossover, choose from TODO
            mutation_method (str): the mutation method, choose from TODO
            problem (problem): A problem object that contains the problem to solve
        """
        self.pop_size = pop_size
        self.genome_size = genome_size
        self.genome_type = genome_type
        self.crossover_type = crossover_type
        self.mutation_method = mutation_method
        self.problem = problem
        self.population = self.create_population(pop_size)
        self.calculate_population_fitness(self.population)
        self.current_gen = 0
        self.history_df = pd.DataFrame(columns = ["Generation","Best fitness","Mean fitness", "Fitness variance","Best genome"])
    
    def get_history(self):
        return self.history_df
    
    def update_history(self):
        fittest = self.get_fittest()
        fitnesses = [individual.get_fitness() for individual in self.population] #create a list of all fitnesses for each individual in the population list.
        self.history_df.loc[self.current_gen] = ([self.current_gen] #generation
                                                 + [fittest.get_fitness()] #best fitness
                                                 + [np.mean(fitnesses)] # mean fitness
                                                + [np.var(fitnesses)] # fitness variance
                                                 +  [fittest.get_genome()]) # best individual genome
                                                 
        
        
    
    def get_population(self) -> list:
        """
        Returns:
            population: a list that contains all individuals in the current population
        """
        return self.population
        
    def create_population(self, size: int) -> list:
        """creates a population of individuals

        Args:
            size (int): The number of individuals in the population

        Returns:
            population: a list that contains all individuals in the current population
        """
        population = [None] * size
        for i in range(0,size):
            population[i] = bitstring_individual(genome_size=self.genome_size)
        return population
    
    
    def calculate_population_fitness(self,population: list = []) -> list:
        """Calculates the fitness of each individual in a population.

        Args:
            population: a list of individuals
            
        Returns:
            fitnesses: (list) a list of the fitnesses for the corresponding individual in the population list at the same index.
        """
        if not len(population)>0:
            population = self.population
        for individual in population:
            fitness = self.problem.get_fitness(genotype = individual.get_genome())
            individual.set_fitness(fitness)
        fitnesses = [individual.get_fitness() for individual in self.population]
        return fitnesses
    
    def roulette_selection(self, gibbs_temperature: int = 0):
        """Performs roulette selection: distribute the probabilities between individuals in the population based on their
        proportional fitness. When the temperature is nonzero, use a gibbs distribution with t equal to the temperature.

        Args:
            gibbs_temperature (int, optional): The temperature to use for the gibbs probability distribution. Use  a fitness
            proportional probability distribution if this is zero. Defaults to 0.

        Returns:
            _type_: a list with the two parents
        """
        fitnesses = self.calculate_population_fitness()
        if gibbs_temperature > 0:
            probabilities = [math.exp(fitness/gibbs_temperature) for fitness in fitnesses]
            probabilities = [prob/sum(probabilities) for prob in probabilities]
        else:
            probabilities = fitnesses/sum(fitnesses)
        
        parents = [0]*2
        for parent_index, parent in enumerate(parents):
            probability_sum = 0
            randfloat = np.random.rand()
            for index, probability in enumerate(probabilities):
                probability_sum = probability_sum+probability
                if randfloat<probability_sum:
                    parents[parent_index] = self.population[index]
                   # print("parent: ",parents[parent_index].get_genome(),"\n", "probability sum", probability_sum, "roulette roll: ", randfloat)
                    break
                    

        return parents
    
    def tournament_selection(self,select_from: list, tournament_size: int, p: float = 1) -> list[individual, individual]:
        """Performs tournament selection: create a tournament by uniformly selecting individuals from the population without replacement.
        Then, select the most fit individual with probability p, the next most fit with probability p*(1-p)^1, the individual after that with probability p*(1-p)^2 etc.
        Args:
            tournament_size (int): The size of the tournament. Cannot be larger than the size of the population.
            p (float, optional): The probability that the most fit individual will be selected. Defaults to 1.

        Returns:
            list[individual, individual]: The two individuals that resulted from the tournament selection
        """
        tournament = np.random.choice(select_from,size = tournament_size,replace = False)
        sorted_tournament = sorted(tournament,key = lambda obj: obj.get_fitness(),reverse = True)
        probabilities = [0]*tournament_size
        if p == 1:
            probabilities[0] = 1
            probabilities[1] = 1
        else:
            for index, individual in enumerate(sorted_tournament):
                if index == 0:
                    probabilities[index] = p
                else:
                    probabilities[index] = np.power(p*(1-p),index)
            
        winners = np.random.choice(self.population,size = 2,replace = False)
        return winners
    
    def make_next_generation(self,ratio: float = 0.5) -> individual: 
        """Create the next generation of the population  by selecting two parents for each child using the 
        crossover and selection method specified in the constructor.

        Args:
            parent_1 (individual): The first parent of the new individual
            parent_2 (individual): the second parent of the new individual
            ratio (float, optional): the probability that a gene comes from parent 1 over parent 2. Defaults to 0.5.

        Returns:
            individual: a new individual from the combined genomes
        """
        self.calculate_population_fitness(self.population)
        children = []
        while len(children) < self.pop_size:
            parents = self.tournament_selection(tournament_size=5,p = 1, select_from=self.population)
            new_children = parents[0].create_child_with(parents[1])
            children.append(new_children[0])
            children.append(new_children[1])
        self.population = children
        self.current_gen = self.current_gen +1
        return self.population
    
    def get_fittest(self) -> individual:
        """returns the fittest individual in the corrent population

        Returns:
            individual: The fittest indiviual in the current population
        """
        self.calculate_population_fitness(self.population)
        sorted_pop = sorted(self.population,key = lambda obj: obj.get_fitness(),reverse = True)
        return sorted_pop[0]
        
    def mutate_population(self,mutation_probability:float, mutation_method):
        """Calls the mutation method for each individual in the current population. For bitstring gene representations
        this will be a bitflip operation. Mutation is implemented in subclasses of the abstract class "individual"

        Args:
            mutation_probability (float): The probability for each gene in the genome to mutate. This value should be between 0 and 1
        """
        for individual in self.population:
            individual.mutate(mutation_probability,mutation_method) 

    
        
if __name__ == "__main__":
    knapsack = knapsack(minimum_item_weight=0.1,maximum_item_weight=10,maximum_total_weight=15,item_count = 8)
    alg = genetic_alg(50,8, mutation_method="bitflip",mutation_probability=0.1,crossover_type="uniform",problem = knapsack)

    pop = alg.tournament_selection(select_from = alg.get_population(), tournament_size=4)
    for i in range(20):
        alg.update_history()
        fittest = alg.get_fittest()
        print(fittest.get_genome(),fittest.get_fitness())
        current_pop = alg.make_next_generation()
        alg.mutate_population(mutation_probability=0.1,mutation_method="bitflip")
  
    # alg.roulette_selection(gibbs_temperature = 1)
    
    # print(alg.history_df.head())
    
       # for index, individual in enumerate(sorted_tournament):
    #     print("fitness {index}: ".format(index=  index), individual.get_fitness())
    
    