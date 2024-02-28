import numpy as np
from abc import ABC, abstractmethod
from problem import problem, knapsack


class individual(ABC):
    @abstractmethod
    def __init__(self,genome_size: int) -> None:
        pass
    @abstractmethod
    def get_genome(self) -> list:
        pass
    @abstractmethod
    def set_fitness(self) -> None:
        pass
    @abstractmethod
    def get_fitness(self) -> float:
        pass
    
class bitstring_individual(individual):
    def __init__(self, genome_size: int) -> None:
        self.genome = np.random.randint(2,size = genome_size)
        self.fitness = 0
    def get_genome(self) -> list:
        return self.genome
    def set_fitness(self,fitness: float) -> None:
        self.fitness = fitness
    def get_fitness(self) -> float:
        return self.fitness

class genetic_alg:
    """Genetic algorithm class
    """
    
    def __init__(self,pop_size : int, genome_size: int,crossover_type: str, mutation_method: str,problem: problem):
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
        self.crossover_type = crossover_type
        self.mutation_method = mutation_method
        self.problem = problem
        self.population = self.create_population(pop_size)
        self.set_population_fitness(self.population)
    
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
    
    
    def set_population_fitness(self,population):
        """Calculates the fitness of each individual in a population.

        Args:
            population: a list of individuals
        """
        for individual in population:
            fitness = self.problem.get_fitness(genotype = individual.get_genome())
            individual.set_fitness(fitness)
        
    def tournament_selection(self,tournament_size: int, p: float = 1) -> list[individual, individual]:
        """Performs tournament selection: create a tournament by uniformly selecting individuals from the population without replacement.
        Then, select the most fit individual with probability p, the next most fit with probability p*(1-p)^1, the individual after that with probability p*(1-p)^2 etc.
        Args:
            tournament_size (int): The size of the tournament. Cannot be larger than the size of the population.
            p (float, optional): The probability that the most fit individual will be selected. Defaults to 1.

        Returns:
            list[individual, individual]: The two individuals that resulted from the tournament selection
        """
        tournament = np.random.choice(self.population,size = tournament_size,replace = False)
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
    
    def uniform_crossover(self,parent_1: individual, parent_2: individual,ratio: float = 0.5) -> individual: 
        """performs uniform crossover to create the genome for a new individual.

        Args:
            parent_1 (individual): The first parent of the new individual
            parent_2 (individual): the second parent of the new individual
            ratio (float, optional): the probability that a gene comes from parent 1 over parent 2. Defaults to 0.5.

        Returns:
            individual: a new individual from the combined genomes
        """
        



        
if __name__ == "__main__":
    knapsack = knapsack()
    alg = genetic_alg(20,5,"bit flip", "mutation method",knapsack)
    for individual in alg.get_population():
        print(individual.get_fitness())
    sorted_tournament = alg.tournament_selection(20,1)
    print("first individual: ",sorted_tournament[0].get_fitness())
    # for index, individual in enumerate(sorted_tournament):
    #     print("fitness {index}: ".format(index=  index), individual.get_fitness())
    
    