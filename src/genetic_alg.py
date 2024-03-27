import numpy as np
from problems import *
from individuals import *
import pandas as pd
import math
class genetic_alg:
    """Genetic algorithm class
    """
    
    def __init__(self,pop_size : int,
                genome_size: int,
                crossover_type: str, 
                mutation_method: str, 
                mutation_probability: float, 
                problem: problem,
                genome_type = "bitstring",
                selection_method = "tournament", tournament_frac = 0.2, tournament_prob = 1,roulette_param: int = 0):
        
        """Constructor for the genetic algorithm. Give it parameters and a problem to solve

        Args:
            pop_size (int): Size of the population
            genome_size (int): The amount of genes in an individual's genome
            crossover_type (str): The method used for crossover, choose from TODO
            mutation_method (str): the mutation method, choose from TODO
            problem (problem): A problem object that contains the problem to solve
        """
        self.selection_method = selection_method
        self.pop_size = pop_size
        self.genome_size = genome_size
        self.genome_type = genome_type
        self.crossover_type = crossover_type
        self.mutation_method = mutation_method
        self.mutation_probability = mutation_probability
        self.problem = problem
        self.tournament_prob = tournament_prob
        self.tournament_frac = tournament_frac
        self.population = self.create_population(pop_size)
        self.calculate_population_fitness(self.population)
        self.current_gen = 0
        self.history_list = []
        self.roulette_param = roulette_param
        
    def reset_alg(self):
        """resets the alg to generation 0 with a new population and a blank history
        """
        self.population = self.create_population(self.pop_size)
        self.calculate_population_fitness(self.population)
        self.current_gen = 0
        self.history_list = []
        
    def get_history(self):
        history_df = pd.DataFrame(self.history_list)
        return history_df
  
    def update_history(self):
        fittest = self.get_fittest()
        fitnesses = [individual.get_fitness() for individual in self.population] #create a list of all fitnesses for each individual in the population list.
        gendict = {"Generation" : self.current_gen,
                   "Best fitness": fittest.get_fitness(),
                   "Mean fitness": np.mean(fitnesses),
                   "Fitness variance": np.var(fitnesses),
                   "Best genome": fittest.get_genome() }
        self.history_list.append(gendict)
    
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
        if self.genome_type == 'bitstring':
            for i in range(0,size):
                population[i] = bitstring_individual(genome_size=self.genome_size)
        if self.genome_type == 'path':
            for i in range(0,size):
                population[i] = path_individual(genome_size=self.genome_size)
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
        selection = []
        if sum(fitnesses)==0:
            selection = np.random.choice(self.population,size = len(self.population),replace = True)
            
        elif gibbs_temperature > 0:
            probabilities = [math.exp(fitness/gibbs_temperature) for fitness in fitnesses]
            probabilities = [prob/sum(probabilities) for prob in probabilities]
            for p in range(len(self.population)):
                probability_sum = 0
                randfloat = np.random.rand()
                for index, probability in enumerate(probabilities):
                    probability_sum = probability_sum+probability
                    if randfloat<probability_sum:
                        selection.append(self.population[index])
                        break
                    
        
        else:
            probabilities = [fitness/sum(fitnesses) for fitness in fitnesses]

            for p in range(len(self.population)):
                probability_sum = 0
                randfloat = np.random.rand()
                for index, probability in enumerate(probabilities):
                    probability_sum = probability_sum+probability
                    if randfloat<probability_sum:
                        selection.append(self.population[index])
                        break
        return selection       
        # for parent_index, parent in enumerate(probabilities):
        #     probability_sum = 0
        #     randfloat = np.random.rand()
        #     for index, probability in enumerate(probabilities):
        #         probability_sum = probability_sum+probability
        #         if randfloat<probability_sum:
        #             parents[parent_index] = self.population[index]
        #            # print("parent: ",parents[parent_index].get_genome(),"\n", "probability sum", probability_sum, "roulette roll: ", randfloat)
        #             break
        

    def tournament_selection(self,select_from: list, tournament_size: int, p: float = 0.8) -> list[individual, individual]:
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
        match self.selection_method: 
            case "tournament":
                while len(children) < self.pop_size:
                        tournament_size = int(self.pop_size*self.tournament_frac)
                        parents = self.tournament_selection(tournament_size=tournament_size,p = self.tournament_prob, select_from=self.population)       
                        new_children = parents[0].create_child_with(parents[1],crossover_method=self.crossover_type)
                        children.append(new_children[0])
                        children.append(new_children[1])
                        
            case "roulette":
                parents = self.roulette_selection(gibbs_temperature=self.roulette_param)
                for i in range(0, len(parents), 2):
                    new_children = parents[i].create_child_with(parents[i+1],crossover_method=self.crossover_type)
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
    
    def mutate_population(self, mutation_method = 'bitstring'):
        """Calls the mutation method for each individual in the current population. For bitstring gene representations
        this will be a bitflip operation. Mutation is implemented in subclasses of the abstract class "individual"

        Args:
            mutation_probability (float): The probability for each gene in the genome to mutate. This value should be between 0 and 1
        """
        for individual in self.population:
            individual.mutate(self.mutation_probability,mutation_method) 

    def run(self,max_generations:int):
        while(self.current_gen<max_generations):
            self.make_next_generation()
            self.mutate_population()
            self.update_history()
        
if __name__ == "__main__":
    def generate_problems(per_class = 2):
        problems = []
        max_total_weights = [10, 20, 30]
        item_counts = [5, 10, 20]
        test_cases = []
        for iteration in range(per_class):
            for total_weight in max_total_weights:
                for count in item_counts:
                
                    test_case = {
                        'maximum_total_weight': total_weight,
                        'item_count': count
                    }
                    test_cases.append(test_case)
                
        for case in test_cases:
            problems.append(knapsack(minimum_item_weight=0,maximum_item_weight=10, **case))
        return problems
    problems = generate_problems()
    print(len(problems))
    
    algs = []
    
    pop_sizes = [6, 10, 20]
    crossover_types = ["one point", "uniform", "two point"]
    mutation_probabilities = [0.05, 0.1, 0.2]
    selection_methods = ["roulette", "tournament"]
    for problem in problems:
        for pop_size in pop_sizes:
            for crossover_type in crossover_types:
                for mutation_probability in mutation_probabilities:
                    for selection_method in selection_methods:
                        alg = genetic_alg(
                            pop_size=pop_size,
                            genome_size=problem.get_item_count(),
                            crossover_type=crossover_type, 
                            mutation_method="gene bitflip", 
                            mutation_probability=mutation_probability,
                            problem=problem,
                            genome_type='bitstring',
                            selection_method=selection_method)
                        algs.append(alg)
                        
    for alg in algs:
        alg.run(50)
    
    