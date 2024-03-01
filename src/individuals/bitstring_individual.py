from .abstract_individual import individual
from abc import ABC, abstractmethod
import numpy as np
class bitstring_individual(individual):
    """
    An individual whose genome is represented by a bitstring
    """
    def __init__(self, genome_size: int = 0, genome: list = []) -> None:
        """Constructor for a bitstring individual

        Args:
            genome_size (int, optional): When specified, this will construct a random bitstring genome with
            genes equal to this value. Defaults to 0.
            genome (list, optional): When specified, construct an individual with this explicit bitstring.
            Defaults to [].
        """
        if len(genome)>0:
            self.genome = genome
        else:
            self.genome = np.random.randint(2,size = genome_size)
            
        self.fitness = 0

    def get_genome(self) -> list:
        """

        Returns:
            list: a list containing the 0 or 1 value for each gene
        """
        return self.genome
    def set_fitness(self,fitness: float) -> None:
        """
        Args:
            fitness (float): sets the fitness of this individual
        """
        self.fitness = fitness
    def get_fitness(self) -> float:
        """
        Returns:
            float: the fitness of this individual
        """
        return self.fitness

    def mutate(self, mutation_probability: float,mutation_method = "bitflip") -> None:
        """mutates the genome of the individual

        Args:
            mutation_probability (float): the probability of mutation for each gene
            mutation_method (str, optional): the method of mutation used. Defaults to "bitflip".
        """
        
        for index, bit in enumerate(self.genome):
            if np.random.random() < mutation_probability:
                if self.genome[index] == 0:
                    self.genome[index] = 1
                else:
                    self.genome[index] = 0
        
    
    def create_child_with(self, other_parent: individual,ratio: float = 0.5,crossover_method: str = "uniform") -> individual: 
        """performs uniform crossover to create the genome for a new individual.

        Args:
            other_parent (individual): the other parent to generate the child with
            ratio (float, optional): the probability that a gene comes from this individual's genome over the other parent.
            Defaults to 0.5.

        Returns:
            individual: a new individual from the combined genomes
        """
        if crossover_method == "uniform":
            # perform uniform crossover to generate the child.
            new_gene = [0]*len(self.get_genome())
            for index, gene in enumerate(new_gene):
                if np.random.random() < ratio:
                    new_gene[index] = self.get_genome()[index]
                else:
                    new_gene[index] = other_parent.get_genome()[index]
                    
            
                child = bitstring_individual(genome = new_gene)
                
        return child
        

    