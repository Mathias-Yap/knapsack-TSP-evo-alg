from abc import ABC, abstractmethod
from .abstract_individual import individual
import random
import pandas as pd

class path_individual(individual):
  
    def __init__(self,genome_size: int = 0, genome: list = []) -> None:
        """Constructor for a path individual

        Args:
            genome_size (int, optional): When specified, this will construct a random bitstring genome with
            genes equal to this value. Defaults to 0.
            genome (list, optional): When specified, construct an individual with this explicit bitstring.
            Defaults to [].
        """
        if len(genome)>0:
            self.genome = genome
        else:
            self.genome = random.sample(range(genome_size),genome_size)
            
        self.fitness = 0
 
    def get_genome(self) -> list:
        return self.genome
     

    def set_fitness(self, fitness) -> None:
        self.fitness = fitness
 
    def get_fitness(self) -> float:
        return self.fitness

    def mutate(self,mutation_probability:float, mutation_method: str) -> None:
        if random.random() < mutation_probability:
            match mutation_method:
                case "random swap":
                        gene1, gene2 = random.sample(range(len(self.genome)), 2)
                        self.genome[gene1], self.genome[gene2] = self.genome[gene2],self.genome[gene1]

            match mutation_method:
                case "neighbor swap":
                        gene = random.sample(range(len(self.genome)-1))
                        self.genome[gene], self.genome[gene+1] = self.genome[gene+1],self.genome[gene]
    
    def partial_match_crossover(self,other_parent,cut_size = 3):
        children = []
        new_gene_1 = ['x']*len(self.get_genome())
        new_gene_2 = ['x']*len(self.get_genome())
        cut1 = random.randint(0,len(self.genome)-cut_size)
        cut2 = cut1+cut_size
        print("cut: ", cut1, cut2)
        
        # create a df containing the mappings
        mapfrom = self.genome[cut1:cut2] + other_parent.get_genome()[cut1:cut2]
        mapto = other_parent.get_genome()[cut1:cut2] + self.genome[cut1:cut2]
        mappings = pd.DataFrame(data = {'from' : mapfrom,'to' : mapto})
        
        for index in range(cut1,cut2):
            # set the genomes within the cut for the children
            new_gene_1[index], new_gene_2[index] = other_parent.get_genome()[index], self.genome[index]

        print ("child 1 after cut: ", new_gene_1)
        print ("child 2 after cut: ", new_gene_2)
        
        # now, let's fill in the indices outside of the cut by their parent's genomes as long as there is no conflict
        for index in range(0, cut1):
            # before the first cut  
            if self.genome[index] not in new_gene_1:
                new_gene_1[index] = self.genome[index]
        
            if other_parent.get_genome()[index] not in new_gene_2:
                new_gene_2[index] = other_parent.get_genome()[index]
            
        for index in range(cut2, len(self.genome)):
            # after the second cut
            if self.genome[index] not in new_gene_1:
                new_gene_1[index] = self.genome[index]
            
            if other_parent.get_genome()[index] not in new_gene_2:
                new_gene_2[index] = other_parent.get_genome()[index]
                
        print ("filled child 1: ", new_gene_1)
        print ("filled child 2: ", new_gene_1)
        
        # repair the genomes: if a number is missing after the operations, use the mappings to fill them
        
        def fill_from_mappings(parent_gene,new_gene,mappings):
            empty_indices = [i for i,x in enumerate(new_gene) if x == 'x']
            for index in empty_indices:
                from_value = parent_gene[index]
                found = False
                to_check = mappings[mappings['from'] == from_value]['to'].tolist()
                visited = set()  # To track visited 'to' values and prevent infinite loops

                while not found:
                    next_check = to_check.pop(0)  # Get the next 'to' value to check
                    if next_check in new_gene:
                        # If 'next_check' is already in genome, we look for its 'to' values
                        if next_check in visited:
                            continue  # Already visited this node, skip to avoid loop
                        visited.add(next_check)  # Mark this 'to' as visited
                        additional_checks = mappings[mappings['from'] == next_check]['to'].tolist()
                        to_check.extend(additional_checks)  # Add new values to check
                    else:
                        # Found a 'to' value not in genome, use it and break
                        new_gene[index] = next_check
                        found = True
                        break
            return new_gene
            
        new_gene_1 = fill_from_mappings(parent_gene = self.genome, new_gene = new_gene_1, mappings = mappings)
        new_gene_2 = fill_from_mappings(parent_gene = other_parent.get_genome(), new_gene = new_gene_2, mappings = mappings)
        
                
        print("finished child 1", new_gene_1)
        print("finished child 2", new_gene_2)
        children = [path_individual(genome = new_gene_1), path_individual(genome = new_gene_2)]
        return children

        
    
        
    def create_child_with(self, other_parent,ratio: float,crossover_method: str):
        children = []
        new_gene_1 = [0]*len(self.get_genome())
        new_gene_2 = [0]*len(self.get_genome())
        
        match crossover_method:
            case "pmx":
                children = self.partial_match_crossover(other_parent=other_parent)
                
if __name__ == "__main__":
    ind1 = path_individual(5)
    ind2 = path_individual(5)
    print(ind1.get_genome())
    print(ind2.get_genome())
    ind1.create_child_with(ind2, ratio = 0, crossover_method="pmx")
    
    
   