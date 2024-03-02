from .abstract_problem import problem
import numpy as np
import pandas as pd
class knapsack(problem):
    
    def __init__(self, minimum_item_weight: float = 0, maximum_item_weight: float = 10, maximum_total_weight: float = 20, item_count: int = 5, copy: bool = False):
        """knapsack problem class

        Args:
            minimum_item_weight (float, optional): the minimum weight an item can have. Defaults to 0.
            maximum_item_weight (float, optional): the maximum weight an item can have. Defaults to 10.
            maximum_total_weight (float, optional): the maximum weight that fits inside the knapsack. Defaults to 20.
            item_count (int, optional): The number of items that are in the problem. Defaults to 5.
            copy (bool, optional): True if items can be selected without replacement, false otherwise. Defaults to False.
        """
        self.maximum_total_weight = maximum_total_weight
        item_rewards = list(range(1,item_count+1))        	
        rng = np.random.default_rng(seed=None)
        item_weights = rng.uniform(low=minimum_item_weight, high=maximum_item_weight, size=(item_count))
        self.problem_df = pd.DataFrame({'rewards': item_rewards,'weights':item_weights})
       
    def get_knapsack_df(self):
        return self.problem_df
    
    def get_fitness(self,genotype: list) ->float:
        """Calculates the fitness that a solution can have for this problem

        Args:
            genotype (list): the genotype of an individual

        Returns:
            gene_value (float): the score that a solution has in this problem
        """
            gene_df = self.problem_df
            gene_df['Genotype'] = genotype
            gene_value = (gene_df.rewards * gene_df.Genotype).sum()
            gene_weight = (gene_df.weights * gene_df.Genotype).sum()
             
            if gene_weight > self.maximum_total_weight:
                return 0
            else:
                return gene_value
            
    
# if __name__ == "__main__":
#     problem = knapsack()
#     print(problem.get_fitness([1,0,0,0,0]))
    