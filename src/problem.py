import pandas as pd
import numpy as np
class Problem:
    def __init__(self, kind = "knapsack"):
        self.kind = kind
        if kind == "knapsack":
                self.problem_df, self.objective_function = self.construct_knapsack()
        
        
    def construct_knapsack(knapsack_size: int = 10, maximum_item_weight: float = 10, maximum_total_weight: float = 20, item_count: int = 5, copy: bool = False):
        item_rewards = list(range(1,item_count+1))
        item_weights = np.random.random_sample(size =item_count)*maximum_item_weight
        items_df = pd.DataFrame({'rewards': item_rewards,'weights':item_weights})
        objective_function = items_df.rewards * items_df.Genotype
        return items_df, objective_function
    
    def get_knapsack_fitness(selection_df):
        
        
        return
if __name__ == "__main__":
    problem = Problem()
    df, objective_function = problem.construct_knapsack()
    df["Genotype"] = 
    print(objective_function)