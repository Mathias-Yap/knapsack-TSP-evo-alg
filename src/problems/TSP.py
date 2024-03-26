from abstract_problem import problem
import random
import math
import pandas as pd
class TSP(problem):
    def __init__(self,n_cities: int):
        """TSP problem with 0,0 at the center point. Cities lie on the unit circle that surrounds this center point. Cities are represented by their angle from this center point"""
        angles = self.make_angles(n_cities)
        print(angles)
        self.problem_df = self.make_distance_df(angles)
    
    def make_angles(self, n_cities):
        angles = []
        for n in range(n_cities):
            angles.append(random.uniform(0,2*math.pi)) 
        return angles
    
    def get_city_distance(self, angle1, angle2):
        anglediff = abs(angle1-angle2)
        distance = math.sqrt(2-(2*math.cos(anglediff)))
        return distance
    
    def make_distance_df(self, angles):
        n = len(angles)
        # Initialize an empty DataFrame
        distance_df = pd.DataFrame(index=range(n), columns=range(n))

        # Iterate through each pair of cities to compute distances
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_df.iloc[i, j] = self.get_city_distance(angles[i], angles[j])
                else:
                    distance_df.iloc[i, j] = 0  # Distance from a city to itself is 0

        return distance_df

    def get_problem_df(self): 
        return self.problem_df
    
    def get_fitness(self,genotype:list) -> float:
        
        total_distance = 0
        # Iterate through the list of cities
        for i in range(len(genotype) - 1):  
            # Add the distance from the current city to the next
            total_distance += self.problem_df.iloc[genotype[i], genotype[i+1]]
        total_distance += self.problem_df.iloc[genotype[-1],genotype[0]] #add the distance from the final city back to the origin
        return total_distance
    
if __name__ == "__main__":
    problem = TSP(4)
    print(problem.get_problem_df())
    print(problem.get_fitness([0,3,2,1]))
    
   