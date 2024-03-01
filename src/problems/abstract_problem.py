import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class problem(ABC):
    
    @abstractmethod
    def get_fitness(self, genotype:list) -> float:
        pass
    
   