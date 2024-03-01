from abc import ABC, abstractmethod
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
    @abstractmethod
    def mutate(self,mutation_probability:float, mutation_method: str) -> None:
        pass
    @abstractmethod
    def create_child_with(self, other_parent,ratio: float,crossover_method: str): 
        pass
   