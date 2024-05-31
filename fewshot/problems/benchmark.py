from abc import ABC, abstractmethod
from typing import Tuple
from ..data import MultiTaskDataset
from ..model import SupervisedLearningProblem

class LearningBenchmark(ABC):
    @abstractmethod
    def problem(self) -> SupervisedLearningProblem:
        pass

    @abstractmethod
    def dataset(self, val_fraction=0.1) -> Tuple[MultiTaskDataset,MultiTaskDataset,MultiTaskDataset]:
        pass

