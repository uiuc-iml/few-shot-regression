from .benchmark import LearningBenchmark
from typing import Dict,Type

def all_benchmarks() -> Dict[str,Type[LearningBenchmark]]:
    from . import cannon 
    from . import toy_gaussian
    return {
        'cannon': cannon.CannonBenchmark,
        'toy_gaussian': toy_gaussian.ToyGaussianBenchmark,
        }