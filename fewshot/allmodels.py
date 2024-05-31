
from .model import ProbabilisticModel,AdaptiveModel
from typing import Dict

def all_models() -> dict:
    """Returns a dict of model name to Model class.
    """
    from .ADKL import ADKLModel
    from .CNP import CNPModel
    from .FewShotSR import FewShotSRModel
    from .DNNFineTuning import DNNFineTuningModel
    from .DNNResidualGP import DNNResidualGPModel
    from .DKL import DKLModel
    from .GP import GPModel
    from .MAML import MAMLModel
    return {
        'ADKL':ADKLModel,
        'CNP':CNPModel,
        'FewShotSR':FewShotSRModel,
        'DNNFineTuning':DNNFineTuningModel,     
        'DNNResidualGP':DNNResidualGPModel,
        'DNNResidualFSGP':DNNResidualGPModel,
        'DKT':DKLModel,
        'DKL':DKLModel,
        'GP':GPModel, 
        'FSGP':GPModel, 
        'MAML':MAMLModel,
    }

def probabilistic_models() -> Dict[str,ProbabilisticModel]:
    from .ADKL import ADKLModel
    from .FewShotSR import FewShotSRModel
    from .DNNResidualGP import DNNResidualGPModel
    from .DKL import DKLModel
    from .GP import GPModel
    return {
        'ADKL':ADKLModel,
        'FewShotSR':FewShotSRModel,
        'DNNResidualGP':DNNResidualGPModel,
        'DNNResidualFSGP':DNNResidualGPModel,
        'DKT':DKLModel,
        'DKL':DKLModel,
        'GP':GPModel, 
        'FSGP':GPModel, 
    }

def adaptive_models() -> Dict[str,AdaptiveModel]:
    return all_models()