from torch import nn, optim
import copy
from util import *
from models import *
import numpy as np


def denorm_obj(cycle, energy, log_obj, norm_obj, norm_path=None):
    """Denormalizes the cycle and energy objectives.

    See denorm_cycle_obj and denorm_energy_obj for more info.
    """
    cycle=denorm_cycle_obj(cycle, log_obj, norm_obj, norm_path)
    energy=denorm_energy_obj(energy, log_obj, norm_obj, norm_path)
    return cycle, energy


def norm_obj(cycle, energy, log_obj, norm_obj, norm_path=None):
    """Normalizes the cycle and energy objectives.

    See norm_cycle_obj and norm_energy_obj for more info.
    """
    cycle=norm_cycle_obj(cycle, log_obj, norm_obj, norm_path)
    energy=norm_energy_obj(energy, log_obj, norm_obj, norm_path)
    return cycle, energy


def denorm_cycle_obj(cycle, log_obj, norm_obj, norm_path=None):
    """Denormalize the cycle objective.

    Similar to denorm_layerfeat_func below. No normalization option (mean-std
    only), and if log and normalization are both not used values are divided
    by a default value.
    """
    if norm_obj:
        data = parse_json(norm_path)  
        cycle = cycle * data['cycle_std'] + data['cycle_mean'] 
    if log_obj:
        cycle = np.exp(cycle)
    if not norm_obj and not log_obj:
        cycle = cycle * 2**28
    return cycle


def norm_cycle_obj(cycle, log_obj, norm_obj, norm_path=None):
    """Normalize the cycle objective.

    Similar to norm_layerfeat_func below. No normalization option (mean-std 
    only), and if log and normalization are both not used values are divided
    by a default value.
    """
    if log_obj:
        cycle = np.log(cycle)
    if norm_obj:
        data = parse_json(norm_path)  
        cycle = (cycle - data['cycle_mean']) / data['cycle_std'] 
    if not norm_obj and not log_obj:
        cycle = cycle / 2**28
    return cycle


def denorm_energy_obj(energy, log_obj, norm_obj, norm_path=None):
    """Denormalize the energy objective.

    Same as denorm_cycle_obj above.
    """
    if norm_obj:
        data = parse_json(norm_path)  
        energy = energy * data['energy_std'] + data['energy_mean'] 
    if log_obj:
        energy = np.exp(energy)
    if not norm_obj and not log_obj:
        energy = energy * 2**38 
    return energy


def norm_energy_obj(energy, log_obj, norm_obj, norm_path=None):
    """Normalize the energy objective.

    Same as norm_cycle_obj above.
    """
    if log_obj:
        energy = np.log(energy)
    if norm_obj:
        data = parse_json(norm_path)  
        energy = (energy - data['energy_mean']) / data['energy_std'] 
    if not norm_obj and not log_obj:
        energy = energy / 2**38 
    return energy


def denorm_layerfeat_func(layerfeat, log_layerfeat, norm_layerfeat, norm_layerfeat_option, norm_path=None):
    """Denormalizes layer dimensions.

    Uses previously populated normalization statistics to restore layer 
    dimensions (N, C, K, etc.) that were previously normalized.

    Args:
        See norm_layerfeat_func below.

    Returns:
        List layerfeat, denormalized using the options selected.
    """
    if norm_obj:
        data = parse_json(norm_path)  
        if norm_layerfeat_option=='mean': 
            for i in range(9):
                layerfeat[i] = layerfeat[i] * (data[f'prob_{i}_std'] + 1e-16) + data[f'prob_{i}_mean'] 
        elif norm_layerfeat_option=='max':
            for i in range(9):
                layerfeat[i] = layerfeat[i] * data[f'prob_{i}_max'] 
        else:
            raise('Invalid norm_layerfeat_option.')
    if log_obj:
        for i in range(9):
            layerfeat[i] = np.exp(layerfeat[i])
    return layerfeat


def norm_layerfeat_func(layerfeat, log_layerfeat, norm_layerfeat, norm_layerfeat_option, norm_path=None):
    """Normalize layer dimensions.

    Normalizes layer dimensions, by taking the log and/or carrying out either 
    mean-std normalization or division by the max. The statistics (mean, std, 
    max) are pulled from a file, norm_path, that contains the aforementioned
    statistics, populated during initial dataset processing.

    Args:
        layerfeat: List containing dimensions (N, C, K, etc.) of a given neural
            network layer.
        log_layerfeat: Boolean, whether to take the log of each layer dimension.
        norm_layerfeat: Boolean, whether to normalize each layer dimension.
        norm_layerfeat_option: String, the method with which to normalize. [mean, max]
        norm_path: String, the path to the previously populated normalization
            statistics.

    Returns:
        List layerfeat, normalized using the options selected.
    """
    if log_layerfeat:
        for i in range(9):
            layerfeat[i] = np.log(layerfeat[i])

    if norm_layerfeat:
        assert(norm_path)
        data = parse_json(norm_path)  
        if norm_layerfeat_option=='mean': 
            for i in range(9):
                layerfeat[i] = (layerfeat[i] - data[f'prob_{i}_mean']) / (data[f'prob_{i}_std'] + 1e-16) 
        elif norm_layerfeat_option=='max':
            for i in range(9):
                assert(data[f'prob_{i}_max'] > 0)
                layerfeat[i] = layerfeat[i] / data[f'prob_{i}_max'] 
        else:
            raise('Invalid norm_layerfeat_option.')

    return layerfeat


def model_add_predictor(model, args, layer_size):
    """Adds performance predictor(s) to VAESA PyTorch model.
    
    Args:
        model: The VAESA PyTorch model object.
        args: Command line arguments from train.py.
        layer_size: Size of neural network layers for which performance is 
            predicted.

    Returns:
        Void, but the model object is modified to include the following keys:

        model.mseloss: MSE loss function used for evaluation
        model.predictor: Latency predictor MLP model 
        model.latency_loss: L1 loss function used for latency predictor
        model.predictor_energy: Latency predictor MLP model 
        model.energy_loss: L1 loss function used for energy predictor
    """
    if args.predictor_model == 'deep':
        predictor = nn.Sequential(
                nn.Linear(args.nz+layer_size, args.hs), 
                nn.Tanh(), 
                nn.Linear(args.hs, 2*args.hs), 
                nn.Tanh(), 
                nn.Linear(2*args.hs, args.hs), 
                nn.Tanh(), 
                nn.Linear(args.hs, 1),
                #nn.Sigmoid()
                )
    elif args.predictor_model == 'deep_1':
        predictor = nn.Sequential(
                nn.Linear(args.nz+layer_size, 64), 
                #nn.BatchNorm1d(64),
                nn.Tanh(), 
                nn.Linear(64, 256), 
                #nn.BatchNorm1d(256),
                nn.Tanh(), 
                nn.Linear(256, 1024), 
                #nn.BatchNorm1d(1024),
                nn.Tanh(), 
                nn.Linear(1024, 2048), 
                #nn.BatchNorm1d(2048),
                nn.Tanh(), 
                #nn.Linear(2048, 2048), 
                #nn.Tanh(), 
                nn.Linear(2048, 1024), 
                #nn.BatchNorm1d(1024),
                nn.Tanh(), 
                nn.Linear(1024, 256), 
                #nn.BatchNorm1d(256),
                nn.Tanh(), 
                nn.Linear(256, 64), 
                #nn.BatchNorm1d(64),
                nn.Tanh(), 
                nn.Linear(64, 1),
                #nn.Sigmoid()
                )
    elif args.predictor_model == 'deep_2':
        predictor = nn.Sequential(
                nn.Linear(args.nz+layer_size, 64), 
                nn.ReLU(), 
                nn.Linear(64, 256), 
                nn.ReLU(), 
                nn.Linear(256, 1024), 
                nn.ReLU(), 
                nn.Linear(1024, 2048), 
                nn.ReLU(), 
                nn.Linear(2048, 2048), 
                nn.ReLU(), 
                nn.Linear(2048, 1024), 
                nn.ReLU(), 
                nn.Linear(1024, 256), 
                nn.ReLU(), 
                nn.Linear(256, 64), 
                nn.ReLU(), 
                nn.Linear(64, 1),
                # nn.Sigmoid()
                )
    elif args.predictor_model == 'orig_1':
        predictor = nn.Sequential(
                nn.Linear(args.nz+layer_size, 256), 
                nn.ReLU(), 
                nn.Linear(256, 1024), 
                nn.ReLU(), 
                nn.Linear(1024, 256), 
                nn.ReLU(), 
                nn.Linear(256, 1),
                # nn.Sigmoid()
                )
    elif args.predictor_model == 'orig_2':
        predictor = nn.Sequential(
                nn.Linear(args.nz+layer_size, 256), 
                nn.ReLU(), 
                nn.Linear(256, 256), 
                nn.ReLU(), 
                nn.Linear(256, 256), 
                nn.ReLU(), 
                nn.Linear(256, 1),
                # nn.Sigmoid()
                )
    elif args.predictor_model == 'orig':
        predictor = nn.Sequential(
                nn.Linear(args.nz+layer_size, args.hs), 
                nn.Tanh(), 
                nn.Linear(args.hs, 1),
                #nn.Sigmoid()
                )
    else:
        raise ValueError("Invalid model.")
    if args.predictor_model == 'deep':
        predictor_energy = nn.Sequential(
                nn.Linear(args.nz+layer_size, args.hs), 
                nn.Tanh(), 
                nn.Linear(args.hs, 2*args.hs), 
                nn.Tanh(), 
                nn.Linear(2*args.hs, args.hs), 
                nn.Tanh(), 
                nn.Linear(args.hs, 1),
                #nn.Sigmoid()
                )
    elif args.predictor_model == 'deep_1':
        predictor_energy = nn.Sequential(
                nn.Linear(args.nz+layer_size, 64), 
                nn.Tanh(), 
                nn.Linear(64, 256), 
                nn.Tanh(), 
                nn.Linear(256, 1024), 
                nn.Tanh(), 
                nn.Linear(1024, 2048), 
                nn.Tanh(), 
                #nn.Linear(2048, 2048), 
                #nn.Tanh(), 
                nn.Linear(2048, 1024), 
                nn.Tanh(), 
                nn.Linear(1024, 256), 
                nn.Tanh(), 
                nn.Linear(256, 64), 
                nn.Tanh(), 
                nn.Linear(64, 1),
                # nn.Sigmoid()
                )
    elif args.predictor_model == 'deep_2':
        predictor_energy = nn.Sequential(
                nn.Linear(args.nz+layer_size, 64), 
                nn.ReLU(), 
                nn.Linear(64, 256), 
                nn.ReLU(), 
                nn.Linear(256, 1024), 
                nn.ReLU(), 
                nn.Linear(1024, 2048), 
                nn.ReLU(), 
                nn.Linear(2048, 2048), 
                nn.ReLU(), 
                nn.Linear(2048, 1024), 
                nn.ReLU(), 
                nn.Linear(1024, 256), 
                nn.ReLU(), 
                nn.Linear(256, 64), 
                nn.ReLU(), 
                nn.Linear(64, 1),
                # nn.Sigmoid()
                )
    elif args.predictor_model == 'orig_1':
        predictor_energy = nn.Sequential(
                nn.Linear(args.nz+layer_size, 256), 
                nn.ReLU(), 
                nn.Linear(256, 1024), 
                nn.ReLU(), 
                nn.Linear(1024, 256), 
                nn.ReLU(), 
                nn.Linear(256, 1),
                # nn.Sigmoid()
                )
    elif args.predictor_model == 'orig_2':
        predictor_energy = nn.Sequential(
                nn.Linear(args.nz+layer_size, 256), 
                nn.ReLU(), 
                nn.Linear(256, 256), 
                nn.ReLU(), 
                nn.Linear(256, 256), 
                nn.ReLU(), 
                nn.Linear(256, 1),
                # nn.Sigmoid()
                )
    elif args.predictor_model == 'orig':
        predictor_energy = nn.Sequential(
                nn.Linear(args.nz+layer_size, args.hs), 
                nn.Tanh(), 
                nn.Linear(args.hs, 1),
                #nn.Sigmoid()
                )
    else:
        raise ValueError("Invalid model.")

    model.mseloss = nn.MSELoss()
    # model.mseloss = nn.MSELoss(reduction='sum')
    model.predictor = predictor
    # torch.nn.init.xavier_uniform(model.predictor.weight)
    # model.mseloss = nn.MSELoss(reduction='sum')
    model.latency_loss= nn.L1Loss()
    # model.latency_loss= nn.L1Loss(reduction='sum')
    #model.latency_loss= nn.SmoothL1Loss()

    # model.predictor_energy = copy.deepcopy(predictor) 
    model.predictor_energy = predictor_energy 
    model.energy_loss= nn.L1Loss()
    # model.energy_loss= nn.L1Loss(reduction='sum')
    # model.energy_loss= nn.SmoothL1Loss(reduction='sum')
    # torch.nn.init.xavier_uniform(model.weight)



