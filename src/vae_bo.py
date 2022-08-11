from bayes_opt import BayesianOptimization
import argparse
import logging
import os
import pathlib
import math
import random
import sys
import torch
import test_util 

sys.path.insert(1, os.environ['COSA_DIR'] + '/src/')
from run_arch import gen_arch_yaml_from_config, gen_data, gen_dataset_col_title, append_dataset_csv, parse_results, fetch_arch_perf_data 

def eval_sample(vae_model, device, next_point_to_probe, obj, base_arch_path, arch_dir, output_dir, dataset_path, model, layer_idx, dnn_def_path):
    target_scale = 1e+14
    latent_config = []
    print("Next point to probe is:", next_point_to_probe)
    for i in range(len(next_point_to_probe)):
        # hw_config.append(next_point_to_probe[i] * scales[i])
        latent_config.append(next_point_to_probe[i] - 1)

    print("Sample latent feature:", latent_config)
    hw_config = vae_model.decode(torch.tensor(latent_config).to(device)).detach().cpu().tolist()
    print(f'Test hw config: {hw_config}')
    cycle, energy, area = test_util.eval(hw_config, base_arch_path, arch_dir, output_dir, dataset_path, model, layer_idx=layer_idx, dnn_def_path=dnn_def_path)
    if obj == 'edp':
        target = cycle * energy / target_scale 
    elif obj == 'latency':
        target = cycle
    elif obj == 'energy':
        target = energy
    else:
        raise
    return target


def bo_latent(vae_model, device, base_arch_path, arch_dir, output_dir, num_samples, model='resnet50', init_samples=0, random_seed=1, obj='edp', layer_idx=None, dnn_def_path=None):
    assert(num_samples > init_samples)

    output_dir = output_dir
    output_dir = f'{output_dir}_{model}_s{random_seed}' 
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arch_dir = arch_dir
    arch_dir = f'{arch_dir}_{model}_s{random_seed}' 
    arch_dir = pathlib.Path(arch_dir).resolve()

    dataset_path = output_dir / f'dataset_{model}_s{random_seed}.csv'
    with open(dataset_path,  'w') as f:
        key = gen_dataset_col_title()
        f.write(f'{key}\n')

    pbounds = {}
    bounds = [2,2,2,2]
    scales = [1,1,1,1]
    
    for i, bound in enumerate(bounds):
        pbounds[i] = (0, bound)

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2,
        random_state=random_seed,
    )
    
    from bayes_opt import UtilityFunction
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    init_data = []
    for iteration in range(init_samples):
        next_point_to_probe = optimizer.suggest(utility)
        target = eval_sample(vae_model, device, next_point_to_probe, obj, base_arch_path, arch_dir, output_dir, dataset_path, model, layer_idx=layer_idx, dnn_def_path=dnn_def_path)
        init_data.append((next_point_to_probe, target))

    for iteration in range(init_samples):
        next_point_to_probe, target = init_data[iteration]
        optimizer.register(
            params=next_point_to_probe,
            target=target,
        )

    min_target = None
    min_it = None
    for iteration in range(num_samples-init_samples):
        next_point_to_probe = optimizer.suggest(utility)
        target = eval_sample(vae_model, device, next_point_to_probe, obj, base_arch_path, arch_dir, output_dir, dataset_path, model, layer_idx=layer_idx, dnn_def_path=dnn_def_path)
        if min_target is None:
            min_target = target
            min_it = iteration
        else:
            if min_target > target:
                min_target = target
                min_it = iteration

        optimizer.register(
            params=next_point_to_probe,
            target=target,
        )
        print(next_point_to_probe)
        print(f'Iteration: {iteration}, Target: {target}, Min Target: {min_target}, Min Iteration: {min_it}')


