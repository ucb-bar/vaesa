#!/usr/bin/env python3 
import argparse
import pathlib
import copy
import itertools
import os
import subprocess
import re
import shutil
import sys
from test_util import parse_dnn_def, gen_dnn_predictors

_COSA_DIR = os.environ['COSA_DIR']
sys.path.insert(1, _COSA_DIR)

import util
import bo
from cosa.cosa_input_objs import Arch, Prob


def eval_arch(hw_config, obj, base_arch_path, arch_dir, output_dir, dataset_path, model, layer_idx, dnn_def_path):
    print(f'Test hw config: {hw_config}')
    cycle, energy, area = bo.eval(hw_config, base_arch_path, arch_dir, output_dir, dataset_path, model, layer_idx=layer_idx, dnn_def_path=dnn_def_path)
    if obj == 'edp':
        target = cycle * energy 
    elif obj == 'latency':
        target = cycle
    elif obj == 'energy':
        target = energy
    else:
        raise
    return target


def gen_existing_layer_json(output_dir='test_layers/exisiting_layers'):
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workload_dir = f'{_COSA_DIR}/configs/workloads' 
    workload_dir = pathlib.Path(workload_dir).resolve()
    
    arch_dir = '/gen_arch/'
    model_strs = ['alexnet', 'resnet50', 'resnext50_32x4d', 'deepbench']

    existing_layers = []
    for model_str in model_strs: 
        model_dir = workload_dir / (model_str+'_graph')
        layer_def_path = model_dir / 'unique_layers.yaml'
        layers = util.parse_yaml(layer_def_path)

        for layer_idx, layer in enumerate(layers): 
            try: 
                prob_path = model_dir / (layer + '.yaml') 
                prob = Prob(prob_path)
                prob_key = prob.config_str()
                existing_layers.append(prob_key)
                entry = prob.prob_bound 
                keys = ['Wstride', 'Hstride']
                entry.extend([prob.prob[key] for key in keys])
                single_layer_network = [entry]

                util.store_json(output_dir / f'{prob_key}.json', single_layer_network)
            except:
                raise

    output_dir = 'test_layers/other_layers'
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_strs = ['densenet161', 'vgg16', 'dlrm']
    existing_layers = []
    for model_str in model_strs: 
        model_dir = workload_dir / (model_str+'_graph')
        layer_def_path = model_dir / 'unique_layers.yaml'
        layers = util.parse_yaml(layer_def_path)

        for layer_idx, layer in enumerate(layers): 
            try: 
                prob_path = model_dir / (layer + '.yaml') 
                prob = Prob(prob_path)
                prob_key = prob.config_str()
                if prob_key in existing_layers:
                    continue
                existing_layers.append(prob_key)
                entry = prob.prob_bound 
                keys = ['Wstride', 'Hstride']
                entry.extend([prob.prob[key] for key in keys])
                single_layer_network = [entry]

                util.store_json(output_dir / f'{prob_key}.json', single_layer_network)
            except:
                raise
 

def search_all_layers(layer_dir, model, target_model, search_samples, search_optimizer, search_lr, search_seed, res_dir, obj, device, data_type, norm_latent, log_layerfeat, norm_layerfeat, norm_layerfeat_option, log_obj, norm_obj, norm_path):
    sgd_steps = 200
    sgd_indice = [0, sgd_steps//2, sgd_steps]
    layer_defs = pathlib.Path(layer_dir).glob('*.json')
   
    _COSA_DIR = os.environ['COSA_DIR']
    base_arch_path = f'{_COSA_DIR}/configs/arch/simba_dse.yaml' 
    arch_dir = pathlib.Path(f'arch_all_layers_{search_seed}')

    for dnn_def_path in layer_defs:
        # load network def
        dnn_def_tensor, num_predictors = parse_dnn_def(dnn_def_path, device, log_layerfeat=log_layerfeat, norm_layerfeat=norm_layerfeat, norm_layerfeat_option=norm_layerfeat_option, norm_path=norm_path)
        
        # generate predictors for inference
        gen_dnn_predictors(model, num_predictors)
        
        # perform search
        postfix = dnn_def_path.name.replace('.json','')

        sgd_design_dicts = model.dnn_search(search_samples, search_optimizer, dnn_def_tensor, lr=search_lr, obj=obj, norm_latent=norm_latent, sgd_indice=sgd_indice, sgd_steps=sgd_steps)

        # print(search_samples)
        for i, sgd_design_dict in enumerate(sgd_design_dicts):
            for sgd_step, sgd_design in sgd_design_dict.items():
                search_dir = os.path.join(res_dir, f'dnn_search_{postfix}_sgd{sgd_step}_s{search_seed}')
                print(f'search_dir {search_dir}')
                if not os.path.exists(search_dir):
                    os.makedirs(search_dir)
                output_dir = pathlib.Path(search_dir) 
                layer_idx = None

                dataset_path = layer_dir / f'dataset_{postfix}_sgd{sgd_step}_s{search_seed}.csv'
                print(f'dataset_path {dataset_path}')
                namei = '{}_graph_{}'.format(search_optimizer, i)
                util.plot_searched_points(sgd_design, search_dir, namei)
                target_model = f'new_{postfix}_sgd{sgd_step}_s{search_seed}'
                eval_arch(sgd_design[0].tolist(), obj, base_arch_path, arch_dir, output_dir, dataset_path, target_model, layer_idx, dnn_def_path)
       
        # finded_graphs_path = os.path.join(search_dir, 'finded_graphs.pkl')
        # with open(finded_graphs_path, 'wb') as f:
        #     pickle.dump(graphs, f)

    sys.exit(0)

if __name__ == "__main__":
    gen_existing_layer_json()

