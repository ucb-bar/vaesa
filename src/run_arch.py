#!/usr/bin/env python3 
import argparse
import pathlib
import copy
import itertools
import os
import subprocess
import re
import shutil
import pickle
import numpy as np
import traceback

import util
from cosa.cosa_input_objs import Arch, Prob
from run_dnn_models import run_dnn_models

_COSA_DIR = os.environ['COSA_DIR']
# _VAESA_SRC_DIR = 

def construct_argparser():

    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default='output_dir_dataset',
                        )
    parser.add_argument('-a',
                        '--arch_dir',
                        type=str,
                        help='Generated Archtecture Folder',
                        default='gen_arch_dataset',
                        )
    parser.add_argument('-bap',
                        '--base_arch_path',
                        type=str,
                        help='Hardware Architecture Path',
                        default=f'{_COSA_DIR}/configs/arch/simba_dse_v3.yaml',
                        )
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        help='DNN Model Workload',
                        default='resnet50',
                        )
    return parser


def gen_arch_yaml_from_config(base_arch_path, arch_dir, hw_configs, config_prefix, arch_v3=False):
    # Get base arch dictionary
    base_arch = util.parse_yaml(base_arch_path)

    # Create directory for new arch files, if necessary
    new_arch_dir = arch_dir
    new_arch_dir.mkdir(parents=True, exist_ok=True)
    new_arch = copy.deepcopy(base_arch)

    if arch_v3: 
        # parse hw config
        arith_meshX,arith_ins, \
            mem1_depth,mem1_blocksize,mem1_ports,mem1_banks, \
            mem2_depth,mem2_blocksize,mem2_ports,mem2_banks, \
            mem3_depth,mem3_blocksize,mem3_ports,mem3_banks, \
            mem4_depth,mem4_blocksize,mem4_ports,mem4_banks = hw_configs

        buf_attributes = {
            1: {"depth": mem1_depth, "blocksize": mem1_blocksize, "ports": mem1_ports, "banks": mem1_banks}, # acc
            2: {"depth": mem2_depth, "blocksize": mem2_blocksize, "ports": mem2_ports, "banks": mem2_banks}, # weight
            3: {"depth": mem3_depth, "blocksize": mem3_blocksize, "ports": mem3_ports, "banks": mem3_banks}, # input
            4: {"depth": mem4_depth, "blocksize": mem4_blocksize, "ports": mem4_ports, "banks": mem4_banks}, # global
        }

        # arith_meshX,arith_ins,mem1_ent,mem2_ent,mem3_ent,mem4_ent = hw_configs 

        arch_invalid = False
        new_arch_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]
        new_meshX = arith_meshX - 1
        new_arch_dict["subtree"][0]["name"] = f"PE[0..{new_meshX}]" 

        new_storage = new_arch_dict["subtree"][0]["local"]
        new_arith = new_arch_dict["subtree"][0]["local"][4]["attributes"]
        new_arith["meshX"] = arith_meshX

        for mem_lvl in range(1, 4): # mem_lvl is mem lvl, i is index in timeloop list of SRAMs
            i = 3 - mem_lvl  # accbuf (lvl 1, index 2), weightbuf (lvl 2, index 1), inputbuf (lvl 3, index 0)
            if "meshX" in new_storage[i]["attributes"]:
                new_storage[i]["attributes"]["meshX"] = arith_meshX
                # Check whether meshX divides num instances of all buffers
                if new_storage[i]["attributes"]["instances"] % new_storage[i]["attributes"]["meshX"] != 0:
                    print("Arch invalid")
                    print("Instances:", new_storage[i]["attributes"]["instances"])
                    print("meshX:", new_storage[i]["attributes"]["meshX"])
                    arch_invalid = True

            depth = buf_attributes[mem_lvl]["depth"]
            blocksize = buf_attributes[mem_lvl]["blocksize"]
            word_bits = new_storage[i]["attributes"]["word-bits"]
            new_storage[i]["attributes"]["entries"] = depth * blocksize
            new_storage[i]["attributes"]["depth"] = depth
            new_storage[i]["attributes"]["width"] = blocksize * word_bits
            new_storage[i]["attributes"]["n_rdwr_ports"] = buf_attributes[mem_lvl]["ports"]
            new_storage[i]["attributes"]["n_banks"] = buf_attributes[mem_lvl]["banks"]

            # # Check whether SRAM size is at least 64
            # entries = new_storage[i]["attributes"]["entries"]
            # banks = new_storage[i]["attributes"]["n_banks"]
            # if (entries // banks) < 64:
            #     print("Arch invalid:")
            #     print("Mem lvl", mem_lvl, "Entries:", entries, "Banks:", banks)
            #     arch_invalid = True

        # global buffer
        new_gb_dict = new_arch_dict["local"][0]
        gb_depth = buf_attributes[4]["depth"]
        gb_blocksize = buf_attributes[4]["blocksize"]
        gb_word_bits = new_gb_dict["attributes"]["word-bits"]
        new_gb_dict["attributes"]["entries"] = gb_depth * gb_blocksize
        new_gb_dict["attributes"]["depth"] = gb_depth
        new_gb_dict["attributes"]["width"] = gb_blocksize * gb_word_bits
        new_gb_dict["attributes"]["n_rdwr_ports"] = buf_attributes[4]["ports"]
        new_gb_dict["attributes"]["n_banks"] = buf_attributes[4]["banks"]
        
        # # Check whether SRAM size is at least 64
        # entries = new_gb_dict["attributes"]["entries"]
        # banks = new_gb_dict["attributes"]["n_banks"]
        # if (entries // banks) < 64:
        #     print("Arch invalid:")
        #     print("Mem lvl", 4, "Entries:", entries, "Banks:", banks)
        #     arch_invalid = True

        if arch_invalid:
            raise('Arch invalid!')

        # MAC
        new_arith["instances"] = int(arith_ins) * 128 
        # Set registers to match MACs
        new_storage[3]["attributes"]["instances"] = int(arith_ins) * 128  
        new_storage[3]["attributes"]["meshX"] = arith_meshX

    else:
        # parse hw config
        arith_meshX,arith_ins,mem1_ent,mem2_ent,mem3_ent,mem4_ent = hw_configs 

        # Get nested dictionaries
        base_arith = base_arch["arch"]["arithmetic"]
        new_arith = new_arch["arch"]["arithmetic"]
        base_storage = base_arch["arch"]["storage"]
        new_storage = new_arch["arch"]["storage"]
        new_arith["meshX"] = arith_meshX
        new_arith["instances"] = int(arith_ins) * 128
        new_storage[0]["instances"] = int(arith_ins) * 128 
        new_storage[1]["entries"] = mem1_ent
        new_storage[2]["entries"] = mem2_ent
        new_storage[3]["entries"] = mem3_ent
        new_storage[4]["entries"] = mem4_ent

        for i in range(5):
            if "meshX" in new_storage[i]:
                new_storage[i]["meshX"] = arith_meshX
            
    hw_configs_arr = [str(i) for i in hw_configs]
    hw_configs_str = "_".join(hw_configs_arr)

    # Construct filename for new arch
    config_str = get_hw_config_str(hw_configs, config_prefix, arch_v3)

    # Save new arch
    new_arch_path = new_arch_dir.resolve() / config_str
    util.store_yaml(new_arch_path, new_arch)
    return config_str


def get_hw_config_str(hw_configs, config_prefix, arch_v3=False):
    hw_configs_arr = [str(i) for i in hw_configs]
    hw_configs_str = "_".join(hw_configs_arr)

    # Construct filename for new arch
    config_str = f"arch_{config_prefix}"
    config_str += hw_configs_str 
    if arch_v3: 
        config_str += "_v3"
    config_str += ".yaml"
    return config_str


def gen_arch_yaml(base_arch_path, arch_dir):
    # Get base arch dictionary
    base_arch = util.parse_yaml(base_arch_path)

    # Create directory for new arch files, if necessary
    new_arch_dir = arch_dir
    new_arch_dir.mkdir(parents=True, exist_ok=True)

    buf_multipliers = [0.5, 1, 2, 4]
    buf_multipliers_perms = [p for p in itertools.product(buf_multipliers, repeat=4)]

    for pe_multiplier in [0.25, 1, 4]:
        for mac_multiplier in [0.25, 1, 4]:
            for buf_multipliers_perm in buf_multipliers_perms:
                new_arch = copy.deepcopy(base_arch)
                base_arch_dict = base_arch["architecture"]["subtree"][0]["subtree"][0]
                new_arch_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]
                
                print(f"{pe_multiplier} {mac_multiplier} {buf_multipliers_perm}")
                base_meshX_str =  base_arch_dict["subtree"][0]["name"]
                m = re.search("PE\[0..(\S+)\]", base_meshX_str)
                if not m:
                    raise ValueError("Wrong mesh-X specification.")
                base_meshX = int(m.group(1)) + 1
                new_meshX = int(base_meshX * pe_multiplier) - 1 
                new_arch_dict["subtree"][0]["name"] = f"PE[0..{new_meshX}]" 

                # Get nested dictionaries
                base_arith = base_arch_dict["subtree"][0]["local"][4]["attributes"] 
                new_arith = new_arch_dict["subtree"][0]["local"][4]["attributes"]


                base_storage = base_arch_dict["subtree"][0]["local"]
                new_storage = new_arch_dict["subtree"][0]["local"]

                arch_invalid = False

                # PE and buffer
                new_arith["meshX"] = int(base_arith["meshX"] * pe_multiplier)
                
                # PE buffers 
                for i in range(3): # Ignoring DRAM
                    if "meshX" in new_storage[i]["attributes"]:
                        new_storage[i]["attributes"]["meshX"] = int(base_storage[i]["attributes"]["meshX"] * pe_multiplier)

                        # Check whether meshX divides num instances of all buffers
                        if new_storage[i]["attributes"]["instances"] % new_storage[i]["attributes"]["meshX"] != 0:
                            print("Arch invalid")
                            print("Instances:", new_storage[i]["attributes"]["instances"])
                            print("meshX:", new_storage[i]["attributes"]["meshX"])
                            arch_invalid = True

                    # if i != 0: # Ignoring registers
                    new_storage[i]["attributes"]["entries"] = int(base_storage[i]["attributes"]["entries"] * buf_multipliers_perm[i])

                # global buffer
                base_gb_dict = base_arch_dict["local"][0]
                new_gb_dict = new_arch_dict["local"][0]
                new_gb_dict["attributes"]["entries"] = int(base_gb_dict["attributes"]["entries"] * buf_multipliers_perm[3])
                    
                if arch_invalid:
                    continue

                # MAC
                new_arith["instances"] = int(base_arith["instances"] * mac_multiplier)

                # Set registers to match MACs
                new_storage[3]["attributes"]["instances"] = new_arith["instances"]
                new_storage[3]["attributes"]["meshX"] = int(base_storage[3]["attributes"]["meshX"] * pe_multiplier)

                # Construct filename for new arch
                config_str = "arch" + "_pe" + str(pe_multiplier) +   \
                                      "_mac" + str(mac_multiplier) + \
                                      "_buf"
                for multiplier in buf_multipliers_perm:
                    config_str += "_" + str(multiplier)
                config_str += "_v3.yaml"
                
                # Save new arch
                new_arch_path = new_arch_dir.resolve() / config_str
                util.store_yaml(new_arch_path, new_arch)


def gen_dataset_col_title(with_layer=False, arch_v3=False, with_buf_ratios=False):
    col_str = ['name', 'unique_cycle_sum', 'unique_energy_sum', 'area', 'arith_meshX', 'arith_ins']
    if arch_v3:
        for i in range(5):
            col_str.extend([f'mem{i}_ins', f'mem{i}_ent', f'mem{i}_dep', f'mem{i}_blksiz', f'mem{i}_port', f'mem{i}_bank'])
    else:
        for i in range(5):
            col_str.extend([f'mem{i}_ins', f'mem{i}_ent'])
    if with_layer:
        for i in range(9):
            col_str.append(f'prob_{i}') 
    if with_buf_ratios:
        for i in range(1, 5):
            for tensor in ['W', 'IA', 'OA']:
                col_str.append(f'ratio_mem{i}_{tensor}')
    key = ','.join(col_str)
    return key


def gen_dataset_csv(data, dataset_path, arch_v3=False, with_buf_ratios=False):
    with open(dataset_path,  'w') as f:
        key = gen_dataset_col_title(arch_v3=arch_v3, with_buf_ratios=with_buf_ratios)
        f.write(f'{key}\n')
        for d in data:
            key = d[0]
            col_str = ','.join(d[1])
            f.write(f'{key},{col_str}\n')


def append_dataset_csv(data, dataset_path):
    with open(dataset_path,  'a') as f:
        for d in data:
            key = d[0]
            col_str = ','.join(d[1])
            f.write(f'{key},{col_str}\n')
            

def parse_results(output_dir, config_str, unique_sum=True, model='resnet50', layer_idx=None, workload_dir=f'{_COSA_DIR}/configs/workloads', arch_v3=False, buf_str=None):
    # if network is None, return sum of 4 networks
    # if layer is None, reuturn sum of specific network
    if not buf_str:
        buf_str = ''
    cycle_path = output_dir / f'results_{config_str}{buf_str}_cycle.json'
    energy_path = output_dir / f'results_{config_str}{buf_str}_energy.json'
    area_path = output_dir /f'results_{config_str}{buf_str}_area.json'
    area = -1
    print(f'path: {cycle_path}') 

    if layer_idx is not None: 
        layer_idx = int(layer_idx)
        workload_dir = pathlib.Path(workload_dir).resolve()
        model_dir = workload_dir / (model+'_graph')
        layer_def_path = model_dir / 'unique_layers.yaml'
        layers = util.parse_yaml(layer_def_path)
        layer = list(layers)[layer_idx]
        prob_path = model_dir / (layer + '.yaml')
        prob = Prob(prob_path)
        prob_key = prob.config_str()

        cycle = util.parse_json(cycle_path)[model][prob_key]
        energy = util.parse_json(energy_path)[model][prob_key]
        if arch_v3: 
            area = util.parse_json(area_path)[model][prob_key]
    else: 
        if unique_sum: 
            workload_dir = pathlib.Path(workload_dir)
            model_dir = workload_dir / (model + '_graph')
            layer_def_path = model_dir / 'unique_layers.yaml'
            layers = util.parse_yaml(layer_def_path)
            num_unique_layers = len(layers)
            print(f'Target model dir {model_dir}, layer def path {layer_def_path}, num layers {num_unique_layers}')

            try:
                num_layers = len(util.parse_json(cycle_path)[model].values())
                if num_layers != num_unique_layers:
                    return -1, -1, -1
                
                cycle = sum(util.parse_json(cycle_path)[model].values())
                energy = sum(util.parse_json(energy_path)[model].values())
                if arch_v3:
                    area = list(util.parse_json(area_path)[model].values())[0]
            except:
                return -1, -1, -1
        else:
            # Load aggregated results JSON files
            cycle_dict = util.parse_json(cycle_path)
            energy_dict = util.parse_json(energy_path)

            # Load the layer count file for the selected model
            workload_dir = pathlib.Path(workload_dir).resolve()
            model_dir = workload_dir / (model+'_graph')
            layer_count_path = model_dir / ('layer_count.yaml')
            layer_counts_model = util.parse_yaml(layer_count_path)
            
            # Compute total cycle count/energy
            cycle = total_layer_values(cycle_dict[model], layer_counts_model)
            energy = total_layer_values(energy_dict[model], layer_counts_model)
            
            # Just one value for area
            if arch_v3:
                area = list(util.parse_json(area_path)[model].values())[0]
    return cycle, energy, area


def fetch_arch_perf_data(new_arch_dir, output_dir, glob_str='arch_pe*_v3.yaml', arch_v3=False, mem_levels=5, model_cycles=False, model='resnet50', layer_idx=None, unique_sum=True, workload_dir=f'{_COSA_DIR}/configs/workloads', obj='edp', buf_ratios=None):
    data, _ = fetch_arch_perf_data_func(new_arch_dir, output_dir, glob_str=glob_str, arch_v3=arch_v3, mem_levels=mem_levels, model_cycles=model_cycles, model=model, layer_idx=layer_idx, unique_sum=unique_sum, workload_dir=workload_dir, obj=obj, buf_ratios=buf_ratios)
    return data


def fetch_arch_perf_data_func(new_arch_dir, output_dir, glob_str='arch_pe*_v3.yaml', arch_v3=False, mem_levels=5, model_cycles=False, model='resnet50', layer_idx=None, unique_sum=True, workload_dir=f'{_COSA_DIR}/configs/workloads', obj='edp', buf_ratios=None):
    # Get all arch files
    arch_files = list(new_arch_dir.glob(glob_str))
    arch_files.sort()
    data = []
    print(len(arch_files))
    
    min_metric = None
    output_files = list(output_dir.glob(glob_str))
    output_files.sort()
    for output_file in output_files:
        if arch_v3:
            arch_file = pathlib.Path(new_arch_dir) / pathlib.Path(output_file.name.split("v3")[0] + "v3.yaml")
            buf_ratios = output_file.name.split("v3_")[-1].split("_")
        else:
            arch_file = pathlib.Path(new_arch_dir) / output_file.name + ".yaml"

        base_arch_str = arch_file.name 
        m = re.search("(\S+).yaml", base_arch_str)
        if not m:
            raise ValueError("Wrong config string format.")
        config_str = m.group(1)
        print(config_str)

        try:
            data_entry, metric = fetch_one_arch_perf_data(arch_file, output_dir, config_str, arch_v3, mem_levels, model, layer_idx, unique_sum, workload_dir, obj, buf_ratios)
            if metric > 0: 
                if min_metric: 
                    if metric < min_metric:
                        print(config_str)
                        min_metric = metric
                else:
                    min_metric = metric
                print(f'{obj}: {metric}')
                print(f'min_{obj}: {min_metric}')

                data.append((config_str, data_entry))
        except:
            print(f"Could not fetch data for {arch_file}, buf ratios {buf_ratios}")
            print(traceback.format_exc())

    return data, min_metric

def fetch_one_arch_perf_data(arch_file, output_dir, config_str, arch_v3, mem_levels, model, layer_idx, unique_sum, workload_dir, obj, buf_ratios):
    new_arch = util.parse_yaml(arch_file)
    config_v3_str = ""
    if arch_v3: 
        base_arch_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]
        base_meshX_str =  base_arch_dict["subtree"][0]["name"]
        m = re.search("PE\[0..(\S+)\]", base_meshX_str)
        if not m:
            raise ValueError("Wrong mesh-X specification.")
        base_meshX = int(m.group(1)) + 1

        base_arith = base_arch_dict["subtree"][0]["local"][4]["attributes"] 
        base_storage = base_arch_dict["subtree"][0]["local"]
        data_entry = [str(base_meshX), str(base_arith["instances"])]
        
        for i in reversed(range(4)): 
            attr = base_storage[i]["attributes"]
            data_entry.extend([
                str(attr["instances"]),
                str(attr["entries"]),
                str(attr["depth"]),
                str(attr["width"] // attr["word-bits"]),
                str(attr["n_rdwr_ports"]),
                str(attr["n_banks"])
            ])
        base_gb_dict = base_arch_dict["local"][0]
        gb_attr = base_gb_dict["attributes"]
        data_entry.extend([
            str(gb_attr["instances"]),
            str(gb_attr["entries"]),
            str(gb_attr["depth"]),
            str(gb_attr["width"] // gb_attr["word-bits"]),
            str(gb_attr["n_rdwr_ports"]),
            str(gb_attr["n_banks"])
        ])
    else:
        # Get nested dictionaries
        new_arith = new_arch["arch"]["arithmetic"]
        new_storage = new_arch["arch"]["storage"]

        data_entry = [str(new_arith["meshX"]), str(new_arith["instances"]), ]
        for i in range(mem_levels): # Ignoring DRAM
            data_entry.extend([str(new_storage[i]["instances"]), str(new_storage[i]["entries"])])

    # Get the labels 
    buf_ratios_str = [str(r) for r in list(np.array(buf_ratios).flatten())]
    buf_str = "_" + "_".join(buf_ratios_str)
    cycle, energy, area = parse_results(output_dir, config_str, unique_sum=unique_sum, model=model, layer_idx=layer_idx, workload_dir=workload_dir, buf_str=buf_str, arch_v3=arch_v3)
    metric = 0
    if obj == 'edp' and cycle > 1 and energy > 1:
        metric = cycle * energy
    elif obj == 'adp' and area > 1 and cycle > 1:
        metric = area * cycle
    elif obj == 'latency':
        metric = cycle
    elif obj == 'energy':
        metric = energy

    # data_entry = [str(cycle), str(energy)] + [str(area), str(edp), str(adp)]  + data_entry
    data_entry = [str(cycle), str(energy), str(area)] + data_entry + buf_ratios_str

    return data_entry, metric

def gen_dataset(new_arch_dir, output_dir, glob_str='arch_pe*_v3.yaml', model='resnet50', arch_v3=False, mem_levels=5, model_cycles=False, postfix='', obj='edp', layer_idx=None):
    config_str = glob_str.replace('_*.yaml', '')
    dataset_path = output_dir / f'dataset{postfix}.csv'
    print(dataset_path)

    data, min_metric = fetch_arch_perf_data_func(new_arch_dir, output_dir, model=model, glob_str=glob_str, arch_v3=arch_v3, mem_levels=mem_levels, model_cycles=model_cycles, obj=obj, layer_idx=layer_idx)
    gen_dataset_csv(data, dataset_path)
    return min_metric


def gen_dataset_per_layer(output_dir='output_dir', model='resnet50', arch_v3=False, mem_levels=5, model_cycles=False, postfix='', arch_dir = 'gen_arch'):
    workload_dir = '../configs/workloads' 
    workload_dir = pathlib.Path(workload_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    
    arch_dir = pathlib.Path(arch_dir).resolve()
    glob_str = 'arch_*' 
    config_str = glob_str.replace('_*.yaml', '')

    # model_strs = ['alexnet', 'resnet50', 'resnext50_32x4d', 'deepbench']
    model_strs = [model]

    for model_str in model_strs: 
        model_dir = workload_dir / (model_str+'_graph')
        layer_def_path = model_dir / 'unique_layers.yaml'
        layers = util.parse_yaml(layer_def_path)

        for layer_idx, layer in enumerate(layers): 
            try: 
                prob_path = model_dir / (layer + '.yaml') 
                prob = Prob(prob_path)
                prob_key = prob.config_str()

                dataset_path = output_dir / f'dataset_{prob_key}.csv'

                data = fetch_arch_perf_data(arch_dir, output_dir, glob_str=glob_str, arch_v3=arch_v3, mem_levels=mem_levels, model_cycles=model_cycles, model=model_str, layer_idx=layer_idx)
                gen_dataset_csv(data, dataset_path, arch_v3=arch_v3, with_buf_ratios=arch_v3)
                print(f'gen: {dataset_path}')
            except:
                # raise
                print(traceback.format_exc())


def gen_dataset_per_network(output_dir='output_dir', arch_v3=False, mem_levels=5, model_cycles=False, postfix='', arch_dir='gen_arch'):
    unique_sum = True 
    workload_dir = '../configs/workloads' 
    workload_dir = pathlib.Path(workload_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    
    arch_dir = pathlib.Path(arch_dir).resolve()
    glob_str = 'arch_*.yaml' 
    config_str = glob_str.replace('_*.yaml', '')

    model_strs = ['alexnet', 'resnet50', 'resnext50_32x4d', 'deepbench']

    for model_str in model_strs: 
        if unique_sum: 
            dataset_path = output_dir / f'dataset_{model_str}.csv'
        else:
            dataset_path = output_dir / f'dataset_{model_str}_actual.csv'
        try:
            data = fetch_arch_perf_data(arch_dir, output_dir, glob_str=glob_str, arch_v3=arch_v3, mem_levels=mem_levels, model_cycles=model_cycles, model=model_str, layer_idx=None, unique_sum=unique_sum)
            gen_dataset_csv(data, dataset_path)
            print(f'gen: {dataset_path}')
        except:
            raise
            # continue


def gen_dataset_all_layer(per_layer_dataset_dir='/scratch/qijing.huang/cosa_dataset/db/layer_db/all', output_dir='/scratch/qijing.huang/cosa_dataset/db/all_layers_db'):
    per_layer_dataset_dir = pathlib.Path(per_layer_dataset_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    per_layer_dataset_files = per_layer_dataset_dir.glob('dataset_*.csv')
    
    dataset_path = output_dir / 'dataset_all_layer.csv'
    with open(dataset_path,  'w') as f:
        key = gen_dataset_col_title(with_layer=True)
        f.write(f'{key}\n')

        per_layer_dataset_files = list(per_layer_dataset_files)
        first_layer_dataset_file = per_layer_dataset_files[0]
        per_arch_data = util.parse_csv(first_layer_dataset_file)
        
        for arch_idx, arch_data in enumerate(per_arch_data[1:]):
            for per_layer_dataset_file in per_layer_dataset_files:
                layer_line = util.parse_csv_line(per_layer_dataset_file, arch_idx+1)
                layer_config_str = per_layer_dataset_file.name
                layer_config_str = layer_config_str.replace('.csv','')
                layer_config_str = layer_config_str.replace('dataset_','')
                layer_config = layer_config_str.split('_')[:-2]
                new_data = layer_line + layer_config
                new_data_str = ','.join(new_data) 
                f.write(f'{new_data_str}\n')


def merge_dataset_per_layer(per_layer_dataset_dir='/scratch/qijing.huang/cosa_dataset/db/layer_db/grid', output_dir='/scratch/qijing.huang/cosa_dataset/db/layer_db/all'):
    per_layer_dataset_dir = pathlib.Path(per_layer_dataset_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_layer_dataset_files = per_layer_dataset_dir.glob('dataset_*.csv')
    per_layer_dataset_files = list(per_layer_dataset_files)
    # first_layer_dataset_file = per_layer_dataset_files[0]
    # per_arch_data = util.parse_csv(first_layer_dataset_file)
    
    seeds = [6,7,888,9999,987654321,123456]
    # target_dir = '/scratch/qijing.huang/cosa_dataset/db/layer_db' 
    target_dir = '/nscratch/qijing.huang/cosa/layer_db' 
    target_dir = pathlib.Path(target_dir)
    per_layer_target_dirs = list(target_dir.glob('*'))
    print(len(per_layer_dataset_files))
    print(len(per_layer_target_dirs))
    # per_layer_dataset_files = per_layer_dataset_files[12:13]
    for per_layer_dataset_file in per_layer_dataset_files:
        per_layer_dataset_file = pathlib.Path(per_layer_dataset_file)
        dataset_path = output_dir / per_layer_dataset_file.name 
        # with open(target_dataset_path, 'r') as f:
        #     data = f.readlines()

        # with open(dataset_path,  'w') as f:
        #     f.write(f'{key}\n')
        shutil.copyfile(per_layer_dataset_file, dataset_path)
        # for seed in seeds:
        print(dataset_path)
        for per_layer_target_dir in per_layer_target_dirs: 
            # target_dataset_path = target_dir / f'random_s{seed}' / per_layer_dataset_file.name  
            target_dataset_path = per_layer_target_dir / per_layer_dataset_file.name  
            if target_dataset_path.is_file():
                print(target_dataset_path)
                with open(target_dataset_path, 'r') as f:
                    data = f.readlines()
                    with open(dataset_path,  'a') as f:
                        for d in data[1:]:
                            f.write(d)



# find layer with max diff
def gen_dataset_max_diff(per_layer_dataset_dir='/scratch/qijing.huang/cosa_dataset/db/layer_db', output_dir='/scratch/qijing.huang/cosa_dataset/db/'):
    per_layer_dataset_dir = pathlib.Path(per_layer_dataset_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    per_layer_dataset_files = per_layer_dataset_dir.glob('dataset_*.csv')
    
    per_layer_dataset_files = list(per_layer_dataset_files)
    first_layer_dataset_file = per_layer_dataset_files[0]
    per_arch_data = util.parse_csv(first_layer_dataset_file)
    
    diffs = []
    variances = [] 
    per_layer_dataset_files = per_layer_dataset_files[12:13]
    for per_layer_dataset_file in per_layer_dataset_files:
    # for arch_idx, arch_data in enumerate(per_arch_data):
        # layer_line = util.parse_csv_line(per_layer_dataset_file, arch_idx+1)
        # new_data = layer_line + layer_config
        # new_data_str = ','.join(new_data) 
        min_perf, _ = parse_best_results(per_layer_dataset_file, n_entries=None, obj='edp', func='min')
        max_perf, _ = parse_best_results(per_layer_dataset_file, n_entries=None, obj='edp', func='max')
        var_perf, _ = parse_best_results(per_layer_dataset_file, n_entries=None, obj='edp', func='var')
        print(f'min: {min_perf}, max: {max_perf}, var: {var_perf}')
        diff = (max_perf - min_perf) / min_perf
        diffs.append(diff)
        variances.append(var_perf)

    import numpy as np
    diffs_np = np.array(diffs)
    vars_np = np.array(variances)
    
    max_diff = np.max(diffs_np)
    max_diff_idx = np.argmax(diffs_np)
    file_name = per_layer_dataset_files[max_diff_idx]
    print(f'max_diff {max_diff}, max_diff_idx {max_diff_idx}, file {file_name}')

    max_var = np.max(vars_np)
    max_var_idx = np.argmax(vars_np)
    file_name = per_layer_dataset_files[max_var_idx]
    print(f'max_var {max_var}, max_var_idx {max_var_idx}, file {file_name}')

    



def get_best_entry(data, metric_idx=[1,2], func='min'):
    import numpy as np
    best_perf = None
    best_entry = None
    data_arr = []
    for line, per_arch_data in enumerate(data): 
        perf_prod = 1.0
        for entry in metric_idx:
            perf_prod *= float(per_arch_data[entry])
        data_arr.append(perf_prod)
   
    data_np = np.array(data_arr)
    if func == 'min':
        print(data_np)
        best_perf = np.min(data_np)
        best_entry_idx = np.argmin(data_np) 
        best_entry = data[best_entry_idx] 
    elif func == 'max':
        best_perf = np.max(data_np)
        best_entry_idx = np.argmax(data_np) 
        best_entry = data[best_entry_idx] 
    elif func == 'mean':
        best_perf = np.mean(data_np)
        best_entry = None
    elif func == 'var':
        best_perf = np.var(data_np)
        best_entry = None 
    else:
        raise ValueError("Func not valid.")
    #     if line == 0:
    #         best_perf = perf_prod
    #         best_entry = per_arch_data
    #     if metric == 'min':
    #         if perf_prod < best_perf:
    #             best_perf = perf_prod
    #             best_entry = per_arch_data
    #     elif metric == 'max':
    #         if perf_prod > best_perf:
    #             best_perf = perf_prod
    #             best_entry = per_arch_data

    return best_perf, best_entry


def parse_best_results(dataset_path, n_entries=None, obj='edp', func='min'):
    data = util.parse_csv(dataset_path)
    if n_entries is None:
        data = data[1:]
    else:
        n_entries = int(n_entries)
        data = data[1: n_entries+1]
    if obj == 'edp':
       metric_idx = [1,2] 
    elif obj == 'latency':
       metric_idx = [1] 
    elif obj == 'energy':
        metric_idx = [2] 

    best_metric, best_entry = get_best_entry(data, metric_idx=metric_idx, func=func) 
    print(f'dataset_path: {dataset_path}') 
    print(f'best_entry: {best_entry}') 
    print(f'best_metric: {best_metric}') 
    return best_metric, best_entry


def gen_dataset_all(per_network_dataset_dir='output_dir', unique_sum=True):
    model_strs = ['alexnet', 'resnet50', 'resnext50_32x4d', 'deepbench']
    
    per_network_dataset_dir = pathlib.Path(per_network_dataset_dir).resolve()
    dataset_path = per_network_dataset_dir / 'dataset_all.csv'
    if not unique_sum: 
        dataset_path = per_network_dataset_dir / 'dataset_all_actual.csv'
    network_data = []
    for model_str in model_strs:
        if unique_sum: 
            path = per_network_dataset_dir / f'dataset_{model_str}.csv' 
        else:
            path = per_network_dataset_dir / f'dataset_{model_str}_actual.csv' 

        per_arch_data = util.parse_csv(path)
        network_data.append(per_arch_data)    
    
    all_data = network_data[0].copy()
    for line, per_arch_data in enumerate(all_data): 
        if line == 0:
            continue
        for entry in [1,2]:
            entry_network_sum = 0.0
            for network_idx in range(len(model_strs)):
                entry_network_sum += float(network_data[network_idx][line][entry])
            all_data[line][entry] = str(entry_network_sum)
    with open(dataset_path, 'w') as f:
        for line in all_data:
            f.write(','.join(line))
    print(f'gen: {dataset_path}')

def gen_data(new_arch_dir, output_dir, glob_str='arch_pe*_v3.yaml', model='resnet50', buf_ratios=None, layer_idx=None, dnn_def_path=None, run_serial=False):
    if not buf_ratios:
        buf_ratios = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0.25, 0.75]]
    print(buf_ratios)

    # Get all arch files
    arch_files = list(new_arch_dir.glob(glob_str))
    arch_files.sort()
    print(arch_files)
    
    if run_serial:
        workload_dir = pathlib.Path(f'{_COSA_DIR}/configs/workloads').resolve()
        mapspace_path = pathlib.Path(f'{_COSA_DIR}/configs/mapspace/mapspace.yaml').resolve()
        for arch_file in arch_files:
            run_dnn_models(workload_dir, arch_file, mapspace_path, output_dir, model, buf_ratios, layer_idx, dnn_def_path=dnn_def_path)
    else:
        buf_ratios_str = "_".join([str(i) for i in np.array(buf_ratios).flatten()])
        buf_ratios_file = output_dir / f"buf_ratios_{buf_ratios_str}.pkl"
        with open(buf_ratios_file, "wb") as f:
            pickle.dump(buf_ratios, f)

        processes = []
        # Start schedule generation script for each layer on each arch
        for arch_file in arch_files:
            cmd = ["python3", f"{_COSA_DIR}/src/run_dnn_models.py", "--output_dir", str(output_dir), "--arch_path", arch_file, "--buf_ratios_file", buf_ratios_file]
            if model:
                cmd += ["--model", model]
            if layer_idx is not None:
                cmd += ["--layer_idx", layer_idx]
            if dnn_def_path is not None:
                cmd += ["--dnn_def_path", str(dnn_def_path)]

            process = subprocess.Popen(cmd)
            processes.append(process)

            # Limit number of active processes
            while len(processes) >= 10:
                # Iterate backward so the list of processes doesn't get messed up
                for i in range(len(processes)-1, -1, -1):
                    if processes[i].poll() is not None: # If process has terminated
                        processes.pop(i)
        
        # Wait for schedule generation to finish
        for process in processes:
            process.wait()

def total_layer_values(layer_values_dict, layer_count):
    """
    Calculate the total cycle/energy value of a network by summing up the values for each unique layer,
    multiplied by the number of times a layer with those dimensions appears in the network.
    """
    total = 0
    for layer in layer_values_dict:
        if layer not in layer_count:
            print(f"ERROR: layer {layer} not found in layer count file")
            exit(1)
        total += layer_values_dict[layer] * layer_count[layer]

    return total


def fetch_data(new_arch_dir, output_dir, glob_str='arch_pe*_v3.yaml', buf_str=''):
    # Get all arch files
    arch_files = new_arch_dir.glob(glob_str)
    workload_dir = pathlib.Path('../configs/workloads').resolve()

    db = {}
    layer_counts = {}
    # Fetch data into DB
    for arch_path in arch_files:
        # Get each file's data
        arch_name = os.path.basename(arch_path).split(".yaml")[0]
        cycle_json = output_dir / f"results_{arch_name}{buf_str}_cycle.json"
        energy_json = output_dir / f"results_{arch_name}{buf_str}_energy.json"
        try:
            cycle_dict = util.parse_json(cycle_json)
            energy_dict = util.parse_json(energy_json)
        except:
            # Data missing for some reason
            continue
        
        arch = Arch(arch_path)
        arch_db = {}
        
        for model in cycle_dict:
            if model not in layer_counts:
                model_dir = workload_dir / (model+'_graph')
                layer_count_path = model_dir / ('layer_count.yaml')
                layer_counts[model] = util.parse_yaml(layer_count_path)
            
            total_cycle = total_layer_values(cycle_dict[model], layer_counts[model])
            total_energy = total_layer_values(energy_dict[model], layer_counts[model])
            arch_db[model] = {
                "cycle": total_cycle,
                "energy": total_energy,
                "cycle_energy_prod": total_energy * total_cycle,
            }
        
        db[arch.config_str()] = arch_db

        if len(db) % 100 == 0:
            print(f"Fetched data for {len(db)} arch")
    util.store_json(output_dir / "all_arch.json", db)
    

if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    base_arch_path = pathlib.Path(args.base_arch_path).resolve()
    arch_dir = pathlib.Path(args.arch_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    # gen_arch_yaml(base_arch_path, arch_dir)
    # gen_data(arch_dir, output_dir)
    # gen_dataset(arch_dir, output_dir, arch_v3=True, postfix='_v3')
    # fetch_data(new_arch_dir, output_dir)

    # models = []
    # output_dirs = []
    # arch_dirs = []
    # for model in ["deepbench"]:
    #     for seed in [6, 888, 123456]:
    #         if model == "deepbench" and (seed == 6 or seed == 888):
    #             continue
    #         models.append(model)
    #         output_dirs.append(f"output_dir_{model}_s{seed}")
    #         arch_dirs.append(f"arch_bo_{model}_s{seed}")
    # for i in range(len(output_dirs)):
    #     gen_dataset_per_layer(output_dir=output_dirs[i], model=models[i], arch_dir=arch_dirs[i])
    
    gen_dataset_per_layer(args.output_dir, arch_dir=args.arch_dir, model=args.model, arch_v3=True)
    # gen_dataset_per_network(output_dir=args.output_dir, arch_dir=args.arch_dir)
    #gen_dataset_all()
    # gen_dataset_all_layer()
