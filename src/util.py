from __future__ import print_function
import gzip
import pickle
import numpy as np
import torch
import json
from torch import nn
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import subprocess
import collections
# import igraph
import argparse
import pdb
# import pygraphviz as pgv
import sys

# create a parser to save graph arguments
cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()

'''load and save objects'''
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret


def load_module_state(model, state_name, device=None):
    if device is None:
        pretrained_dict = torch.load(state_name)
    else:
        pretrained_dict = torch.load(state_name, map_location=device)
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    print(pretrained_dict.keys())
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return


def graph_with_energy(g_list):
    g_list_new = []
    energy_list = []
    width = 48
    energy_max = 0
    for (g, y) in g_list:
        energy = 0
        for i in range(1, g.vcount()):
            if g.vs[i]['type'] == 2 or g.vs[i]['type'] == 3:
                m = 9 + 1
            elif g.vs[i]['type'] == 4 or g.vs[i]['type'] == 5:
                m = 25 + 1
            elif g.vs[i]['type'] == 6 or g.vs[i]['type'] == 7:
                m = 1
            else:
                m = width
            num_precessors = len(g.predecessors(i))
            energy += width * m * num_precessors
        if energy > energy_max:
            energy_max = energy
        energy_list.append(energy)
    for i, (g, y) in enumerate(g_list):
        g_list_new.append((g, y, energy_list[i] / energy_max))
    return g_list_new


'''Data preprocessing'''
def load_ENAS_graphs(name, n_types=6, fmt='igraph', rand_seed=0, with_y=True, burn_in=1000):
    # load ENAS format NNs to igraphs or tensors
    g_list = []
    max_n = 0  # maximum number of nodes
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if i < burn_in:
                continue
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                g, n = decode_ENAS_to_igraph(row)
            elif fmt == 'string':
                g, n = decode_ENAS_to_tensor(row, n_types)
            max_n = max(max_n, n)
            g_list.append((g, y)) 
    graph_args.num_vertex_type = n_types + 2  # original types + start/end types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1 # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng*0.9)], g_list[int(ng*0.9):], graph_args


# return igraph type samples of NASBench
def load_NASBench_graphs(name, n_types=5, fmt='igraph', rand_seed=0, with_y=True):
    g_list = []
    max_n = 0  # maximum number of nodes
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            row, labeling, cmplx,  y= eval(row)
            if fmt == 'igraph':
                g, n = decode_NASBench_to_igraph(row, labeling)
            max_n = max(max_n, n)
            g_list.append((g, y, cmplx/10000000))
    
    graph_args.num_vertex_type = 7 + 2  # original types + start/end types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1 # predefined end vertex type
    ng = len(g_list)        
    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng*0.9)], g_list[int(ng*0.9):], graph_args


def one_hot(idx, length):
    idx = torch.LongTensor([idx]).unsqueeze(0)
    x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x


def decode_ENAS_to_tensor(row, n_types):
    n_types += 2  # add start_type 0, end_type 1
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)  # n+2 is the real number of vertices in the DAG
    g = []
    # ignore start vertex
    for i, node in enumerate(row):
        node_type = node[0] + 2  # assign 2, 3, ... to other types
        type_feature = one_hot(node_type, n_types)
        if i == 0:
            edge_feature = torch.zeros(1, n+1)  # a node will have at most n+1 connections
        else:
            edge_feature = torch.cat([torch.FloatTensor(node[1:]).unsqueeze(0), 
                                     torch.zeros(1, n+1-i)], 1)  # pad zeros
        edge_feature[0, i] = 1 # ENAS node always connects from the previous node
        g.append(torch.cat([type_feature, edge_feature], 1))
    # process the output node
    node_type = 1
    type_feature = one_hot(node_type, n_types)
    edge_feature = torch.zeros(1, n+1)
    edge_feature[0, n] = 1  # output node only connects from the final node in ENAS
    g.append(torch.cat([type_feature, edge_feature], 1))
    return torch.cat(g, 0).unsqueeze(0), n+2


def plot_config(g, res_dir, name, backbone=False, data_type='cosa', pdf=False):
    # backbone: puts all nodes in a straight line 
    file_name = os.path.join(res_dir, name+'.txt')
    g_np_array = g.detach().cpu().numpy()
    g_str = np.array2string(g_np_array, formatter={'float_kind':lambda x: "%.2f" % x})
    with open(file_name, 'w') as f:
        f.write(g_str)

def plot_searched_points(g, res_dir, name):
    file_name = os.path.join(res_dir, name+'.json')
    print(file_name)
    g_key = ['decoded_sample', 'sample_latent', 'optimized_sample_latent', 'pred_cycle_before', 'pred_cycle_after', 'pred_energy_before', 'pred_energy_after']
    data = {}
    for idx, entry in enumerate(g):
        entry = entry.detach().cpu().numpy().tolist()
        data[g_key[idx]] = entry  
    print(data)
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=2)

def parse_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

'''optimizers for search for one objective'''
def Newton_method_single_obj(x, f, lr):
    x = x.view(-1)
    optimization_list = []
    for i in range(20):
        # flip f to minimize cycle perf
        grad_x = torch.autograd.grad(-f(x), x, create_graph=True)
        if len(torch.nonzero(grad_x[0])) < 1:
            break
        # grad_grad_x = torch.zeros(len(x), len(x), device='cuda:0')
        grad_grad_x = torch.zeros(len(x), len(x))
        for j in range(len(x)):
                grad_grad_x[j,:] = torch.autograd.grad(grad_x[0][j], x, create_graph=True)[0]
        x += torch.Tensor(lr) * torch.mm(grad_x[0].view(1, -1), torch.inverse(grad_grad_x)).view(-1)
        optimization_list.append(f(x))
    plot_fig(optimization_list, 'tmp.pdf')
    return x


'''optimizers for search for one objective'''
def Newton_method(x, f, f_comp, lr, log=False, obj='edp'):
    x = x.view(-1).cpu()
    f.cpu()
    f_comp.cpu()
    optimization_list = []
    for i in range(20):
        # flip f to minimize cycle perf
        # grad_x = torch.autograd.grad(-f(x)*f_comp(x), x, create_graph=True)
        if log:
            grad_x = torch.autograd.grad(-(torch.log(f(x)) + torch.log(f_comp(x))), x, create_graph=True)
        else:
            if obj == 'edp':
                grad_x = torch.autograd.grad(-f_pred(x) * f_comp(x), x, create_graph=True)
            elif obj == 'latency' or obj == 'edp_only':
                grad_x = torch.autograd.grad(-f_pred(x), x, create_graph=True)
            elif obj == 'energy':
                grad_x = torch.autograd.grad(-f_comp(x), x, create_graph=True)

        if len(torch.nonzero(grad_x[0])) < 1:
            break
        # grad_grad_x = torch.zeros(len(x), len(x), device='cuda:0')
        grad_grad_x = torch.zeros(len(x), len(x)).cpu()
        for j in range(len(x)):
                grad_grad_x[j,:] = torch.autograd.grad(grad_x[0][j], x, create_graph=True)[0]
        x += torch.Tensor(lr) * torch.mm(grad_x[0].view(1, -1), torch.inverse(grad_grad_x)).view(-1)
        optimization_list.append(f(x))
    plot_fig(optimization_list, 'tmp.pdf')
    return x


def sgd_method_dnn(x, latency_predictors, energy_predictors, dnn_def, device, lr, obj='edp', norm_latent=False, sgd_indice=[200], sgd_steps=200):
    assert(sgd_indice[-1]==sgd_steps)
    x = x.view(-1)
    latency_list = []
    energy_list = []
    x_new = x.data
    x = torch.autograd.Variable(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    # optimizer = torch.optim.SGD([x], lr=lr,nesterov=True, momentum=0.9)
    max_iter = sgd_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0)
    
    sgd_latent_designs = []
    for i in range(max_iter):
    #for i in range(1):
        # flip f_pred to minimize cycle perf
        latency = 0
        energy = 0
        if i in sgd_indice:
            sgd_latent_designs.append(x.clone())

        for idx, latency_predictor in enumerate(latency_predictors):
            energy_predictors[idx].zero_grad()
            latency_predictor.zero_grad()
            optimizer.zero_grad()
            #energy_predictors[idx].train()
            #latency_predictor.train()
        if x.grad is not None:
            x.grad.zero_()
            x.grad = None
        for idx, latency_predictor in enumerate(latency_predictors):

            #test_x = torch.autograd.Variable(torch.cat((x, dnn_def[idx]), dim=0).to(device), requires_grad=True)
            if norm_latent: 
                test_x = torch.cat((x/10, dnn_def[idx]), dim=0).to(device)
            else:
                test_x = torch.cat((x, dnn_def[idx]), dim=0).to(device)
            if obj == 'latency' or obj == 'edp':
                latency += latency_predictor(test_x)
            if obj == 'energy' or obj == 'edp':
                energy += energy_predictors[idx](test_x)
            
        if obj == 'latency':
            target = latency
        if obj == 'energy':
            target = energy
        if obj == 'edp':
            # target = latency * energy
            # if log 
            target = latency + energy
            
        print(f'iter {i}, latency {latency}, energy {energy}, target {target}')
        target.backward()

        grad_x = x.grad.data
        print(f'x_grad: {grad_x}')
        optimizer.step()
        scheduler.step()
        print(f'x_new: {x}')
        if obj == 'latency' or obj == 'edp':
            latency_list.append(latency.detach().cpu().numpy())
        if obj == 'energy' or obj == 'edp':
            energy_list.append(energy.detach().cpu().numpy())
    if sgd_steps in sgd_indice:
        sgd_latent_designs.append(x.clone())
    plot_fig(latency_list, 'latency_sgd.png')
    plot_fig(energy_list, 'energy_sgd.png')
    plot_fig(np.array(energy_list) + np.array(latency_list), 'eds_sgd.png')
    plot_fig(np.array(energy_list) * np.array(latency_list), 'edp_sgd.png')
    return sgd_latent_designs


def sgd_method(x, f_pred, f_comp, lr, log=False, obj='edp'):
    x = x.view(-1)
    optimization_list = []
    energy_list = []
    for i in range(200):
        # flip f_pred to minimize cycle perf
        if log:
            grad_x = torch.autograd.grad(-(torch.log(f_pred(x)) + torch.log(f_comp(x))), x, create_graph=True)
        else:
            if obj == 'edp':
                grad_x = torch.autograd.grad(-f_pred(x) * f_comp(x), x, create_graph=True)
            elif obj == 'latency' or obj == 'edp_only':
                grad_x = torch.autograd.grad(-f_pred(x), x, create_graph=True)
            elif obj == 'energy':
                grad_x = torch.autograd.grad(-f_comp(x), x, create_graph=True)

        print(f'x_before: {x}')
        print(f'grad_x: {grad_x}')
        x +=  grad_x[0] * lr

        print(f'x_after: {x}')
        optimization_list.append(f_pred(x))
        energy_list.append(f_comp(x))
    plot_fig(optimization_list, 'tmp.pdf')
    plot_fig(energy_list, 'tmp1.pdf')
    plot_fig(np.array(energy_list) + np.array(optimization_list), 'tmp2.pdf')

    return x


def sgd_method_direct(x, f_pred, lr, obj='edp'):
    x = x.view(-1)
    optimization_list = []
    energy_list = []
    for i in range(200):
        # flip f_pred to minimize cycle perf
        latency, energy = f_pred(x)
        if obj == 'edp':
            grad_x = torch.autograd.grad(-latency * energy, x, create_graph=True)
        elif obj == 'latency':
            grad_x = torch.autograd.grad(-latency, x, create_graph=True)
        elif obj == 'energy':
            grad_x = torch.autograd.grad(-energy, x, create_graph=True)

        print(f'x_before: {x}')
        print(f'grad_x: {grad_x}')
        x +=  grad_x[0] * lr

        print(f'x_after: {x}')
        optimization_list.append(latency)
        energy_list.append(energy)
    plot_fig(optimization_list, 'latency_sgd.pdf')
    plot_fig(energy_list, 'energy_sgd.pdf')
    plot_fig(np.array(energy_list) * np.array(optimization_list), 'edp_sgd.pdf')

    return x


def Newton_method_direct(x, f_pred, lr, obj='edp'):
    x = x.view(-1).cpu()
    f_pred.cpu()
    optimization_list = []
    for i in range(20):
        latency, energy = f_pred(x)
        if obj == 'edp':
            grad_x = torch.autograd.grad(-f_pred(x) * f_comp(x), x, create_graph=True)
        elif obj == 'latency':
            grad_x = torch.autograd.grad(-f_pred(x), x, create_graph=True)
        elif obj == 'energy':
            grad_x = torch.autograd.grad(-f_comp(x), x, create_graph=True)

        if len(torch.nonzero(grad_x[0])) < 1:
            break
        # grad_grad_x = torch.zeros(len(x), len(x), device='cuda:0')
        grad_grad_x = torch.zeros(len(x), len(x)).cpu()
        for j in range(len(x)):
                grad_grad_x[j,:] = torch.autograd.grad(grad_x[0][j], x, create_graph=True)[0]
        x += torch.Tensor(lr) * torch.mm(grad_x[0].view(1, -1), torch.inverse(grad_grad_x)).view(-1)
        optimization_list.append(f(x))
    plot_fig(optimization_list, 'latency_Newton.pdf')
    return x


def plot_fig(arr, file_path):    
    fig = plt.figure()
    plt.plot(range(1, len(arr) + 1), arr, label='Total')
    plt.xlabel('Epoch')
    plt.ylabel('Predicted acc')
    plt.legend()
    plt.savefig(file_path)


def get_perf_name(type):
    if type == "cycle":
        return "Latency (MCycles)"
    elif type == "energy":
        return "Energy (mJ)"
    elif type == "edp":
        return "EDP (MCycles * mJ)"

def get_arch_name(idx):
    if idx == 0:
        return "mesh_x"
    elif idx == 1:
        return "arith_instances"
    elif idx == 2:
        return "accbuf_entries"
    elif idx == 3:
        return "weightbuf_entries"
    elif idx == 4:
        return "inputbuf_entries"
    elif idx == 5:
        return "globalbuf_entries"
    else:
        return "check arch idx"

def get_arch_name_pretty(idx):
    if idx == 0:
        return "# of PEs"
    elif idx == 1:
        return "# of MAC Units"
    elif idx == 2:
        return "AccBuf Size (KB)"
    elif idx == 3:
        return "WeightBuf Size (KB)"
    elif idx == 4:
        return "InputBuf Size (KB)"
    elif idx == 5:
        return "GlobalBuf Size (KB)"
    else:
        return "check arch idx"

def get_arch_feat_name(idx):
    if idx == 0:
        return "arith_meshX"
    elif idx == 1:
        return "arith_ins"
    elif idx == 2:
        return "mem1_ent"
    elif idx == 3:
        return "mem2_ent"
    elif idx == 4:
        return "mem3_ent"
    elif idx == 5:
        return "mem4_ent"
    else:
        return "check arch idx"

def is_same_config(g0, g1):
    # note that it does not check isomorphism
    if g0.vcount() != g1.vcount():
        return False
    for vi in range(g0.vcount()):
        if g0.vs[vi]['type'] != g1.vs[vi]['type']:
            return False
        if set(g0.neighbors(vi, 'in')) != set(g1.neighbors(vi, 'in')):
            return False
    return True

