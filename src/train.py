"""Contains training utilities the VAESA model.

Training script that covers all training options for VAESA. The encoder, decoder,
and performance predictor(s) are all trained together here. The user can configure
typical neural network model training parameters such as the initial learning rate,
random seed, and number of epochs to train. The user can also configure VAESA-specific
parameters such as the latent space dimensionality (--nz) or set predictor/autoencoder 
model options (--predictor, --predictor-model, --VAE-model). Use the --help option for 
more info.

    Typical usage example:

    python train.py --data-name cosa_data \
                    --save-interval 10 \
                    --epochs 2000 \
                    --lr 1e-4 \
                    --model VAE \
                    --predictor \
                    --nz 4 \
                    --batch-size 64 \
                    --data-type cosa \
                    --train-from-scratch \
                    --seed 1234 \
                    --dataset-size 131328 \
                    --predictor-model orig_1 \
                    --dataset-path ../db/dataset_all_layer.csv \
                    --obj edp \
                    --VAE-model model_1 \
                    --reprocess
"""

import os
import sys
import pathlib
import math
import pickle
import pdb
import argparse
import random
import shutil
import copy
import logging
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

from util import *
from models import *
from train_util import model_add_predictor, denorm_cycle_obj, denorm_energy_obj 
from test_util import parse_dnn_def, gen_dnn_predictors, get_percent_diff, pred_arch_perf, pred_vis, encoded_data, encoded_vis
from eval_sgd import search_all_layers 
import vae_bo
import dataset_prob

parser = argparse.ArgumentParser(description='Train VAE')
parser.add_argument('--data-type', default='cosa')
parser.add_argument('--data-name', default='', help='dataset name')
parser.add_argument('--save-appendix', default='', 
                    help='postfix for result file names')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=20, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--no-test', action='store_true', default=False,
                    help='if True, merge test with train, i.e., no held-out set')
parser.add_argument('--reprocess', action='store_true', default=True,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=True,
                    help='if True, do not remove any old data in the result folder')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--only-search', action='store_true', default=False,
                    help='if True, perform search on latent space')
parser.add_argument('--search-strategy', default='random',
                    help='search strategy, including random and optimal, bo_latent')
parser.add_argument('--bo-config-path', default=None, help='vae configs')
parser.add_argument('--search-optimizer', default='sgd',
                    help='optimizer, including sgd and Newton')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--predictor-model', default='orig', help='set predictor model. options [orig, deep, orig_1, deep_1]')
parser.add_argument('--VAE-model', default='orig', help='set VAE hidden_dims model. options [orig, model_1, model_2]')
parser.add_argument('--obj', default='edp', help='valid options [edp, latency, energy]')
parser.add_argument('--model', default='VAE', help='model to use VAE')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--log-obj', action='store_true', default=False,
                    help='whether to use log perf label')
parser.add_argument('--norm-obj', action='store_true', default=False,
                    help='whether to use log perf label')
parser.add_argument('--log-layerfeat', action='store_true', default=False,
                    help='whether to log layer features')
parser.add_argument('--norm-layerfeat', action='store_true', default=False,
                    help='whether to normalize layer features')
parser.add_argument('--norm-layerfeat-option', default='', help='valid options [mean, max]')
parser.add_argument('--norm-latent', action='store_true', default=False,
                    help='whether to normalize latent features')
parser.add_argument('--new-loss', action='store_true', default=False,
                    help='whether to normalize latent features')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--hs', type=int, default=1024, metavar='N',
                    help='hidden size')
parser.add_argument('--nz', type=int, default=4, metavar='N',
                    help='latent vectors z dimension')
parser.add_argument('--predictor', action='store_true', default=False,
                    help='whether to train a performance predictor from latent\
                    encodings and a VAE at the same time')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--search-lr', type=float, default=1e+1, metavar='LR',
                    help='search learning rate (default: 1e+1)')
parser.add_argument('--dataset-size', default=None,
                    help='Training sample size')
parser.add_argument('--dataset-path', default='arch_dataset_12.csv',
                    help='Dataset path')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=64, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--search-samples', type=int, default=10, metavar='N',
                    help='the number of samples for searching')
parser.add_argument('--train-from-scratch', action='store_true', default=False,
                    help='if True, perform train on selected architectures')
parser.add_argument('--new-dnn-path', default='new_dnn.json',
                    help='new DNN definition')
parser.add_argument('--only-dnn-search', action='store_true', default=False,
                    help='if True, perform search on latent space')
parser.add_argument('--search-all-layers', action='store_true', default=False,
                    help='if True, search all layers')
parser.add_argument('--loglevel', type=str, default="INFO",
                    help='set log level, options [NOTSET, DEBUG, INFO,\
                    WARNING, ERROR, CRITICAL] (default: INFO)')

args = parser.parse_args()
logging.basicConfig(level=args.loglevel.upper(), format='%(asctime)s %(levelname)s %(message)s')
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)
logging.info(args)

"""Prepare data"""
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name, args.save_appendix))
if args.predictor:
    args.res_dir += '{}'.format('_predictor')

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

pkl_name = os.path.join(args.res_dir, args.data_name + '.pkl')

# check whether to load pre-stored pickle data
if os.path.isfile(pkl_name) and not args.reprocess:
    with open(pkl_name, 'rb') as f:
        train_data, test_data, vae_args = pickle.load(f)
else:
    if args.data_type == 'cosa':
        if not args.only_search and not args.search_strategy=='bo_latent':
            train_dataset = dataset_prob.CoSADataset(split= "train", transform=torch.Tensor(), target_transform=torch.Tensor(), train_samples=args.dataset_size, dataset_path=args.dataset_path, target_log=args.log_obj, target_norm=args.norm_obj, layerfeat_log=args.log_layerfeat, layerfeat_norm=args.norm_layerfeat,layerfeat_norm_option=args.norm_layerfeat_option)
            test_dataset = dataset_prob.CoSADataset(split= "test", transform=torch.Tensor(), target_transform=torch.Tensor(), train_samples=args.dataset_size, dataset_path=args.dataset_path, target_log=args.log_obj, target_norm=args.norm_obj, layerfeat_log=args.log_layerfeat, layerfeat_norm=args.norm_layerfeat, layerfeat_norm_option=args.norm_layerfeat_option)
            train_data, test_data = train_dataset, test_dataset 
        cmd_opt = argparse.ArgumentParser()
        vae_args, _ = cmd_opt.parse_known_args()

vae_args.max_n = 6
def train(epoch):
    """Trains the VAESA model for one epoch.
    
    Training includes the encoder, decoder, and performance predictors. 
    Arguments are received through the command line.

    Args:
        epoch: An int representing the number of the current epoch.

    Returns:
        A tuple with loss values of all neural networks that are part of VAESA.
        Loss values are floats. The loss values returned:
        
        (train_loss, recon_loss, kld_loss, pred_loss, energy_loss)
        
        If performance predictors not used, pred_loss and energy_loss will be 0.
        If the search objective is latency, energy_loss will be 0.
        If the search objective is energy, pred_loss will be 0.
    """
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    pred_loss = 0
    energy_loss = 0
    # shuffle(train_data)
    # pbar = tqdm(train_data)
    input_batch = []
    y_batch = []
    energy_batch = []
    layer_feat_batch = []
    # for i, pb in enumerate(pbar):
    for i, pb in enumerate(train_data):
        inp, y, energy, layer_feat = pb
        input_batch.append(inp.tolist())
        y_batch.append(y)
        energy_batch.append(energy)
        layer_feat_batch.append(layer_feat.tolist())
        optimizer.zero_grad()
        if len(input_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()

            input_batch = torch.Tensor(input_batch).to(device)
            with torch.no_grad():
                mu, logvar = model.encode(input_batch)
                # mu, logvar = torch.rand(len(input_batch), args.nz).to(model.get_device()), torch.rand(len(input_batch), args.nz).to(model.get_device())
                _, recon, kld = model.loss(mu, logvar, input_batch, epoch)
            loss = 0
            #kld = 0
            loss_energy = 0
            if args.predictor:
                layer_feat_batch_tensor = torch.Tensor(layer_feat_batch).to(device)
                mu = mu  
                if args.norm_latent: 
                    pred_input = torch.cat((mu/10, layer_feat_batch_tensor), dim=1).to(device)
                else:
                    pred_input = torch.cat((mu, layer_feat_batch_tensor), dim=1).to(device)

                if args.obj in ['edp', 'latency']:
                    y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
                    y_pred = model.predictor(pred_input).to(device)
                    if args.new_loss:
                        ones = torch.full(list(y_batch.size()), 1).to(device)
                        loss_y = model.latency_loss(torch.div(y_pred, y_batch), ones).to(device) * 100
                    else:
                        loss_y = model.latency_loss(y_pred, y_batch).to(device) * 100
                    loss += loss_y

                if args.obj in ['edp', 'energy']:
                    energy_batch = torch.FloatTensor(energy_batch).unsqueeze(1).to(device)
                    energy_pred = model.predictor_energy(pred_input).to(device)
                    if args.new_loss:
                        ones = torch.full(list(energy_batch.size()), 1).to(device)
                        loss_energy = model.energy_loss(torch.div(energy_pred, energy_batch), ones).to(device) * 100
                    else:
                        loss_energy = model.energy_loss(energy_pred, energy_batch).to(device) * 100
                    loss += loss_energy 
            else:
                pbar.set_description('Epoch: %d, loss: %0.8f, recon: %0.8f, kld: %0.8f' % (
                                     epoch, loss.item()/len(input_batch), recon.item()/len(input_batch), 
                                     kld.item()/len(input_batch)))
            
            loss_y.backward()
            loss_energy.backward()
            
            train_loss += float(loss)
            recon_loss += float(recon)
            kld_loss += float(kld)
            if args.predictor:
                pred_loss += float(loss_y)
            energy_loss += float(loss_energy)
            optimizer.step()

            input_batch = []
            layer_feat_batch = [] 
            y_batch = []
            energy_batch = []

    logging.info('==========> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_data)))

    return train_loss, recon_loss, kld_loss, pred_loss, energy_loss


def test(log_obj, norm_obj, norm_path=None):
    """Evaluates the VAESA model.
    
    Evaluates reconstruction and prediction accuracy of VAESA model.

    Args:
        log_obj: Boolean; set true if the log of the objectives (latency, 
            energy) was taken during normalization
        norm_obj: Boolean; set true if the objectives (latency, energy) 
            were normalized 
        norm_path: String file pathname of file containing normalization
            statistics (mean/std) of dataset

    Returns:
        A tuple with loss values of all neural networks that are part of VAESA.
        Loss values are floats. The loss values returned:
        
        (Nll, acc, pred_rmse)
        
        Nll: MSE loss value of autoencoder reconstruction
        acc: Number of exactly reconstructed test points TODO: implement
        pred_rmse: Root mean squared error of performance predictions. 0 if
            performance predictor(s) not used.
    """
    model.eval()
    encode_times = 10
    decode_times = 10
    Nll = 0
    pred_loss = 0
    energy_loss = 0
    n_perfect = 0
    logging.info('Testing begins...')
    pbar = tqdm(test_data)
    input_batch = []
    y_batch = []
    energy_batch = []
    layer_feat_batch = []
    test_data_size = len(test_data)
    input_diff_percent = None
    latency_diff_percent = None
    energy_diff_percent = None
    mu_all = None 
    logging.info('Predictor weight value: %s', model.predictor_energy[0].weight.data[1])
    with torch.no_grad():
        for i, pb in enumerate(pbar):
            g, y, energy, layer_feat = pb
            input_batch.append(g.tolist())
            y_batch.append(y)
            energy_batch.append(energy)
            layer_feat_batch.append(layer_feat.tolist())
            if len(input_batch) == args.infer_batch_size or i == len(test_data) - 1:
                # g = model._collate_fn(input_batch)
                input_batch = torch.Tensor(input_batch).to(device)
                mu, logvar = model.encode(input_batch)
                all_loss, nll, _ = model.loss(mu, logvar, input_batch, i)
                pbar.set_description('recons loss: {:.8f}'.format(nll.item()/len(input_batch)))
                logging.info('total loss: {:.8f}'.format(all_loss.item()/len(input_batch)))
                Nll += nll.item()
                if mu_all is None:
                    # mu_all  = torch.sum(mu, 0)
                    mu_all  = mu 
                else:
                    logging.debug('Number of test latent values forward propagated so far: %s', mu_all.size())
                    
                    #mu_all = torch.stack((torch.sum(mu, 0), mu_all), 0)
                    mu_all = torch.cat((mu, mu_all), 0)
                
                if args.predictor:
                    layer_feat_batch_tensor = torch.Tensor(layer_feat_batch).to(device)

                    if args.norm_latent: 
                        pred_input = torch.cat((mu / 10, layer_feat_batch_tensor), dim=1).to(device)
                    else:
                        pred_input = torch.cat((mu, layer_feat_batch_tensor), dim=1).to(device)
     
                    if args.obj in ['edp', 'latency']:
                        y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
                        y_pred = model.predictor(pred_input)
                        y_batch_np = y_batch.cpu().detach().numpy()
                        y_pred_np = y_pred.cpu().detach().numpy()

                        y_batch_np = denorm_cycle_obj(y_batch_np, log_obj, norm_obj, norm_path)
                        y_pred_np = denorm_cycle_obj(y_pred_np, log_obj, norm_obj, norm_path)

                        latency_diff_sum = get_percent_diff(y_batch_np, y_pred_np, 'latency') 
                        if latency_diff_percent is None:
                            latency_diff_percent = latency_diff_sum 
                        else:
                            latency_diff_percent += latency_diff_sum
                           
                        loss_y = model.mseloss(y_pred, y_batch)
                        pred_loss += loss_y
                        logging.debug(f'y_batch: {y_batch_np}, y_pred: {y_pred_np}')
                        logging.debug('latency_pred loss: {:.8f}'.format(loss_y.item()/len(input_batch)))

                    if args.obj in ['edp', 'energy']:
                        energy_batch = torch.FloatTensor(energy_batch).unsqueeze(1).to(device)
                        energy_pred = model.predictor_energy(pred_input)
                        energy_batch_np = energy_batch.cpu().detach().numpy()
                        energy_pred_np = energy_pred.cpu().detach().numpy()

                        energy_batch_np = denorm_energy_obj(energy_batch_np, log_obj, norm_obj, norm_path)
                        energy_pred_np = denorm_energy_obj(energy_pred_np, log_obj, norm_obj, norm_path)

                        energy_diff_sum = get_percent_diff(energy_batch_np, energy_pred_np, 'energy') 
                        if energy_diff_percent is None:
                            energy_diff_percent = energy_diff_sum 
                        else:
                            energy_diff_percent += energy_diff_sum

                        loss_energy = model.mseloss(energy_pred, energy_batch)
                        logging.debug(f'energy_batch: {energy_batch_np}, energy_pred: {energy_pred_np}')
                        logging.debug('energy_pred loss: {:.8f}'.format(loss_energy.item()/len(input_batch)))
                        pred_loss += loss_energy 
     

                g_recon = model.decode(mu)

                input_batch_np = input_batch.cpu().detach().numpy() 
                g_recon_np = g_recon.cpu().detach().numpy() 

                g_diff_avg = get_percent_diff(input_batch_np, g_recon_np, 'recon') 
                if input_diff_percent is None:
                    input_diff_percent = g_diff_avg 
                else:
                    input_diff_percent += g_diff_avg

                input_batch = []
                y_batch = []
                energy_batch = []
                layer_feat_batch = [] 
        Nll /= len(test_data)
        pred_loss /= len(test_data)
        pred_rmse = math.sqrt(pred_loss)
        acc = n_perfect / (len(test_data))
        input_diff_percent /= len(test_data)
        latency_diff_percent /= len(test_data)
        energy_diff_percent /= len(test_data)
        mu_mean = torch.mean(mu_all, 0)
        mu_std = torch.std(mu_all, 0)
        logging.debug(f'mu mean value: {mu_mean}')
        logging.debug(f'mu std value: {mu_std}')
        logging.debug(f'avg recon diff : {input_diff_percent}')
        logging.debug(f'avg latency diff: {latency_diff_percent}')
        logging.debug(f'avg energy diff: {energy_diff_percent}')
        if args.predictor:
            logging.info('Test average recon loss: {0}, recon accuracy: {1:.8f}, pred rmse: {2:.8f}'.format(Nll, acc, pred_rmse))
        else:
            logging.info('Test average recon loss: {0}, recon accuracy: {1:.8f}'.format(Nll, acc))
        return Nll, acc, pred_rmse


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if not args.only_test and not args.only_search:
    # delete old files in the result directory
    remove_list = [f for f in os.listdir(args.res_dir) if not f.endswith(".pkl") and 
            not f.endswith('.pth')]
    for f in remove_list:
        tmp = os.path.join(args.res_dir, f)
        if not os.path.isdir(tmp) and not args.keep_old:
            os.remove(tmp)

    if not args.keep_old:
        # backup current .py files
        shutil.copy('train.py', args.res_dir)
        shutil.copy('models.py', args.res_dir)
        shutil.copy('util.py', args.res_dir)

    # save command line input
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    logging.info('Command line input: ' + cmd_input + ' is saved.')


# prepare training data
if args.no_test:
    train_data = train_data + test_data

if args.small_train:
    train_data = train_data[:100]

'''Prepare the model'''

if args.VAE_model == 'orig':
    hidden_dims =  [1024, 2048]
elif args.VAE_model == 'model_1':
    hidden_dims = [1024, 2048, 1024]
elif args.VAE_model == 'model_2':
    hidden_dims = [64,256,1024,2048,1024,256,64]

layer_size = 9
model = VAE(in_channels=6, latent_dim=args.nz, dataset_size=args.dataset_size, batch_size=args.batch_size, hidden_dims=hidden_dims)

if args.predictor:
    model_add_predictor(model, args, layer_size)
model.to(device)

if args.all_gpus:
    net = custom_DataParallel(model, device_ids=range(torch.cuda.device_count()))

if args.load_latest_model:
    load_module_state(model, os.path.join(args.res_dir, 'latest_model.pth'), device=device)
else:
    if args.continue_from is not None:
        if args.continue_from == 2500: 
            predictor = nn.Sequential(
                    nn.Linear(args.nz+layer_size, args.hs), 
                    nn.Tanh(), 
                    nn.Linear(args.hs, 2*args.hs), 
                    nn.Tanh(), 
                    nn.Linear(2*args.hs, args.hs), 
                    nn.Tanh(), 
                    nn.Linear(args.hs, 1),
                    nn.Sigmoid()
                    )
            model.predictor = predictor
            predictor_energy = nn.Sequential(
                    nn.Linear(args.nz+layer_size, args.hs), 
                    nn.Tanh(), 
                    nn.Linear(args.hs, 2*args.hs), 
                    nn.Tanh(), 
                    nn.Linear(2*args.hs, args.hs), 
                    nn.Tanh(), 
                    nn.Linear(args.hs, 1),
                    nn.Sigmoid()
                    )
            model.predictor_energy = predictor_energy

        epoch = args.continue_from
        load_module_state(model, os.path.join(args.res_dir, 
                                              'model_checkpoint{}.pth'.format(epoch)), device=device)
if args.predictor and args.continue_from==2500:
    model_add_predictor(model, args, layer_size)
    model.to(device)

# make sure pred parameter is added 
# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)


if not args.load_latest_model:
    if args.continue_from is not None:
        if args.continue_from != 2500:
            load_module_state(optimizer, os.path.join(args.res_dir, 
                                          'optimizer_checkpoint{}.pth'.format(epoch)), device=device)
            load_module_state(scheduler, os.path.join(args.res_dir, 
                                          'scheduler_checkpoint{}.pth'.format(epoch)), device=device)


'''only test to output reconstruct loss'''
if args.only_test:
    epoch = args.continue_from
    #sampled = model.generate_sample(args.sample_number)
    # save_latent_representations(epoch)
    #visualize_recon(300)
    #interpolation_exp2(epoch)
    #interpolation_exp3(epoch)
    # prior_validity(True)
    dataset_name = args.dataset_path.split('/')[-1].replace('.csv', '')  
    norm_path = f'dataset_stats_{dataset_name}.json'

    #test(log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    # pred_arch_perf(args.new_dnn_path, model, device, log_layerfeat=args.log_layerfeat, norm_layerfeat=args.norm_layerfeat, norm_layerfeat_option=args.norm_layerfeat_option, norm_latent=args.norm_latent, log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    # pred_vis(args.new_dnn_path, model, device, args.nz, save_to="png", log_layerfeat=args.log_layerfeat, norm_layerfeat=args.norm_layerfeat, norm_layerfeat_option=args.norm_layerfeat_option, norm_latent=args.norm_latent, log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    latent_points = encoded_data(args, train_data, model, num_points=10000)
    save_to = "pdf"
    encoded_vis(args, train_data, model, latent_points, plot_color="Blues", plot_range=None, perf_type="", arch_idx=0, save_to=save_to, log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    encoded_vis(args, train_data, model, latent_points, plot_color="Blues", plot_range=None, perf_type="", arch_idx=1, save_to=save_to, log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    encoded_vis(args, train_data, model, latent_points, plot_color="Oranges", plot_range=None, perf_type="", arch_idx=2, save_to=save_to, log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    encoded_vis(args, train_data, model, latent_points, plot_color="Oranges", plot_range=None, perf_type="", arch_idx=3, save_to=save_to, log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    encoded_vis(args, train_data, model, latent_points, plot_color="Oranges", plot_range=None, perf_type="", arch_idx=4, save_to=save_to, log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    encoded_vis(args, train_data, model, latent_points, plot_color="Oranges", plot_range=None, perf_type="", arch_idx=5, save_to=save_to, log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    encoded_vis(args, train_data, model, latent_points, plot_color="viridis", plot_range=None, perf_type="cycle", arch_idx=0, save_to=save_to, log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    encoded_vis(args, train_data, model, latent_points, plot_color="viridis", plot_range=None, perf_type="energy", arch_idx=0, save_to=save_to, log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    encoded_vis(args, train_data, model, latent_points, plot_color="viridis", plot_range=None, perf_type="edp", arch_idx=0, save_to=save_to, log_obj=args.log_obj, norm_obj=args.norm_obj, norm_path=norm_path)
    sys.exit(0)


search_seed = args.seed


assert (not (args.search_all_layers and args.only_dnn_search))
if args.only_dnn_search:
    torch.manual_seed(search_seed)
    if args.cuda:
        torch.cuda.manual_seed(search_seed)
    np.random.seed(search_seed)
    random.seed(search_seed)

    dataset_name = args.dataset_path.split('/')[-1].replace('.csv', '')  
    norm_path = f'dataset_stats_{dataset_name}.json'

    # load network def
    dnn_def_tensor, num_predictors = parse_dnn_def(args.new_dnn_path, device, log_layerfeat=args.log_layerfeat, norm_layerfeat=args.norm_layerfeat, norm_layerfeat_option=args.norm_layerfeat_option, norm_path=norm_path)
    
    # generate predictors for inference
    gen_dnn_predictors(model, num_predictors)
    
    # perform search
    search_dir = os.path.join(args.res_dir, f'dnn_search_s{search_seed}')

    configs = model.dnn_search(args.search_samples, args.search_optimizer, dnn_def_tensor, lr=args.search_lr, obj=args.obj, norm_latent=args.norm_latent)
    logging.info(f'search_dir {search_dir}')
    if not os.path.exists(search_dir):
        os.makedirs(search_dir)

    for i, g in enumerate(configs):
        namei = '{}_config_{}'.format(args.search_optimizer, i)
        plot_config(g[0], search_dir, namei, data_type=args.data_type)
        plot_searched_points(g, search_dir, namei)
    
    finded_configs_path = os.path.join(search_dir, 'finded_configs.pkl')
    with open(finded_configs_path, 'wb') as f:
        pickle.dump(configs, f)
    sys.exit(0)


if args.search_all_layers:
    torch.manual_seed(search_seed)
    if args.cuda:
        torch.cuda.manual_seed(search_seed)
    np.random.seed(search_seed)
    random.seed(search_seed)

    dataset_name = args.dataset_path.split('/')[-1].replace('.csv', '')  
    norm_path = f'dataset_stats_{dataset_name}.json'
    # layer_dir = pathlib.Path('test_layers/exisiting_layers')
    layer_dir = pathlib.Path('test_layers/nonex_layers')
    target_model = 'new'
    
    search_all_layers(layer_dir, model, target_model, args.search_samples, args.search_optimizer, args.search_lr, search_seed, args.res_dir, args.obj, device, args.data_type, args.norm_latent, args.log_layerfeat, args.norm_layerfeat, args.norm_layerfeat_option, args.log_obj, args.norm_obj, norm_path)


'''search network based on latent space'''
if args.only_search:
    torch.manual_seed(search_seed)
    if args.cuda:
        torch.cuda.manual_seed(search_seed)
    np.random.seed(search_seed)
    random.seed(search_seed)

    logging.info('Begin searching ...')
    if args.search_strategy == 'random':
        configs = model.random_search(args.search_samples, args.search_optimizer, args.search_lr, args.obj)
        search_dir = os.path.join(args.res_dir, f'random_search_s{search_seed}')
    elif args.search_strategy == 'optimal':
        configs = model.optimal_search(args.search_samples, args.search_optimizer, train_data + test_data, args.search_lr, args.obj)
        search_dir = os.path.join(args.res_dir, f'optimal_search_s{search_seed}')
    elif args.search_strategy == 'bo_latent': 
        bo_config = parse_json(args.bo_config_path)
        bo_config['random_seed'] = search_seed
        # if bo_config['dnn_def_path'] is not None:
        #     bo_config['model'] += "_" + str(search_seed)
        vae_bo.bo_latent(model, device, **bo_config)
    
    if args.search_strategy != 'bo_latent':
        if not os.path.exists(search_dir):
            os.makedirs(search_dir)
        for i, g in enumerate(configs):
            namei = '{}_config_{}'.format(args.search_optimizer, i)
            plot_config(g[0], search_dir, namei, data_type=args.data_type)
            plot_searched_points(g, search_dir, namei)
        
        finded_configs_path = os.path.join(search_dir, 'finded_configs.pkl')
        with open(finded_configs_path, 'wb') as f:
            pickle.dump(configs, f)
    sys.exit(0)


'''train from scratch'''
if args.train_from_scratch:
    if args.search_strategy == 'random':
        search_dir = os.path.join(args.res_dir, 'random_search')
    else:
        search_dir = os.path.join(args.res_dir, 'optimal_search')
    finded_configs_path = os.path.join(search_dir, 'finded_configs.pkl')
    if os.path.exists(finded_configs_path):
        with open(finded_configs_path, 'rb') as f:
            configs = pickle.load(f)

# plot sample train/test configs
if not os.path.exists(os.path.join(args.res_dir, 'train_sample_0.json')) or args.reprocess:
    if not args.keep_old:
        for data in ['train_data', 'test_data']:
            # G = [g for g, y in eval(data)[:10]]
            for i, g in enumerate(data):
                name = 'train_sample_{}'.format(i)
                # plot_config(g, args.res_dir, name, data_type=args.data_type)
                with open(os.path.join(args.res_dir, f'{name}.json'), 'w') as f:
                    f.write(f'{name}: {g}\n')

'''Training begins here'''
min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')
if os.path.exists(loss_name) and not args.keep_old:
    os.remove(loss_name)

start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, args.epochs + 1):
    train_loss, recon_loss, kld_loss, pred_loss, energy_loss = train(epoch)

    logging.info("Epoch {}: {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(
        epoch,
        train_loss/len(train_data), 
        recon_loss/len(train_data), 
        kld_loss/len(train_data), 
        pred_loss/len(train_data), 
        energy_loss/len(train_data)
        ))

    with open(loss_name, 'a') as loss_file:
        loss_file.write("{:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(
            train_loss/len(train_data), 
            recon_loss/len(train_data), 
            kld_loss/len(train_data), 
            pred_loss/len(train_data), 
            energy_loss/len(train_data)
            ))

    scheduler.step(train_loss)
    if epoch % args.save_interval == 0:
        logging.info("save current model...")
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
        scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)
        torch.save(optimizer.state_dict(), optimizer_name)
        torch.save(scheduler.state_dict(), scheduler_name)
        # logging.info("visualize reconstruction examples...")
        # visualize_recon(epoch)
        # logging.info("extract latent representations...")
        # save_latent_representations(epoch)
        logging.info("sample from prior...")
        sampled = model.generate_sample(args.sample_number)
        logging.info("plot train loss...")
        losses = np.loadtxt(loss_name)
        if losses.ndim == 1:
            continue
        fig = plt.figure()
        num_points = losses.shape[0]
        plt.plot(range(1, num_points+1), losses[:, 0], label='Total')
        plt.plot(range(1, num_points+1), losses[:, 1], label='Recon')
        plt.plot(range(1, num_points+1), losses[:, 2], label='KLD')
        plt.plot(range(1, num_points+1), losses[:, 3], label='Pred')
        plt.xlabel('Epoch')
        plt.ylabel('Train loss')
        plt.legend()
        plt.savefig(loss_plot_name)


sys.exit(0)

