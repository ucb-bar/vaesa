"""Contains models used in VAESA."""

import math
import random
import torch
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
from torch.nn import functional
import torch.nn.init as init
import numpy as np
import pdb
from util import Newton_method, sgd_method, is_same_config, sgd_method_dnn

Tensor = TypeVar('torch.tensor')

# This file implements the NAS method with graph VAE.
class VAE(nn.Module):
    """
    The main PyTorch object, extending torch.nn.Module, used in VAESA.
    This class includes the autoencoder network. Predictor models can be added
    later (and are added in train.py). This class can also be used to run
    architecture search using the functions random_search(), optimal_search(),
    and grid_search().
    """
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None, log=True, norm=True, dataset_size=2303, batch_size=16): 

        self.max_n = in_channels
        self.nz = latent_dim
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.device = None
        self.scale = torch.tensor([64, 32, 256, 2**16, 4096, 2**18]).float() # 4096 / 128
        self.log_scale = torch.log(self.scale.float())

        self.norm = norm
        self.log = log

        self.dataset_size = dataset_size
        self.batch_size = batch_size

        modules = []
        if hidden_dims is None:
            hidden_dims = [1024, 2048]

        # Build Encoder
        in_dim = in_channels
        for h_dim in hidden_dims:
            modules.append(
                 nn.Sequential(
                     nn.Linear(in_dim, h_dim),
                     nn.LeakyReLU())
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_dims[-1], in_channels),
                            nn.Sigmoid()
                            )

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device


    def encode_decode(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        encode_input = input 
        self.scale = self.scale.to(self.get_device())
        self.log_scale = self.log_scale.to(self.get_device())
        encode_scale = self.scale
        if self.log:
            encode_input = torch.log(encode_input.float())
            encode_scale = self.log_scale
        if self.norm:
            encode_input = torch.div(encode_input, encode_scale)
            max_norm_input = torch.max(encode_input).item()
            assert(max_norm_input <= 1)

        result = self.encoder(encode_input.float())
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)

        self.scale = self.scale.to(self.get_device())
        self.log_scale = self.log_scale.to(self.get_device())

        encode_scale = self.scale
        decode_result = result
        if self.log:
            encode_scale = self.log_scale
        if self.norm:
            decode_result = torch.mul(result.float(), encode_scale)
        if self.log:
            # decode_result = torch.exp2(decode_result)
            decode_result = torch.exp(decode_result)

        return decode_result

    def loss(self, mu: Tensor, logvar: Tensor, input: Tensor, batch_idx, optimizer_idx=0,  beta=0.005):
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        # print('input', input[0][-1])
        # print('output', output[0][-1])
        # print('mu', mu[0][-1])
        train_loss = self.loss_function(output, input, mu, logvar,
                                            # M_N = self.batch_size / self.dataset_size,
                                            M_N = 0.00001,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx = batch_idx)

        return train_loss['loss'], train_loss['Reconstruction_Loss'], train_loss['KLD']

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input.float())
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        norm_recons = torch.div(recons.float(), self.scale)
        norm_input = torch.div(input.float(), self.scale)

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = functional.mse_loss(norm_recons, norm_input)

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        if kld_weight != 0:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        else:
            kld_loss = torch.tensor(0)

        if torch.isnan(recons_loss).item() and kld_weight != 0:
            mu_2 = mu ** 2 
            log_var_exp = log_var.exp()
            print(f'mu^2 {mu_2}')
            print(f'log_var.exp {log_var_exp}')
            print(f'kld_loss {kld_loss}')
            print(f'recons_loss {recons_loss}')
            raise
            
        # print(kld_weight * kld_loss)
        if kld_weight == 0:
            loss = recons_loss 
        else:
            loss = recons_loss + kld_weight * kld_loss


        loss_dict = {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_weight*kld_loss}
        # print(loss_dict)
        return loss_dict

    def generate_sample(self,
               num_samples: int,
               current_device: int = 0, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(self.get_device())

        samples = self.decode(z)
        return samples

    def dnn_perf(self, sample_lentent, dnn_def, norm_latent=False, batched=False):
        pred_latency = 0
        pred_energy = 0
        for pred_idx, layer_def in enumerate(dnn_def):
            if norm_latent:
                sample_lentent = sample_lentent / 10

            # print(f'layer_def: {layer_def}')
            print(f'sample_lentent: {sample_lentent}')
            print(f'sample_lentent for predictor: {sample_lentent}')
            if batched:
                layer_def = layer_def.repeat(sample_lentent.size()[0], 1)
                sample_latent_cat = torch.cat((sample_lentent, layer_def), dim=1).to(self.get_device())
            else:
                sample_latent_cat = torch.cat((sample_lentent, layer_def), dim=0).to(self.get_device())

            pred_latency += self.latency_predictors[pred_idx](sample_latent_cat)
            pred_energy += self.energy_predictors[pred_idx](sample_latent_cat)
        return pred_latency, pred_energy

    '''search methods'''
    def dnn_search(self, k, search_optimizer, dnn_def, lr=1e-2, obj='edp', norm_latent=False, sgd_indice=[200], sgd_steps=200):
        assert(sgd_indice[-1]==sgd_steps)
        # start_points_latent = torch.randn([k, self.nz], device='cuda:0', requires_grad=True)
        start_points_latent = torch.rand([k, self.nz], device='cuda:0', requires_grad=True) * 2 - 1
        # start_points_latent = torch.zeros((k, self.nz), requires_grad=True).to(self.get_device())
        # start_points_latent = torch.normal( mean=0.0, std=0.25, size=(k, self.nz), device='cuda:0', requires_grad=True)
        for pred_idx, layer_def in enumerate(dnn_def):
            self.latency_predictors[pred_idx].to(self.get_device())
            self.energy_predictors[pred_idx].to(self.get_device())

        pred_samples = []
        for i, sample_latent in enumerate(start_points_latent):            
            sample_latent_copy = sample_latent.clone()
            pred_latency_before, pred_energy_before = self.dnn_perf(sample_latent, dnn_def, norm_latent) 
            latent_samples = sgd_method_dnn(sample_latent, self.latency_predictors, self.energy_predictors, dnn_def, self.get_device(), lr, obj, norm_latent, sgd_indice, sgd_steps)
                
            decoded_samples = []
            for latent_sample in latent_samples: 
                latent_sample = latent_sample.to(self.get_device())
                decoded_sample = self.decode(latent_sample.reshape(1, self.nz))
                decoded_samples.append(decoded_sample)
            
            sgd_decode_samples = {}
            for sgd_idx, decoded_sample in zip(sgd_indice, decoded_samples):
                sgd_decode_samples[sgd_idx] = decoded_sample

            optimized_sample_latent = latent_samples[-1].to(self.get_device())

            pred_latency_after, pred_energy_after = self.dnn_perf(optimized_sample_latent, dnn_def, norm_latent)
            print('{} graph {} : cycle {} -> {}, energy {} -> {}'.format('sgd', i, pred_latency_before[0].data, pred_latency_after[0].data, pred_energy_before[0].data, pred_energy_after[0].data))
            
            pred_samples.append(sgd_decode_samples)
            # # print(sample_input)
            # sample_exist = False
            # for g in pred_outputs:
            #     # if g[0] == sample_input[0]:
            #     if torch.equal(g[0], sample_input[0]):
            #         sample_exist = True
            #         break
            # if not sample_exist:
            #     pred_outputs.append((sample_input[0], sample_latent_copy, optimized_sample_latent, pred_latency_before[0].data, pred_latency_after[0].data, pred_energy_before[0].data, pred_energy_after[0].data))
        return pred_samples


    '''search methods'''
    def random_search(self, k, search_optimizer, lr=1e-2, obj='edp'):
        """
        Name is misleading. Starting from k random points in the latent space,
        performs gradient descent or Newton's method on the performance
        predictors to optimize the objective.
        """
        pred_outputs = []
        # start_points_latent = torch.randn([k, self.nz], device='cuda:0', requires_grad=True)
        # start_points_latent = torch.tensor(torch.randn(k, self.nz), device='cuda:0', requires_grad=True)
        start_points_latent = torch.tensor(torch.randn(k, self.nz), requires_grad=True).to(self.get_device())
        for i, sample_latent in enumerate(start_points_latent):            
            sample_latent_copy = sample_latent.clone()
            pred_acc_before = self.predictor(sample_latent)
            pred_comp_before = self.predictor_energy(sample_latent)

            #pred_acc = self.predictor(sample_latent)
            if search_optimizer == 'Newton':
                optimized_sample_latent = Newton_method(sample_latent, self.predictor, self.predictor_energy, lr, obj) # cpu only
            else:
                optimized_sample_latent = sgd_method(sample_latent, self.predictor, self.predictor_energy, lr, obj)
            optimized_sample_latent = optimized_sample_latent.to(self.get_device())
            sample_input = self.decode(optimized_sample_latent.reshape(1, self.nz))
            self.predictor.to(self.get_device())
            self.predictor_energy.to(self.get_device())
            pred_acc_after = self.predictor(optimized_sample_latent)
            pred_comp_after = self.predictor_energy(optimized_sample_latent)
            print('{} graph {} : cycle {} -> {}, energy {} -> {}'.format(search_optimizer, i, pred_acc_before[0].data, pred_acc_after[0].data, pred_comp_before[0].data, pred_comp_after[0].data))
            
            sample_exist = False
            for g in pred_outputs:
                if torch.equal(g[0], sample_input[0]):
                    sample_exist = True
                    break
            if not sample_exist:
                pred_outputs.append((sample_input[0], sample_latent_copy, optimized_sample_latent, pred_acc_before[0].data, pred_acc_after[0].data, pred_comp_before[0].data, pred_comp_after[0].data))
        return pred_outputs


    def optimal_search(self, k, search_optimizer, data, lr=1e-2, obj='edp'):
        """
        Assumes we have existing data for the given workload. Starting from the
        best k known points in the latent space, performs gradient descent or
        Newton's method on the performance predictors to optimize the objective.
        """
        pred_outputs = []
        # start_points_latent = torch.randn([k, self.nz], device='cuda:0', requires_grad=True)
        start_points_latent = self.best_k_performance(k, data, obj=obj)
        # print("search samples: {}".format(start_points_latent))

        for i, sample_latent in enumerate(start_points_latent):            
            sample_latent_copy = sample_latent.clone()
            pred_acc_before = self.predictor(sample_latent)
            pred_comp_before = self.predictor_energy(sample_latent)

            if search_optimizer == 'Newton':
                optimized_sample_latent = Newton_method(sample_latent, self.predictor, self.predictor_energy, lr, obj)
            else:
                optimized_sample_latent = sgd_method(sample_latent, self.predictor, self.predictor_energy, lr, obj)
            
            optimized_sample_latent = optimized_sample_latent.to(self.get_device())
            self.predictor.to(self.get_device())
            self.predictor_energy.to(self.get_device())

            pred_acc_after = self.predictor(optimized_sample_latent)
            pred_comp_after = self.predictor_energy(optimized_sample_latent)
            print('{} graph {} : accuracy {} -> {}, energy {} -> {}'.format(search_optimizer, i, pred_acc_before[0].data, pred_acc_after[0].data, pred_comp_before[0].data, pred_comp_after[0].data))
            sample_input = self.decode(optimized_sample_latent.reshape(1, self.nz))

            # print(f'generated sample: {sample_input}')
            # sample_input = decode_from_latent_space(optimized_sample_latent.reshape(1, self.nz), self)
            
            sample_exist = False
            for g in pred_outputs:
                # if is_same_config(g[0], sample_input[0]):
                if torch.equal(g[0], sample_input[0]):
                    sample_exist = True
                    break
            if not sample_exist:
                pred_outputs.append((sample_input[0], sample_latent_copy, optimized_sample_latent, pred_acc_before[0].data, pred_acc_after[0].data, pred_comp_before[0].data, pred_comp_after[0].data))

        return pred_outputs

    def grid_search(self, samples, obj='edp', search_range=4, points_per_dim=40):
        """
        Searches all points within a hypercube centered around the origin.
        """
        print(f"Starting grid search with {samples} samples, searching in range {search_range}")
        linspace = np.linspace(-search_range, search_range, points_per_dim)
        # Generate list of all points in grid
        z_points = list(itertools.product(*((linspace,)*args.nz)))

        # Predict cycle/energy/edp
        pbar = tqdm(z_points)
        g_batch = []
        cycle = None
        energy = None
        for i, g in enumerate(pbar):
            g_batch.append(list(g))
            if len(g_batch) == 256 or i == len(z_points) - 1:
                g_batch = torch.Tensor(g_batch)
                cycle_batch = self.predictor(torch.tensor(g_batch).float()) * 2**28
                energy_batch = self.predictor_energy(torch.tensor(g_batch).float()) * 2**38
                if cycle is None:
                    cycle = cycle_batch.detach().numpy()
                    energy = energy_batch.detach().numpy()
                else:
                    cycle = np.concatenate((cycle, cycle_batch.detach().numpy()), axis=0)
                    energy = np.concatenate((energy, energy_batch.detach().numpy()), axis=0)
                g_batch = []

        cycle = cycle.flatten()
        energy = energy.flatten()
        edp = cycle * energy

        # Find indices of smallest predicted EDPs
        # indices = np.argpartition(edp, samples)[:samples] # Gives indices of N smallest values (but not in order)
        indices = np.argsort(edp)[:samples] # Gives indices of N smallest values (in order)
        pred_cycle = cycle[indices]
        pred_energy = energy[indices]
        print("Predicted cycle:", pred_cycle)
        print("Predicted energy:", pred_energy)
        print("Predicted EDP:", edp[indices])
        
        # Decode to arch
        points_to_eval = np.array(z_points)[indices]
        arch_to_eval = self.decode(torch.tensor(points_to_eval).float())

        print("Latent points searched:", points_to_eval)
        print("Arch to evaluate:", arch_to_eval)

        pred_outputs = []
        for i, sample_latent in enumerate(points_to_eval):
            print('{} graph {} : cycle {}, energy {}'.format('grid', i, pred_cycle[i], pred_energy[i]))
            pred_outputs.append((arch_to_eval[i], points_to_eval[i], points_to_eval[i], pred_cycle[i], pred_cycle[i], pred_energy[i], pred_energy[i]))

        return pred_outputs

    def best_k_performance(self, k, data, obj='edp'):
        perform_dict = {}
        worst_latency = 0
        for i, d in enumerate(data):
            # latency = d[1]
            if obj == 'latency':
                latency = d[1]
            elif obj == 'energy':
                latency = d[2]
            elif obj == 'edp' or obj == 'edp_only':
                latency = d[1] * d[2]
            else:
                raise
            if latency in perform_dict:
                perform_dict[latency] += [i]
            elif len(perform_dict) < k:
                perform_dict[latency] = [i]
            else:                
                worst_latency = np.max(list(perform_dict.keys()))
                if latency < worst_latency:
                    del perform_dict[worst_latency]
                    perform_dict[latency] = [i]

        sample_list = []
        for (_, index_set) in perform_dict.items():
            for index in index_set:
                sample_list.append(data[index][0])
        
        print(f'best k config: {sample_list}')
        samples_lentent_list = torch.zeros([len(sample_list), self.nz], requires_grad=True).to(self.get_device())
        for i, sample in enumerate(sample_list):
            sample = sample.to(self.get_device())
            sample = torch.reshape(sample, (1, -1))
            sample_lentent, _ = self.encode(sample)
            samples_lentent_list[i] = sample_lentent
        print(f'compressed best k config: {samples_lentent_list}')
        return samples_lentent_list


