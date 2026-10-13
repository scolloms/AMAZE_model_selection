"""
Storm Colloms 21/6/23

Defines Class to instantiate and train noramlising flow for each channel used in inference.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.stats import entropy
from scipy.stats import norm, gaussian_kde
from scipy.special import logit

import copy
import torch
from  glasflow import RealNVP, CouplingNSF
from torch import nn
import wandb


class NFlow():

    #initialise flow with inputs, conditionals, including the type of network, real non-volume preserving,
    #or neural spline flow
    #spline flow increases the flexibility in the flow model
    def __init__(self, no_trans, no_neurons, no_blocks, no_bins, training_inputs, cond_inputs,
                total_smdls, RNVP=False, device="cpu"):
                
        """
        Initialise Flow with inputed data, either RNVP or Spline flow.

        Parameters
        ----------
        no_trans : int
            number of transforms to give the flow
        no_neurons : int
            number of neurons of the flow per layer/transform
        no_blocks : int
            number of blocks
        training_inputs : int
            number of parameters in dataspace (binary parameters)
        cond_inputs : int
            number of population hyperparameters
        batch_size : int
            number of training and validation samples to use in each batch
        total_smdls : int
            total number of subpopulation models
        num_bins : int
            number of bins to use for a spline flow
        RNVP : bool
            whether or not to use realNVP flow, if False use spline
        device : str
            device on which to run pytorch operations, default is CPU, otherwise GPU enabled with device = 'cuda:0'
        """
        self.no_params = training_inputs

        self.total_smdls = total_smdls
        self.cond_inputs = cond_inputs

        self.device_sample = None
        self.device_mapped_sample = None

        self.device = device # cuda:X where X is the slot of the GPU. run nvidia-smi in the terminal to see gpus

        if RNVP:
            self.network = RealNVP(n_inputs = training_inputs, n_conditional_inputs= cond_inputs,
                                    n_neurons = no_neurons, n_transforms = no_trans, n_blocks_per_transform = no_blocks,
                                    linear_transform = None, batch_norm_between_transforms=True)
        else:
            self.network = CouplingNSF(n_inputs = training_inputs, n_conditional_inputs= cond_inputs,
                                        n_neurons = no_neurons, n_transforms = no_trans,
                                        n_blocks_per_transform = no_blocks, batch_norm_between_transforms=True,
                                        linear_transform = None, num_bins=no_bins)

        self.network.to(device)

    def to_torch(self, a):
        return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(self.device)

    def trainval(self, lr, epochs, batch_size, filename, training_data, val_data, update_every, frac_dataset_per_epoch, verbose):
        """
        Train the normalising flow for the specified number of epochs using a set of training and validation data,
        and save the model with the best validation loss

        Parameters
        ----------
        lr : float
            the initial learning rate used to train the normalising flow, which is then reduced with cosine annealing
        epochs : int
            number iterations to train for  - 1 epoch goes through through entire dataset
        batch_size : int
            number of training samples in one batch of training
        filename : str
            directory and filename of where to save best flow model
        training_data : array
            set of training data points for the normlising flow
        val_data : array
            set of validation data points for the normlising flow
        update_every : int or None
            number of epochs after which to plot the loss so far during training. 
            if None will not create updating loss plot and only plot at the end
        frac_dataset_per_epoch : float
            fraction of training and validation data to train on per training epoch. if less than one, speeds up training.
            recommendation is to set this s.t. max number of samples per epoch is O(1e5)
        """
        #set optimiser for flow, optimises flow parameters:
        #affine - s and t that shift and scale the transforms
        #spline - nodes used to model the distribution of CDFs
        optimiser = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=0)
        #set learning rate cheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=0, last_epoch=- 1)

        #unpack training and validation data and assign to torch tensors
        training_samps, training_conds, training_weights = [self.to_torch(a) for a in training_data]
        val_samps, val_conds, val_weights = [self.to_torch(a) for a in val_data]

        #determine batch size in order to use all available training samples
        ytrainsize = len(training_samps)
        n_batches = ytrainsize // batch_size

        #update batch size if not using all samples in training/val set every epoch
        if frac_dataset_per_epoch <1.:
            frac_dataset_per_epoch = min(1.,frac_dataset_per_epoch)
            if frac_dataset_per_epoch < 0.:
                if verbose:
                    print('Cannot use less training samples than 0, setting frac_dataset_per_epoch=0.01')
                frac_dataset_per_epoch = 0.01
            n_train_per_epoch = int(ytrainsize * frac_dataset_per_epoch)
            n_batches = n_train_per_epoch // batch_size
            n_val_per_epoch = int(val_samps.shape[0] * frac_dataset_per_epoch)
            if verbose:
                print(f'Downsampling no. training samples per epoch to {n_train_per_epoch} and no. validation samples per epoch to {n_val_per_epoch}')
        val_batch_size = batch_size
        n_val_batches = n_val_per_epoch // val_batch_size
        if verbose:
            print(f'Using a batch size of {batch_size} for {ytrainsize} total training samples.')

        n_epochs = epochs

        #initialize best flow model
        best_epoch = 0
        best_val_loss = np.inf

        #record network values and outputs in dictionary as training
        self.history = {'train': [], 'val': [], 'lr': []}

        #training loop
        for n in range(n_epochs): 
            shuffled_inds = torch.randperm(ytrainsize, device=self.device)[:n_train_per_epoch]

            train_loss_torch = torch.zeros((), device=self.device)
            unweighted_KL_train = 0

            #set flow into training mode
            self.network.train()
            self.history['lr'].append(scheduler.get_last_lr())
            
            #Training
            for i in range(n_batches):
                idx = shuffled_inds[i * batch_size:(i + 1)*batch_size]
                #calculate the training loss function for flow as -mean(weight(x)*log_prob(x))
                unweighted_KL = self.network.log_prob(training_samps[idx], conditional=training_conds[idx])
                loss = -(training_weights[idx]*unweighted_KL).mean()

                optimiser.zero_grad(set_to_none=True)
                #computes gradient of flow network parameters
                loss.backward()
                #steps optimtiser down gradient of loss surface
                optimiser.step()
                #track flow losses
                #unweighted_KL_train += -unweighted_KL.mean().cpu().item()
                train_loss_torch += loss.detach()
            scheduler.step()

            #track and average losses
            train_loss = train_loss_torch.item() / n_batches
            self.history['train'].append(train_loss)
         
            #evaluate flow parameters for validation
            self.network.eval()
            
            # Validate
            with torch.no_grad(): #disables gradient calculation
                val_idx = torch.randperm(val_samps.shape[0], device=self.device)[:n_val_per_epoch]
                val_samps_batch, val_conds_batch, val_weights_batch = val_samps[val_idx], val_conds[val_idx], val_weights[val_idx]
                weighted_sum = torch.zeros((), device=self.device)

                #loop over all validation data to avoid memory issues, but validate on all available data per epoch
                for i in range(n_val_batches):
                    #calculate flow validation loss
                    unweighted_KL_loss = self.network.log_prob(val_samps_batch[i*val_batch_size:(i + 1)*val_batch_size], conditional=val_conds_batch[i*val_batch_size:(i + 1)*val_batch_size])
                    weighted_sum += (val_weights_batch[i*val_batch_size:(i + 1)*val_batch_size] * unweighted_KL_loss).mean()

                total_val_loss =  - weighted_sum / n_val_batches
                #total_unweighted_KL_val = -unweighted_KL_loss.mean().cpu().numpy()
                #save the loss value of the training data
                self.history['val'].append(total_val_loss.cpu().numpy())

            #print history
            sys.stdout.write(
                    '\r Epoch: {} || Training loss: {} || Validation loss: {}'.format(
                    n+1, train_loss, total_val_loss))
            sys.stdout.flush()
            
            #copy the best flow model
            if total_val_loss < best_val_loss:
                best_epoch = n
                best_val_loss = total_val_loss
                best_model = copy.deepcopy(self.network.state_dict())
            
            if update_every is not None and n % update_every == 0:
                    self.plot_history(filename)
                    torch.save(best_model, f'{filename}_{update_every}.pt')

        #save best model
        print(f'\n Best epoch: {best_epoch}')
        self.network.load_state_dict(best_model)
        torch.save(best_model, f'{filename}.pt')
        self.plot_history(filename)

    def plot_history(self,filename):
        """
        Plots losses for training of network

        filename : str
            directory and filename of where to save loss data and figures
            alongside best flow model
        """

        #loss plot
        plt.rcParams.update({'font.size': 10})
        fig, ax = plt.subplots(figsize = (10,5))
        ax.plot(self.history['val'][5:], label = 'Val loss', color='tab:orange')
        ax.plot(self.history['train'][5:], label = 'Train loss', color='tab:blue')
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_xlabel('Epochs', fontsize=10)
        ax.tick_params(axis='both', labelsize=10)
        text = ax.yaxis.get_offset_text()
        text.set_size(10)
        ax.legend(loc = 'lower left', prop={'size':10})

        #inset log plot
        axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
        valloss = np.asarray(self.history['val'][1:])
        trainloss = np.asarray(self.history['train'][1:])
        axins.plot(valloss, color='tab:orange')
        axins.plot(trainloss, color='tab:blue')
        axins.set_xscale('log')
        axins.tick_params(axis='both', labelsize=10)
        text = axins.yaxis.get_offset_text()
        text.set_size(10)

        #save loss data
        plt.savefig(f'{filename}loss.pdf')
        pd.DataFrame.to_csv(pd.DataFrame.from_dict(self.history),f'{filename}_loss_history.csv')


    def sample(self, conditional,no_samples):
        """
        Pull samples from flow given one pair of population hyperparameters.

        Parameters
        ----------
        coditional : array
            list/array of hyperparameter values to use as conditional values from which to sample from the flow, 
            of shape [Nhyperparams]
        no_samples : int
            number of samples to take for each conditional
        
        Returns
        -------
        samples : array 
            Flow samples in the logistically-mapped space of shape [no_samples, Nparams]
        """
        samples = np.zeros((no_samples, self.no_params))

        #check that requested conditional are the same shape as Nhyperparameters
        if conditional.astype(np.float32).shape[0] != self.cond_inputs:
            raise ValueError(f"Expected shape {self.cond_inputs} but got {conditional.astype(np.float32).shape}")

        with torch.no_grad():
            conditional = torch.from_numpy(conditional.astype(np.float32))
            #tile as many conditional hyperparameter values as no samples
            conditional = conditional.tile(no_samples,1)
            samples = self.network.sample(no_samples, conditional=conditional.to(self.device))

        return(samples.cpu())

    def load_model(self,filename):
        """
        Load pre-trained flow from saved model, and set flow to evaluation mode
        """
        self.network.load_state_dict(torch.load(filename, map_location=torch.device(self.device)))
        self.network.eval()

    def log_jacobian(self,sample, param_dict):
        """
        Calculate the log jacobian term to add to the log likelihood to account for the logistic transforms
        of the samples of [mchirp, q, chieff, z]

        returns the sum of the log of the absolute value of the jacobian term
        """
        #dtheta prime by dtheta
        jac = torch.zeros(sample.shape[0], self.no_params).to(self.device)

        #loop over number of params and add jacobian term, according to the type of transformation specified for that dimension
        for i, param in enumerate(param_dict):
            if param_dict[param]['transf'] == 'logit':
                jac[:,i] = param_dict[param]['max']/((sample[:,i])*(param_dict[param]['max']-(sample[:,i]))*param_dict[param]['logit_max'])
            elif param_dict[param]['transf'] == 'tanh':
                jac[:,i] = 1/(1-sample[:,i]**2)
            else:
                print(f'No transformation type specified for {param} dimension, attempting jacobian calculate for logistic transform')
                jac[:,i] = param_dict[param]['max']/((sample[:,i])*(param_dict[param]['max']-(sample[:,i]))*param_dict[param]['logit_max'])
            
        return torch.sum(torch.log(torch.abs(jac)), dim=1)

    def store_obs_data(self, sample, mapped_sample, param_dict):
        """stores obsdata and mapped data as class attributes for faster evaluation"""

        #save jacobian calculated from float64 to avoiding infs from rounding
        sample_64 = torch.from_numpy(sample.astype(np.float64)).to(self.device)
        self.device_jcb = self.log_jacobian(sample_64[~torch.isnan(sample_64)].reshape(-1,4), param_dict)

        self.device_sample = torch.from_numpy(sample.astype(np.float32)).to(self.device)
        self.device_mapped_sample = torch.from_numpy(mapped_sample.astype(np.float32)).to(self.device)
        self.shape = mapped_sample.shape

        #flatten samples given they are have dimensions Nsamples x Nobs x Nparams
        self.device_sample = torch.flatten(self.device_sample, start_dim=0, end_dim=1)
        self.device_mapped_sample = torch.flatten(self.device_mapped_sample, start_dim=0, end_dim=1)
        self.device_sample = self.device_sample.reshape(-1,self.no_params)
        self.device_mapped_sample = self.device_mapped_sample.reshape(-1,self.no_params)

    def get_logprob(self, sample, mapped_sample, param_dict, conditionals, test_mode=False):
        """
        get log_prob p(theta|Lambda) given a sample of gw observables theta given conditional hyperparameters Lambda

        Parameters
        ----------
        sample : array
            posterior samples of GW observations in unmapped data-space
            [Nobs x Nsamples x Nparams] shape array
        mapped_sample : array
            posterior samples mapped into logistic space with Nflow.map_obs function
            [Nobs x Nsamples x Nparams] shape array
        conditionals : array
            values of population hyperparameters
            [Nobs x Nsamples x Nconditionals] shapped array
        test_mode : bool
            If True uses sample and mapped_sample provided, rather than preiously stored self.device_sample and self.device_mapped_sample

        Returns
        ----------
        log_prob : array
            the log probability of each sample
            [Nobs x Nsamples] shaped array
        """
        #save samples as class method to prevent recasting to torch every call, unless test_mode=True
        #make sure samples in right format
        #sample is now float64 to prevent rounding infs in jacobian
        if test_mode==False and self.device_sample is None:
            self.store_obs_data(sample, mapped_sample, param_dict)
        elif test_mode==True:
            self.store_obs_data(sample, mapped_sample, param_dict)

        hyperparams = torch.from_numpy(conditionals.astype(np.float32)).to(self.device)
        hyperparams = torch.flatten(hyperparams, start_dim=0, end_dim=1)[~torch.isnan(self.device_mapped_sample)[:,0]]
        hyperparams = hyperparams.reshape(-1,self.cond_inputs)

        log_prob = torch.zeros((self.shape[0]*self.shape[1])) - torch.inf
        log_prob = log_prob.to(self.device)
        with torch.no_grad():
            log_prob[~torch.isnan(self.device_mapped_sample)[:,0]] = self.network.log_prob(self.device_mapped_sample[~torch.isnan(self.device_mapped_sample)].reshape(-1,4), hyperparams)
            log_prob[~torch.isnan(self.device_mapped_sample)[:,0]] += self.device_jcb

            #reshape
            log_prob = torch.reshape(log_prob, [self.shape[0],self.shape[1]])

            if torch.any(torch.isnan(log_prob)):
                sys.exit('Unexpected nans in log prob evaluation, quitting code.')
            #log_prob = log_prob.cpu().numpy() 
        return log_prob

    def get_latent_samps(self, samps, conditionals):
        """
        Returns samps of shape [Nsamps, Nparams] mapped to the latent space conditional on conditionals of shape [Nsamps, Nconditionals]
        """
        
        samps = torch.from_numpy(samps.astype(np.float32))
        conditionals = torch.from_numpy(conditionals.astype(np.float32))

        latent_samples, _= self.network.forward(samps.reshape(-1,self.no_params), conditional=conditionals.reshape(-1,self.cond_inputs))
        return latent_samples