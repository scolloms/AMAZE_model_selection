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

import copy
import torch
from  glasflow import RealNVP, CouplingNSF
from torch import nn
import pdb
import time


class NFlow():

    #initialise flow with inputs, conditionals, including the type of network, real non-volume preserving,
    #or neural spline flow
    #spline flow increases the flexibility in the flow model
    def __init__(self, no_trans, no_neurons, training_inputs, cond_inputs,
                no_binaries, batch_size, total_hps, RNVP=True, num_bins=4, device="cpu"):
        """
        Initialise Flow with inputed data, either RNVP or Spline flow.

        Parameters
        ----------
        no_trans : int
            number of transforms to give the flow
        no_neurons : int
            number of neurons to give the flow
        training_inputs : int
            number of parameters in dataspace (binary parameters)
        cond_inputs : int
            number of conditional population hyperparameters
        no_binaries : int
            number of binaries in each population
        batch_size : int
            number of training + validation samples to use in each batch
        total_hps : int
            number of subpopulations
        RNVP : bool
            whether or not to use realNVP flow, if False use spline
        num_bins : int
            number of bins to use for a spline flow
        """
        self.no_params = training_inputs
        self.no_binaries = no_binaries
        self.batch_size = batch_size

        self.total_hps = total_hps

        self.cond_inputs = cond_inputs

        self.device = device # cuda:X where X is the slot of the GPU. run nvidia-smi in the terminal to see gpus

        if RNVP:
            self.network = RealNVP(n_inputs = training_inputs, n_conditional_inputs= cond_inputs,
                                    n_neurons = no_neurons, n_transforms = no_trans, n_blocks_per_transform = 2,
                                    linear_transform = None, batch_norm_between_transforms=True)
        else:
            self.network = CouplingNSF(n_inputs = training_inputs, n_conditional_inputs= cond_inputs,
                                        n_neurons = no_neurons, n_transforms = no_trans,
                                        n_blocks_per_transform = 2, batch_norm_between_transforms=True,
                                        linear_transform = None, num_bins=num_bins)

        self.network.to(device)

    #training and validation loop for the flow
    def trainval(self, lr, epochs, batch_no, filename, training_data, val_data):


        #set optimiser for flow, optimises flow parameters:
        #(affine - s and t that shift and scale the transforms)
        #(spline - nodes used to model the distribution of CDFs)
        optimiser = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=0)

        n_epochs = epochs #number of iterations to train  - 1 epoch goes through through entire dataset
        n_batches = batch_no #number of batches of data in one iteration

        #initialize best flow model
        best_epoch = 0
        best_val_loss = np.inf

        #record network values and outputs in dictionary as training
        self.history = {'train': [], 'val': []}

        #training loop
        for n in range(n_epochs): 
            train_loss = 0
            #set flow into training mode
            self.network.train()
            
            #Training
            for _ in range(n_batches):
                #split training data into - train: binary params; conditional: pop hyperparams
                x_train, x_conditional, xweights = self.get_training_data(training_data)

                #sets flow optimisers gradients to zero
                optimiser.zero_grad()
                #calculate the training loss function for flow as -log_prob
                loss = -(xweights*self.network.log_prob(x_train, conditional=x_conditional)).mean()
                #computes gradient of flow network parameters
                loss.backward()
                #steps optimtiser down gradient of loss surface
                optimiser.step()
                #track flow losses
                train_loss += loss.item()

            #track and average losses
            train_loss /= n_batches
            self.history['train'].append(train_loss)
            
            # Validate
            with torch.no_grad(): #disables gradient caluclation
                #call validation data
                x_val, x_conditional_val, x_weights_val = self.get_val_data(val_data)
                
                #evaluate flow parameters
                self.network.eval()
                #calculate flow validation loss
                val_loss_f = - (x_weights_val*self.network.log_prob(x_val, conditional=x_conditional_val)).mean().numpy()
                total_val_loss=val_loss_f
                self.history['val'].append(total_val_loss)#save the loss value of the training data

            #print history
            sys.stdout.write(
                    '\r Epoch: {} || Training loss: {} || Validation loss: {}'.format(
                    n+1, train_loss, total_val_loss))

            #copy the best flow model 
            if total_val_loss < best_val_loss:
                best_epoch = n
                best_val_loss = total_val_loss
                best_model = copy.deepcopy(self.network.state_dict())

        #save best model
        print(f'\n Best epoch: {best_epoch}')
        self.network.load_state_dict(best_model)
        torch.save(best_model, filename)

    def plot_history(self, filename=None):
        """
        Plots losses, KL, and latent space

        Parameters
        ----------
        filename : str
            where to save history values
        """

        #loss plot
        fig, ax = plt.subplots(figsize = (10,5))
        ax.plot(self.history['train'][5:], label = 'Train loss')
        ax.plot(self.history['val'][5:], label = 'Validation loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        ax.legend(loc = 'best')

        #inset log plot
        axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
        trainloss = np.asarray(self.history['train'][:])
        axins.plot(trainloss, label = 'Train loss')#-np.min(trainloss)
        valloss = np.asarray(self.history['val'][:])
        axins.plot(valloss, label = 'Validation loss')#-np.min(trainloss)
        axins.set_xscale('log')
        #axins.set_yscale('log')
        plt.show()

        pd.DataFrame.to_csv(pd.DataFrame.from_dict(self.history),f'{filename}_means.csv')

    def sample(self, no_samples, conditional):
        """
        Pull samples from flow given one pair of population hyperparameters

        Parameters
        ----------
        no_samples : int
            number of samples to take for each conditional
        coditional : array
            [chi_b, alpha]
        
        Returns
        -------
        array 
            [no_samples, self.no_params]
        """
        samples = np.zeros((no_samples, self.no_params))

        with torch.no_grad():
            conditional = torch.from_numpy(conditional.astype(np.float32))
            #tile as many conditional chi_b alpha pairs as no samples
            conditional = conditional.tile(no_samples,1)
            samples = self.network.sample(no_samples, conditional=conditional)

        return(samples)

    def get_training_data(self, training_samples):
        """
        Get random batch training data from self.training_samples
        
        Returns
        -------
        xdata : tensor 
            [no_samples, self.no_params]
        x_hyperparams : tensor

        """
        #treatment of separation of training and validation data is different for 2d CE channel than 1d channels
        #differentiated by size of conditional inputs
        #2D channel has seperate populations for training and validation data, 1D mixes up samples
        if self.cond_inputs >=2:
            random_samples = np.random.choice(np.shape(training_samples)[0],size=(int(self.batch_size)))
            batched_hp_pairs = training_samples[random_samples,-2:]
            batch_weights = training_samples[random_samples,-3]
        else:
            random_samples = np.random.choice(self.no_binaries,size=(int(self.batch_size)))
            batched_hp_pairs = training_samples[random_samples, -1]
            batch_weights = training_samples[random_samples,-2]

        batched_samples = training_samples[random_samples,:(self.no_params)]

        #reshape tensors
        xdata=torch.from_numpy(batched_samples.astype(np.float32))
        #xhyperparams = np.concatenate(batched_hp_pairs)
        xhyperparams = torch.from_numpy(batched_hp_pairs.astype(np.float32))
        xhyperparams = xhyperparams.reshape(-1,self.cond_inputs)
        xweights = torch.from_numpy(batch_weights.astype(np.float32))

        return(xdata, xhyperparams,xweights)

    def get_val_data(self, validation_data):
        """
        Get random batch validation data from self.validation_data
        
        Returns
        -------
        xdata : tensor 
            [no_samples, self.no_params]
        x_hyperparams : tensor

        """
        random_samples = np.random.choice(np.shape(validation_data)[0], size=(int(self.batch_size)))
        validation_hp_pairs = validation_data[random_samples,-self.cond_inputs:]
        validation_samples = validation_data[random_samples,:self.no_params]

        if self.cond_inputs >=2:
            val_weights = validation_data[random_samples,-3]
        else:
            val_weights = validation_data[random_samples,-2]

        #reshape
        xval=torch.from_numpy(validation_samples.astype(np.float32))
        xhyperparams = torch.from_numpy(validation_hp_pairs.astype(np.float32)) 
        xhyperparams = xhyperparams.reshape(-1,self.cond_inputs)
        xweights = torch.from_numpy(val_weights.astype(np.float32)) 
        return(xval, xhyperparams, xweights)

    def load_model(self,filename):
        """
        Load pre-trained flow from saved model
        """
        self.network.load_state_dict(torch.load(filename))
        self.network.eval()

    def get_logprob(self, sample, conditionals):
        """
        get log_prob given a sample of [mchirp,q,chieff,z] given conditional hyperparameters
        """
        #make sure samples in right format
        sample = torch.from_numpy(sample.astype(np.float32)).to(self.device)
        hyperparams = torch.from_numpy(conditionals.astype(np.float32)).to(self.device)
        #store shape
        shape = sample.shape

        #flatten samples given they are have dimensions Nsampels x Nobs x Nparams
        sample = torch.flatten(sample, start_dim=0, end_dim=1)
        hyperparams = torch.flatten(hyperparams, start_dim=0, end_dim=1)

        hyperparams = hyperparams.reshape(-1,self.cond_inputs)
        sample = sample.reshape(-1,4)

        with torch.no_grad():
            log_prob = self.network.log_prob(sample, hyperparams)
            #reshape
            log_prob = torch.reshape(log_prob, [shape[0],shape[1]])

            log_prob = log_prob.cpu().numpy() 
            if np.any(np.isnan(log_prob)):
                print('nans!')
            log_prob[np.isnan(log_prob)] = -np.inf

        return log_prob