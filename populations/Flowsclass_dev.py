import sys
import os
import pickle
import itertools
import copy
from tqdm import tqdm
import multiprocessing
from functools import partial
import warnings
import pdb
import time
import wandb

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import norm, truncnorm
from scipy.special import logit
from scipy.special import logsumexp
from scipy.special import expit
from sklearn.model_selection import train_test_split
from .utils.selection_effects import projection_factor_Dominik2015_interp, _PSD_defaults
from .utils.flow import NFlow

from astropy import cosmology
from astropy.cosmology import z_at_value
import astropy.units as u
cosmo = cosmology.Planck18

# Get the interpolation function for the projection factor in Dominik+2015
# which takes in a random number and spits out a projection factor 'w'
projection_factor_interp = projection_factor_Dominik2015_interp()

_param_bounds = {"mchirp": (0,100), "q": (0,1), "chieff": (-1,1), "z": (0,10)}

"""
Set of classes used to construct statistical models of populations.
"""

class Model(object):
    """
    Base model class. Mostly used to root the inheritance tree.
    """
    def __init__(self):
        pass

    def __call__(self, data):
        return None


class FlowModel(Model):
    @staticmethod
    def from_samples(channel, samples, params, sensitivity, normalize=False, detectable=False, device='cpu', no_bins=4, use_unityweights=False, randch_weights=False):
        """
        Generate a Flow model instance from `samples`, where `params` are series in the `samples` dataframe. 
        
        If `weight` is a column in your population model, will assume this is the cosmological weight of each sample,
        and will include this in the construction of all your KDEs. If `sensitivity` 
        is provided, samples used to generate the detection-weighted KDE will be 
        weighted according to the key in the argument `pdet_*sensitivity*`.

        Inputs
        ----------
        channel : str
            channel label of form 'CE'
        samples : Dataframe
            samples from population synthesis.
            contains all binary parameters in 'params' array, cosmo_weights, pdet weights, optimal snrs
        params : list of str
            subset of mchirp, q, chieff, z
        sensitivity : str
            Desired detector sensitivity consistent with the string following the `pdet` and `snropt` columns in the population dataframes.
            Used to construct a detection-weighted population model, as well as for drawing samples from the underlying population
        normalize : bool
            state of normalisation, used in sampling for the KDE model, but defunt here.
        detectable : bool
            whether or not to construct a detection weighted flow model
        deivce : str
            Device on which to run the flow. default is 'cpu', otherwise choose 'cuda:X' where X is the GPU slot.
        
        Returns
        ----------
        FlowModel : obj
        """

        return FlowModel(channel, samples, params, sensitivity, normalize=normalize, detectable=detectable, device=device, no_bins=no_bins,\
            use_unityweights=use_unityweights, randch_weights=randch_weights)


    def __init__(self, channel, samples, params, sensitivity=None, normalize=False, detectable=False, device='cpu', no_bins=4, use_unityweights=False, randch_weights=False):
        """
        Initialisation for FlowModel object. Sets self.flow as instance of Nflow class, of which FlowModel is wrapper of that object.

        Parameters
        ----------
        channel : str
            channel label of form 'CE'
        samples : Dataframe
            samples from population synthesis.
            contains all binary parameters in 'params' array, cosmo_weights, pdet weights, optimal snrs
        params : list of str
            subset of [mchirp, q, chieff, z]
        sensitivity : str
            Desired detector sensitivity consistent with the string following the `pdet` and `snropt` columns in the population dataframes.
            Used to construct a detection-weighted population model, as well as for drawing samples from the underlying population
        normalize : bool
            state of normalisation, used in sampling for the KDE model, but defunt here.
        detectable : bool
            whether or not to construct a detection weighted flow model
        deivce : str
            Device on which to run the flow. default is 'cpu', otherwise choose 'cuda:X' where X is the GPU slot.
        """
        
        super()
        self.channel_label = channel
        self.samples = samples
        self.params = params
        self.sensitivity = sensitivity
        self.normalize = normalize
        self.detectable = detectable

        #initialises list of population hyperparameter values
        self.hps = [[0.,0.1,0.2,0.5]]
        self.param_bounds = [_param_bounds[param] for param in self.params]

        #additional alpha dimension for CE channel, else dummy dimension
        if self.channel_label=='CE':
            self.hps.append([0.2,0.5,1.,2.,5.])
        else:
            self.hps.append([1])
        
        #number of binary parameters
        self.no_params = np.shape(params)[0]
        #dimensionailty of non-branching ratio hyperparameters
        self.conditionals = 2 if self.channel_label =='CE' else 1

        #initialise dictionaries of alpha, cosmo_weights, pdet, optimal_snrs, and combined_weights for each submodel
        alpha = dict.fromkeys(samples.keys())
        cosmo_weights= dict.fromkeys(samples.keys())
        pdets= dict.fromkeys(samples.keys())
        optimal_snrs= dict.fromkeys(samples.keys())
        combined_weights= dict.fromkeys(samples.keys())

        #loop over submodels
        for chib_id, chib in enumerate(self.hps[0]):
            for alphaid, alphaCE in enumerate(self.hps[1]):

                #grab samples for each submodel depending on how many hyperparameter dimensions there are (CE vs non-CE)
                if self.channel_label=='CE':
                    sbml_samps = samples[chib_id,alphaid]
                    key = chib_id,alphaid
                else:
                    sbml_samps = samples[chib_id]
                    key= chib_id

                #check that defined sensitivity exists in submodel dataframe
                if sensitivity is not None:
                    if 'pdet_'+sensitivity not in sbml_samps.columns:
                        raise ValueError("{0:s} was specified for your detection weights, but cannot find this column in the samples datafarme!")

                # get *\alpha* for each model, defined as \int p(\theta|\lambda) Pdet(\theta) d\theta
                if sensitivity is not None:
                    # if cosmological weights are provided, do mock draws from the pop
                    if 'weight' in sbml_samps.keys():
                        mock_samp = sbml_samps.sample(int(1e6), weights=(sbml_samps['weight']/len(sbml_samps)), replace=True)
                    else:
                        mock_samp = sbml_samps.sample(int(1e6), replace=True)
                    alpha[key]=np.sum(mock_samp['pdet_'+sensitivity]) / len(mock_samp)
                else:
                    alpha[key]=1.0

                ### GET WEIGHTS ###
                # if cosmological weights are provided...
                if 'weight' in sbml_samps.keys():
                    cosmo_weights[key] = np.asarray(sbml_samps['weight'])
                else:
                    cosmo_weights[key] = np.ones(len(sbml_samps))
                # if detection weights are provided...
                if sensitivity is not None:
                    pdets[key] = np.asarray(sbml_samps['pdet_'+sensitivity])
                else:
                    pdets[key] = np.ones(len(sbml_samps))

                # get optimal SNRs for this sensitivity
                if sensitivity is not None:
                    optimal_snrs[key] = np.asarray(sbml_samps['snropt_'+sensitivity])
                else:
                    optimal_snrs[key] = np.nan*np.ones(len(sbml_samps))

                #if not using cosmo_weights then set to None, later sets combined weights to be 1
                if use_unityweights == True:
                    cosmo_weights[key] = None

                # Combine the cosmological and detection weights
                # detectable only used for plotting
                if self.detectable == True:
                    if (cosmo_weights[key] is not None) and (pdets[key] is not None):
                        combined_weights[key] = (cosmo_weights[key] / np.sum(cosmo_weights[key])) * (pdets[key] / np.sum(pdets[key]))
                    elif pdets[key] is not None:
                        combined_weights[key] = (pdets[key] / np.sum(pdets[key]))
                    elif (cosmo_weights[key] is not None):
                        combined_weights[key] = (cosmo_weights[key] / np.sum(cosmo_weights[key]))
                    else:
                        combined_weights[key] = np.ones(len(sbml_samps))
                else:
                    if (cosmo_weights[key] is not None):
                        combined_weights[key] = (cosmo_weights[key] / np.sum(cosmo_weights[key]))
                    else:
                        combined_weights[key] = np.ones(len(sbml_samps))
                if use_unityweights == True:
                    pass
                else:
                    combined_weights[key] /= np.sum(combined_weights[key])

        #sets weights as class properties
        self.combined_weights = combined_weights
        self.pdets = pdets
        self.optimal_snrs = optimal_snrs
        self.alpha = alpha
        self.cosmo_weights = cosmo_weights
        
        #flow network parameters
        self.no_trans = 6
        self.no_neurons = 128
        batch_size=10000
        self.total_hps = np.shape(self.hps[0])[0]*np.shape(self.hps[1])[0]

        channel_ids = {'CE':0, 'CHE':1,'GC':2,'NSC':3, 'SMT':4}
        channel_id = channel_ids[self.channel_label] #will be 0, 1, 2, 3, or 4

        #number of data points (total) for each channel
        #no_binaries is total number of samples across sub-populations for non-CE channels, and no samples in each sub-population for CE channel
        if use_unityweights==True:
            channel_samples=[20e6,864124,896611,582961, 4e6]
        else:
            channel_samples = [19912038,864124,896611,582961, 4e6]
        self.no_binaries = int(channel_samples[channel_id])

        #initislises network
        flow = NFlow(self.no_trans, self.no_neurons, self.no_params, self.conditionals, self.no_binaries, batch_size, 
                    self.total_hps, self.channel_label, RNVP=False, device=device, no_bins=no_bins, randch_weights=randch_weights)
        self.flow = flow


    def map_samples(self, samples, params, filepath, testCEsmdl=True):
        """
        Maps samples with logistic mapping (mchirp, q, z samples) and tanh (chieff).
        Stacks data by [mchirp,q,chieff,z,weight,chi_b,(alpha)].
        Handles any channel.

        Parameters
        ----------
        samples : dict
            dictionary of data in form 
            ['mchirp', 'q', 'chieff', 'z', 'm1' 'm2' 's1x' 's1y' 's1z' 's2x' 's2y' 's2z'
            'weight' 'pdet_midhighlatelow_network' 'snropt_midhighlatelow_network'
            'pdet_midhighlatelow' 'snropt_midhighlatelow']
        params : list of str
            list of parameters to be used for inference, typically ['mchirp', 'q', 'chieff', 'z']
        filepath : str
            the filepath to the flow models and mappings to be loaded/saved
        
        Returns
        -------
        training_data : array
            data samples to be used for training the normalising flow.
            [mchirp, q, chieff, z, weights, chi_b,(alpha)]
        val_data : array
            data samples to be used for validating the normalising flow.
            for the non-CE channels this is the same as the training data.
            for the CE channel this is set to 2 of the 20 sub-populations
        mappings : array
            constants used to map the mchirp, q, and z distributions.
        """

        print('Mapping population synthesis samples for training...')
        
        channel_ids = {'CE':0, 'CHE':1,'GC':2,'NSC':3, 'SMT':4}
        channel_id = channel_ids[self.channel_label] #will be 0, 1, 2, 3, or 4

        if self.channel_label != 'CE':
            #Channels with 1D hyperparameters: SMT, GC, NSC, CHE

            #put samples from binary parameters for all chi_bs into model_stack
            models = np.zeros((self.no_binaries, self.no_params))
            weights = np.zeros((self.no_binaries))
            model_size = np.zeros(self.no_params)
            cumulsize = np.zeros(self.no_params)

            #moves binary parameter samples and weights from dictionaries into arrays, reducing dimension in hyperparameter space
            for chib_id, xb in enumerate(self.hps[0]):

                model_size[chib_id] = np.shape(samples[(chib_id)][params])[0]
                cumulsize[chib_id] = np.sum(model_size)
                models[int(cumulsize[chib_id-1]):int(cumulsize[chib_id])]=np.asarray(samples[(chib_id)][params])
                weights[int(cumulsize[chib_id-1]):int(cumulsize[chib_id])]=np.asarray(self.combined_weights[(chib_id)])
            models_stack = np.copy(models)

            #scatter
            for idx, param in enumerate(models_stack.T):
                if len(np.unique(param))==1:
                    models_stack[:,idx] += np.random.normal(loc=0.0, scale=1e-5, size=models_stack.shape[0])

            #map samples before dividing into training and validation data
            models_stack[:,0], max_logit_mchirp, max_mchirp = self.logistic(models_stack[:,0],wholedataset=True, \
                rescale_max=self.param_bounds[0][1])
            if channel_id == 2:
                #add extra tiny amount to GC mass ratios as q=1 samples exist
                models_stack[:,1], max_logit_q, max_q = self.logistic(models_stack[:,1],wholedataset=True, \
                rescale_max=self.param_bounds[1][1]+0.001)
            else:
                models_stack[:,1], max_logit_q, max_q = self.logistic(models_stack[:,1],wholedataset=True, \
                rescale_max=self.param_bounds[1][1])
            models_stack[:,2] = np.arctanh(models_stack[:,2])
            models_stack[:,3],max_logit_z, max_z = self.logistic(models_stack[:,3],wholedataset=True, \
                rescale_max=self.param_bounds[3][1])

            training_hps_stack = np.repeat(self.hps[0], (model_size).astype(int), axis=0)
            training_hps_stack = np.reshape(training_hps_stack,(-1,self.conditionals))
            weights = np.reshape(weights,(-1,1))
            train_models_stack, validation_models_stack, train_weights, validation_weights, training_hps_stack, validation_hps_stack = \
                    train_test_split(models_stack, weights, training_hps_stack, shuffle=True, train_size=0.8)
            
        else:
            #CE channel with alpha_CE parameter

            #put data from required parameters for all alphas and chi_bs into model_stack

            if testCEsmdl:
                self.no_binaries = self.no_binaries - 996688
                test_model_id = [1,2]
                test_model_id_flat = 7
                self.total_hps = self.total_hps - 1

            models = np.zeros((self.no_binaries, self.no_params))
            weights = np.zeros((self.no_binaries,1))

            model_size = np.zeros((4,5))
            cumulsize = np.zeros(self.total_hps)

            #format which chi_bs and alphas match which parameter values being read in
            chi_b_alpha_pairs= np.zeros((20, 2))
            chi_b_alpha_pairs[:,0] = np.repeat(self.hps[0],np.shape(self.hps[1])[0])
            chi_b_alpha_pairs[:,1] = np.tile(self.hps[1], np.shape(self.hps[0])[0])
            if testCEsmdl:
                chi_b_alpha_pairs = np.delete(chi_b_alpha_pairs, test_model_id_flat, axis=0)

            #stack data
            i=0
            for chib_id in range(4):
                for alpha_id in range(5):
                    if testCEsmdl:
                        if [chib_id, alpha_id] == test_model_id:
                            continue
                    weights_temp=np.asarray(self.combined_weights[(chib_id, alpha_id)])
                    weights_idxs = np.argwhere((weights_temp) > np.finfo(np.float32).tiny)
                    model_size[chib_id, alpha_id] = np.shape(weights_idxs)[0]
                    cumulsize[i] = np.sum(model_size)

                    models[int(cumulsize[i-1]):int(cumulsize[i])]=np.reshape(np.asarray(samples[(chib_id, alpha_id)][params])[weights_idxs],(-1,len(params)))
                    weights[int(cumulsize[i-1]):int(cumulsize[i])]=np.reshape(np.asarray(self.combined_weights[(chib_id, alpha_id)])[weights_idxs],(-1,1))
                    i+=1

            flat_model_size = np.reshape(model_size, 20)
            if testCEsmdl:
                flat_model_size = np.delete(flat_model_size, test_model_id_flat)

            all_chi_b_alphas = np.repeat(chi_b_alpha_pairs, (flat_model_size).astype(int), axis=0)

            for idx, param in enumerate(models.T):
                if len(np.unique(param))==1:
                    models[:,idx] += np.random.normal(loc=0.0, scale=1e-5, size=models.shape[0])

            #reshaping popsynth samples into array of shape [Nsmdls,Nbinaries,Nparams]
            joined_chib_samples = models

            #map samples before dividing into training and validation data

            #TO CHANGE - needs to account for different sets of parameters
            #chirp mass original range 0 to inf
            joined_chib_samples[:,0], max_logit_mchirp, max_mchirp = self.logistic(joined_chib_samples[:,0], wholedataset=True, \
                rescale_max=self.param_bounds[0][1])

            #mass ratio - original range 0 to 1
            joined_chib_samples[:,1], max_logit_q, max_q = self.logistic(joined_chib_samples[:,1], wholedataset=True, \
                rescale_max=self.param_bounds[1][1])

            #chieff - original range -1 to +1
            joined_chib_samples[:,2] = np.arctanh(joined_chib_samples[:,2])

            #redshift - original range 0 to inf
            joined_chib_samples[:,3], max_logit_z, max_z = self.logistic(joined_chib_samples[:,3],wholedataset=True, \
                rescale_max=self.param_bounds[3][1])

            weights = np.reshape(weights,(-1,1))
            train_models_stack, validation_models_stack, train_weights, validation_weights, training_hps_stack, validation_hps_stack = \
                    train_test_split(joined_chib_samples, weights, all_chi_b_alphas, shuffle=True, train_size=0.8)

        #concatenate data and weights and hyperparams
        training_data = np.concatenate((train_models_stack, train_weights, training_hps_stack), axis=1)
        val_data = np.concatenate((validation_models_stack, validation_weights, validation_hps_stack), axis=1)

        #save mapping constants
        mappings = np.asarray([max_logit_mchirp, max_mchirp, max_logit_q, max_q, max_logit_z, max_z])
        np.save(f'{filepath}{self.channel_label}_mappings.npy',mappings)
        
        return(training_data, val_data, mappings)

    #TO CHANGE - for fake observations. 
    def sample(self, conditional, N=1):
        """
        Samples Flow
        """
        logit_samps = self.flow.sample(N,conditional)

        #map samples back from logit space
        samps = np.zeros(np.shape(logit_samps))
        samps[:,0] = self.expistic(logit_samps[:,0], self.mappings[0], self.mappings[1])
        samps[:,1] = self.expistic(logit_samps[:,1], self.mappings[2], self.mappings[3])
        samps[:,2] = np.tanh(logit_samps[:,2])
        samps[:,3] = self.expistic(logit_samps[:,3], self.mappings[4], self.mappings[5])
        return samps

    def __call__(self, data, conditional_hp_idxs, smallest_N, prior_pdf=None, proc_idx=None, return_dict=None):
        """
        Calculate the likelihood of the observations give a particular hypermodel (given by conditional_hps).
        (this is the hyperlikelihood).

        Parameters
        ----------
        data : array
            posterior samples of observations or mock observations for which to calculate the likelihoods,
            shape[Nobs x Nsample x Nparams]
        conditional_hp_idxs : array
            indices of hyperparameters for require submodel. of shape [self.conditionals]
        prior_pdf : array
            p(x) prior on the data
            If prior_pdf is None, each observation is expected to have equal
            posterior probability. Otherwise, the prior weights should be
            provided as the dimemsions [samples(Nobs), samples(Nsamps)].
        proc_idx : int
            index of return_dict for multiprocessing
        return_dict : dict
            stores a dictionary of likelihoods for multiprocessing

        Returns
        -------
        likelihood : array
        the log likelihoods obtained from the flow model for each event, shape [Nobs]
        """
        
        likelihood = np.ones(data.shape[0]) * -np.inf
        prior_pdf = prior_pdf if prior_pdf is not None else np.ones((data.shape[0],data.shape[1]))
        prior_pdf[prior_pdf==0] = 1e-50

        conditional_hps = []

        #gets values of hyperparameters at indices given by conditional_hp_idxs
        #self.conditionals is number of hyperparameters, self.hps is list of hyperparameters [[chi_b],[alpha]]
        for i in range(self.conditionals):
            conditional_hps.append(self.hps[i][conditional_hp_idxs[i]])
        conditional_hps = np.asarray(conditional_hps)

        #maps observations into the logistically mapped space
        mapped_obs = self.map_obs(data)


        #conditionals tiled into shape [Nobs x Nsamples x Nconditionals]
        conditionals = np.repeat([conditional_hps],np.shape(mapped_obs)[1], axis=0)
        conditionals = np.repeat([conditionals],np.shape(mapped_obs)[0], axis=0)

        #calculates likelihoods for all events and all samples
        likelihoods_per_samp = self.flow.get_logprob(data, mapped_obs, self.mappings, conditionals) - np.log(prior_pdf)


        if smallest_N is not None:
            #LSE population probability plus uniform regularisation
            pi_reg = np.log(1/(smallest_N+1))
            q_weight = np.log(smallest_N/(smallest_N+1))
            likelihoods_per_samp = logsumexp([q_weight + likelihoods_per_samp, pi_reg*np.ones(likelihoods_per_samp.shape)], axis=0)
            
        #checks for nans
        if np.any(np.isnan(likelihoods_per_samp)):
            raise Exception('Obs data is outside of range of samples for channel - cannot logistic map.')

        #adds likelihoods from samples together and then sums over events, normalise by number of samples
        #likelihood in shape [Nobs]
        likelihood = logsumexp([likelihood, logsumexp(likelihoods_per_samp, axis=1) - np.log(data.shape[1])], axis=0)
        # store value for multiprocessing
        if return_dict is not None:
            return_dict[proc_idx] = likelihood
        
        return likelihood

    def map_obs(self,data):
        """
        Maps oberservational data into logistically mapped space for flows to handle.

        Parameters
        -------
        data : array
            observations in array Nobs x Nsamples x Nparams

        Returns
        -------
        mapped_data : array
            observational binary parameters logistically mapped

        TO CHANGE - account for different subsets of parameters.
        mappings in form [max_logit_mchirp, max_mchirp, max_q, None, max_logit_z, max_z]

        """
        mapped_data = np.zeros((np.shape(data)[0],np.shape(data)[1],np.shape(data)[2]))

        mapped_data[:,:,0],_,_= self.logistic(data[:,:,0], False, max=self.mappings[0], rescale_max=self.mappings[1])
        mapped_data[:,:,1],_,_= self.logistic(data[:,:,1], False, max=self.mappings[2], rescale_max=self.mappings[3])
        mapped_data[:,:,2]= np.arctanh(data[:,:,2])
        mapped_data[:,:,3],_,_= self.logistic(data[:,:,3], False, max=self.mappings[4], rescale_max=self.mappings[5])

        return mapped_data


    def logistic(self, data, wholedataset, max =1, rescale_max=1):
        """
        Logistically maps sample in non-logistsic space
        input is [Nsamps] shape array
        if the whole training set is passed to the function, this determines the maximum rescaling values
        """

        #rescales samples so that they lie between 0 to 1, according to the upper bound of the parameter space
        rescale_max = rescale_max
        d = data/rescale_max
        
        #sample must be within bounds for logistic function to return definite value
        if np.logical_or(d <= 0, d >= 1).any():
            raise Exception('Data out of bounds for logistic mapping')

        #takes the logistic of sample
        d = logit(d)

        #scales the distribution in logistic space, so that the samples can have spread O(1), easier for flow to learn
        if wholedataset:
            max = np.max(d)
        else:
            max = max
        d /= max
        return([d, max, rescale_max])

    def expistic(self, data, max, rescale_max=None):
        data*=max
        data = expit(data)
        if rescale_max != None:
            data *=rescale_max
        return(data)

    def wandb_init(self, epochs):
        """
        Initialises a wandb sweep and then uses a wandb agent to train a flow under that sweeps regimes
        current process optimises the number of epochs over validation loss, 
        the prior for the number of epochs being uniform between argument's number of epochs and 3*args.epochs
        """
        #setting up wandb sweep parameters

        sweep_config = {
            'method': 'bayes'
            }

        metric = {
            'name': 'val_loss',
            'goal': 'minimize'   
            }

        sweep_config['metric'] = metric

        parameters_dict = {
            'lr': {
                'value':0.001
            },
            'epochs': {
                'value':10000
            },
            'batch_no': {
                'value':10
            },
            'no_trans': {
                'distribution': 'int_uniform',
                'min': 4,
                'max': 10
            },
            'no_neurons': {
                'distribution': 'int_uniform',
                'min': 12,
                'max': 128
            },
            'no_bins': {
                'distribution': 'int_uniform',
                'min': 3,
                'max': 8
            }                
            }

        sweep_config['parameters'] = parameters_dict

        sweep_id = wandb.sweep(sweep_config, project=f"{self.channel_label}_transneubins_sweep")
        
        wandb.agent(sweep_id, self.wandbtrain, count=20)
    
    def wandbtrain(self, config=None):
        with wandb.init(config=config):
            config = wandb.config

            batch_size=10000
            total_hps = np.shape(self.hps[0])[0]*np.shape(self.hps[1])[0]
            device='cuda:0'

            flow = NFlow(config.no_trans, config.no_neurons, self.no_params, self.conditionals, self.no_binaries, batch_size, 
                    total_hps, self.channel_label, RNVP=False, device=device, no_bins=config.no_bins)
            self.flow = flow
            self.train(config.lr, config.epochs, config.batch_no, f"./wandb_models/{wandb.run.id}_wandb", self.channel_label, True)

    def train(self, lr, epochs, batch_no, filepath, channel, use_wandb):

        training_data, val_data, self.mappings = self.map_samples(self.samples, self.params, filepath)
        save_filename = f'{filepath}{channel}'
        self.flow.trainval(lr, epochs, batch_no, save_filename, training_data, val_data, use_wandb)

    def load_model(self, filepath, channel):
        self.flow.load_model(f'{filepath}{channel}.pt')
        self.mappings = np.load(f'{filepath}{channel}_mappings.npy', allow_pickle=True)
        if self.channel_label == 'GC':
            self.mappings[self.mappings==None] = 1.001
        else:
            self.mappings[self.mappings==None] = 1.


    ######CURRENTLY don't worry about functions below here - theyre used for plotting or simulated events
    """
    def marginalize(self, params, alpha, bandwidth=_kde_bandwidth):

        #Generate a new, lower dimensional, KDEModel from the parameters in [params]

        label = self.label
        for p in params:
            label += '_'+p
        label += '_marginal'

        return KDEModel(label, self.samples[params], params, bandwidth, self.cosmo_weights, self.sensitivity, self.pdets, self.optimal_snrs, alpha, self.normalize, self.detectable)


    def generate_observations(self, Nobs, uncertainty, sample_from_kde=False, sensitivity='design_network', multiproc=True, verbose=False):

        #Generates samples from KDE model. This will generated Nobs samples, storing the attribute 'self.observations' with dimensions [Nobs x Nparam]. 

        if verbose:
            print("   drawing {} observations from channel {}...".format(Nobs, self.label))

        ### If sample_from_KDE is specified... ###
        # draw samples from the detection-weighted KDE, which is quicker,
        # but not compatible with SNR-dependent uncertainty
        if sample_from_kde==True:
            if uncertainty=='snr':
                raise ValueError("You cannot sample from the detection-weighted KDE with an SNR-dependent measurement uncertainty, since we need the detection probabilities and optimal SNRs of individual systems! If you wish to use SNR-weighted uncertainties, please do not use the argument 'sample-from-kde'.")
            observations = self.sample(Nobs)
            self.observations = observations
            return observations

        ### Otherwise, draw samples from the population used to construct the KDEs ###
        self.snr_thresh = _PSD_defaults['snr_network'] if 'network' in sensitivity else _PSD_defaults['snr_single']

        # allocate empty arrays
        observations = np.zeros((Nobs, self.samples.shape[-1]))
        snrs = np.zeros(Nobs)
        Thetas = np.zeros(Nobs)

        # find indices for systems that can potentially be detected
        # loop until we have enough systems with SNRs greater than the SNR threshold
        recovered_idxs = []
        for idx in tqdm(np.arange(Nobs), total=Nobs):
            detected = False
            while detected==False:
                sys_idx = np.random.choice(np.arange(len(self.pdets)), p=(self.cosmo_weights/np.sum(self.cosmo_weights)))
                pdet = self.pdets[sys_idx]
                snr_opt = self.optimal_snrs[sys_idx]
                Theta = float(projection_factor_interp(np.random.random()))

                # if the SNR is greater than the threshold, the system is "observed"
                if snr_opt*Theta >= self.snr_thresh:
                    if sys_idx in recovered_idxs:
                        continue
                    detected = True
                    observations[idx,:] = np.asarray(self.samples.iloc[sys_idx])
                    snrs[idx] = snr_opt*Theta
                    Thetas[idx] = Theta
                    recovered_idxs.append(sys_idx)

        self.observations = observations
        self.snrs = snrs
        self.Thetas = Thetas
        return observations


    def measurement_uncertainty(self, Nsamps, method='delta', observation_noise=False, verbose=False):

        #Mocks up measurement uncertainty from observations using specified method

        if verbose:
            print("   mocking up observation uncertainties for the {} channel using the '{}' method...".format(self.label, method))

        params = self.params

        if method=='delta':
            # assume a delta function measurement
            obsdata = np.expand_dims(self.observations, 1)
            return obsdata

        # set up obsdata as [obs, samps, params]
        obsdata = np.zeros((self.observations.shape[0], Nsamps, self.observations.shape[-1]))
        
        # for 'gwevents', assume snr-independent measurement uncertainty based on the typical values for events in the catalog
        if method == "gwevents":
            for idx, obs in tqdm(enumerate(self.observations), total=len(self.observations)):
                for pidx in np.arange(self.observations.shape[-1]):
                    mu = obs[pidx]
                    sigma = [_posterior_sigmas[param] for param in self.samples.columns][pidx]
                    low_lim = self.param_bounds[pidx][0]
                    high_lim = self.param_bounds[pidx][1]

                    # construnct gaussian and drawn samples
                    dist = norm(loc=mu, scale=sigma)

                    # if observation_noise is specified, wiggle around the observed value
                    if observation_noise==True:
                        mu_obs = dist.rvs()
                        dist = norm(loc=mu_obs, scale=sigma)

                    samps = dist.rvs(Nsamps)

                    # reflect samples if drawn past the parameters bounds
                    above_idxs = np.argwhere(samps>high_lim)
                    samps[above_idxs] = high_lim - (samps[above_idxs]-high_lim)
                    below_idxs = np.argwhere(samps<low_lim)
                    samps[below_idxs] = low_lim + (low_lim - samps[below_idxs])

                    obsdata[idx, :, pidx] = samps


        # for 'snr', use SNR-dependent measurement uncertainty following procedures from Fishbach, Holz, & Farr 2018 (2018ApJ...863L..41F)
        if method == "snr":

            # to use SNR-dependent uncertainty, we need to make sure that the correct parameters are supplied

            for idx, (obs,snr,Theta) in tqdm(enumerate(zip(self.observations, self.snrs, self.Thetas)), total=len(self.observations)):
                # convert to mchirp, q
                if set(['mchirp','q']).issubset(set(params)):
                    mc_true = obs[params.index('mchirp')]
                    q_true = obs[params.index('q')]
                elif set(['mtot','q']).issubset(set(params)):
                    mc_true = mtotq_to_mc(obs[params.index('mtot')], obs[params.index('q')])
                    q_true = obs[params.index('q')]
                elif set(['mtot','eta']).issubset(set(params)):
                    mc_true, q_true = mtoteta_to_mchirpq(obs[params].index('mtot'), obs[params].index('q'))
                else:
                    raise ValueError("You need to have a mass and mass ratio parameter to to SNR-weighted uncertainty!")

                z_true = obs[params.index('z')]
                mcdet_true = mc_true*(1+z_true)
                eta_true = q_true * (1+q_true)**(-2)
                Theta_true = Theta
                dL_true = cosmo.luminosity_distance(z_true).to(u.Gpc).value

                # apply Gaussian noise to SNR
                snr_obs = snr + np.random.normal(loc=0, scale=1)

                # get the snr-weighted sigma for the detector-frame chirp mass, and draw samples
                mc_sigma = _snrscale_sigmas['mchirp']*self.snr_thresh / snr_obs
                if observation_noise==True:
                    mcdet_obs = float(10**(np.log10(mcdet_true) + norm.rvs(loc=0, scale=mc_sigma, size=1)))
                else:
                    mcdet_obs = mcdet_true
                mcdet_samps = 10**(np.log10(mcdet_obs) + norm.rvs(loc=0, scale=mc_sigma, size=Nsamps))

                # get the snr-weighted sigma for eta, and draw samples
                eta_sigma = _snrscale_sigmas['eta']*self.snr_thresh / snr_obs
                if observation_noise==True:
                    eta_obs = float(truncnorm.rvs(a=(0-eta_true)/eta_sigma, b=(0.25-eta_true)/eta_sigma, loc=eta_true, scale=eta_sigma, size=1))
                else:
                    eta_obs = eta_true
                eta_samps = truncnorm.rvs(a=(0-eta_obs)/eta_sigma, b=(0.25-eta_obs)/eta_sigma, loc=eta_obs, scale=eta_sigma, size=Nsamps)

                # get samples for projection factor (use the true value as the observed value)
                # Note that our Theta is the projection factor (between 0 and 1), rather than the Theta from Finn & Chernoff 1993
                snr_opt = snr/Theta
                Theta_sigma = 0.3 / (1.0 + snr_opt/self.snr_thresh)
                Theta_samps = truncnorm.rvs(a=(0-Theta)/Theta_sigma, b=(1-Theta)/Theta_sigma, loc=Theta, scale=Theta_sigma, size=Nsamps)

                # get luminosity distance and redshift observed samples
                dL_samps = dL_true * (Theta_samps/Theta)
                z_samps = np.asarray([z_at_value(cosmo.luminosity_distance, d) for d in dL_samps*u.Gpc])

                # get source-frame chirp mass and other mass parameters
                mc_samps = mcdet_samps / (1+z_samps)
                q_samps = eta_to_q(eta_samps)
                m1_samps, m2_samps = mchirpq_to_m1m2(mc_samps,q_samps)
                mtot_samps = (m1_samps + m2_samps)

                for pidx, param in enumerate(params):
                    if param=='mchirp':
                        obsdata[idx, :, pidx] = mc_samps
                    elif param=='mtot':
                        obsdata[idx, :, pidx] = mtot_samps
                    elif param=='q':
                        obsdata[idx, :, pidx] = q_samps
                    elif param=='eta':
                        obsdata[idx, :, pidx] = eta_samps
                    elif param=='chieff':
                        chieff_true = obs[params.index('chieff')]
                        chieff_sigma = _snrscale_sigmas['chieff']*self.snr_thresh / snr_obs
                        if observation_noise==True:
                            chieff_obs = float(truncnorm.rvs(a=(-1-chieff_true)/chieff_sigma, b=(1-chieff_true)/chieff_sigma, loc=chieff_true, scale=chieff_sigma, size=1))
                        else:
                            chieff_obs = chieff_true
                        chieff_samps = truncnorm.rvs(a=(-1-chieff_obs)/chieff_sigma, b=(1-chieff_obs)/chieff_sigma, loc=chieff_obs, scale=chieff_sigma, size=Nsamps)
                        obsdata[idx, :, pidx] = chieff_samps
                    elif param=='z':
                        obsdata[idx, :, pidx] = z_samps

        return obsdata
    """