import sys
import pandas as pd
import numpy as np

from scipy.special import logsumexp

sys.path.append('../')
from populations.bbh_models import get_models
from populations import bbh_models as read_models
from populations.utils.flow import NFlow
from populations.Flowsclass_dev import FlowModel
from populations import gw_obs

def get_lnlikelihood(new_likelikelihoods=False):
    #set posterior indxs here?
    params = ['mchirp','q', 'chieff', 'z']
    file_path='/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5'
    gw_path = '/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/gw_events'
    observations, obsdata, p_theta, events = gw_obs.generate_observations(params, gw_path, \
                                            3, 'test', None)
    
    #use dictionary of flows for different channels - only CE channel atm
    flows = {}
    lnlikelihood ={}
    channels = ['CE', 'CHE', 'GC', 'NSC', 'SMT']
    for chnl in channels:
        popsynth_outputs = read_models.read_hdf5(file_path, chnl)
        flows[chnl] = FlowModel.from_samples(chnl, popsynth_outputs, params, device='cpu')
        flows[chnl].load_model('/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/flow_models/', chnl)
        conditional_idxs =[3,1]
        lnlikelihood[chnl] = flows[chnl](obsdata,conditional_idxs)

    if new_likelikelihoods:
        lnl_df=pd.DataFrame.from_dict(lnlikelihood)
        lnl_df.to_csv('/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/tests/likelihoods.csv')
    return lnlikelihood

def get_llh_all_chnls(new_likelikelihoods, channels):

    likes_per_chnl = get_lnlikelihood(new_likelikelihoods)

    #set branching franctions to test
    betas_tmp = np.asarray([0.1,0.4,0.08,0.22])
    betas_tmp = np.append(betas_tmp, 1-np.sum(betas_tmp))

    # Likelihood
    lnprob = np.zeros(3)-np.inf

    # Detection effiency for this hypermodel
    alpha = 0

    # Iterate over channels in this submodel, return likelihood of population model
    for channel, beta in zip(channels, betas_tmp):
        lnprob = logsumexp([lnprob, np.log(beta) + likes_per_chnl[channel]])
        alpha += beta * alpha

    #TO CHANGE - use log likelihood throughout
    return logsumexp(lnprob-np.log(alpha))

#Create test class when multiple tests

def test_likelihood(chnl):
    lnlikelihoods = pd.read_csv('/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/tests/likelihoods.csv')
    check_lnls = pd.DataFrame.from_dict(get_lnlikelihood())[chnl]
    print(check_lnls-lnlikelihoods[chnl])
    #check why its this number
    assert (check_lnls - lnlikelihoods[chnl] <= 3.552714e-15).all()

#Next unit test - test that correct samples are read in for each channel

def test_alpha(chnl):
    #tests that alpha is calculated correctly
    file_path='/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5'
    params = ['mchirp','q', 'chieff', 'z']
    device='cpu'
    detectable=True

    if chnl=='CE':
        #CE channel
        popsynth_outputs = read_models.read_hdf5(file_path, chnl)
        sensitivity ='midhighlatelow_network'
        alpha = np.zeros((4,5))

        for chib_id in range(4):
            for alphaCE_id in range(5):
                samples = popsynth_outputs[(chib_id,alphaCE_id)]
                mock_samp = samples.sample(int(1e6), weights=(samples['weight']/len(samples)), replace=True)
                alpha[chib_id,alphaCE_id] = np.sum(mock_samp['pdet_'+sensitivity]) / len(mock_samp)

        PopModel = get_models(chnl, popsynth_outputs, params, device=device, sensitivity=sensitivity, detectable=detectable)
        alpha_model =PopModel.alpha

        #reshape alpha_model into same shape array not dict
        #find difference and assert not more than some error