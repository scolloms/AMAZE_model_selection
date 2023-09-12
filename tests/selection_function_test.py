import sys
import numpy as np

from scipy.special import logsumexp

sys.path.append('../')
from populations.bbh_models import get_models
from populations import bbh_models as read_models
from populations.utils.flow import NFlow
from populations.Flowsclass_dev import FlowModel
from populations import gw_obs

def test_alpha(chnl, use_flows):
    #tests that alpha is calculated correctly
    file_path='/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5'
    params = ['mchirp','q', 'chieff', 'z']
    device='cpu'
    detectable=True
    popsynth_outputs = read_models.read_hdf5(file_path, chnl)
    sensitivity ='midhighlatelow_network'

    if chnl=='CE':
        #CE channel
        alpha = np.zeros((4,5))

        for chib_id in range(4):
            for alphaCE_id in range(5):
                samples = popsynth_outputs[(chib_id,alphaCE_id)]
                mock_samp = samples.sample(int(1e6), weights=(samples['weight']/len(samples)), replace=True)
                alpha[chib_id,alphaCE_id] = np.sum(mock_samp['pdet_'+sensitivity]) / len(mock_samp)

        FlowPop = FlowModel.from_samples(chnl, popsynth_outputs, params, device='cpu', sensitivity=sensitivity, detectable=detectable)
        alpha_flow =FlowPop.alpha

        #reshape alpha_model into array from dict
        alpha_flow = np.reshape(list(alpha_flow.values()),(4,5))

    else:
        #non-CE channel
        alpha = np.zeros((4))

        for chib_id in range(4):
                samples = popsynth_outputs[(chib_id)]
                mock_samp = samples.sample(int(1e6), weights=(samples['weight']/len(samples)), replace=True)
                alpha[chib_id] = np.sum(mock_samp['pdet_'+sensitivity]) / len(mock_samp)

        FlowPop = FlowModel.from_samples(chnl, popsynth_outputs, params, device='cpu', sensitivity=sensitivity, detectable=detectable)
        alpha_flow =FlowPop.alpha

        #reshape alpha_model into array from dict
        alpha_flow = np.reshape(list(alpha_flow.values()),(4))

    #calculate difference
    alpha_difference=alpha-alpha_flow
    percent_difference=(alpha-alpha_flow)/alpha
    print(alpha_difference)
    print(percent_difference)

    assert (percent_difference <= 0.001).all()

