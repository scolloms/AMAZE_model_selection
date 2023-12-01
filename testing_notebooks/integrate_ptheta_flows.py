import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys

sys.path.append('../')
from populations.bbh_models import get_models
import populations.bbh_models as read_models
from populations.utils.flow import NFlow
from populations.Flowsclass_dev import FlowModel
from populations import gw_obs

def f(mchirp, q,chieff,z):
    params= np.asarray([mchirp, q,chieff,z])
    params = np.reshape(params, (1,1,4))
    mapped_params = flow.map_obs(params)
    return np.exp(flow.flow.get_logprob(params, mapped_params, flow.mappings, np.asarray([[[0.,0.2]]])))

params = ['mchirp','q', 'chieff', 'z']
chi_b = [0.0,0.1,0.2,0.5]
alpha = [0.2,0.5,1.,2.,5.]
file_path='/data/wiay/2297403c/models_reduced.hdf5'
channels = ['CE']

model_names, flow = read_models.get_models(file_path, channels, params, use_flows=True, device='cpu', no_bins=[5])

flow_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/rns/flows_271023/flow_models/"


for chnl in tqdm(channels):
    flow[chnl].load_model(flow_path, chnl)
    #logprob args: param_grid, mapped_param_grid, flow['CE'].mappings, conds
    flow = flow[chnl]
    int =scipy.integrate.nquad(f, [[0.,100.],[0.,1.],[-1.,1.],[0.,10.]], args=None, opts={'epsrel': 0.1, 'epsabs': 0.1})
    print(chnl)
    print(int)