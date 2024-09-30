import numpy as np
import pandas as pd
import os
import operator
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import corner
import matplotlib.lines as mlines
from scipy.stats import dirichlet

from matplotlib import gridspec

import sys
sys.path.append('../../')
from populations.bbh_models import *
from populations.Flowsclass_dev import FlowModel

colors = sns.color_palette("colorblind", n_colors=10)
cp = [colors[0], colors[2], colors[4], colors[1], colors[3], colors[6], colors[9], colors[5], colors[8]]
_basepath, _ = os.path.split(os.path.realpath(__file__))
plt.style.use(_basepath+"/mpl.sty")

_params = ['mchirp','q', 'chieff', 'z']
_channels = ['CE','CHE','GC','NSC','SMT']

_param_label = ['Chirp Mass /$M_{\odot}$','Mass Ratio', 'Effective Spin', 'Redshift']
_param_bounds = {"mchirp": (0,70), "q": (0.25,1), "chieff": (-0.5,1), "z": (0,1.25)}
_param_ticks = {"mchirp": [0,10,20,30,40,50,60,70], "q": [0.25,0.5,0.75,1], "chieff": [-0.5,0,0.5,1], "z": [0,0.25,0.5,0.75,1.0,1.25]}
_pdf_bounds = {"mchirp": (0,0.075), "q": (0,32), "chieff": (0,17), "z": (0,4)}
_pdf_ticks = {"mchirp": [0.0,0.025,0.050,0.075], "q": [0,10,20,30], "chieff": [0,4,8,12,16], "z": (0,1,2,3,4)}
_labels_dict = {"mchirp": r"$\mathcal{M}_{\rm c}$ [$M_{\odot}$]", "q": r"$q$", \
"chieff": r"$\chi_{\rm eff}$", "z": r"$z$", "chi00": r"$\chi_\mathrm{b}=0.0$", \
"chi01": r"$\chi_\mathrm{b}=0.1$", "chi02": r"$\chi_\mathrm{b}=0.2$", \
"chi05": r"$\chi_\mathrm{b}=0.5$", "alpha02": r"$\alpha_\mathrm{CE}=0.2$", \
"alpha05": r"$\alpha_\mathrm{CE}=0.5$", "alpha10": r"$\alpha_\mathrm{CE}=1.0$", \
"alpha20": r"$\alpha_\mathrm{CE}=2.0$", "alpha50": r"$\alpha_\mathrm{CE}=5.0$", \
"CE": r"$\texttt{CE}$", "CHE": r"$\texttt{CHE}$", "GC": r"$\texttt{GC}$", \
"NSC": r"$\texttt{NSC}$", "SMT": r"$\texttt{SMT}$"}
_Nsamps = 100000

_models_path ='/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5'

pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.

jour_sizes = {"AAS": {"onecol": 246.*pt, "twocol": 513.*pt},
              # Add more journals below. Can add more properties to each journal
             }

figure_width = jour_sizes["AAS"]["twocol"]


_base_corner_kwargs = dict(
    bins=64,
    smooth=0.9,
    #quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    plot_density=True,
    plot_datapoints=True,
    fill_contours=True,
    show_titles=False,
    hist_kwargs=dict(density=True),
    labels=_param_label,
)

def sample_pop_corner(flow_dir, channel_label, conditional, KDE_hyperparam_idxs=None):
    #models_path ='/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/testing_notebooks/flow_samples.hdf5'
    
    popsynth_outputs = read_hdf5(_models_path, channel_label) # read all data from hdf5 file

    weighted_flow = FlowModel(channel_label, popsynth_outputs, _params, no_bins=5)
    model_names, KDE_models = get_models(_models_path, [channel_label], _params, use_flows=False, spin_distr=None, normalize=False, detectable=False)

    hyperparams = list(set([x.split('/', 1)[1] for x in model_names]))
    Nhyper = np.max([len(x.split('/')) for x in hyperparams])

    # construct dict that relates submodels to their index number
    submodels_dict = {} #dummy index dict keys:0,1,2,3, items: particular models
    ctr=0 #associates with either chi_b or alpha (0 or 1)
    while ctr < Nhyper:
        submodels_dict[ctr] = {}
        hyper_set = sorted(list(set([x.split('/')[ctr] for x in hyperparams])))
        for idx, model in enumerate(hyper_set): #idx associates with 0,1,2,3,(4) keys
            submodels_dict[ctr][idx] = model
        ctr += 1

    weighted_flow.load_model(flow_dir, channel_label)

    #sample flow

    print('sampling flow...')
    if channel_label=='CE':
        flow_samples_stack = weighted_flow.flow.sample(np.array([conditional[0], np.log(conditional[1])]),_Nsamps)
    else:
        flow_samples_stack = weighted_flow.flow.sample(np.array([conditional[0]]),_Nsamps)
    flow_samples_stack[:,0] = weighted_flow.expistic(flow_samples_stack[:,0], weighted_flow.mappings[0], weighted_flow.mappings[1])
    flow_samples_stack[:,1] = weighted_flow.expistic(flow_samples_stack[:,1], weighted_flow.mappings[2])
    flow_samples_stack[:,2] = np.tanh(flow_samples_stack[:,2])
    flow_samples_stack[:,3] = weighted_flow.expistic(flow_samples_stack[:,3], weighted_flow.mappings[4], weighted_flow.mappings[5])

    print('sampling KDE...')
    if type(KDE_hyperparam_idxs) == type(None):
        pass
    else:
        if channel_label=='CE':
            kde_samples = KDE_models[channel_label][submodels_dict[0][KDE_hyperparam_idxs[0]]][submodels_dict[1][KDE_hyperparam_idxs[1]]].sample(_Nsamps)
        else:
            kde_samples = KDE_models[channel_label][submodels_dict[0][KDE_hyperparam_idxs[0]]].sample(_Nsamps)
        np.save(f"{_basepath}/data/{channel_label}_KDEs_cornersample.npy", kde_samples)
        
    print('saving samples...')
    np.save(f"{_basepath}/data/{channel_label}_flows_cornersample.npy", flow_samples_stack.numpy())
    print('samples saved')


def make_pop_corner(channel_label, hyperparam_idxs, justplot=True, flow_dir=None, conditional=None):

    if justplot==False:
        sample_pop_corner( flow_dir, channel_label, conditional, KDE_hyperparam_idxs=hyperparam_idxs)
    if type(hyperparam_idxs) == type(None):
        pass
    else:
        kde_samples = np.load(f"{_basepath}/data/{channel_label}_KDEs_cornersample.npy")
    flow_samples= np.load(f"{_basepath}/data/{channel_label}_flows_cornersample.npy")

    popsynth_outputs = read_hdf5(_models_path, channel_label) # read all data from hdf5 file
    models_dict = dict.fromkeys(popsynth_outputs.keys())
    weights_dict = dict.fromkeys(popsynth_outputs.keys())

    for key in popsynth_outputs.keys():
        models_dict[key] = popsynth_outputs[key][_params]
        weights_dict[key]= popsynth_outputs[key]['weight']

    colors=['C1', 'purple', 'royalblue']
    labels=['Underlying Model', 'KDE', 'Normalising Flow']

    model_kwargs = deepcopy(_base_corner_kwargs)
    model_kwargs["color"] = colors[0]
    model_kwargs["hist_kwargs"]["color"] = colors[0]
    corner_kwargs_kde = deepcopy(_base_corner_kwargs)
    corner_kwargs_kde["color"] = colors[1]
    corner_kwargs_kde["hist_kwargs"]["color"] = colors[1]
    corner_kwargs_flow = deepcopy(_base_corner_kwargs)
    corner_kwargs_flow["color"] = colors[2]
    corner_kwargs_flow["hist_kwargs"]["color"] = colors[2]

    plt.rcParams['figure.figsize'] = [figure_width, figure_width]
    if type(hyperparam_idxs) == type(None):
        fig=corner.corner(flow_samples, **corner_kwargs_flow)
    else:
        if channel_label == 'CE':
            fig =corner.corner(models_dict[tuple(hyperparam_idxs)],  weights=weights_dict[tuple(hyperparam_idxs)], **model_kwargs)
        else:
            fig =corner.corner(models_dict[hyperparam_idxs[0]],  weights=weights_dict[hyperparam_idxs[0]], **model_kwargs)
        corner.corner(kde_samples, fig=fig, **corner_kwargs_kde)
        corner.corner(flow_samples, fig=fig, **corner_kwargs_flow)
    plt.legend(
            handles=[
                mlines.Line2D([], [], color=colors[i], label=labels[i])
                for i in range(3)
            ],
            frameon=False,
            bbox_to_anchor=(1, 4), loc="upper right"
        )
    plt.savefig(f"{_basepath}/pdfs/{channel_label}_flowKDEmodel_corner_chib{hyperparam_idxs[0]}.pdf")

def make_1D_result_discrete(filenames, second_files=None, labels = [None,None], figure_name='Discrete'):
    channels = _channels
    submodels_dict= {0: {0: 'chi00', 1: 'chi01', 2: 'chi02', 3: 'chi05'}, \
    1: {0: 'alpha02', 1: 'alpha05', 2: 'alpha10', 3: 'alpha20', 4: 'alpha50'}}
    channel_label =[r'$\beta_{\mathrm{CE}}$',r'$\beta_{\mathrm{CHE}}$',r'$\beta_{\mathrm{GC}}$',r'$\beta_{\mathrm{NSC}}$',r'$\beta_{\mathrm{SMT}}$']

    Nhyper =2
    _concentration = np.ones(len(channels))
    beta_p0 =  dirichlet.rvs(_concentration, size=100000)

    samples_allchains = np.array([])
    samples_allchains_comp = np.array([])
    fig, ax_margs = plt.subplots(2,5)
    fig.tight_layout(h_pad=3)

    #loop over astrophysical parameters
    for hyper_idx in [0, 1]:

        #add together samples from multiple files
        for i, filename in enumerate(filenames):
            
            #skip over file if it doesn't exist
            try:
                result = h5py.File(filename, 'r')
            except:
                continue
            samples_file = np.hstack([result['model_selection']['samples']['block1_values'], result['model_selection']['samples']['block0_values']])
            samples_allchains = np.append(samples_allchains, samples_file)

            if second_files:
                try:
                    comp_file = h5py.File(second_files[i], 'r')
                except:
                    continue
                samples_file_comp = np.hstack([comp_file['model_selection']['samples']['block1_values'], comp_file['model_selection']['samples']['block0_values']])
                samples_allchains_comp = np.append(samples_allchains_comp, samples_file_comp)

        samples_allchains = np.reshape(samples_allchains, (-1, Nhyper+len(channels)))
        samples_allchains_comp = np.reshape(samples_allchains_comp, (-1, Nhyper+len(channels)))

        #loop over population models and plot histograms
        for midx, model in submodels_dict[hyper_idx].items():
            smdl_locs = np.argwhere(samples_allchains[:,hyper_idx]==midx).flatten()
            if second_files:
                comp_smdl_locs = np.argwhere(samples_allchains_comp[:,hyper_idx]==midx).flatten()

            for cidx, channel in enumerate(channels):
                factor = 50/(len(samples_allchains[:, cidx+Nhyper]))
                h, bins, _ = ax_margs[hyper_idx,cidx].hist(samples_allchains[smdl_locs, cidx+Nhyper], \
                    histtype='step', color=cp[midx], bins=50, ls='-', lw=1.5, \
                    label=_labels_dict[model]+labels[0],\
                    weights=factor*np.ones_like(samples_allchains[smdl_locs, cidx+Nhyper]))
                if second_files:
                    factor_comp = 50/(len(samples_allchains_comp[:, cidx+Nhyper]))
                    h, bins, _ = ax_margs[hyper_idx,cidx].hist(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper], \
                        histtype='stepfilled', color=cp[midx], bins=50, \
                        alpha=0.3, label=_labels_dict[model]+labels[1],\
                        weights=factor_comp*np.ones_like(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper]))
                    h, bins, _ = ax_margs[hyper_idx,cidx].hist(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper], \
                        histtype='step', color=cp[midx], bins=50, \
                        alpha=0.7, weights=factor_comp*np.ones_like(samples_allchains_comp[comp_smdl_locs, cidx+Nhyper]))

        # format plot
        for cidx, (channel, ax_marg) in enumerate(zip(channels, ax_margs.T)):
            #median branching fractions
            lower_5 = np.percentile(samples_allchains[:, cidx+Nhyper], 5)
            upper_95 = np.percentile(samples_allchains[:, cidx+Nhyper], 95)
            median = np.percentile(samples_allchains[:, cidx+Nhyper], 50)

            ax_marg[hyper_idx].vlines([lower_5, median, upper_95], 0,20, color='black', alpha=0.5, lw=0.5)

            #plot prior
            h, bins, _ = ax_marg[hyper_idx].hist(beta_p0[:,cidx], \
                    histtype='step', color='grey', bins=20, alpha=0.7, density=True)
            #plot total BF
            h, bins, _ = ax_marg[hyper_idx].hist(samples_allchains[:, cidx+Nhyper], \
                    histtype='step', color='black', bins=50, ls='--', lw=1.0, \
                    alpha=0.7, density=True)

            ax_marg[1].set_xlabel(channel_label[cidx], fontsize=15)
            ax_marg[hyper_idx].set_yscale('log')

            ax_marg[hyper_idx].set_xlim(0,1)
            ax_marg[hyper_idx].set_ylim(int(1e-3),20)
            if cidx == 0:
                ax_marg[hyper_idx].set_ylabel(r"p($\beta$)", fontsize=15)
            else:
                ax_marg[hyper_idx].tick_params(labelleft=False)
        # legend
        if hyper_idx == 0:
            ax_margs[0,0].legend(loc='lower left', bbox_to_anchor=(0.5, 1.02), ncol=4, prop={'size':10})
        if hyper_idx ==1:
            ax_margs[1,0].legend(loc='lower left', bbox_to_anchor=(-1.0, 1.02), ncol=5, prop={'size':10})
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{_basepath}/pdfs/{figure_name}_flowKDE_infresults.pdf")
        