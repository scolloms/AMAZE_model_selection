#! /bin/bash
#runs model_select with KDE likelihoods, and delta uncertainty on events for all channels
#for paths on local directory
#22/11/23 running KDEs CE channel only with posteriors and ptheta prior ro check lnprob values in output

model_path="/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5"
gw_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/gw_events"
flow_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/flow_models/cosmo_weights/"

python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
	--verbose \
	--channels 'CE' \
        --param 'mchirp' 'q' 'chieff' 'z' \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --uncertainty 'posteriors' \
        --prior 'p_theta' \
        --name 'burnin'

