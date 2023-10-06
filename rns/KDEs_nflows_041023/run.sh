#! /bin/bash
#runs model_select with KDE likelihoods, and posterior uncertainty on events for all channels with 100 posterior samples. includes prior on posterior samples from the GWTC data release
#for paths on local directory
#04/10/23 no CE channel to compare with flow run ()

model_path="/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5"
gw_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/gw_events"
flow_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/flow_models/cosmo_weights/"

python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
	--verbose \
        --param 'mchirp' 'q' 'chieff' 'z' \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --uncertainty 'posteriors' \
        --prior 'p_theta' \
        --channels 'CHE' 'GC' 'NSC' 'SMT'

