#! /bin/bash
#runs model_select with GPU specified in bash call, flows, and preset file paths. all channels
#for paths on local directory

model_path="/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5"
gw_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/gw_events"
flow_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/flow_models/cosmo_weights/"

python model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
        --use-flows \
		--verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
        --device 'cpu' \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \