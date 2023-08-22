#! /bin/bash
#runs model select with GPU specified in bach call, flows, and preset file paths, CE channel only

model_path="/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5"
gw_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/gw_events"
flow_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/flow_models/cosmo_weights/"

python model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
		--verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
        --use-flows True \
        --train-flows True \
        --device 'cpu' \
        --sensitivity 'midhighlatelow_network'
