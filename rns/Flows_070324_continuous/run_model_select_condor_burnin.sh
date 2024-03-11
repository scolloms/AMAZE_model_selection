#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#running inference with continuous flows, this time not flooring samples before saving

model_path="/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5"
gw_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/gw_events"
flow_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/rns/Flows_150124_extralong/flow_models/"

/Users/stormcolloms/opt/anaconda3/envs/amaze/bin/python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	--verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
        --epochs 15000 10000 10000 10000 10000 \
        --spline-bins 5 4 4 5 4 \
        --use-flows \
        --device cpu \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --prior 'p_theta' \
        --random-seed 8675309 \
        --name burnintest \
