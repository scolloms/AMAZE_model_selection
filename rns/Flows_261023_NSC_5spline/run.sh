#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, NSC channel training
#training for 500 epochs and with 5 spline bins

model_path="/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5"
gw_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/gw_events"
flow_path="./flow_model/"

python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	--verbose \
        --channels 'NSC'\
        --epochs 500 \
        --spline-bins 5 \
        --train-flows \
        --use-flows \
        --device cpu \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --prior 'p_theta'
