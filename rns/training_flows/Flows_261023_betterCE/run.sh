#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#pull the merge from no0weights branch: updated CE training so that training and val data is shuffle split and weights>fmin
#training for 700 epochs and with 5 spline bins

model_path="/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5"
gw_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/gw_events"
flow_path="./flow_model/"

python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	--verbose \
        --channels 'CE'\
        --epochs 700 \
        --spline-bins 5 \
        --train-flows \
        --use-flows \
        --device cpu \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --prior 'p_theta'
