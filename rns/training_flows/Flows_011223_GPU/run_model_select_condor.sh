#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#acctidentally run without this update: updated CE training so that training and val data is shuffle split and weights>fmin
#training for 700 epochs and with 5 spline bins

model_path="/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5"
gw_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/gw_events"
flow_path="./flow_model/"

python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	    --verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
        --epochs 1500 1000 1000 1000 1000 \
        --spline-bins 5 4 4 4 4 \
        --train-flows \
        --use-flows \
        --device cuda:0 \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --prior 'p_theta'
