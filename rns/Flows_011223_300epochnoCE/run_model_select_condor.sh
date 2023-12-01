#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#training flow models on GPU and then inference with all events

model_path="/data/wiay/2297403c/models_reduced.hdf5"
gw_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/gw_events"
flow_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/rns/Flows_011223_GPU/flow_model/"

/data/wiay/2297403c/conda_envs/amaze/bin/python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	--verbose \
        --channels 'CHE' 'GC' 'NSC' 'SMT' \
        --epochs 300 300 300 300 \
        --spline-bins 4 4 4 4 \
        --train-flows \
        --use-flows \
        --device cuda:0 \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --prior 'p_theta'
