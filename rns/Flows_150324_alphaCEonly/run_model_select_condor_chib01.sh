#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#running inference with only CE channel, inference only on alphaCE continuously. this uses the chib=0.1 models for training.

model_path="/data/wiay/2297403c/models_reduced.hdf5"
gw_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/gw_events"
flow_path="./flow_models/"

/data/wiay/2297403c/conda_envs/amaze/bin/python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	--verbose \
        --channels 'CE' \
        --epochs 10000 \
        --spline-bins 5 \
        --use-flows \
        --device cuda:0 \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --prior 'p_theta' \
        --random-seed 8675309 \
        --name chib01
