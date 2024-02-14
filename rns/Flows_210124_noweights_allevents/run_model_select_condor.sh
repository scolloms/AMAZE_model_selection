#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#training flows with all weights=1. and then inference with GW150914, GW151226, GW170608, GW190519, GW190412

model_path="/data/wiay/2297403c/models_reduced.hdf5"
gw_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/gw_events"
flow_path='/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/rns/Flows_120123_noweights_5events/flow_models/'

/data/wiay/2297403c/conda_envs/amaze/bin/python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	--verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
        --epochs 1500 1000 1000 1000 1000 \
        --spline-bins 5 4 4 5 4 \
        --use-flows \
        --use-unityweights \
        --device cuda:0 \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --prior 'p_theta' \
        --random-seed 8675309
