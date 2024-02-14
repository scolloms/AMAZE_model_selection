#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#training flow models for extra long no. of epochs and then inference with GW150914, GW151226, GW170608, GW190412

model_path="/data/wiay/2297403c/models_reduced.hdf5"
gw_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/gw_events"
flow_path="./flow_models/"

/data/wiay/2297403c/conda_envs/amaze/bin/python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	--verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
        --epochs 100000 10000 10000 10000 10000 \
        --params 'mchirp' 'q' 'z' \
        --spline-bins 5 4 4 5 4 \
        --use-flows \
        --device cuda:0 \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --prior 'p_theta' \
        --random-seed 8675309 \
        --name 'all_chnls'
