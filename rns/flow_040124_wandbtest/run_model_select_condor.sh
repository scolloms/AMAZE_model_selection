#! /bin/bash
#testing wandb on local gpu, varying number of transforms, bins, and neurons


model_path="/data/wiay/2297403c/models_reduced.hdf5"
gw_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/gw_events"
flow_path="./flowmodel/"

/data/wiay/2297403c/conda_envs/amaze/bin/python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	--verbose \
        --channels 'SMT' 'GC' 'CHE'\
        --epochs 10000 10000 10000\
        --spline-bins 4 4 4\
        --train-flows \
        --device cuda:0 \
        --use-flows \
        --use-wandb \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --prior 'p_theta' \
        --random-seed 8675309
