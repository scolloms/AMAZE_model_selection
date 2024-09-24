#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#running inference with CE flow from trainlogalpha flows, and other flows from scatter trained flows.

model_path="/data/wiay/2297403c/models_reduced.hdf5"
gw_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/notebooks/gwtc3_events_oldpriorrange/events"
flow_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/flow_models/mixedmodels_0924/"

/data/wiay/2297403c/conda_envs/amaze/bin/python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	--verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
        --epochs 15000 10000 10000 10000 10000 \
        --spline-bins 5 4 4 5 4 \
        --use-flows \
        --device cuda:0 \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --prior 'p_theta_jcb' \
        --regularisation_N '990903' \
        --Nsamps 1000 \
        --random-seed 12 \
        --name seed12
