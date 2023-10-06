#! /bin/bash
#runs model select with GPU on condor, with flows, and preset file paths, CE channel only

model_path="/data/wiay/2297403c/models_reduced.hdf5"
gw_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/gw_events"
flow_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/flow_models/cosmo_weights/newmappings_2/"

/data/wiay/2297403c/conda_envs/amaze/bin/python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
        --flow-model-filename ${flow_path} \
	--verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
        --use-flows \
        --device cuda:0 \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --prior 'p_theta'
