#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#running inference with fixed regularisation added per event sample, using extra-long trained flows

model_path="/data/wiay/2297403c/models_reduced.hdf5"
gw_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/gwtc3_events/events_processed/events_processed"
flow_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/rns/Flows_210524_scatter/flow_models/flow_models/"

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
        --make-plots \
        --prior 'p_theta_jcb' \
        --regularisation_N '990903' \
        --Nsamps 1000 \
        --random-seed 86 \
