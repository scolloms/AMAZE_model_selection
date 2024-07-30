#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#includes /data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/gw_events/GW190521_074359.hdf5 but not GW190521

model_path="/data/wiay/2297403c/models_reduced.hdf5"
gw_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/gwtc3_events/events_processed/events_processed"

/data/wiay/2297403c/conda_envs/amaze/bin/python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
	    --verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
        --params 'mchirp' 'q' 'chieff' 'z' \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --prior 'p_theta_jcb' \
        --random-seed 12 \
        --regularisation_N '990903' \
        --Nsamps 1000 \
        --name 'seed12'
