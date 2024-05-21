#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#KDE construction with _kde_bandwidth=0.005 in __init__.py

model_path="/data/wiay/2297403c/models_reduced.hdf5"
gw_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/gw_events"

/data/wiay/2297403c/conda_envs/amaze/bin/python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
	    --verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
        --params 'mchirp' 'q' 'chieff' 'z' \
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --make-plots \
        --prior 'p_theta' \
        --regularisation_N '990903' \
        --random-seed 8675309 \
