#!/bin/bash

gw_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/notebooks/gwtc3_events_oldpriorrange/events"
flow_path="/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/flow_models/mixedmodels_0924/"

/data/wiay/2297403c/conda_envs/amaze/bin/python calculate_KL.py \
    --flow-path ${flow_path} \
    --channel-label 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
    --gw-path ${gw_path} \
    --no-samps 100