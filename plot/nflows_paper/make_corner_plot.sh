flow_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/flow_models/mixedmodels_0924/"

python make_corner_plot.py \
    --flow-path ${flow_path} \
    --channel-label 'CHE' 'GC' 'NSC' 'SMT' \
    --hyperparam-idxs 3 2 \
    --conditional 0.5 1. \