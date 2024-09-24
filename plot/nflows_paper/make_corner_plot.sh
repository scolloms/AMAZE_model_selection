flow_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/rns/Flows_040924_trainlogalpha/flow_models/"

python make_corner_plot.py \
    --flow-path ${flow_path} \
    --channel-label 'CE' \
    --hyperparam-idxs 1 2 \
    --conditional 0.1 1. \