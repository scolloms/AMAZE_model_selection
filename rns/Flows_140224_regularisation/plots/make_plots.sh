#! /bin/bash
mkdir llh_ratio
mkdir flow_KDE_samples

/Users/stormcolloms/opt/anaconda3/envs/amaze/bin/python /Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/plot/makeplots_forrun.py \
        --file-path '/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/rns/Flows_140224_regularisation' \
        --flow-path "/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/rns/Flows_150124_extralong" \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
        --flow-bins 5 4 4 5 4 \
        --use-reg