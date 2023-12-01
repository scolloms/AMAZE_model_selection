#! /bin/bash
#runs model_select with KDE likelihoods, and posterior uncertainty on events for all channels with 100 posterior samples. includes prior on posterior samples from the GWTC data release
#for paths on local directory
#02/11/23 using samples from flow model (flows_271023/flow_models) saved to file to construct the KDEs for inference

model_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/testing_notebooks/flow_samples.hdf5"
gw_path="/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/gw_events"

python ../../model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
	--verbose \
        --param 'mchirp' 'q' 'chieff' 'z' \
        --save-samples \
        --make-plots \
        --uncertainty 'posteriors' \
        --prior 'p_theta' \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT'