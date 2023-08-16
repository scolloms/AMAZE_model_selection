#! /bin/bash

python model_select --file-path '../../models_reduced.hdf5' --model0 'gwobs' --gw-path '/gw_events' --flow-model-filename '/flow_models/' --verbose --channels 'CE' --use-flows True --train-flows True --device 'coda NVIDIA GeForce'