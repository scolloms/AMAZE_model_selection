import unittest
import numpy as np
import sys

sys.path.append('../')
from populations.Flowsclass_dev import FlowModel
from populations.bbh_models import read_hdf5


class TestFlowModel(unittest.TestCase):

    def load_flow(self,chnl):
        file_path='/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5'
        params = ['mchirp','q', 'chieff', 'z']
        detectable=True
        popsynth_outputs = read_hdf5(file_path, chnl)
        sensitivity ='midhighlatelow_network'

        FlowPop = FlowModel.from_samples(chnl, popsynth_outputs, params, device='cpu', sensitivity=sensitivity, detectable=detectable)
        return FlowPop

    def test_network_loaded(self):
        FlowPop = self.load_flow('GC')
        FlowPop.flow

    
    def test_training(self):
        FlowPop = self.load_flow('GC')
        FlowPop.train(0.001,1,10,'test_flow','GC')

    def test_samples(self):
        pass

    def test_obs_mapping(self):
        #test map_obs returned in correct format for given mock samples
        mock_samples = np.array([[[40.,0.7,-0.5,1.],[0.2,0.1,0.0,1.9],[99.,0.1,-0.5,1.9]], [[40.,0.7,-0.5,1.],[0.2,0.1,0.0,1.9],[99.,0.1,-0.5,1.9]]])
        FlowPop = self.load_flow('GC')
        FlowPop.load_model(filepath='/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/flow_models/cosmo_weights/newmappings_2/',channel='GC')
        FlowPop.mappings[3] = 1.
        mapped_obs = FlowPop.map_obs(mock_samples)
        self.assertTrue(np.shape(mapped_obs)==np.shape(mock_samples))

        
    def test_weights(self):
        #test weights are calculated correctly for detectable and non-detectable cases
        pass

    def test_channel_label(self):
        #tests channel label is set correctly
        for chnl in ['CE','CHE', 'GC','NSC','SMT']:
            FlowPop = self.load_flow(chnl)
            self.assertEqual(FlowPop.channel_label, chnl)

    def test_mapping(self):
        #tests the upper bound for mchirp as set in the flowmodel param bounds is equal to the saved mappings
        for chnl in ['CE','CHE', 'GC','NSC','SMT']:
            FlowPop = self.load_flow(chnl)
            params = ['mchirp','q', 'chieff', 'z']
            file_path='/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5'
            popsynth_outputs = read_hdf5(file_path, chnl)
            _,_,mappings = FlowPop.map_samples(popsynth_outputs, params, 'testing_flow')
            self.assertEqual (mappings[1], FlowPop.param_bounds[0][1])

    def test_alphas(self):
        for chnl in ['CE','CHE', 'GC','NSC'
        ,'SMT']:
            #tests that alpha is calculated correctly
            file_path='/Users/stormcolloms/Documents/PhD/Project_work/OneChannel_Flows/models_reduced.hdf5'
            params = ['mchirp','q', 'chieff', 'z']
            detectable=True
            popsynth_outputs = read_hdf5(file_path, chnl)
            sensitivity ='midhighlatelow_network'

            if chnl=='CE':
                #CE channel
                alpha = np.zeros((4,5))

                for chib_id in range(4):
                    for alphaCE_id in range(5):
                        samples = popsynth_outputs[(chib_id,alphaCE_id)]
                        mock_samp = samples.sample(int(1e6), weights=(samples['weight']/len(samples)), replace=True)
                        alpha[chib_id,alphaCE_id] = np.sum(mock_samp['pdet_'+sensitivity]) / len(mock_samp)

                FlowPop = FlowModel.from_samples(chnl, popsynth_outputs, params, device='cpu', sensitivity=sensitivity, detectable=detectable)
                alpha_flow =FlowPop.alpha

                #reshape alpha_model into array from dict
                alpha_flow = np.reshape(list(alpha_flow.values()),(4,5))

            else:
                #non-CE channel
                alpha = np.zeros((4))

                for chib_id in range(4):
                        samples = popsynth_outputs[(chib_id)]
                        mock_samp = samples.sample(int(1e6), weights=(samples['weight']/len(samples)), replace=True)
                        alpha[chib_id] = np.sum(mock_samp['pdet_'+sensitivity]) / len(mock_samp)

                FlowPop = FlowModel.from_samples(chnl, popsynth_outputs, params, device='cpu', sensitivity=sensitivity, detectable=detectable)
                alpha_flow =FlowPop.alpha

                #reshape alpha_model into array from dict
                alpha_flow = np.reshape(list(alpha_flow.values()),(4))

            #calculate difference
            alpha_difference=alpha-alpha_flow
            percent_difference=(alpha-alpha_flow)/alpha
            print(alpha_difference)
            print(percent_difference)

            assert (percent_difference <= 0.001).all()

class TestNFlow(unittest.TestCase):
    def test_jacobian(self):
        #test that samples match samples fed in
        #test that jacobian is right datatype
        #test that the jacobian is calculated correctly for 1 test sample
        pass


if __name__ == '__main__':
    unittest.main()