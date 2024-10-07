from plot_functions import *
import glob

"""result_path = '/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/rns/Flows_040924_trainlogalpha'
result_files = glob.glob(f'{result_path}/*.hdf5')

make_1D_result_continuous(result_files)"""

det_result_files = ['/Users/stormcolloms/Documents/PhD/Project_work/AMAZE_model_selection/plot/nflows_paper/data/test_detectable_betas.hdf5']
make_1D_result_continuous(det_result_files, detectable=True, figure_name='det_cont')