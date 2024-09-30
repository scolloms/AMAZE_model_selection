
import argparse
from plot_functions import *
import glob

argp = argparse.ArgumentParser()
argp.add_argument("--flow-path", type=str, default=None, help="Directory from where to load flow models. Default=None.")
argp.add_argument("--discrete-result-path", type=str, default=None, help="Directory from where to load discrete result files. Default=None.")
argp.add_argument("--KDE-result-path", type=str, default=None, help="Directory from where to load discrete KDE result files. Default=None.")

argp.add_argument("--hyperparam-idxs", nargs="+", type=int, default=None, help="")
argp.add_argument("--channel-label", type=str, nargs="+", default="CE", help="")
argp.add_argument("--conditional", type=float,  nargs="+", help="")


args = argp.parse_args()
flow_dir = args.flow_path
discrete_result_path = args.discrete_result_path
discrete_result_path_KDE = args.KDE_result_path
channel_label = args.channel_label
hyperparam_idxs = args.hyperparam_idxs
conditional = np.array(args.conditional)


for channel in channel_label:
    make_pop_corner(channel, hyperparam_idxs, justplot=False, flow_dir=flow_dir, conditional=conditional)

discrete_result_files = glob.glob(f'{discrete_result_path}/*.hdf5')
try:
    KDE_result_files = glob.glob(f'{discrete_result_path_KDE}/*.hdf5')
    make_1D_result_discrete(discrete_result_files, second_files=KDE_result_files, labels = [' flow', ' KDE'], figure_name='DiscreteKDE')
except FileNotFoundError():
    make_1D_result_discrete(discrete_result_files, second_files=None, labels = [' flow', None], figure_name='Discrete')