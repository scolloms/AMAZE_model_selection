
import argparse
from plot_functions import *

argp = argparse.ArgumentParser()
argp.add_argument("--flow-path", type=str, default=None, help="Directory from where to load flow models. Default=None.")
argp.add_argument("--hyperparam-idxs", nargs="+", type=int, default=None, help="")
argp.add_argument("--channel-label", type=str, default="CE", help="")
argp.add_argument("--conditional", type=float,  nargs="+", help="")


args = argp.parse_args()
flow_dir = args.flow_path
channel_label = args.channel_label
hyperparam_idxs = np.array(args.hyperparam_idxs)
conditional = np.array(args.conditional)

print(channel_label)

make_pop_corner(channel_label, hyperparam_idxs, justplot=False, flow_dir=flow_dir, conditional=conditional)