import sys
sys.path.append("..")

from DatasetCreator.loadGraphDatasets import get_dataset_generator

import argparse

RB_datasets = ["RB_iid_200", "RB_iid_100", "RB_iid_small", "RB_iid_large", "RB_iid_giant", "RB_iid_huge", "RB_iid_dummy"]
BA_datasets = ["BA_small", "BA_large", "BA_huge", "BA_giant", "BA_dummy"]
Gset = ["Gset"]
KS_datasets = ["KS_2", "KS_3", "KS_4", "KS_5", "KS_one_3", "KS_d_re_5000"]
# dataset_choices =  RB_datasets + BA_datasets  + Gset
dataset_choices =  RB_datasets + BA_datasets + Gset + KS_datasets

parser = argparse.ArgumentParser()

parser.add_argument('--licence_path', default="/home/chenhaojun/", type = str, help='licence base path')
parser.add_argument('--seed', default=[123], type = int, help='Define dataset seed', nargs = "+")
parser.add_argument('--parent', default=False, type = bool, help='use parent directory or not')
parser.add_argument('--save', default=False, type = bool, help='save the entire dataset in a pickle file or not')
parser.add_argument('--gurobi_solve', default=True, type = bool, help='whether to solve instances with gurobi or not')
parser.add_argument('--pulp_solve', default=False, type = bool, help='whether to solve instances with gulp or not')
parser.add_argument('--datasets_name', default=['KS_3'], choices = dataset_choices, help='Define the dataset', nargs="+")
parser.add_argument('--diff_ps', default=False, type = bool, help='')
parser.add_argument('--problems', default=['MaxCut'], choices = ["MIS", "MVC", "MaxCl", "MaxCut", "MDS", "TSP"], help='Define the CO problem', nargs="+")
parser.add_argument('--modes', default=[ "test", "train", "val"], type = str, help='Define dataset split', nargs = "+")
# parser.add_argument('--modes', default=["test"], type = str, help='Define dataset split', nargs = "+")
parser.add_argument('--time_limits', default=["inf", "0.1", "1."], type = str, help='Gurobi Time Limit for each [mode]', nargs = "+")
# parser.add_argument('--time_limits', default=["inf"], type = str, help='Gurobi Time Limit for each [mode]', nargs = "+")
parser.add_argument('--datasets_path', default="/home/haojun/DIffUCO/only_one_Data_for_solver.pkl", type = str, help='datasets path')
parser.add_argument('--uniform_generate_data', default=False, type = bool, help='uniformly generate data')
parser.add_argument('--dim', default=3, type = int, help='generate ndim graphs')
parser.add_argument('--num_samples', default=1000, type = int, help='the number of nodes of generated graphs')
parser.add_argument('--st_idx', default=0, type = int)
parser.add_argument('--ed_idx', default=6000, type = int)
parser.add_argument('--GPUs', default=["6"], type = str, help='Define Nb', nargs = "+")
parser.add_argument('--thread_fraction', default=0.5, type = float)
#parser.add_argument('--n_graphs', default=[100, 10, 10], type = int, help='Number of graphs for each [mode]', nargs = "+")
args = parser.parse_args()


def create_dataset(config: dict, modes: list, time_limits: list):
	"""
	Create a dataset with the specified configuration

	:param config: config dictionary specifying the dataset that should be generated
	:param modes: ["train", "val", "test"] modes for which the dataset should be generated
	:param sizes: [int, int, int] number of graphs for each mode
	:param time_limits: [float, float, float] time limit for each mode
	"""
	if len(modes) != len(time_limits):
		raise ValueError("Length of modes, sizes and time_limits should be the same")

	for mode,  time_limit in zip(modes, time_limits):
		config["mode"] = mode
		#config["n_graphs"] = size
		config["time_limit"] = float(time_limit)
		dataset_generator = get_dataset_generator(config)
		dataset_generator.generate_dataset()

devices = args.GPUs

device_str = ""
for idx, device in enumerate(devices):
    if (idx != len(devices) - 1):
        device_str += str(devices[idx]) + ","
    else:
        device_str += str(devices[idx])

print(device_str)

if(len(args.GPUs) > 1):
    device_str = ""
    for idx, device in enumerate(devices):
        if (idx != len(devices) - 1):
            device_str += str(devices[idx]) + ","
        else:
            device_str += str(devices[idx])

    print(device_str, type(device_str))
else:
    device_str = str(args.GPUs[0])

import os
os.environ['CUDA_VISIBLE_DEVICES'] = device_str
print(f"device_str={device_str}")


if __name__ == "__main__":
	# print(f"args={args}")
	# sys.exit()
	# from argparse import Namespace
	# args=Namespace(licence_path='/system/user/sanokows/', seed=[123], parent=False, save=False, gurobi_solve=True, datasets=['RB_iid_100'], diff_ps=False, problems=['MIS'], modes=['test', 'train', 'val'], time_limits=['inf', '0.1', '1.'])
	

	for dataset in args.datasets_name:
		for problem in args.problems:
			for seed in args.seed:
				base_config = {
					"licence_base_path": args.licence_path,
					"seed": seed,
					"parent": args.parent,
					"save": args.save,
					"gurobi_solve": args.gurobi_solve,
					"pulp_solve": args.pulp_solve,
					"diff_ps": args.diff_ps,
					"dataset_name": dataset,
					"problem": problem,
					"time_limit": None,
					"n_graphs": None,
					"datasets_path": args.datasets_path,
					'uniform_generate_data': args.uniform_generate_data,
					'dim': args.dim,
					'num_samples': args.num_samples,
                    'thread_fraction': args.thread_fraction,
                    'st_idx': args.st_idx,
                    'ed_idx': args.ed_idx,
				}
				print(f"base_config={base_config}")
				create_dataset(base_config, args.modes, args.time_limits)