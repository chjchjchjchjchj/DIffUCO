import os
import argparse

parser = argparse.ArgumentParser()
name = "4-GPU_RB_small_MaxCl_no_clip"
name_MIS_small_124 = "4-GPU_RB_small_MIS_124"
name_MIS_small_125 = "4-GPU_RB_small_MIS_125"
name_3 = "4-GPU_RB_large_MaxCl_no_clip"
name2 = "1-GPU_TSP_20"
continue_train = "continue_training_MIS_123"
evaluate = "run_CE"
name_TPS = "1-GPU_TSP_100"
name_TPS_1 = "1-GPU_TSP_100_1"
name_MVC = "2-GPU_RB_200_MVC"
name_MVC_1 = "2-GPU_RB_200_MVC_1"
name_MVC_2 = "2-GPU_RB_200_MVC_2"
name_MIS = "2-GPU_RB_small_MIS"
name_MIS_1 = "2-GPU_RB_small_MIS_1"
name_MIS_large = "4-GPU_RB_large_MIS_0"
name_MaxCl_small = "2-GPU_RB_small_MaxCl_0"
name_MaxCl_small_1 = "2-GPU_RB_small_MaxCl_1"
continue_MVC = "2-GPU_RB_200_MVC_continue"
continue_MIS_0 = "2-GPU_RB_small_MIS_continue_0"
continue_MIS_1 = "2-GPU_RB_small_MIS_continue_1"
name_Gset_MaxCut = "2-GPU_Gset_MaxCut"
parser.add_argument('--script_name', default=[name_Gset_MaxCut], help='Define the script', nargs = "+")

args = parser.parse_args()

def push_data():
    rsa_path = "/system/user/sanokows/.ssh/id_rsa"

    for script_name in args.script_name:
        local_script_path_txt = os.getcwd() + f"/argparse/scripts/{script_name}.txt"
        remote_script_path = "~/code/DiffUCO" + f"/argparse/scripts/{script_name}.txt"

        # local_CE = os.getcwd() + f"/ConditionalExpectation.py"
        # CE_file_path = "~/code/meanfield_annealing" + f"/ConditionalExpectation.py"
        # transfer_CE_command = f"scp -i {rsa_path} {local_CE} it4i-sanokow@karolina.it4i.cz:{CE_file_path}"
        ### TODO transfer script to server
        transfer_command = f"scp -i {rsa_path} {local_script_path_txt} it4i-sanokow@karolina.it4i.cz:{remote_script_path}"
        run_script = f'ssh -i {rsa_path} it4i-sanokow@karolina.it4i.cz "sbatch {remote_script_path}"'

        # os.system(f"mv {local_script_path_txt} {local_script_path_sh}")
        os.system(f"dos2unix {local_script_path_txt}")
        os.system(f"cat {local_script_path_txt}")
        os.system(transfer_command)
        #os.system(transfer_CE_command)
        os.system(run_script)

    ### TODO make similar script that starts runs on the server
    ### 1. transfer .pbs script -> run .pbs script


def load_data():
    rsa_path = "/system/user/sanokows/.ssh/id_rsa"

    Checkpoint_path = "~/code/DiffUCO/Checkpoints"
    Download_folder_path = "/system/user/publicwork/sanokows/Downloads"

    ### TODO transfer script to server
    load_command = f"scp -r -i {rsa_path} it4i-sanokow@karolina.it4i.cz:{Checkpoint_path} {Download_folder_path}"

    os.system(load_command)

    ### TODO make similar script that starts runs on the server
    ### 1. transfer .pbs script -> run .pbs script


if(__name__ == "__main__"):
    push_data()