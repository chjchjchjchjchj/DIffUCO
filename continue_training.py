import os
import argparse
from train import TrainMeanField


parser = argparse.ArgumentParser()
parser.add_argument('--wandb_id', default = "",type = str, help='Switch ray into local mode for debugging')
parser.add_argument('--GPUs', default=["0"], type = str, help='Define Nb', nargs = "+")
parser.add_argument('--memory', default=0.92, type = float, help="GPU memory")
parser.add_argument('--continue_dataset', default="KS_3_1000", type = str)
parser.add_argument('--energy_A', default=1., type = float)
parser.add_argument('--energy_B', default=2., type = float)
parser.add_argument('--batch_size', default=20, type = int)
parser.add_argument('--add_epoch', default=2000, type = int)
parser.add_argument('--N_basis_states', default=2, type = int)
# parser.add_argument('--n_diffusion_steps', default=6, type = int)

### TODO add gradient clipping?
args = parser.parse_args()

### TODO add MaxCut

def meanfield_run():
    devices = args.GPUs

    device_str = ""
    for idx, device in enumerate(devices):
        if (idx != len(devices) - 1):
            device_str += str(devices[idx]) + ","
        else:
            device_str += str(devices[idx])

    print(device_str)

    if (len(args.GPUs) > 1):
        device_str = ""
        for idx, device in enumerate(devices):
            if (idx != len(devices) - 1):
                device_str += str(devices[idx]) + ","
            else:
                device_str += str(devices[idx])

        print(device_str, type(device_str))
    else:
        device_str = str(args.GPUs[0])

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str


    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.memory)
    #os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    config = {"jit": True}
    continue_dataset = args.continue_dataset
    train = TrainMeanField(config, load_wandb_id=args.wandb_id, continue_dataset=continue_dataset, continue_energy_A=args.energy_A, continue_energy_B=args.energy_B, continue_batch_size=args.batch_size, add_epoch=args.add_epoch, continue_N_basis_states=args.N_basis_states)

    train.train()

if(__name__ == "__main__"):
    meanfield_run()