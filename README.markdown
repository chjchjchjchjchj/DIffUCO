# Official DiffUCO Repository

## Installing the environment

To install the environment fist run
```
conda env create -f environment.yml
```

Some packages will be installed but the installation of jax will run into an error.
Therefore, continue isntalling all missing packages by following the instructions below:

```
conda activate rayjay_clone
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tqdm jraph matplotlib tqdm optax
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install flax==0.8.1 igraph unipath wandb==0.15.0
```

For the creation of the TSP dataset `pyconcorde` has to be installed.
For that follow the instructions on:
https://github.com/jvkersch/pyconcorde


## Getting started
- get started by creating a dataset with the DatasetCreator
```
cd /DatasetCreator
```
and run for example
```setup
python prepare_datasets.py --dataset RB_iid_100 --problem MIS
```

For more details read
```
DatasetCreator/README.md
```


## Running experiments

To run an experiment on the created dataset (see above) do the following:
```
python argparse_ray_main.py --lrs 0.002 --GPUs 0 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_100 --EnergyFunction MIS --N_anneal 2000 
--n_diffusion_steps 3 --batch_size 20 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123 
```

### parameter explanation
`--IsingMode` Dataset to train on. In this case the `RB_iid_100`dataset \
`--EnergyFunction` CO problem to train on. In this case the `MIS`problem
`--noise_potential` Noise Distribution that is used during training \
`--batch_size` Number of different CO Instances in each batch\
`--n_basis_sates` Number of sampled Diffusion Trajectories in each CO Instance\
`--temps` Starting temperature for annealing\




