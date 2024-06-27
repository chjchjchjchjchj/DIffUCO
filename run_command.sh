# preparing datasets
python prepare_datasets_knp.py --dataset KS_3 --problem MIS


# python argparse_ray_main.py --lrs 0.002 --GPUs 0 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_100 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 20 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123
python argparse_ray_main.py --lrs 0.002 --GPUs 0 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_large --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 20 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123



# python argparse_ray_main.py --lrs 0.002 --GPUs 0,1,2,3,4,5,6,7 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_100 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 160 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123 




python argparse_ray_main.py --lrs 0.002 --GPUs 6 --n_GNN_layers 8 --temps 0.6 --IsingMode KS_iid_1000 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 1 --n_basis_states 10 --noise_potential bernoulli --project_name test_KS --seed 123



# python argparse_ray_main.py --lrs 0.002 --GPUs 0,1,2,3,4,5,6,7 --n_GNN_layers 8 --temps 0.6 --IsingMode KS_iid_1000 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 8 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123


