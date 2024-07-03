# preparing datasets
python prepare_datasets_knp.py --datasets_name KS_4 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl
python prepare_datasets.py --dataset RB_iid_100 --problem MIS

# python argparse_ray_main.py --lrs 0.002 --GPUs 0 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_100 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 20 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123
python argparse_ray_main.py --lrs 0.002 --GPUs 0 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_large --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 20 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123



# python argparse_ray_main.py --lrs 0.002 --GPUs 0,1,2,3,4,5,6,7 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_100 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 160 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123 




python argparse_ray_main.py --lrs 0.002 --GPUs 0 --n_GNN_layers 8 --temps 0.6 --IsingMode KS_iid_1000 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 1 --n_basis_states 10 --noise_potential bernoulli --project_name one_gpu_test --seed 123



python argparse_ray_main.py --lrs 0.002 --GPUs 0,5,6,7 --n_GNN_layers 8 --temps 0.6 --IsingMode KS_iid_1000 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 14 --n_basis_states 4 --noise_potential bernoulli --project_name KS3 --seed 123




# evaluation
python ConditionalExpectation.py --wandb_id ergzkyr5 --dataset KS_iid_1000 --GPU 0 --evaluation_factor 3 --n_samples 8
python MIS_visualize.py --wandb_id z5xmvb6v --dataset KS_4 --GPU 7 --evaluation_factor 3 --n_samples 8




# original paper

# train
python argparse_ray_main.py --lrs 0.002 --GPUs 1,2,3,4 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_100 --EnergyFunction MIS --N_anneal 2000  --n_diffusion_steps 3 --batch_size 20 --n_basis_states 10 --noise_potential bernoulli --project_name RB_small --seed 123 











