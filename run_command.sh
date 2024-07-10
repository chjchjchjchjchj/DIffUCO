# preparing datasets
# python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun
python prepare_datasets_knp.py --datasets_name KS_4 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 4 --num_samples 3000 --thread_fraction 1 --GPUs ['1'] 
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --GPUs ['2'] 
python prepare_datasets_knp.py --datasets_name KS_4 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 4 --num_samples 3000 --thread_fraction 1 --GPUs ['3','4','5','6','7'] 
python prepare_datasets_knp.py --datasets_name KS_4 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 4 --num_samples 1000 --thread_fraction 1 --GPUs ['6'] 
python prepare_datasets_knp.py --datasets_name KS_4 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 4 --num_samples 1000 --GPUs ['7']
# python prepare_datasets.py --dataset RB_iid_100 --problem MIS
python prepare_datasets.py --dataset RB_iid_small --problem MIS

# python argparse_ray_main.py --lrs 0.002 --GPUs 0 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_100 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 20 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123
python argparse_ray_main.py --lrs 0.002 --GPUs 0 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_large --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 20 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123



# python argparse_ray_main.py --lrs 0.002 --GPUs 0,1,2,3,4,5,6,7 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_100 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 160 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123 




python argparse_ray_main.py --lrs 0.002 --GPUs 0 --n_GNN_layers 8 --temps 0.6 --IsingMode KS_iid_1000 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 1 --n_basis_states 10 --noise_potential bernoulli --project_name one_gpu_test --seed 123



python argparse_ray_main.py --lrs 0.002 --GPUs 0,5,6,7 --n_GNN_layers 8 --temps 0.6 --IsingMode KS_iid_1000 --EnergyFunction MIS --N_anneal 2000 --n_diffusion_steps 3 --batch_size 14 --n_basis_states 4 --noise_potential bernoulli --project_name KS3 --seed 123




# evaluation
python ConditionalExpectation.py --wandb_id xvrrfwsg --dataset RB_iid_small --GPU 0 --evaluation_factor 3 --n_samples 8
python MIS_evaluate.py --wandb_id z5xmvb6v --dataset KS_4_1000 --GPU 7 --evaluation_factor 3 --n_samples 8
python MIS_evaluate.py --wandb_id xvrrfwsg --dataset KS_3_1000 --GPU 2 --evaluation_factor 3 --n_samples 1
python MIS_evaluate.py --wandb_id spcrakpu --dataset KS_3_1000 --GPU 1 --evaluation_factor 3 --n_samples 1




# original paper

# train
python argparse_ray_main.py --lrs 0.001 --GPUs 1,2,3,4 --n_GNN_layers 7 --temps 0.4 --IsingMode RB_iid_small --EnergyFunction MIS --N_anneal 2000  --n_diffusion_steps 4 --batch_size 12 --n_basis_states 7 --noise_potential bernoulli --project_name RB_small --seed 123 



python ConditionalExpectation.py --wandb_id xvrrfwsg --dataset RS_iid_small --GPU 0 --evaluation_factor 3 --n_samples 8



export GUROBI_HOME=/home/chenhaojun/opt/gurobi1102/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export GRB_LICENSE_FILE =/home/chenhaojun/gurobi.lic
# /home/chenhaojun/scratch/gurobi1102

python continue_training.py --wandb_id z5xmvb6v --GPUs 2,3,6,7 --continue_dataset KS_3_1000