# preparing datasets
# python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun
python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --GPUs ['5'] --st_idx 0 --ed_idx 500 
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --GPUs ['2'] 
python prepare_datasets_knp.py --datasets_name KS_4 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 4 --num_samples 3000 --thread_fraction 1 --GPUs ['3','4','5','6','7'] 
python prepare_datasets_knp.py --datasets_name KS_4 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 4 --num_samples 1000 --thread_fraction 1 --GPUs ['6'] 
python prepare_datasets_knp.py --datasets_name KS_4 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 4 --num_samples 1000 --GPUs ['7']
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 4 --num_samples 1000 --GPUs ['7']
# python prepare_datasets.py --dataset RB_iid_100 --problem MIS
python prepare_datasets.py --dataset RB_iid_small --problem MIS

python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/haojun/DIffUCO/only_one_Data_for_solver.pkl --licence_path /home/chenhaojun --dim 3 --num_samples 1000 --thread_fraction 1 --GPUs 3
python prepare_datasets_knp.py --datasets_name KS_d_re_5000 --problem MIS --datasets_path /home/haojun/DIffUCO/only_one_Data_for_solver.pkl --licence_path /home/chenhaojun --dim 3 --num_samples 1000 --thread_fraction 1 --GPUs 0

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
python MIS_evaluate.py --wandb_id 71qtteyy --dataset KS_3_1000_train --GPU 1 --evaluation_factor 3 --n_samples 1 --GPU 7
python MIS_evaluate.py --wandb_id mr6slup1 --dataset KS_3_1000_train --GPU 1 --evaluation_factor 3 --n_samples 1 --GPU 0
python MIS_evaluate.py --wandb_id ojqw39ks --dataset KS_3_1000_train --GPU 0 --evaluation_factor 5 --n_samples 1
python MIS_evaluate.py --wandb_id f3ad84ee --dataset KS_one_3_1000 --GPU 0 --evaluation_factor 5 --n_samples 1
python MIS_evaluate.py --wandb_id 8360d925 --dataset KS_one_3_1000 --GPU 0 --evaluation_factor 5 --n_samples 1
python MIS_evaluate.py --wandb_id 2cdb71a6 --dataset KS_one_3_1000 --GPU 0 --evaluation_factor 2 --n_samples 1
python MIS_evaluate.py --wandb_id 2a4a042f --dataset KS_one_3_1000 --GPU 0 --evaluation_factor 3 --n_samples 1
python MIS_evaluate.py --wandb_id 2efec818 --dataset KS_one_3_1000 --GPU 0 --evaluation_factor 3 --n_samples 1
python MIS_evaluate.py --wandb_id spcrakpu --dataset KS_one_3_1000 --GPU 0 --evaluation_factor 20 --n_samples 1
python MIS_evaluate.py --wandb_id de63df5d --dataset KS_one_3_1000 --GPU 5 --evaluation_factor 9 --n_samples 1
python MIS_evaluate.py --wandb_id 0ba01942 --dataset KS_5_5000 --GPU 7 --evaluation_factor 4 --n_samples 1
python MIS_evaluate.py --wandb_id f53af0da --dataset KS_5_5000 --GPU 4 --evaluation_factor 5 --n_samples 1
python MIS_evaluate.py --wandb_id 026ae66b --dataset KS_5_5000 --GPU 0 --evaluation_factor 3 --n_samples 1


python MIS_evaluate.py --wandb_id 34e406b9 --dataset KS_d_re_5000 --GPU 1 --evaluation_factor 3 --n_samples 1
python MIS_evaluate.py --wandb_id 2992ad1d --dataset KS_5_5000 --GPU 0 --evaluation_factor 3 --n_samples 1
python MIS_evaluate.py --wandb_id 31e75545 --dataset KS_5_5000 --GPU 0 --evaluation_factor 3 --n_samples 1




# original paper

# train
python argparse_ray_main.py --lrs 0.001 --GPUs 1,2,3,4 --n_GNN_layers 7 --temps 0.4 --IsingMode RB_iid_small --EnergyFunction MIS --N_anneal 2000  --n_diffusion_steps 4 --batch_size 12 --n_basis_states 7 --noise_potential bernoulli --project_name RB_small --seed 123 



python ConditionalExpectation.py --wandb_id xvrrfwsg --dataset RS_iid_small --GPU 0 --evaluation_factor 3 --n_samples 8



export GUROBI_HOME=/home/chenhaojun/opt/gurobi1102/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export GRB_LICENSE_FILE =/home/chenhaojun/gurobi.lic
# /home/chenhaojun/scratch/gurobi1102

# python continue_training.py --wandb_id z5xmvb6v --GPUs 2,3,6,7 --continue_dataset KS_3_1000
# python continue_training.py --wandb_id spcrakpu --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_one_3_1000 --add_epoch 2000
python continue_training.py --wandb_id 34e406b9 --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_5_5000 --add_epoch 2000 --batch_size 7 --N_basis_states 1




# Ours
# prepare dataset
cd DatasetCreator/
python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 0 --modes "test" --time_limits "inf" --GPUs ['0']
python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 0 --modes "val" --time_limits "1" --GPUs ['1']
python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 5000 --ed_idx 5500 --modes "train" --time_limits "0.1" --GPUs ['0']
python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 6000 --ed_idx 6500 --modes "train" --time_limits "0.1" --GPUs ['1']

python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 3250 --ed_idx 3500 --modes "train" --time_limits "0.1" --GPUs ['2']
python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 5000 --ed_idx 5250 --modes "train" --time_limits "0.1" --GPUs ['3']
python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 5250 --ed_idx 5500 --modes "train" --time_limits "0.1" --GPUs ['4']
python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 6000 --ed_idx 6250 --modes "train" --time_limits "0.1" --GPUs ['5']
python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 6250 --ed_idx 6500 --modes "train" --time_limits "0.1" --GPUs ['6']
python prepare_datasets_knp.py --datasets_name KS_3 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 3000 --ed_idx 3250 --modes "train" --time_limits "0.1" --GPUs ['7']

python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --st_idx 0 --modes "test" --time_limits "inf" --GPUs ['0']
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --st_idx 0 --modes "val" --time_limits "1" --GPUs ['1']
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --st_idx 2500 --ed_idx 2750 --modes "train" --time_limits "0.1" --GPUs ['0']
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --st_idx 2750 --ed_idx 3000 --modes "train" --time_limits "0.1" --GPUs ['1']
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --st_idx 3000 --ed_idx 3250 --modes "train" --time_limits "0.1" --GPUs ['2']
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --st_idx 3250 --ed_idx 3500 --modes "train" --time_limits "0.1" --GPUs ['3']
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --st_idx 3500 --ed_idx 3750 --modes "train" --time_limits "0.1" --GPUs ['4']
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --st_idx 3750 --ed_idx 4000 --modes "train" --time_limits "0.1" --GPUs ['5']
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --st_idx 4000 --ed_idx 4250 --modes "train" --time_limits "0.1" --GPUs ['6']
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --st_idx 4250 --ed_idx 4500 --modes "train" --time_limits "0.1" --GPUs ['7']
python prepare_datasets_knp.py --datasets_name KS_5 --problem MIS --datasets_path /home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 5 --num_samples 5000 --thread_fraction 1 --st_idx 3495 --ed_idx 3495 --modes "train" --time_limits "0.1" --GPUs ['7']

python prepare_datasets_knp.py --datasets_name KS_d_re_5000 --problem MIS --datasets_path /home/haojun/DIffUCO/only_one_Data_for_solver.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 200 --ed_idx 500 --modes "test" --time_limits "inf" --GPUs ['0']
python prepare_datasets_knp.py --datasets_name KS_d_re_5000 --problem MIS --datasets_path /home/haojun/DIffUCO/only_one_Data_for_solver.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 0 --ed_idx 500 --modes "val" --time_limits "1" --GPUs ['1']
python prepare_datasets_knp.py --datasets_name KS_d_re_5000 --problem MIS --datasets_path /home/haojun/DIffUCO/only_one_Data_for_solver.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 0 --ed_idx 833 --modes "train" --time_limits "0.1" --GPUs ['2']
python prepare_datasets_knp.py --datasets_name KS_d_re_5000 --problem MIS --datasets_path /home/haojun/DIffUCO/only_one_Data_for_solver.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 833 --ed_idx 1666 --modes "train" --time_limits "0.1" --GPUs ['3']
python prepare_datasets_knp.py --datasets_name KS_d_re_5000 --problem MIS --datasets_path /home/haojun/DIffUCO/only_one_Data_for_solver.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 1666 --ed_idx 2499 --modes "train" --time_limits "0.1" --GPUs ['4']
python prepare_datasets_knp.py --datasets_name KS_d_re_5000 --problem MIS --datasets_path /home/haojun/DIffUCO/only_one_Data_for_solver.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 2499 --ed_idx 3332 --modes "train" --time_limits "0.1" --GPUs ['5']
python prepare_datasets_knp.py --datasets_name KS_d_re_5000 --problem MIS --datasets_path /home/haojun/DIffUCO/only_one_Data_for_solver.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 3332 --ed_idx 4165 --modes "train" --time_limits "0.1" --GPUs ['6']
python prepare_datasets_knp.py --datasets_name KS_d_re_5000 --problem MIS --datasets_path /home/haojun/DIffUCO/only_one_Data_for_solver.pkl --licence_path /home/chenhaojun --uniform_generate_data True --dim 3 --num_samples 1000 --thread_fraction 1 --st_idx 4165 --ed_idx 5000 --modes "train" --time_limits "0.1" --GPUs ['7']


# train
python argparse_ray_main.py --lrs 0.002 --GPUs 5,6,7 --n_GNN_layers 7 --temps 0.3 --IsingMode KS_3_1000 --EnergyFunction MIS --energy_A 1.0 --energy_B 1.2 --N_anneal 3000  --n_diffusion_steps 4 --batch_size 10 --n_basis_states 10 --noise_potential annealed_obj --project_name A800 --seed 123 
python argparse_ray_main.py --lrs 0.002 --GPUs 1,2,3,4,5,6,7 --n_GNN_layers 8 --temps 0.6 --IsingMode KS_3_1000 --EnergyFunction MIS --energy_A 1.0 --energy_B 1.01 --N_anneal 5000  --n_diffusion_steps 6 --batch_size 20 --n_basis_states 10 --noise_potential annealed_obj --project_name A800 --seed 123 


# python continue_training.py --wandb_id 71qtteyy --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_3_1000 --energy_A 1.0 --energy_B 2.0 --batch_size 20
python continue_training.py --wandb_id xzuk0835 --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_3_1000 --energy_A 1.0 --energy_B 2.0 --batch_size 20
python continue_training.py --wandb_id mr6slup1 --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_3_1000 --energy_A 1.0 --energy_B 2.0 --batch_size 10 --wandb False
python continue_training.py --wandb_id ojqw39ks --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_3_1000 --energy_A 1.0 --energy_B 1.01 --batch_size 20
python continue_training.py --wandb_id f3ad84ee --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_one_3_1000 --energy_A 1.0 --energy_B 1.3 --batch_size 20
python continue_training.py --wandb_id a2e20b8c --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_one_3_1000 --energy_A 1.0 --energy_B 4.0 --batch_size 20
python continue_training.py --wandb_id 8360d925 --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_one_3_1000 --energy_A 1.0 --energy_B 1.01 --batch_size 20
python continue_training.py --wandb_id 2cdb71a6 --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_one_3_1000 --energy_A 1.0 --energy_B 8 --batch_size 20
python continue_training.py --wandb_id 2efec818 --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_one_3_1000 --energy_A 1.0 --energy_B 2.0 --batch_size 20
python continue_training.py --wandb_id spcrakpu --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_one_3_1000 --energy_A 1.0 --energy_B 1.01 --batch_size 20 --add_epoch 2000
python continue_training.py --wandb_id 0ba01942 --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_5_5000 --energy_A 1.0 --energy_B 1.3 --batch_size 7 --add_epoch 2000
python continue_training.py --wandb_id 026ae66b --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_5_5000 --energy_A 1.0 --energy_B 1.01 --batch_size 7 --add_epoch 2000

python continue_training.py --wandb_id 2992ad1d --GPUs 1,2,3,4,5,6,7 --continue_dataset KS_5_5000 --energy_A 1.0 --energy_B 1.01 --batch_size 7 --add_epoch 2000




# 分卡
python argparse_ray_main.py --lrs 0.002 --GPUs 1,2 --n_GNN_layers 8 --temps 0.6 --IsingMode KS_3_1000 --EnergyFunction MIS --energy_A 1.0 --energy_B 1.01 --N_anneal 3000  --n_diffusion_steps 8 --batch_size 16 --n_basis_states 10 --noise_potential bernoulli --project_name A800 --seed 123 

# python argparse_ray_main.py --lrs 0.002 --GPUs 1,2,3 --n_GNN_layers 8 --temps 0.6 --IsingMode KS_3_1000 --EnergyFunction MIS --energy_A 1.0 --energy_B 1.01 --N_anneal 5000  --n_diffusion_steps 8 --batch_size 8 --n_basis_states 10 --noise_potential annealed_obj --project_name A800 --seed 123 
python argparse_ray_main.py --lrs 0.002 --GPUs 1,2,3,4,5,6,7 --n_GNN_layers 8 --temps 0.8 --IsingMode KS_3_1000 --EnergyFunction MIS --energy_A 1.0 --energy_B 1.3 --N_anneal 6000  --n_diffusion_steps 10 --batch_size 12 --n_basis_states 10 --noise_potential annealed_obj --project_name A800 --seed 123 



python argparse_ray_main.py --lrs 0.002 --GPUs 1,2,3,4,5,6,7 --n_GNN_layers 8 --temps 0.6 --IsingMode KS_one_3_1000 --EnergyFunction MIS --energy_A 1.0 --energy_B 1.01 --N_anneal 5000  --n_diffusion_steps 6 --batch_size 20 --n_basis_states 10 --noise_potential annealed_obj --project_name A800 --seed 123 
python argparse_ray_main.py --lrs 0.002 --GPUs 1,2,3,4,5,6,7 --n_GNN_layers 8 --temps 0.6 --IsingMode KS_one_3_1000 --EnergyFunction MIS --energy_A 1.0 --energy_B 1.01 --N_anneal 2000  --n_diffusion_steps 4 --batch_size 20 --n_basis_states 10 --noise_potential bernoulli --project_name A800 --seed 123 
python argparse_ray_main.py --lrs 0.002 --GPUs 1,2,3,4,5,6,7 --n_GNN_layers 7 --temps 0.6 --IsingMode KS_5_5000 --EnergyFunction MIS --energy_A 1.0 --energy_B 1.2 --N_anneal 2000  --n_diffusion_steps 4 --batch_size 8 --n_basis_states 4 --noise_potential bernoulli --project_name A800 --seed 123 
python argparse_ray_main.py --lrs 0.002 --GPUs 1,2,3,4,5,6,7 --n_GNN_layers 7 --temps 0.6 --IsingMode KS_5_5000 --EnergyFunction MIS --energy_A 1.0 --energy_B 1.2 --N_anneal 2000  --n_diffusion_steps 4 --batch_size 7 --n_basis_states 2 --noise_potential bernoulli --project_name A800 --seed 123 
python argparse_ray_main.py --lrs 0.002 --GPUs 1,2,3,4,5,6,7 --n_GNN_layers 7 --temps 0.6 --IsingMode KS_5_5000 --EnergyFunction MIS --energy_A 1.0 --energy_B 1.2 --N_anneal 1000  --n_diffusion_steps 8 --batch_size 7 --n_basis_states 1 --noise_potential bernoulli --project_name A800 --seed 123 
python argparse_ray_main.py --lrs 0.002 --GPUs 1,2,3,4,5,6,7 --n_GNN_layers 7 --temps 0.6 --IsingMode KS_d_re_5000 --EnergyFunction MIS --energy_A 1.0 --energy_B 1.2 --N_anneal 1000  --n_diffusion_steps 4 --batch_size 8 --n_basis_states 4 --noise_potential bernoulli --project_name A800 --seed 123 
