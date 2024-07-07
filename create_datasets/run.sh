cd create_datasets/
CUDA_VISIBLE_DEVICES=1 python create_solution_for_subgraph.py --sub_graph_group /home/chenhaojun/DIffUCO/Data/KS_3_split/folder_2
CUDA_VISIBLE_DEVICES=2 python create_solution_for_subgraph.py --sub_graph_group /home/chenhaojun/DIffUCO/Data/KS_3_split/folder_3
CUDA_VISIBLE_DEVICES=3 python create_solution_for_subgraph.py --sub_graph_group /home/chenhaojun/DIffUCO/Data/KS_3_split/folder_4
CUDA_VISIBLE_DEVICES=4 python create_solution_for_subgraph.py --sub_graph_group /home/chenhaojun/DIffUCO/Data/KS_3_split/folder_5
CUDA_VISIBLE_DEVICES=5 python create_solution_for_subgraph.py --sub_graph_group /home/chenhaojun/DIffUCO/Data/KS_3_split/folder_6
CUDA_VISIBLE_DEVICES=6 python create_solution_for_subgraph.py --sub_graph_group /home/chenhaojun/DIffUCO/Data/KS_3_split/folder_7
CUDA_VISIBLE_DEVICES=7 python create_solution_for_subgraph.py --sub_graph_group /home/chenhaojun/DIffUCO/Data/KS_3_split/folder_8


# Define the range of folder numbers
start=2
end=8

# Loop through the specified range
for i in $(seq $start $end); do
  # Set the folder name
  FOLDER_NAME="folder_$i"
  WINDOW_NAME="window_$i"
  CUDA_DEVICE=$((i - 1))

  # Create a new tmux session with the specified window name and detach
  tmux new-session -d -s $WINDOW_NAME

  # Send commands to the tmux session
  tmux send-keys -t $WINDOW_NAME "conda activate rayjay_clone" C-m
  tmux send-keys -t $WINDOW_NAME "cd create_datasets/" C-m
  tmux send-keys -t $WINDOW_NAME "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python create_solution_for_subgraph.py --sub_graph_group /home/chenhaojun/DIffUCO/Data/KS_3_split/${FOLDER_NAME}" C-m

  # Optionally attach to the tmux session (uncomment the line below if you want to attach)
  # tmux attach-session -t $WINDOW_NAME
done

