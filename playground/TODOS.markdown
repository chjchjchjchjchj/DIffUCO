#### TODO
- AnnealedNoise + forward Kl does not work
- test if other energy functions work as intended
- add rand nodes to forwardkl and ppo

### TODO set up GPU clusters

### TODO average prob figure properly over batch; prob figures show something different than expected

### TODOS test loss change in REINFORCE, test combined noise distr in REINFORCE
### TODO dest annealed dist with forward KL and then combined

### TODO implement CE-ST

### TODO implement gradient accumulation




#### TODOs find out influence of lr schedule
### find influence of annealed + noise distr
### does DiffUCO ICML perform as well as DiffUCO Rl with same number of diffusion steps?
### gradient accumulation in DIffUCO is only possible across graph batch size