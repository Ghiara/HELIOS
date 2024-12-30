# Pretrained Bayesian Non-parametric Knowledge Prior in Robotic Long-Horizon Reinforcement Learning


#### [[Project Website]](https://ghiara.github.io/helios/) [[Paper]](docs/static/helios_paper.pdf)

[Yuan Meng](https://github.com/Ghiara)<sup>1,</sup>, 
[Xiangtong Yao](https://www.ce.cit.tum.de/air/people/xiangtong-yao/)<sup>1</sup>, 
[Kejia Chen](https://kifabrik.mirmi.tum.de/team/)<sup>1</sup>,
[Yansong Wu](https://kifabrik.mirmi.tum.de/team/)<sup>1</sup>,
[Liding Zhang](https://kifabrik.mirmi.tum.de/team/)<sup>1</sup>,
[Achim Lilienthal](https://kifabrik.mirmi.tum.de/team/)<sup>1</sup>,
[Zhenshan Bing](https://github.com/zhenshan-bing)<sup>1,2</sup>, 
[Alois Knoll](https://www.ce.cit.tum.de/air/people/prof-dr-ing-habil-alois-knoll/)<sup>1</sup>,

<sup>1</sup>The School of Computation, Information and Technology, Technical University of Munich, Germany

<sup>2</sup>State Key Laboratory for Novel Software Technology, Nanjing University, China


The official implementation of robotic long-horizon manipulation refincement learning framework -- **HELIOS**: Hierarchical Encoding of Long-horizon Inference with Off-policy Bayesian Non-parametric Skills Prior

<p align="center">
<img src="docs/static/images/helios_framework.png" width="800"></p></img>





## Requirements

- python 3.7+
- mujoco 2.0 (for RL experiments)
- Ubuntu 20.04 LTS or 22.04 LTS

## Installation Instructions

### 1. Create and activate a virtual environment, install all requirements

```bash
# Setup the environment

conda create -n helios python=3.8

# Install dependencies and packages

cd helios
pip3 install -r requirements.txt
pip3 install -e .
```

### 2. Define environment variables to specify the root experiment and data directories

```bash
# Experiments folder stores trained models
# Data folder stores external data libraries

mkdir ./experiments
mkdir ./data
export EXP_DIR=./experiments
export DATA_DIR=./data
```

### 3. Install the Fork of D4RL benchmark

Follow the [D4RL Fork link](https://github.com/kpertsch/d4rl) and install the fork according to instructions.
This fork includes the new key 'completed_tasks' in the Kitchen environment, which is **necessary for the correct RL phase**.

### 4. Log in to WandB to track results

[WandB](https://www.wandb.com/) is used for **logging the training process**. Before running any of the commands below, 
create an account and then change the WandB entity and project name at the top of [train.py](helios/train.py) and
[rl/train.py](helios/rl/train.py) to match your account.

## CLI for Training

### 1. Train DPM based Skill Prior

To train a **DPM based Generalized Skill Prior** model, run:
```bash
python3 helios/train.py --path=helios/configs/skill_prior_learning/kitchen/helios_h_cl --val_data_size=160 --gpu=0

```

### 2. Train HELIOS for downstream Long-Horizon RL manipulation (e.g., Franka kitchen)

After DPM based Skill Prior model is trained, to train **HELIOS** agent on the franka kitchen long-horizon tasks, run (change the index number of your available GPU device accordingly):
```bash
python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/helios_cl --seed=0 --prefix=helios_kitchen_seed0 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/helios_cl --seed=1 --prefix=helios_kitchen_seed1 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/helios_cl --seed=2 --prefix=helios_kitchen_seed2 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/helios_cl --seed=3 --prefix=helios_kitchen_seed3 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/helios_cl --seed=4 --prefix=helios_kitchen_seed4 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/helios_cl --seed=5 --prefix=helios_kitchen_seed5 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/helios_cl --seed=6 --prefix=helios_kitchen_seed6 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/helios_cl --seed=7 --prefix=helios_kitchen_seed7 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/helios_cl --seed=8 --prefix=helios_kitchen_seed8 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/helios_cl --seed=9 --prefix=helios_kitchen_seed9 --gpu=0

```



### 3. Train Baseline Models

- Run **Vanilla SAC**:
```bash
python3 helios/rl/train.py --path=helios/configs/rl/kitchen/SAC --seed=0 --prefix=SAC_kitchen_seed0 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/SAC --seed=1 --prefix=SAC_kitchen_seed1 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/SAC --seed=2 --prefix=SAC_kitchen_seed2 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/SAC --seed=3 --prefix=SAC_kitchen_seed3 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/SAC --seed=4 --prefix=SAC_kitchen_seed4 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/SAC --seed=5 --prefix=SAC_kitchen_seed5 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/SAC --seed=6 --prefix=SAC_kitchen_seed6 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/SAC --seed=7 --prefix=SAC_kitchen_seed7 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/SAC --seed=8 --prefix=SAC_kitchen_seed8 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/SAC --seed=9 --prefix=SAC_kitchen_seed9 --gpu=0
```

- Run **SPIRL (closed-loop)**: 
```bash
# for Skill Prior learning (single Gaussian)
python3 helios/train.py --path=helios/configs/skill_prior_learning/kitchen/hierarchical_cl --val_data_size=160 --gpu=0

# for downstream hierarchical RL policy
# Note: change the load path of pretrained prior before you run training.
python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/spirl_cl --seed=0 --prefix=SPIRL_kitchen_seed0 --gpu=0 

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/spirl_cl --seed=1 --prefix=SPIRL_kitchen_seed1 --gpu=0 

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/spirl_cl --seed=2 --prefix=SPIRL_kitchen_seed2 --gpu=0 

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/spirl_cl --seed=3 --prefix=SPIRL_kitchen_seed3 --gpu=0 

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/spirl_cl --seed=4 --prefix=SPIRL_kitchen_seed4 --gpu=0 

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/spirl_cl --seed=5 --prefix=SPIRL_kitchen_seed5 --gpu=0 

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/spirl_cl --seed=6 --prefix=SPIRL_kitchen_seed6 --gpu=0 

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/spirl_cl --seed=7 --prefix=SPIRL_kitchen_seed7 --gpu=0 

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/spirl_cl --seed=8 --prefix=SPIRL_kitchen_seed8 --gpu=0 

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/spirl_cl --seed=9 --prefix=SPIRL_kitchen_seed9 --gpu=0 
```


- Train **Single-step action prior**:
```bash
python3 helios/train.py --path=helios/configs/skill_prior_learning/kitchen/flat --val_data_size=160 --gpu=0
```

- Run **SAC w/ single-step action prior**:
```bash
python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/flat_prior/ --seed=0 --prefix=flatPrior_kitchen_seed0 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/flat_prior/ --seed=1 --prefix=flatPrior_kitchen_seed1 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/flat_prior/ --seed=2 --prefix=flatPrior_kitchen_seed2 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/flat_prior/ --seed=3 --prefix=flatPrior_kitchen_seed3 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/flat_prior/ --seed=4 --prefix=flatPrior_kitchen_seed4 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/flat_prior/ --seed=5 --prefix=flatPrior_kitchen_seed5 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/flat_prior/ --seed=6 --prefix=flatPrior_kitchen_seed6 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/flat_prior/ --seed=7 --prefix=flatPrior_kitchen_seed7 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/flat_prior/ --seed=8 --prefix=flatPrior_kitchen_seed8 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/flat_prior/ --seed=9 --prefix=flatPrior_kitchen_seed9 --gpu=0
```

- Run **BC + finetune**:
```bash
python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/bc_finetune/ --seed=0 --prefix=bcFinetune_kitchen_seed0 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/bc_finetune/ --seed=1 --prefix=bcFinetune_kitchen_seed1 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/bc_finetune/ --seed=2 --prefix=bcFinetune_kitchen_seed2 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/bc_finetune/ --seed=3 --prefix=bcFinetune_kitchen_seed3 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/bc_finetune/ --seed=4 --prefix=bcFinetune_kitchen_seed4 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/bc_finetune/ --seed=5 --prefix=bcFinetune_kitchen_seed5 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/bc_finetune/ --seed=6 --prefix=bcFinetune_kitchen_seed6 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/bc_finetune/ --seed=7 --prefix=bcFinetune_kitchen_seed7 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/bc_finetune/ --seed=8 --prefix=bcFinetune_kitchen_seed8 --gpu=0

python3 helios/rl/train.py --path=helios/configs/rl/kitchen/prior_initialized/bc_finetune/ --seed=9 --prefix=bcFinetune_kitchen_seed9 --gpu=0
```

- Run **Skill Space Policy w/o prior**:
```bash
python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/no_prior/ --seed=0 --prefix=SSP_noPrior_kitchen_seed0 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/no_prior/ --seed=1 --prefix=SSP_noPrior_kitchen_seed1 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/no_prior/ --seed=2 --prefix=SSP_noPrior_kitchen_seed2 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/no_prior/ --seed=3 --prefix=SSP_noPrior_kitchen_seed3 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/no_prior/ --seed=4 --prefix=SSP_noPrior_kitchen_seed4 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/no_prior/ --seed=5 --prefix=SSP_noPrior_kitchen_seed5 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/no_prior/ --seed=6 --prefix=SSP_noPrior_kitchen_seed6 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/no_prior/ --seed=7 --prefix=SSP_noPrior_kitchen_seed7 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/no_prior/ --seed=8 --prefix=SSP_noPrior_kitchen_seed8 --gpu=0

python3 helios/rl/train.py --path=helios/configs/hrl/kitchen/no_prior/ --seed=9 --prefix=SSP_noPrior_kitchen_seed9 --gpu=0

```
Again, all commands can be run on `maze / block stacking` by replacing `kitchen` with the respective environment in the paths
(after downloading the datasets).

## Starting to Modify the Code

### Modifying the hyperparameters
The default hyperparameters are defined in the respective model files, e.g. in [```SkillPriorMdl```](helios/models/skill_prior_mdl.py#L28) for the HELIOS model. [```SPiRL_DPMM_Mdl```](helios/models/CL_SPIRL_DPMM_mdl.py#L21) defines DPM related hyperparameters. Modifications to these parameters can be defined through the experiment config files (passed to the respective command via the `--path` variable). For an example, see [```kitchen/hierarchical/conf.py```](helios/configs/skill_prior_learning/kitchen/hierarchical/conf.py).


### Adding a new dataset for model training
All code that is dataset-specific should be placed in a corresponding subfolder in `helios/data`. 
To add a data loader for a new dataset, the `Dataset` classes from [```data_loader.py```](helios/components/data_loader.py) need to be subclassed
and the `__getitem__` function needs to be overwritten to load a single data sample. The output `dict` should include the following
keys:

```bash
dict({
    'states': (time, state_dim)                 # state sequence (for state-based prior inputs)
    'actions': (time, action_dim)               # action sequence (as skill input for training prior model)
    'images':  (time, channels, width, height)  # image sequence (for image-based prior inputs)
})
```

All datasets used with the codebase so far have been based on `HDF5` files. The `GlobalSplitDataset` provides functionality to read all
HDF5-files in a directory and split them in `train/val/test` based on percentages. The `VideoDataset` class provides
many functionalities for manipulating sequences, like randomly cropping subsequences, padding etc.

### Adding a new RL environment
To add a new RL environment, simply define a new environent class in `helios/rl/envs` that inherits from the environment interface
in [```helios/rl/components/environment.py```](helios/rl/components/environment.py).


### Modifying the skill prior model architecture
Start by defining a model class in the `helios/models` directory that inherits from the `BaseModel` or `SkillPriorMdl` class. 
The new model needs to define the architecture in the constructor (e.g. by overwriting the `build_network()` function), 
implement the forward pass and loss functions,
as well as model-specific logging functionality if desired. For an example, see [```helios/models/skill_prior_mdl.py```](helios/models/skill_prior_mdl.py).

Note, that most basic architecture components (MLPs, CNNs, LSTMs, GRUs, Flow models etc) are defined in `helios/modules` and can be 
conveniently reused for easy architecture definitions. Below are some links to the most important classes.

|Component        | File         | Description |
|:------------- |:-------------|:-------------|
| MLP | [```Predictor```](helios/modules/subnetworks.py#L33) | Basic N-layer fully-connected network. Defines number of inputs, outputs, layers and hidden units. |
| CNN-Encoder | [```ConvEncoder```](helios/modules/subnetworks.py#L66) | Convolutional encoder, number of layers determined by input dimensionality (resolution halved per layer). Number of channels doubles per layer. Returns encoded vector + skip activations. |
| CNN-Decoder | [```ConvDecoder```](helios/modules/subnetworks.py#L145) | Mirrors architecture of conv. encoder. Can take skip connections as input, also versions that copy pixels etc. |
| Processing-LSTM | [```BaseProcessingLSTM```](helios/modules/recurrent_modules.py#L126) | Basic N-layer LSTM for processing an input sequence. Produces one output per timestep, number of layers / hidden size configurable.|
| Processing-GRU | [```BaseProcessingGRU```](helios/modules/recurrent_modules.py#L138) | Basic N-layer GRU for processing an input sequence. Produces one output per timestep, number of layers / hidden size configurable.|
| Prediction-LSTM | [```RecurrentPredictor```](helios/modules/recurrent_modules.py#L428) | Same as processing LSTM, but for autoregressive prediction. |
| Mixture-Density Network | [```MDN```](helios/modules/mdn.py#L10) | MLP that outputs GMM distribution. |
| Normalizing Flow Model | [```NormalizingFlowModel```](helios/modules/flow_models.py#L9) | Implements normalizing flow model that stacks multiple flow blocks. Implementation for RealNVP block provided. |
| HELIOS Skill Prior |[```SPiRL_DPMM_Mdl```](helios/models/CL_SPIRL_DPMM_mdl.py#L22)| Model of DPM-based Skill Prior learning|
| DPM Learning |[```ModelTrainer```](helios/train.py#L30)| Train loop modified for DPM fitting |
| DPM-based Loss functions |[```DPMM_KLDivLoss```](helios/modules/losses.py#L59)| modified loss functions, involving weighted sum of KL-divergencies |


### Adding a new RL algorithm
The core RL algorithms are implemented within the `Agent` class. For adding a new algorithm, a new file needs to be created in
`helios/rl/agents` and [```BaseAgent```](helios/rl/components/agent.py#L19) needs to be subclassed. In particular, any required
networks (actor, critic etc) need to be constructed and the `update(...)` function needs to be overwritten. For an example, 
see the SAC implementation in [```SACAgent```](helios/rl/agents/ac_agent.py#L67).

The main SPIRL skill prior regularized RL algorithm is implemented in [```ActionPriorSACAgent```](helios/rl/agents/prior_sac_agent.py#L12).





### Detailed Code Structure

```bash
helios

  |- components            # reusable infrastructure for model training
  |    |- base_model.py    # basic model class that all models inherit from
  |    |- checkpointer.py  # handles storing + loading of model checkpoints
  |    |- data_loader.py   # basic dataset classes, new datasets need to inherit from here
  |    |- evaluator.py     # defines basic evaluation routines, eg top-of-N evaluation, + eval logging
  |    |- logger.py        # implements core logging functionality using tensorboardX
  |    |- params.py        # definition of command line params for model training
  |    |- trainer_base.py  # basic training utils used in main trainer file
  |
  |- configs               # all experiment configs should be placed here
  |    |- data_collect     # configs for data collection runs
  |    |- default_data_configs   # defines one default data config per dataset, e.g. state/action dim etc
  |    |- hrl              # configs for hierarchical downstream RL
  |    |- rl               # configs for non-hierarchical downstream RL
  |    |- skill_prior_learning   # configs for skill embedding and prior training (both hierarchical and flat)
  |
  |- data                  # any dataset-specific code (like data generation scripts, custom loaders etc)
  |- models                # holds all model classes that implement forward, loss, visualization
  |- modules               # reusable architecture components (like MLPs, CNNs, LSTMs, Flows etc)
  |- rl                    # all code related to RL
  |    |- agents           # implements core algorithms in agent classes, like SAC etc
  |    |- components       # reusable infrastructure for RL experiments
  |        |- agent.py     # basic agent and hierarchial agent classes - do not implement any specific RL algo
  |        |- critic.py    # basic critic implementations (eg MLP-based critic)
  |        |- environment.py    # defines environment interface, basic gym env
  |        |- normalization.py  # observation normalization classes, only optional
  |        |- params.py    # definition of command line params for RL training
  |        |- policy.py    # basic policy interface definition
  |        |- replay_buffer.py  # simple numpy-array replay buffer, uniform sampling and versions
  |        |- sampler.py   # rollout sampler for collecting experience, for flat and hierarchical agents
  |    |- envs             # all custom RL environments should be defined here
  |    |- policies         # policy implementations go here, MLP-policy and RandomAction are implemented
  |    |- utils            # utilities for RL code like MPI, WandB related code
  |    |- train.py         # main RL training script, builds all components + runs training
  |
  |- utils                 # general utilities, pytorch / visualization utilities etc
  |- train.py              # main model training script, builds all components + runs training loop and logging
```



## Implementation & Acknowledgement

The implementation of this work inherits from three repositories:

- [DIVA: A Dirichlet Process Mixtures Based Incremental Deep Clustering Algorithm via Variational Auto-Encoder](https://github.com/Ghiara/DIVA)
- [Memoized online variational inference for Dirichlet process mixture models](https://github.com/bnpy/bnpy)
- [Accelerating Reinforcement Learning with Learned Skill Priors](https://github.com/clvrai/spirl)

