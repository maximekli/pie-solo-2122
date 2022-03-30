# pie-solo-2122 - Reinforcement Learning Trajectories

This folder contains all the software work and documentation associated with the reinforcement learning (RL) part of the 2021-2022 PIE on Solo quadruped platform. 

## Description

This part of the project aims to enable the quadruped robot Solo to perform backflips after learning it in simulation through reinforcement learning. The package RL_solo contains:
- the description of the robot needed for simulation in MuJoCo
- a OpenAI Gym environment to train the robot
- the RL algorithm and configurations, used to train the robot

## Installation

This project mainly relies on the following libraries and projects:
- [OpenAI Gym](https://gym.openai.com/)
- [MuJoCo](https://mujoco.org/)
- [Spinning Up](https://spinningup.openai.com/en/latest/)
- [Open Dynamic Robot Initiative](https://github.com/open-dynamic-robot-initiative)

[Here](https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952) is the tutorial that I followed to setup this environment, except for the Open Dynamic Robot Initiative.

### Install mujoco

Download the MuJoCo archive and extract it in a .mujoco folder:
```bash
mkdir ~/.mujoco
cd .mujoco
```

Set MuJoCo environment variables:
```bash
gedit ~/.bashrc
export LD_LIBRARY_PATH=/home/${USER}/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export MUJOCO_KEY_PATH=/home/${USER}/.mujoco${MUJOCO_KEY_PATH}
```

Test MuJoCo:
```bash
cd ~/.mujoco/mujoco210/bin
./simulate ../model/humanoid.xml
```

### Install mujoco-py (to cmmunicate with MuJoCo in Python)

Create a virtual environment:
```bash
conda create -n mujoco-gym
conda activate mujoco-gym
```

Install mujoco-py:
```bash
cd ~/.mujoco
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py 
pip3 install -e .
```

Test mujoco-py:
```bash
python3
>> import mujoco_py
>> import os
>> mj_path = mujoco_py.utils.discover_mujoco()
>> xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
>> model = mujoco_py.load_model_from_path(xml_path)
>> sim = mujoco_py.MjSim(model)

>> print(sim.data.qpos)
# You should have as output something close to this:
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

>> sim.step()
>> print(sim.data.qpos)
# [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
#   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
#   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
#  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
#  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
#  -2.22862221e-05]
```

If you have the error "no file named 'patchelf'", download [the archive](https://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.gz) and
```bash
cd patchelf-0.9
./configure
make
sudo make install
patchelf --version
```

### Install (OpenAI) Gym

Get requirements.txt from pie-solo-2122/RL :
```bash
pip3 install -r requirements.txt
```

If there is an error on build of box2d, do:
```bash
sudo apt-get install swig build-essential python-dev python3-dev
```

### Install SpinningUp

Clone the repository [here](https://github.com/openai/spinningup).
Modify the setup.py file in this cloned repository: remove me = t, what is after torch or remove torch if it was already installed for Gym for example.
Execute setup.py:
```bash
python setup.py install
```

Go to the spinup folder and modify __init__.py to not take into account tf algorithms as they don't work (add an if False: you should get):
```python
# Algorithms
if False:
    from spinup.algos.tf1.ddpg.ddpg import ddpg as ddpg_tf1
    from spinup.algos.tf1.ppo.ppo import ppo as ppo_tf1
    from spinup.algos.tf1.sac.sac import sac as sac_tf1
    from spinup.algos.tf1.td3.td3 import td3 as td3_tf1
    from spinup.algos.tf1.trpo.trpo import trpo as trpo_tf1
    from spinup.algos.tf1.vpg.vpg import vpg as vpg_tf1

from spinup.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from spinup.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from spinup.algos.pytorch.sac.sac import sac as sac_pytorch
from spinup.algos.pytorch.td3.td3 import td3 as td3_pytorch
from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from spinup.algos.pytorch.vpg.vpg import vpg as vpg_pytorch
```

Set SpinninpUp environment variables, go to the root of the SpinningUp repository:
```bash
pwd
export PYTHONPATH=$PYTHONPATH:<PATH_SPINNINGUP>
```

## Run

Now you should be able to train your robot in simulation by launching the main script:
```bash
cd RL_solo
conda activate mujoco-gym
python main.py
```




