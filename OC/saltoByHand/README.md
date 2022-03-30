# SALTO BY HAND
## Implementation "by hand" of salto trajectory for Solo12.
To only generate thoses pushing phases trajectory, run:
```
python3 inverse_kinematics.py
```
It will generate the pushes phases and store it in ```trajectory_npy/``` directory under the name of ```ik_Q.npy``` for configurations and ```ik_vQ.npy``` for joint angular velocities.

To generate the entire salto and/or simulate it on pybullet, run:
```
python3 main.py
```
or
```
source run.sh
```
It will generate the entire salto and store it in ```trajectory_npy/``` directory under the name of ```salto_Q.npy``` for configurations and ```salto_vQ.npy``` for joint angular velocities.

To generate the entire salto from scratch and simulate it on pybullet, run:
```
rm ./trajectory_npy/*
python3 main.py
```
or
```
source run_from_scratch.sh
```
It will generate the entire salto and store it in ```trajectory_npy/``` directory under the name of ```salto_Q.npy``` for configurations and ```salto_vQ.npy``` for joint angular velocities.

To generate the npz file (used to simulate further Solo12 in its environment), run:
```
python3 generate_npz.py
```
It will generate two files in ```trajectory_npz/``` directory, one called ```backflip_by_hand.npz``` and one ```with_gains.backflip_by_hand.npz``` which is the same trajectory with the gains.




