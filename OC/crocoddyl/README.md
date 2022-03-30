### Trajectory generation with Crocoddyl
The code in this directory uses the optimal control library Crocoddyl to generate 3 trajectories:
* simple jump
* yaw jump: jump with rotation along the *z* axis
* half backflip: backflip with a 180° rotation along the *y* axis

The problem descriptions and cost functions are written in *quadrupedal_jumping_problem.py*.

The problem to be solved is to be configured in *main.py*.

usage: `python3 main.py <display|plot|save>`
options:
* `display` displays the trajectory in gepetto-viewer
* `plot` generate plots about the generated trajectory
* `save` saves the configuration and control trajectories in a *npz* archive that can be converted and simulated with the scripts in the *quadruped-replay* directory