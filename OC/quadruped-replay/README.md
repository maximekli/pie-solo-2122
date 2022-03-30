### Useful scripts
*convert_npz_from_crocoddyl.py*: takes a trajectory generated with Crocoddyl and turns it into something usable with the PyBullet replayer. It generates an archive prefixed with "converted".

usage: `python3 convert_npz_from_crocoddyl.py simple_jumping.npz`

*trim_pad_npz.py*: takes a trajectory archive, cuts it at *knots1*, pads it from *knots2* to *knots* with the landing configuration and generates the transition trajectory between *knots1* and *knots2*. *knots1*, *knots2*, *knots* and the landing configuration are to be configured in the code. It generates an archive prefixed with "trimmed_padded".

usage: `python3 trim_pad_npz.py converted.simple_jumping.npz`

*add_gains_npz.py*: add gains to a trajectory archive. The pushing and landing gains, alongside the index at which to change them are to be configured in the code. It generates an archive prefixed with "with_gains".

usage: `python3 add_gains_npz.py trimmed_padded.converted.simple_jumping.npz`

### Main simulation script
*main_solo12_replay.py*: replays the trajectory from the archive configured in *Params.py*.

usage: `python3 main_solo12_replay.py`