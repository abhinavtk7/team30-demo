# TEAM-30 model

This repository contains a DOLFINx implementation of the [TEAM 30 model](http://www.compumag.org/jsite/images/stories/TEAM/problem30a.pdf).

- `generate_team30_meshes.py`: A script that generates the two TEAM 30 models (single and three phase) meshes and saves them to xdmf format. To learn about input parameters, run `python3 generate_team30_meshes.py --help`.
- `team30_A_phi.py`: Script for solving the TEAM 30 model for either a single phase or three phase engine. To learn about input parameters, run `python3 team30_A_phi.py --help`.

## Dependencies
### Progress bar
We use `tqdm` for progress bar plots. This package can be installed with 
```bash
pip3 install tqdm
```
### Mesh generation
To generate the meshes, `gmsh>=4.8.0` is required, alongside with `mpi4py`, `h5py` and `meshio`. 
To install the `meshio` and `h5py` in the `DOLFINx` docker container call:
```bash
export HDF5_MPI="ON"
export CC=mpicc
export HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/mpich/"
pip3 install --no-cache-dir --no-binary=h5py h5py meshio
```