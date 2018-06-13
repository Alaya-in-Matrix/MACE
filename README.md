# README

## About

Parallel bayesian optimization via multi-objective acquisition ensemble

## Python version

The code in this repo is the code I used to run the experiments for the paper,
I recently also implemented a python version that supports MCMC integration of
the GP hyperparameters, and it has less dependencies. The code is hosted [here](https://github.com/Alaya-in-Matrix/MACE_MCMC)

## Dependencies

- Publicly available:
    - Cmake (for build and install)
    - Eigen
    - Boost
    - OpenMP
    - nlopt
    - gsl
- Libraries written by me, used as git submodules:
    - [GP](https://github.com/Alaya-in-Matrix/GP)
    - [MVMO](https://github.com/Alaya-in-Matrix/MVMO)
    - [MOO](https://github.com/Alaya-in-Matrix/MOO)

## Build and install

```bash
mkdir _build
cd _build
cmake .. -DCMAKE_BUILD_TYPE=release                             \
         -DMYDEBUG=OFF                                          \ 
         -DBOOST_ROOT=/path/to/your/boost/library               \
         -DEigen3_DIR=/path/to/your/eigen/share/eigen3/cmake    \
         -DGSL_ROOT_DIR=/path/to/your/gsl                       \
         -DNLOPT_PATH=/path/to/your/nlopt                       \
         -DCMAKE_INSTALL_PREFIX=/path/you/want/to/install

make
make install
```
## Run

After successfully installed the MACE package, you should already have `mace_bo` in your path, you can go to `demo` and run the `run.sh` script

- Configurations are written in `conf`, the first line `workdir` should be modified
- The objective function is defined in `run.pl`
    - `run.pl` read the `param` file as design variables
    - `run.pl` write the objective value into `result.po`

## TODO

- Use TOML as config
- Constraint handling
