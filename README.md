# README

## About

Parallel bayesian optimization via multi-objective acquisition ensemble

## Dependencies

- Publicly available:
    - Eigen
    - Boost
    - OpenMP
    - nlopt
    - gsl
- Private libraries written by me, used as git submodules:
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

## TODO

- Use TOML as config
- Constraint handling
