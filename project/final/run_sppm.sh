#!/usr/bin/env bash
ROUNDS=5000
PHOTONS=1000000
CKPT_INTERVAL=10
METHOD=sppm
# If project not ready, generate cmake file.
if [[ ! -d build ]]; then
    mkdir -p build
    cd build
    cmake ..
    cd ..
fi

# Build project.
cd build
make -j
cd ..

# Run all testcases. 
# You can comment some lines to disable the run of specific examples.
# mkdir -p sppm_output
# mkdir -p sppm_output/scene06_bunny_1k_vn
# time bin/PA1 testcases/scene06_bunny_1k_vn.txt sppm_output/scene06_bunny_1k_vn $METHOD $ROUNDS $PHOTONS $CKPT_INTERVAL
# mkdir -p sppm_output/scene19_sibenik
# time bin/PA1 testcases/scene19_sibenik.txt sppm_output/scene19_sibenik $METHOD $ROUNDS $PHOTONS $CKPT_INTERVAL
# mkdir -p sppm_output/scene21_livingroom
# time bin/PA1 testcases/scene21_livingroom.txt sppm_output/scene21_livingroom $METHOD $ROUNDS $PHOTONS $CKPT_INTERVAL
# mkdir -p sppm_output/scene19_sibenik_lucy
# time bin/PA1 testcases/scene19_sibenik.txt sppm_output/scene19_sibenik_lucy $METHOD $ROUNDS $PHOTONS $CKPT_INTERVAL
mkdir -p sppm_output/scene19_sibenik_lucy_halton
time bin/PA1 testcases/scene19_sibenik.txt sppm_output/scene19_sibenik_lucy_halton $METHOD $ROUNDS $PHOTONS $CKPT_INTERVAL