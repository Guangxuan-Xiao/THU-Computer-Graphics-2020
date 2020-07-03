#!/usr/bin/env bash
SAMPLES=10
ROUNDS=1500
PHOTONS=200000
CKPT_INTERVAL=100
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
# time bin/PA1 testcases/scene20_diamond.txt output/scene20_diamond_rc.bmp rc $SAMPLES
# pt
# time bin/PA1 testcases/scene20_diamond.txt output/scene20_diamond_pt.bmp pt $SAMPLES

# time bin/PA1 testcases/scene20_diamond_r.txt output/scene20_diamond_r.bmp pt $SAMPLES
# time bin/PA1 testcases/scene20_diamond_g.txt output/scene20_diamond_g.bmp pt $SAMPLES
# time bin/PA1 testcases/scene20_diamond_b.txt output/scene20_diamond_b.bmp pt $SAMPLES
# python src/mergeRGB.py --r output/scene20_diamond_r.bmp --g output/scene20_diamond_g.bmp --b output/scene20_diamond_b.bmp --out output/scene20_diamond_dispersion.bmp

# # sppm
# mkdir -p sppm_output/scene20_diamond
# time bin/PA1 testcases/scene20_diamond.txt sppm_output/scene20_diamond sppm $ROUNDS $PHOTONS $CKPT_INTERVAL

# mkdir -p sppm_output/scene20_diamond_r
# time bin/PA1 testcases/scene20_diamond_r.txt sppm_output/scene20_diamond_r sppm $ROUNDS $PHOTONS $CKPT_INTERVAL

# mkdir -p sppm_output/scene20_diamond_g
# time bin/PA1 testcases/scene20_diamond_g.txt sppm_output/scene20_diamond_g sppm $ROUNDS $PHOTONS $CKPT_INTERVAL

mkdir -p sppm_output/scene20_diamond_b
time bin/PA1 testcases/scene20_diamond_b.txt sppm_output/scene20_diamond_b sppm $ROUNDS $PHOTONS $CKPT_INTERVAL

python src/mergeRGB.py --r sppm_output/scene20_diamond_r/result.bmp --g sppm_output/scene20_diamond_g/result.bmp --b sppm_output/scene20_diamond_b/result.bmp --out sppm_output/scene20_diamond_dispersion.bmp