cd ~
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
git submodule init
git submodule update

conda env create --file conda/build-environment.yaml
conda activate tvm-build

mkdir build
cp ~/Optimization/TVM/config.cmake ~/tvm/build/
cd ~/tvm/build
cmake ..
make -j9