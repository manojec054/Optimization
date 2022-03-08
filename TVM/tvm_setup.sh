# https://github.com/manojec054/Optimization.git
# sudo apt-get install llvm-10*

cd ~
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
git submodule init
git submodule update

conda env create --file conda/build-environment.yaml
conda activate tvm-build

mkdir -p ~/tvm/build
cp ~/Optimization/TVM/config.cmake ~/tvm/build/
cd ~/tvm/build

sudo apt-get -y install llvm-10*

cmake ..
make -j9
conda activate pytorch_p38
export LD_LIBRARY_PATH=~/tvm/build:$LD_LIBRARY_PATH
export TVM_HOME=~/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}