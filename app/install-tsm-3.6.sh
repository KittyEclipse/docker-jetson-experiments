#!/bin/bash
set -e
TEMP_DIR=$(mktemp -d)
pushd "$TEMP_DIR"

apt update && apt-get install -y \
	python3-pip \
	libopenblas-base \
	libopenmpi-dev \
	libomp-dev \
	libjpeg-dev \
	zlib1g-dev \
	libpython3-dev \
	libopenblas-dev \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	llvm \
	protobuf-compiler \
	libprotoc-dev

wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3.6 install 'Cython<3'
pip3.6 install numpy torch-1.10.0-cp36-cp36m-linux_aarch64.whl

git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
pushd torchvision
export BUILD_VERSION=0.11.1
python3.6 setup.py install --user
popd

git clone -b v0.6 https://github.com/apache/incubator-tvm.git incubator-tvm
pushd incubator-tvm
git submodule update --init
mkdir build && cp cmake/config.cmake build/ && pushd build
sed -i config.cmake 's/USE_CUDA OFF/USE_CUDA ON/' && sed -i config.cmake 's/USE_LLVM OFF/USE_LLVM ON/'
cmake ..
make -j4
popd
pushd python && python3.6 setup.py install && popd
pushd topi/python && python3.6 setup.py install && popd

pip3.6 install onnx==1.11.0 onnxoptimizer==0.2.7 onnxruntime==1.7.0 onnx-simplifier==0.3.5
export PATH=$PATH:/usr/local/cuda/bin

popd
