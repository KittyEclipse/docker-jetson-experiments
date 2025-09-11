#!/bin/bash
set -e
TEMP_DIR=$(mktemp -d)
pushd "$TEMP_DIR"

apt update && apt-get install -y \
	python3-pip \
	libhdf5-serial-dev \
	hdf5-tools \
	libhdf5-dev \
	zlib1g-dev \
	zip \
	libjpeg8-dev \
	liblapack-dev \
	libblas-dev \
	gfortran \
	libportaudio2

pip3 install --upgrade pip
pip3 install -U testresources setuptools==65.5.0
pip3 install -U numpy==1.22 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging h5py==3.7.0

#export JP_VERSION=461
#export TF_VERSION=2.7.0
#export NV_VERSION=22.01
#pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION tensorflow==$TF_VERSION+nv$NV_VERSION
#pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512 tensorflow==2.12.0+nv23.06
pip3 install tensorflow[and-cuda]
pip3 install --upgrade tflite-runtime

popd
