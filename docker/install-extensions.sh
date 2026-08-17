#!/bin/bash
set -euo pipefail

mkdir -p /tmp/extensions

if [ ! -d /tmp/extensions/nvdiffrast/.git ]; then
  git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
fi
pip install /tmp/extensions/nvdiffrast --no-build-isolation

if [ ! -d /tmp/extensions/nvdiffrec/.git ]; then
  git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
fi
pip install /tmp/extensions/nvdiffrec --no-build-isolation

if [ ! -d /tmp/extensions/CuMesh/.git ]; then
  git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
fi
pip install /tmp/extensions/CuMesh --no-build-isolation

if [ ! -d /tmp/extensions/FlexGEMM/.git ]; then
  git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
fi
pip install /tmp/extensions/FlexGEMM --no-build-isolation

if [ ! -d /tmp/extensions/o-voxel/.git ]; then
  cp -r /app/o-voxel /tmp/extensions/o-voxel
fi
pip install /tmp/extensions/o-voxel --no-build-isolation
