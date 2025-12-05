#!/usr/bin/env bash
# command to install this enviroment: source init.sh

# install miniconda3 if not installed yet.
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
#source ~/.bashrc

# install PyTorch
conda deactivate
conda env remove --name openpoints
conda create -n openpoints -y python=3.7 numpy=1.20 numba
conda activate openpoints

# please always double check installation for pytorch and torch-scatter from the official documentation
conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia
#torch-scatter is a must
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
# pip install torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl

pip install -r requirements.txt

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# grid_subsampling library. necessary only if interested in S3DIS_sphere
# cd subsampling
# python setup.py build_ext --inplace
# cd ..


# # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
# cd pointops/
# python setup.py install
# cd ..

# Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
# cd chamfer_dist
# python setup.py install --user
# cd ../emd
# python setup.py install --user
# cd ../../../