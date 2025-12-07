pip install gdown

mkdir -p data/ModelNet40Ply2048/
cd data/ModelNet40Ply2048
gdown --fuzzy https://drive.google.com/file/d/1kjmbqe-cPCojxzVJDlHCbIl6lCluD_EM/view?usp=share_link
unzip modelnet40_ply_hdf5_2048.zip
cd ../../
mkdir -p data/S3DIS/
cd data/S3DIS
gdown --fuzzy https://drive.google.com/file/d/14FdvE02kMUde4dLWlCH_ZVsHVpgwdBlP/view?usp=share_link
tar -xvf s3disfull.tar
cd ../../