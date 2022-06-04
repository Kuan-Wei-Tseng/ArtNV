# Download RAFT models
cd RAFT
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
cd ..

# Install Quaternion Package for Synsin
conda install -y -c conda-forge quaternion 

# Download SynSin models
cd synsin
mkdir ./modelcheckpoints/
cd modelcheckpoints/
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/readme.txt

mkdir realestate
cd realestate
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/realestate/synsin.pth
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/realestate/zbufferpts.pth
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/realestate/viewappearance.pth
wget https://dl.fbaipublicfiles.com/synsin/checkpoints/realestate/tatarchenko.pth