# ArtNV
This is the official implementation of "Artistic Style Novel View Synthesis Based on A Single Image" (CVPR 2022 Workshop). The code will be available in **mid June**, 2022.

## Installation
Our codes are tested on the following environemnt:
* Python=3.8
* PyTorch=1.8.0
* PyTorch3D = 0.6.2

1. Create a new virtual environment with conda.
```
conda create -n artnv python=3.8
conda activate artnv
```
2. Install PyTorch with CUDA support.

Remember to check the compatbility between Python, PyTorch, and Pytorch3D. Do not use PyTorch 1.8.3 (LTS) as it is not compatbible with PyTorch3D.
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
3. Install PyTorch3D
 
Follow the installation guideline in the official documents. 
```
pip install scikit-image matplotlib imageio plotly opencv-python
pip install black usort flake8 flake8-bugbear flake8-comprehensions
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

4. Clone the project.
git clone --recurse-submodules

### Reference
If you find this code helpful, please cite
```
@InProceedings{Tseng_2022_CVPRW,
    author    = {Tseng, Kuan-Wei and Lee, Yao-Chih and Chen, Chu-Song},
    title     = {Artistic Style Novel View Synthesis Based on A Single Image},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
}
```
