# CMFN
A novel point cloud completion method guided by rotational image
This repository contains the official implementation for "Research on Multimodal Point Cloud Completion Algorithm Guided by Image Rotation Attention" paper, which is submitted to Romote Sensing

##Introduction
This paper proposes a novel LiDAR point cloud multi-scale completion algorithm guided by image rotation attention mechanisms to address the challenges of severe information loss and suboptimal fusion effects in multimodal feature extraction and integration during point cloud shape completion. The proposed network employs an encoder-decoder structure, integrating a Rotating Channel Attention (RCA) module for enhanced image feature extraction and a multi-scale feature extraction method for point clouds to improve both local and global feature information. The network also utilizes multi-level self-attention mechanisms to achieve effective multimodal feature fusion. The decoder employs a multi-branch completion method, guided by chamfer distance loss, to accomplish the point cloud completion task. Extensive experiments on the ShapeNet-ViPC and ModelNet40ViPC datasets demonstrate the effectiveness of the proposed algorithm. Compared to eight related algorithms, the proposed method achieves superior performance in terms of completion accuracy and efficiency. Specifically, compared to the state-of-the-art XMFnet, the average Chamfer Distance (CD) value is reduced by 11.71%. The algorithm also shows significant improvements in visual comparisons, with fewer outliers, more distinct structural details, and a more uniform density distribution in the completed point clouds. The ablation studies further validate the effectiveness of the RCA module and the multi-scale module, highlighting their complementary nature in enhancing point cloud completion accuracy. Future work will focus on improving the network's performance and exploring its application in more complex 3D vision tasks.

##Requirements
The code has been developed with the following dependencies:
Python 3.8
CUDA version 10.2
G++ or GCC 7.5.0
PyTorch 1.10.2


To setup the environment and install all the required packages run:

sh setup.sh


It automatically creates the environment and installs all the required packages.
If something goes wrong, please consider following the steps in setup manually.


##Dataset
The dataset is borrowed from "View-guided point cloud completion".
First, please download the ShapeNetViPC-Dataset (Dropbox, Baidu (143GB, code: ar8l)). Then run cat ShapeNetViPC-Dataset.tar.gz* | tar zx, you will get ShapeNetViPC-Dataset contains three folders: ShapeNetViPC-Partial, ShapeNetViPC-GT, and ShapeNetViPC-View.
For each object, the dataset includes partial point cloud (ShapeNetViPC-Partial), complete point cloud (ShapeNetViPC-GT), and corresponding images (ShapeNetViPC-View) from 24 different views. You can find the details of 24 camera views in /ShapeNetViPC-View/category/object_name/rendering/rendering_metadata.txt.
In the "dataset" folder of this project, you can find the train and test lists, which are the same as the original ones, except for the formatting style.
Further partialized inputs used for the weakly supervised setting are also available for download (Dropbox).


##Training
The file config.py contains the configuration for all the training parameters.
To train the models in the paper, run this command:


python train.py


##Evaluation

To evaluate the models (select the specific category in config.py):

python eval.py


##Acknowledgements
Some of the code is borrowed from AXform.


Visualizations have been created using Mitsuba 2.
##Citation
If you find our work useful in your research, please consider citing:


##bibtex
@inproceedings{gu2025cmfn,
 author = {Gu, Shangtai and Xu, Ke and Wan, Jianwei and Hou, Baolin and Ma, Yanxin}
 booktitle = {IMDPI Remote Sensing},
 title = {Research on Multimodal Point Cloud Completion Algorithm Guided by Image Rotation Attention},
 year = {2025}
}
##License
Our code is released under MIT License (see LICENSE file for details).
