# PoseNet-Keras-Modified
**Implementation of [PoseNet](http://mi.eng.cam.ac.uk/projects/relocalisation/)** using Keras/Tensorflow.

Modified to be able to train and predict between images with different aspect ratios accurately.

Predicts to within 3m/3° for images with the same aspect ratio and 10m/20° for images with different aspect ratios. Original Posenet architecture returns over 25m/30° of error for images with different aspect ratios. 

If the area you are using this model on is large, divide it into cells using get_data.py and train them individually.

As it was removed from the original paper's repository, posenet.npy can be downloaded here: https://drive.google.com/drive/u/0/folders/1POHVS1i_kBaeg0OQeZ9De3sQYc47fSqJ

## Setup
- Python version 3.6 or 3.7
- h5py                   2.10.0
- Keras                  2.0.6
- matplotlib             3.5.2
- numpy                  1.19.5
- opencv-python          4.6.0
- pandas                 1.3.5
- Pillow                 9.2.0
- scipy                  1.7.3
- tensorflow             1.15.0
- torch		           1.8.2
- transforms3d           0.3.1

### Files/Folders
    keras-posenet
    ├── images/
    ├── augmented_images/
    ├── query_imgs_grayscale/
    └── keras-posenet/
        ├── NewData/
        ├── trained_weights
        ├── posenet.npy
        ├── augment.py
        ├── evaluate.py
        ├── get_data.py
        ├── helper.py
        ├── posenet.py
        ├── predict.py
        ├── train.py
        ├── README.md
        ├── all_img_data.csv
        └── query_img_data.csv
1. keras_posenet: Working directory with all .py, .h5, .npy and .csv files 
2. NewData: Folder where get_data.py sends the csv files with cell data to 
3. images: Folder containing original training images
4. augmented_images: Folder containing augmented training images
5. query_imgs_grayscale: Folder containing query images after being converted to grayscale
