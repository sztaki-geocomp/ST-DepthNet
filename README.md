# ST-DepthNet: A spatio-temporal deep network for depth completion using a single non-repetitive circular scanning Lidar
Supplementary material to our **[paper](https://ieeexplore.ieee.org/document/10100860)** in the IEEE Robotics and Automation Letters (RAL).

## Motivation
The project's purpose is effective deep learning based depth completion for sparse measurements captured by the **[Livox AVIA](https://www.livoxtech.com/avia)** sensor.
<p align="center">
<img src="https://user-images.githubusercontent.com/25935749/204334825-178055ee-0e52-41e4-bb4a-b0790be0846c.png" width="250" height="250">
<img src="https://user-images.githubusercontent.com/25935749/204334800-da2080f4-640f-402f-835d-9601a795c2b1.png" width="250" height="250">
</p> 
<p align="center">
A sparse measurement by the Livox AVIA sensor (left) and the completed depth image by ST-DepthNet (right)
</p> 

## Citation
If you found this work helpful for your research, or use some part of the code or the datasets, please cite our paper:

```text
@article{st-depthnet,
	author = {Örkény Zováthi and Balázs Pálffy and Zsolt Jankó and Csaba Benedek},
	title = {ST-DepthNet: A spatio-temporal deep network for depth completion using a single non-repetitive circular scanning Lidar},
	journal = {IEEE Robotics and Automation Letters},
	year={2023},
	volume={},
	number={},
	pages={1-8},
	doi={10.1109/LRA.2023.3266670},
}
```

## LivoxCarla Dataset
We provide training, validation and test data with ground truth information from the **[Carla simulator](https://carla.org/)**. The dataset consists of 11726 randomly sampled sparse input-dense output range image pairs. Each sample consists of a sequence of five consecutive range images with height and width of 400 pixels. The dataset is arranged in three folders named:
* Train: 10000 sparse samples with ground truth data,
* Validation: 500 sparse samples with ground truth data,
* Test: 1226 sparse samples with ground truth data.
### Dataset availability

The LivoxCarla Dataset can be downloaded from this **[link](https://drive.google.com/drive/folders/1l2_r4Ajf5xX2x3AFeYUUOMG6SP4kl33W?usp=sharing)**.

### Dataset structure
We provide the data both in raw image and in video format. The video format can be directly used by our code for training and inference, just copy the .avi files in the dataset subfolder of the project.

    ├── LivoxCarla                    # Main dataset folder
    │   ├── Train                     # Contains 10000 samples
    │   │   ├── train.zip             # Samples in raw image format
    │   │   ├── train.avi             # Samples in video format (directly usable by the code)
    │   ├── Validation                # Contains 500 samples 
    │   │   ├── validation.zip        # Samples in raw image format
    │   │   ├── validation.avi        # Samples in video format (directly usable by the code)
    │   ├── Test                      # Contains 1226 samples 
    │   │   ├── test.zip              # Samples in raw image format
    │   │   ├── test.avi              # Samples in video format (directly usable by the code)
    └── ...


### Creating the dataset
The dataset was generated from the **[Carla simulator](https://carla.org/)** that gives the opportunity to export perfect depth images without any distortion or blurring. To simulate realistic Livox AVIA measurements, the dense depth images were sampled with rosetta scanning pattern of the Livox AVIA sensor. During the whole data recording, the capturing platform (a simulated) vehicle was dynamically moving. To augment on the extractable information (e.g., vary the ground level), the capturing sensor’s position was randomly rotated along the up axis by [−22.5°, 22.5°], and its height was randomly adjusted between [1.5m, 2.5m].
The final dataset consists of 11726 randomly sampled input-output data pairs. 
* In each sample, the **input data** is a sparse depth image sequence, that consists of five consecutive sparse depth images, each sampled after 200 ms. The implementation of the process is shown in the figure below, where the patterns of the Livox AVIA sensor (displayed in the middle) are used to filter the depth image exported from the simulator resulting in realistic, Livox-like depth images.
<p align="center">
<img src="https://user-images.githubusercontent.com/50795664/186405171-e2f94d63-1101-4050-ab0a-dae11379186c.png" width="250" height="250">
<img src="https://user-images.githubusercontent.com/50795664/186404712-14b84618-45e5-4ad4-8309-77635e8dcdd0.gif" width="250" height="250">
<img src="https://user-images.githubusercontent.com/50795664/186405633-cd98fcea-b37c-44cc-a148-271231454a1b.png" width="250" height="250">
</p> 

* The **output (ground truth) data** was generated at the end of each input sequence using the mask with the full field of view of the Livox AVIA sensor.

<p align="center">
<img src="https://user-images.githubusercontent.com/50795664/186405171-e2f94d63-1101-4050-ab0a-dae11379186c.png" width="250" height="250">
<img src="https://user-images.githubusercontent.com/50795664/186407960-4635bd37-bdfa-417b-aa46-d1571ca995be.png" width="250" height="250">
<img src="https://user-images.githubusercontent.com/50795664/186408003-466c4198-4bde-4cf5-b090-7f6e87d6037b.png" width="250" height="250">
</p>

A sample of the training data presenting the five consequtive sparse depth images and one ground truth data is displayed in the following video:
<div align="center">
  <video src="https://user-images.githubusercontent.com/50795664/202659837-8fe5a97b-5061-4583-be04-46769d074856.mp4" width=400/>
</div>

## LivoxBudapest Dataset for real-life testing

We also provide three real-life measurement sequences in video format and code for testing the trained models in real-world scenarios. The LivoxBudapest Dataset can be downloaded from this **[link](https://drive.google.com/drive/folders/1KOE8xZWPQ06S3T-QlLQ0NvDEs9QACB1z?usp=sharing)**.

### Demo videos

The depth map prediction results by the considered reference techniques and the ST-DepthNet method can be viewed for the complete sequences in the enclosed **[YouTube video streams](https://www.youtube.com/playlist?list=PLpBYjniN2wQnVvwYMHWXiljRP6pjHGalU)**, which also contain an RGB image channel for visual verification.

Methods displayed in the videos:
* Narrow integration time (200ms): Measurements of the Livox Avia sensor grouped by each 200 ms.
* Large integration time (1000ms): Measurements of the Livox Avia sensor grouped by each 1000 ms.
* **[IP-Basic++](https://github.com/kujason/ip_basic)**: An improved version of the IP-Basic reference depth completion method.
* **[Sparse-to-Dense](https://github.com/kocchop/depth-completion-gan)**: A GAN-based reference depth completion method.
* Proposed ST-DepthNet: Depth estimations by the proposed model.

There is an RGB camera recording in each video, only for better visualization. A few frame difference can be experienced compared to the depth streams.

#### Example video

<div align="center">
  <video src="https://user-images.githubusercontent.com/50795664/220680457-20ad23f9-de75-441d-b82a-813ec7012e10.mp4" width=400/>
</div>





## Implementation

### Environment setup
The project was implemented on Ubuntu 18.04 with CUDA 10.2 (with compatible CUDNN) using a 8GB GeForce 1080 Ti GPU. All codes were implemented in python 3.7.0 with packages tensorflow-gpu 1.13.1 and keras-gpu 2.3.1 using conda virtual environment.

### Training
To train the proposed ST-DepthNet model on our LivoxCarla training dataset, simply run:

```sh
python train.py --n_epochs=10 --batch_size=1 --image_size=400 --save_model=./models/model
```

### Pretrained model
A pretrained model of the proposed ST-DepthNet can be downloaded from the following **[link](https://drive.google.com/file/d/1koARFH28yDTci8RElugh-oM2Kso89Vgo/view?usp=sharing)**. 

### Inference
There are three ways to plot the model's predictions using the **[visualize.py](visualize.py)** file.

* For plotting input, ground truth and prediction in one row, run:
```sh
python visualize.py --plot_type=0
```
<p align="center">
<img src="https://user-images.githubusercontent.com/50795664/194721902-f70814d0-3b50-4126-9f6d-607b57bf3040.png" width="1000">
</p>

* For plotting and saving only the model predictions, run:
```sh
python visualize.py --plot_type=1
```
<p align="center">
<img src="https://user-images.githubusercontent.com/50795664/194721935-2f53202c-3cac-4cd1-8a9b-538151252769.png" width="300" height="300">
</p>

* For plotting the prediction of two model variants for comparison, run:
```sh
python visualize.py --plot_type=2
```
<p align="center">
<img src="https://user-images.githubusercontent.com/50795664/194721974-405b3d85-9111-4275-9f5e-719e0d4dde98.png" width="500" height="500">
</p>

### Additional scripts
Additional scripts are provided in the **[utils](utils)** folder for supporting more easier understanding and usage of this repository. Using this may require installing further libraries like openCV.
* To project the grayscale depth images into a 3D point clouds, run the **[Depth_to_cloud.py](utils/Depth_to_cloud.py)** file:
```sh
python Depth_to_cloud.py ./path/depth_image.png
```
<p align="center">
<img src="https://user-images.githubusercontent.com/50795664/199698783-c3283eb7-0ccc-4342-abf2-9c1949a36598.png" width="250" height="250">
<img src="https://user-images.githubusercontent.com/50795664/199698688-1200f920-8dd4-4045-8c55-80b413030b0f.png" width="320" height="250">
</p>

* For constructing the required video formats for your custom data, you can use the **[Video_maker.py](utils/Video_maker.py)** file.

## Authorship declaration
This repository was implemented in the [Machine Perception Research Laboratory](https://www.sztaki.hu/en/science/departments/mplab), Institute of Computer Science and Control (SZTAKI), Budapest.<br>
<img src="https://user-images.githubusercontent.com/50795664/195994236-1579001a-e78e-4638-9cbe-496d4b9a73d2.png" width="200">
