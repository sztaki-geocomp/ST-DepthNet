# ST-DepthNet: A spatio-temporal deep network for depth completion using a single non-repetitive circular scanning Lidar
Supplementary material to our submitted paper in the IEEE Robotics and Automation Letters (RAL).

## Motivation
The project's purpose is effective deep learning based depth completion for sparse measurements captured by the **[Livox AVIA](https://www.livoxtech.com/avia)** sensor.
<p align="center">
<img src="https://user-images.githubusercontent.com/25935749/204334825-178055ee-0e52-41e4-bb4a-b0790be0846c.png" width="250" height="250">
<img src="https://user-images.githubusercontent.com/25935749/204334800-da2080f4-640f-402f-835d-9601a795c2b1.png" width="250" height="250">
</p> 
<p align="center">
A sparse measurement by the Livox AVIA sensor (left) and the completed depth image by ST-DepthNet (right)
</p> 

## LivoxCarla Dataset
We provide training, validation and test data with ground truth information from the **[Carla simulator](https://carla.org/)**. The dataset consists of 11726 randomly sampled sparse input-dense output range image pairs. Each sample consists of a sequence of five consecutive range images with height and width of 400 pixels. The dataset is arranged in three folders named:
* Train: 10000 sparse samples with ground truth data,
* Validation: 500 sparse samples with ground truth data,
* Test: 1226 sparse samples with ground truth data.
### Dataset availability

The dataset will be available after publication.
### Dataset structure
We provide the data both in raw image and in video format. The video format can be directly used by our code for training and inference.

    ├── Dataset                       # Main dataset folder
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

We also provide three real-life measurement sequences in video format and code for testing the trained models in real-world scenarios. The dataset will be available after publication.

## Models and training
The architecture of the proposed ST-DepthNet model with pretrained weights and the code for training and inference will be available after publication.

## Authorship declaration
This repository was implemented in the [Machine Perception Research Laboratory](https://www.sztaki.hu/en/science/departments/mplab), Institute of Computer Science and Control (SZTAKI), Budapest.<br>
<img src="https://user-images.githubusercontent.com/50795664/195994236-1579001a-e78e-4638-9cbe-496d4b9a73d2.png" width="200">
