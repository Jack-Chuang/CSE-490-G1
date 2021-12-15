# CSE 490G1 Final Project — Novel View Synthesis on real world scenes

## Final Presentation Video

[Here is the link to my final presentation](https://youtu.be/nZx4aAZw_n0)

## Github repository

[This is the link to the Github repository](https://github.com/Jack-Chuang/CSE-490-G1/tree/main/final%20project)

## Project summary

- Implemented [NeRF]([final project/NeRF.pdf](https://github.com/Jack-Chuang/CSE-490-G1/blob/49632543203b1f6369a5d04d41dfba719e90b655/final%20project/NeRF.pdf)) and [D-NeRF]([https://github.com/Jack-Chuang/CSE-490-G1/blob/49632543203b1f6369a5d04d41dfba719e90b655/final project/D-NeRF.pdf](https://github.com/Jack-Chuang/CSE-490-G1/blob/49632543203b1f6369a5d04d41dfba719e90b655/final%20project/D-NeRF.pdf)) using [Pytorch]([https://pytorch.org/](https://pytorch.org/)) based on their official Pytorch implementation.
- Recorded seven videos including two static scenes and five dynamic scenes.
- These videos were processed into training data.
- Then, trained different NeRF and D-NeRF models on different scenes.
- D-NeRF: ≈ **200 hours**, ≈ **40 hours per scene**, 5 dynamic scenes, 400k iterations
- NeRF: ≈ **175 hours**, ≈ **25 hours per scene**, 5 dynamic scenes, 2 static scenes, 400k iterations.
- GPU: Nvidia Tesla P100
- Quantitatively and qualitatively, compared each novel view synthesis of each scene between its trained NeRF and D-NeRF models.

## Problem Description

This project looks at reproducing the implementation of NeRF and D-NeRF on static and dynamic scenes and comparing the two models on real-world data. One of the novelties of this project is performing the D-NeRF model on real-world scenes, where the original code from the author only deals with blender-type data(in virtual scenes). I am hoping to render a novel view synthesis of dynamic scenes of real-world objects by using D-NeRF and comparing it with the result of NeRF model on real-world data to prove that D-NeRF is indeed a real-world solution for dynamic scenes.

## Project setup

This project uses Pytorch as the machine learning framework and uses [Tensorboard]([https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)) to track training progress. For training, I used Google Colab to run my training scripts.

The basis of my code is NeRF and D-NeRF projects, so my codebase can be set up by following the installation instructions from the [NeRF repo]([https://github.com/bmild/nerf](https://github.com/bmild/nerf)) and [D-NeRF repo]([https://github.com/albertpumarola/D-NeRF](https://github.com/albertpumarola/D-NeRF)). 

## Dataset

The dataset I used is all recorded by myself. [Here is the link to all the raw data] ([https://drive.google.com/drive/folders/1AFTXyHYdJQlx8aowH7IWz50mXpVNmEXd?usp=sharing](https://drive.google.com/drive/folders/1AFTXyHYdJQlx8aowH7IWz50mXpVNmEXd?usp=sharing)) I used for this project.

## Techniques

I used my mobile phone to record video as my raw dataset. Next, I crop those videos into frames and feed them into [COLMAP]([https://colmap.github.io/](https://colmap.github.io/)). COLMAP is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline with a graphical and command-line interface. It offers a wide range of features for the reconstruction of ordered and unordered image collections. NeRF and an older view synthesis technology [LLFF]([https://github.com/Fyusion/LLFF](https://github.com/Fyusion/LLFF)) also use COLMAP to generate estimated camera poses in each frame and the focal length of the camera.

 ![focal length]([https://github.com/Jack-Chuang/CSE-490-G1/blob/da7f114a298da0237033e84eb4b84b5d4ba3140d/final project/figures/ray direction.PNG](https://github.com/Jack-Chuang/CSE-490-G1/blob/da7f114a298da0237033e84eb4b84b5d4ba3140d/final%20project/figures/ray%20direction.PNG))

With this information, we are able to calculate the ray directions according to each camera pose and its focal length. Next, feed these data into NeRF model and D-NeRF model for training purposes. 

The NeRF model is composed of two embedding layers and a ten-layer MLP with ReLU in between each layer and Sigmoid as the activation function of the last layer. The first embedding layer augments a position vector of dimension D to a (N * 2 + 1) * D dimension and feeds it into the fifth MLP layer. N is a hyperparameter that we can tune for. The second embedding layer augments a direction vector of dimension D to an (M * 2 + 1) * D dimension and feeds it into the ninth MLP layer. Here M is also a hyperparameter that we can tune for. The sigma value is output from the ninth MLP layer and the RGB value is output from the last MLP layer. 

![The NeRF model]([https://github.com/Jack-Chuang/CSE-490-G1/blob/0f09258228837dcc340dc4e89a8a07966bd582dd/final project/figures/NeRF model.PNG](https://github.com/Jack-Chuang/CSE-490-G1/blob/0f09258228837dcc340dc4e89a8a07966bd582dd/final%20project/figures/NeRF%20model.PNG))   

D-NeRF is basically a NeRF model adding a deformation model in front. If the input is the frame at time t = 0, it will be passed into the NeRF model directly, or else, it will be first passed into the deformation model and get the position change with respect to time. This delta position is then fed into the NeRF model to get the sigma value and the RGB value. The sigma values and the RGB values are then used for rendering the views. 

![The D-NeRF model]([https://github.com/Jack-Chuang/CSE-490-G1/blob/8d22c3da0bb159ff355baffa64fe43da527f95de/final project/figures/D-NeRF model.PNG](https://github.com/Jack-Chuang/CSE-490-G1/blob/8d22c3da0bb159ff355baffa64fe43da527f95de/final%20project/figures/D-NeRF%20model.PNG))

For the rendering, the sigma value and the RGB value output from the NeRF and D-NeRF model is then passed into the following equation, where the result is the color of the specific pixel: 

![rendering formula]([https://github.com/Jack-Chuang/CSE-490-G1/blob/f2d595cd83eabc56ff84bd234eea1af58252d0b7/final project/figures/rendering formula.png](https://github.com/Jack-Chuang/CSE-490-G1/blob/f2d595cd83eabc56ff84bd234eea1af58252d0b7/final%20project/figures/rendering%20formula.png))

The full picture of the processes is shown below:

![rendering]([https://github.com/Jack-Chuang/CSE-490-G1/blob/f2d595cd83eabc56ff84bd234eea1af58252d0b7/final project/figures/rendering.PNG](https://github.com/Jack-Chuang/CSE-490-G1/blob/f2d595cd83eabc56ff84bd234eea1af58252d0b7/final%20project/figures/rendering.PNG)) 

## Results

The results are better illustrated with videos. Here is the original video of one of the datasets 

[![kim_move1_original]([https://github.com/Jack-Chuang/CSE-490-G1/blob/75274cb096e9000894198538653488e6e1695708/final project/figures/kim_move1_original.PNG](https://github.com/Jack-Chuang/CSE-490-G1/blob/75274cb096e9000894198538653488e6e1695708/final%20project/figures/kim_move1_original.PNG))]([https://youtu.be/ir6A_BDieqQ](https://youtu.be/ir6A_BDieqQ))

and here is the result of NeRF and D-NeRF trained on this dataset.

[![kim_move1_with_psnr]([https://github.com/Jack-Chuang/CSE-490-G1/blob/b4e02ca9b9afe7bb8c3b7e31b716c3284e0de616/final project/figures/kim_move1_with_psnr.PNG](https://github.com/Jack-Chuang/CSE-490-G1/blob/b4e02ca9b9afe7bb8c3b7e31b716c3284e0de616/final%20project/figures/kim_move1_with_psnr.PNG))]([https://youtu.be/4-jw85CfGeY](https://youtu.be/4-jw85CfGeY)) 

The psnr value is a measure of the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representations. The formula is as follow: 

![psnr formula]([https://github.com/Jack-Chuang/CSE-490-G1/blob/f2d595cd83eabc56ff84bd234eea1af58252d0b7/final project/figures/psnr formula.png](https://github.com/Jack-Chuang/CSE-490-G1/blob/f2d595cd83eabc56ff84bd234eea1af58252d0b7/final%20project/figures/psnr%20formula.png)) 

Here is an example of rendering the dynamic scenes from a fixed camera pose.

[![kim_move1_fixed]([https://github.com/Jack-Chuang/CSE-490-G1/blob/75274cb096e9000894198538653488e6e1695708/final project/figures/kim_move1_fixed.PNG](https://github.com/Jack-Chuang/CSE-490-G1/blob/75274cb096e9000894198538653488e6e1695708/final%20project/figures/kim_move1_fixed.PNG))]([https://youtu.be/M1ecNLSBhBE](https://youtu.be/M1ecNLSBhBE))

As you can see that the D-Nerf model had successfully captured the movement of the person's arm, while NeRF can only render a static representation of the view from that particular camera pose.

Here is another result of D-NeRF. 

[![cat D-NeRF]([https://github.com/Jack-Chuang/CSE-490-G1/blob/d37af92e4982c28d83a921600cca43b4d35afdb7/final project/figures/cat D-NeRF.jpg](https://github.com/Jack-Chuang/CSE-490-G1/blob/d37af92e4982c28d83a921600cca43b4d35afdb7/final%20project/figures/cat%20D-NeRF.jpg))]([https://youtu.be/0sZmIKpSa7c](https://youtu.be/0sZmIKpSa7c)) 

and here is the original video for comparison.

[![cat original]([https://github.com/Jack-Chuang/CSE-490-G1/blob/a98a167c6bbe1359e7fc4aa09d675526288b612f/final project/figures/cat original.jpg](https://github.com/Jack-Chuang/CSE-490-G1/blob/a98a167c6bbe1359e7fc4aa09d675526288b612f/final%20project/figures/cat%20original.jpg))]([https://youtu.be/QtX6AZ-7hLU](https://youtu.be/QtX6AZ-7hLU)) 

As you can see the D-NeRF model has successfully captured the cat turning its head. 

The following is the psnr for every iteration of training. I used this to track the training progress. I save an event file for Tensorboard every 100k iterations. As you can see the value of psnr is increasing, which means the model is learning better and better through each iteration. The loss, on the other hand is decreasing over each iteration, this is also an indication of how well the model is performing. 

![psnr]([https://github.com/Jack-Chuang/CSE-490-G1/blob/a98a167c6bbe1359e7fc4aa09d675526288b612f/final project/figures/psnr.PNG](https://github.com/Jack-Chuang/CSE-490-G1/blob/a98a167c6bbe1359e7fc4aa09d675526288b612f/final%20project/figures/psnr.PNG))

![loss]([https://github.com/Jack-Chuang/CSE-490-G1/blob/a98a167c6bbe1359e7fc4aa09d675526288b612f/final project/figures/loss.PNG](https://github.com/Jack-Chuang/CSE-490-G1/blob/a98a167c6bbe1359e7fc4aa09d675526288b612f/final%20project/figures/loss.PNG))

After successfully rendering the scenes, I've rendered the static scenes in Unity. 

[![Unity rendering]([https://github.com/Jack-Chuang/CSE-490-G1/blob/14873844f4eee90f8bffa29061bb8a2dc8e18ebe/final project/figures/Unity rendering.jpg](https://github.com/Jack-Chuang/CSE-490-G1/blob/14873844f4eee90f8bffa29061bb8a2dc8e18ebe/final%20project/figures/Unity%20rendering.jpg))]([https://youtu.be/SfB9ONEs3os](https://youtu.be/SfB9ONEs3os))

## Future aspects

I followed a [tutorial]([https://github.com/kwea123/nerf_Unity](https://github.com/kwea123/nerf_Unity)) that I found on Github to render my trained NeRF models for two static scenes in [Unity]([https://unity.com](https://unity.com/)) and plan to render D-NeRF models for real-world dynamic scenes in Unity.
