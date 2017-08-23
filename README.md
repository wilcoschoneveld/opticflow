# Optical Flow with Convolutional Neural Networks for Vision-Based Guidance of UAS

Modern research in the area of unmanned aerial systems (UAS) is pushing the level of autonomy to new
limits. Scenarios without GPS, such as indoor environments, are especially challenging for autonomous
navigation. These days, most aerial vehicles are equipped with camera sensors which make vision-
based guidance an appealing solution. Velocity state can be estimated from optical flow, which is
typically implemented with feature based methods. These methods rely on detectable features and lack
in robustness. Current advancements in deep learning and convolutional neural networks (CNNs) have
led to many achievements in the area of computer vision. This repository houses two CNN architectures
which can estimate optical flow from two input frames. The models are trained and evaluated on a
custom generated dataset and are shown to outperform the Lucas-Kanade method. The networks are
sized such that they can run in real time on a Parrot Bebop 2 quadrotor. No in-the-loop validation has
been performed.

![Results](https://github.com/wilcoschoneveld/opticflow/raw/master/checkpoints/results.png)

The project has been developed with Python 3.6 and TensorFlow 1.2.1. The following table provides a list of all
the runnable scripts in this repository with a description of it's purpose. You can run a script by installing
and activating a virtualenv and running the command `python -m scripts.<script>`.

| Script        | Description           | 
| ------------- |---------------|
| `accuracy.py`  | Attempt to add an accuracy estimator to the network architecture |
| `evaluate.py`    | Evaluate CNN and CNN-split architecture against FAST+LK and zero prediction    | 
| `overfit.py` | Overfit the network on a very small subset of the data      |
| `plotting.py` | Two examples of how to plot data stored with tensorflow summaries |
| `prototype.py` | Training of the initial prototype architecture |
| `prune.py` | Prune ~15% of the network architecture weights and evaluate on the test data |
| `train.py` | Train a CNN to predict optical flow from a data generator |
| `video.py` | Evaluate the CNN model in real-time on a webcam and compare with FAST+LK |
| `weights.py` | Visualize the weights of the convolution layers in the network |
