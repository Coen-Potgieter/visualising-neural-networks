
# Visualising Neural Networks Learn 🧠🎓


## Overview 📌

In this project I explored the idea of treating Neural Networks as function approximators. Given some samples from a target function that the neural network knows nothing about, the model attempts to approximate the function. This project visualises that process and graphs its attempt at every iteration, resulting in a satisfying snap into place. 

The model neural network behind all these function approximations was developed by me using nothing but [NumPy](https://numpy.org/). This project gave me a great appreciation for neural networks (see my love letter below) as well as some interesting insights into model convergence and how much complexity in a model plays a role

## A Love Letter to Neural Networks ❤️

A function is a system of inputs that maps to some outputs, and we can plot this function on a graph. If we don’t know the function exactly but have some data available to us, that is what outputs produce what inputs, then we can reverse engineer the original function leveraging neural networks as function approximators, ie. Estimate the function based on patterns in the data. Neural networks are universal function approximators, meaning they can learn any function to any desired accuracy by adding more complexity to the model. This complexity includes adding more layers (depth), adding more neurons to each layer (width), using more complex activation functions, etc. 

What really fascinates me is how the complexity of a neural network is achieved. At their core, neural networks build complex functions by combining many linear equations that are stuck through some other non-linear functions (activation functions). I won't go into the equations for how the forward pass of a neural network is calculated, however it amazes me how simplistic it actually is while capturing the complexity does.

What's more is its applicability. Our entire world, generally, is made of different functions and anything a computer can processes (images, text, audio, video) can be represented as numbers, neural networks can learn to perform tasks like image classification, language translation, and more, as long as the problem can be expressed mathematically.

## Results 📸

This section is about various discoveries or observations while playing around with this program. But if you're just here for something to look at then enjoy the GIFs :) 

### Configurable Hyperparameters

- Here I am speaking about the hyperparameters that make a very prominent visual impact, these hyperparameters include:
    1. **Learning Rate**
    2. **Sample Size**
    3. **Model Architecture**

### 1. Learning Rate

Adjusting the learning rate will influences the rate of convergence (of course). However as shown below, a high learning rate results in erratic fluctuations in predictions, while a lower learning rate ensures a smooth and steady convergence toward the true function.

- **Function Index** = 8 | **Learning Rate** = 0.08

<p align="center">
    <img src="Assets/demo/sum-of-sins-high-lr.gif" width="700" alt="Demo">
</p>

- **Function Index** = 8 | **Learning Rate** = 0.0005

<p align="center">
    <img src="Assets/demo/sum-of-sins-low-lr.gif" width="700" alt="Demo">
</p>

### 2. Sample Size

Again, our neural network requires samples from the target function in order to approximate it. Thus, the number of random samples taken from this target function directly impacts how much the model knows about and constrains what it can infer. 

With a reasonably sparse set of samples, the network makes logical approximations of the true function. However, the real surprise comes when we push sample sparsity to the extreme.

If the network is given only two data points, it will connect them using a sigmoid curve. This is no coincidence as my implementation of these neural networks use sigmoid activations, absolutely gorgeous. 

- **Function Index** = 0 | **Sample Size** = 200

<p align="center">
    <img src="Assets/demo/line.gif" width="700" alt="Demo">
</p>

- **Function Index** = 0 | **Sample Size** = 2

<p align="center">
    <img src="Assets/demo/sigmoid.gif" width="700" alt="Demo">
</p>

### 3. Model Architecture

Model Architecture here refers to the complexity of our neural network. The level of complexity has various effects (some subtle, others more pronounced) which we explore below.

1. **Predicting Ability**
    - Let’s start with the obvious: a more complex model can approximate more complex functions. The comparison below clearly demonstrates this:
    - **Function Index** = 10 | **Simple Vs Complex**

<p style="display: flex; align-items:center">
    <img src="Assets/demo/simple-model.gif" width="400" alt="Demo">
    <img src="Assets/demo/complex-model.gif" width="400" alt="Demo">
</p>


2. **Granularity**
    - This is the more nuanced one, when we increase the complexity we can see the predictions becoming more "granular". Not entirely sure how to explain this one:
    - **Function Index** = 0 | **Simple Vs Complex**

<p style="display: flex; align-items:center">
    <img src="Assets/demo/simple-model-line.gif" width="400" alt="Demo">
    <img src="Assets/demo/complex-model-line.gif" width="400" alt="Demo">
</p>

### The Question of Dimensionality

At this point I hope I have convinced you that neural networks are pretty good at approximating functions. A person may question the power of neural networks saying that it cannot generalise to higher dimensions. However, this is could not be further from the truth, in fact its ability to generalise to higher dimensions is actually a key strength of the neural network architecture. If you're reading this you probably already know this, I just wanted a reason to make nice spinning 3D graphs...

<p align="center">
    <img src="Assets/demo/parabola-3d.gif" width="700" alt="Demo">
</p>

*More to Come...*

### Predicting an Image

I might as well add this here as well. I setup a neural network to predict the greyscale intensity of a pixel given the row and column of that pixel. It would then change its prediction to move close to the ground truth. The ground truth in question is a city skyline. Thus every pixel you see is the output of a neural network trying to emulate some image and this is why it is a static blur that slowly comes into focus. 

<p align="center">
    <img src="Assets/demo/skyline.gif" width="700" alt="Demo">
</p>

## Setup ⚙️

- Ensure you are in the root folder: `visualising-neural-nets/`

### Automated Setup

Run the appropriate script based on your OS

- **Linux/MacOS:**
    ```bash
    ./scripts/demo.sh
    ```

- **Windows:**
    ```bash
    ./scripts/demo.bat
    ```
This script will:
1. Create a virtual environment
2. Install dependencies
3. Run the selected script
4. Finally, clean up

### Manual

Alternatively, you could create & activate a virtual environment yourself (this would offer more flexibility of course).

- Create Virtual Environment
    ```bash
    python -m venv env
    ```
- Activate Virtual Environment
    ```bash
    source env/bin/activate
    ```
- Upgrade pip within the Virtual Environment
    ```bash
    pip install --upgrade pip
    ```
- Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

- Run function approximator in 2D
    ```bash
    python main.py
    ```

- Run function approximator in 3D
    ```bash
    python 3d-main.py
    ```

- See the output of a neural network
    ```bash
    python ml-img.py
    ```

- Deactivate Virtual Environment
    ```bash
    deactivate
    ```

