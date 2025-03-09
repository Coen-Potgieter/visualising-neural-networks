
# Visualising Neural Networks Learn


## Overview 📌

In this project I explored the idea of treating Neural Networks as function approximators. Given some samples from a target function that the neural network knows nothing about, the model attempts to approximate the function. This project visualises that process and graphs its attempt at every iteration, resulting in a satisfying snap into place. 

The model neural network behind all these function approximations was developed by me using nothing but [NumPy](https://numpy.org/). This project gave me a great appreciation for neural networks (see my love letter below) as well as some interesting insights into model convergence and how much complexity in a model plays a role

## A Love Letter to Neural Networks ❤️

A function is a system of inputs that maps to some outputs, and we can plot this function on a graph. If we don’t know the function exactly but have some data available to us, that is what outputs produce what inputs, then we can reverse engineer the original function leveraging neural networks as function approximators, ie. Estimate the function based on patterns in the data. Neural networks are universal function approximators, meaning they can learn any function to any desired accuracy by adding more complexity to the model. This complexity includes adding more layers (depth), adding more neurons to each layer (width), using more complex activation functions, etc. 

What really fascinates me is how the complexity of a neural network is achieved. At their core, neural networks build complex functions by combining many linear equations that are stuck through some other non-linear functions (activation functions). I won't go into the equations for how the forward pass of a neural network is calculated, however it amazxing how simplistic it actually is, and the complexity that it can capture.

What's more is its applicability. Our entire world, generally, is made of different functions and anything a computer can processes (images, text, audio, video) can be represented as numbers, neural networks can learn to perform tasks like image classification, language translation, and more, as long as the problem can be expressed mathematically.

## Results 📸

This section is about various discoveries or observations while playing around with this program. However if you just want to look at some cool looking things then enjoy the gifs :)

### Configurable Hyperparameters

- Here I am speaking about the hyperparameters that make a very prominent visual impact, these hyperparameters include:
    1. **Learning Rate**
    2. **Sample Size**
    3. **Model Architecture**

### Learning Rate

Adjusting the learning rate will influences the rate of convergence (of course), However as shown below, a high learning rate results in erratic fluctuations in predictions, while a lower learning rate ensures a smooth and steady convergence toward the true function.

- **Function Index** = 8 | **Learning Rate** = 0.08

<p align="center">
    <img src="Assets/demo/sum-of-sins-high-lr.gif" width="700" alt="Demo">
</p>

- **Function Index** = 8 | **Learning Rate** = 0.0005

<p align="center">
    <img src="Assets/demo/sum-of-sins-low-lr.gif" width="700" alt="Demo">
</p>

### 2D


<p style="display: flex; flex-direction: column;">
    <img src="Assets/demo/funcs-2d-batch-1.png" width="400" alt="Demo">
    <img src="Assets/demo/funcs-2d-batch-2.png" width="400" alt="Demo">
    <p align="center">
        <img src="Assets/demo/funcs-2d-batch-3.png" width="400" alt="Demo">
    </p>
</p>

<p align="center">
    <img src="Assets/demo/parabola-3d.gif" width="500" alt="Demo">
</p>


<p align="center">
    <img src="Assets/demo/sigmoid.gif" width="500" alt="Demo">
</p>

<p align="center">
    <img src="Assets/demo/sum-of-sins.gif" width="500" alt="Demo">
</p>

<p align="center">
    <img src="Assets/demo/porabola.gif" width="500" alt="Demo">
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

