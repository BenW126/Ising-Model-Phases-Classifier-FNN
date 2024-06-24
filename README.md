## Introduction
This statistical mechanics model aims to classify phases of the 2-dimensional (2D) 10 x 10 Ising Model using a Feedforward Neural Network (FNN) built from scratch without using machine learning libraries like PyTorch or TensorFlow. The FNN was trained on spin configurations from the 2D Ising model data provided by [Min Long](https://github.com/DavidGoing). 

The 2D Ising model exhibits a phase transition at a critical temperature ($T_c ≈ 2.269$), transitioning from a disordered paramagnetic phase to an ordered ferromagnetic phase with spontaneous magnetization. Below the critical temperature, the spins spontaneously align, while above it, thermal fluctuations destroy the spontaneous magnetization.

For more information about the 2D Ising Model, here is the reference from Stanford University: http://micro.stanford.edu/~caiwei/me334/Chap12_Ising_Model_v04.pdf. This document covers the definition of the Ising model, solving the 1D Ising model, and the 2D Ising model, including its analytic solution, Monte Carlo simulation, and qualitative behavior.

## Dataset and Network Structure
The dataset consists of 6000 spin configurations of a 10 x 10 2D Ising model, obtained by Monte Carlo simulation. The spin configurations cover a temperature range from 0.2 to 4.0 with a step of 0.2. Note that 1000 configurations were used for training, and the remaining 5000 configurations were for evaluating the trained model. 

Our network has an input layer of 100 neurons, a hidden layer of 3 neurons with sigmoid activations, and an output layer of 1 neuron, also with a sigmoid activation. 

## Results
The first image shows the neural network's output as a function of temperature, while the second image shows the training and validation losses for the neural network model over the training epochs. 

![Losses](Losses.png) ![Results](Result.png)
