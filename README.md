# SpikingNN with Predictive Feedback
Implementation of Spiking Neural Networks with Predictive Feedback for Increased Input Sparsity

Disclaimer: This code is an extension of the work on training Spiking NNs published in ICLR, 2020. The paper can be found [here](https://openreview.net/forum?id=B1xSperKvH) and the original repo can be found [here](https://github.com/nitin-rathi/hybrid-snn-conversion).

# Motivation
The motivation of this work is to increase input spike sparsity based on the sample prediction difficulty by enabling predictive feedback connections.

# Implementation
The mechanism is implemented by adding feedback connections which are trained to predict/reconstruct input signal (2D image in this case) and block the input neurons if its corresponding pixel value is predicted within some epsilon bound.

# Extensions in this repo
1. Added feedback connections to both ANN model and SNN model.
2. Added scripts to train both types of models to predict inputs.
3. Added script to (qualitatively) analyse predictions.
4. Added script to perform input neuron blocking based on the predictions as well as measure accuracy and average number of spikes required by both models.

Note: Files pertaining to this repo, but not the original training spiking NN work, are distinguished by having '\_wFeedback' postfix in their names.
