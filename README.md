# SpikingNN with Predictive Feedback
Implementation of Spiking Neural Networks with Predictive Feedback for Increased Input Sparsity

Disclaimer: This code is an extension of the work on training Spiking NNs published in ICLR, 2020. The paper can be found [here](https://openreview.net/forum?id=B1xSperKvH) and the original repo can be found [here](https://github.com/nitin-rathi/hybrid-snn-conversion).

# Motivation
The motivation of this work is to increase input spike sparsity based on the sample prediction difficulty by enabling predictive feedback connections.

# Implementation
The mechanism is implemented by adding feedback connections which are trained to predict/reconstruct input signal (2D image in this case) and block the input neurons if its corresponding pixel value is predicted within some epsilon bound.

![Gif example of input neuron blocking](https://github.com/TimurIbrayev/SpikingNN_with_PredictiveFeedback/blob/main/trained_models/snn/snn_vgg5_wfeedback_mnist_50_tradeoff1.0_typelast_t_posthoc_analysis_predictions_and_binarydiff_epsilon%3D0.1.gif)


# Extensions in this repo
1. Added both ANN and SNN models with feedback connections.
2. Added scripts to train both types of models to predict inputs.
3. Added script to (qualitatively) analyse predictions.
4. Added script to perform input neuron blocking based on the predictions as well as measure accuracy and average number of spikes required by both models.

# Extended files
Note: Files pertaining to this repo, but not the original training spiking NN work, are distinguished by having '\_wFeedback' postfix in their names.
These files are:
* self_models/vgg_wFeedback.py:
  * ANN model with feedback. MaxPool/AvgPool are replaced with strided convolution according to common guides of GANs
* self_models/vgg_wFeedback_spiking.py:
  * SNN model with feedback.
* self_models/vgg_wFeedback_spiking_target_objective_1.py:
  * SNN model with feedback and input neuron blocking mechanism based on the reconstructed predictions.
* self_models/vgg_wFeedback_spiking_target_objective_3.py:
  * SNN model with feedback, input neuron blocking, and test of implicit attention through contrast. 
  * (failed attempt, requires more work).
* ann_wFeedback.py and snn_wFeedback.py:
  * Scripts to train ANN and SNN models, respectively, with the corresponding losses to include prediction generation.
* snn_wFeedback_posthoc_model_analysis.py:
  * Simplified script for SNN model inference and qualitative assessment of generated predictions of the inputs.
  * Utilizes model specified in self_models/vgg_wFeedback_spiking.py.
* snn_wFeedback_posthoc_model_inference_obj1.py:
  * Simplified script for SNN model inference with enabled input neuron blocking. 
  * Utilizes model specified in self_models/vgg_wFeedback_spiking_target_objective_1.py.
* snn_wFeedback_posthoc_model_inference_obj3.py:
  * Simplified script for SNN model inference with attempt to try implicit attention through contrast.
  * Utilizes model specified in self_models/vgg_wFeedback_spiking_target_objective_3.py.
  * (failed attempt, requires more work).

Default parameters used for simulations can be found at the begining of the corresponding log files, which can be found in logs/ann and logs/snn for ANN and SNN models with feedback.

# Acknowledgement
Timur would like to thank Nitin Rathi for the clear explanations of how models in original work are implemented and perform SNN processing.
