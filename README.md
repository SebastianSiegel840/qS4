# Quantized structured State-Space sequential (S4 and S4D) model
This repository builds upon the work on S4 and S4D models implementing integer quantization of the models parameters and activations along with quantization-aware training.

## Quantization method and quantization-aware training

The quantization method is inspired by the integrer quantization method of BitNet (https://github.com/microsoft/BitNet). 
To perform quantization-aware training, a Straight-Through-Estimator is used, performing the forward-pass with the quantized value and using the identity as the surrogate for the quantization in the backward-pass. 

## Results

### Parameter-wise quantization and quantization-aware training (QAT)
To check the impact of quantization and the benefits of QAT, the senquential CIFAR10 (grayscale) benchmark dataset can be use. Especially for the kernel parameters, A and C, the model performs close to the unquantized baseline for more aggressive quantization levels than performing unquantized training and quantizing after training only.
![CIFAR10_quantization](images/Figure_CIFAR10_S4D_new.png)

### Quantization reduces hardware cost
![metrics](images/Figure_Metrics.png)

### Quantization helps with structured pruning
![Pruning](images/Figure_Pruning.png)

### Quantizability vs. model size and quantization reducing noise impact
![NoiseSize](images/Figure_NoiseSize.png)

