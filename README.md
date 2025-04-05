# Post-Training-Quantization

Quantization is a process by which a model trained with floating-point weights and activations is converted to a reduced bit-width representation. Most commonly, this is done by converting the model weights and biases trained with 32-bit/16-bit floating-point (FP32/FP16) to an 8-bit integer (INT8) representation for storage and inference on hardware optimized for integer arithmetic. The quantization process reduces the model's memory footprint and speeds up inference. We can expect a 4× reduction in memory footprint and a 2–4× speedup in inference time when quantizing the model from FP32 to INT8.

PyTorch supports three primary types of quantization techniques:

1) Static Post-training Quantization (static PTQ): The quantization is applied after training. Both weights and activations are quantized. Activations are quantized using calibration data (a small subset of the test/train data is used to calibrate the quantization parameters).

2) Dynamic Post-training Quantization (dynamic PTQ): The quantization is applied after training. Weights are quantized before inference, while activations are quantized dynamically during inference based on the statistics of the activations.

3) Quantization-Aware Training (QAT): The model is trained with quantized weights and activations. Quantization modules are inserted into the model to simulate the quantization process during training, helping to create a model that is more robust to quantization.

Most of the code in this notebook is taken from the post_training_quantization notebook provided by Umar Jamil as part of the tutorial [Quantization explained with PyTorch - Post-Training Quantization, Quantization-Aware Training](https://youtu.be/0VdNflU08yA?feature=shared). I strongly recommend watching the video to understand the concepts better before diving into the code.

This notebook uses the MNIST dataset to train a simple deep learning model and then quantizes the model using post-training quantization. The methods explored include:
1) Static Post-training Quantization (with and without layer fusion)
2) Dynamic Post-training Quantization

I have made following modifications to the code provdied by Umar Jamil:
1) Added functions to perform model quantization using mathematical equations rather than PyTorch's built-in quantization functions. The values obtained are then compared with the values obtained using the PyTorch's quantization functions.  This helps in understanding the Pytorch quantization process better.
2) Added layer fusion to the static quantization method.
3) Explored dynamic quantization.

The code is primarly based on the paper [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) by Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.