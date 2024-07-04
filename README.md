# Overview

Welcome to the repository! Here, you will find implementations of three different versions of the Mixture of Experts (MoE) layer in a Transformer model.

## Code Structure

### Building Blocks for Transformer

The `model_classes.py` file includes essential components of the Transformer model, such as:

- **Attention Mechanism**
- **Rotary Positional Encoding**
- **Layer Normalization (LayerNorm)**
- **Transformer Block**
- **Router**
- **Vectorized MoE**

The `dataloader.py` file contains the implementation of our custom dataloader. For training our models, we used the "20220301.simple" subset from the Wikipedia dataset. You can download this dataset by following the instructions in the `dataloader` module.


### Standard MoE Layer

For the implementation of a conventional Mixture of Experts layer, refer to the `VectorizedMoE()` class located in the `model_classes.py` file.

### Multi-Head MoE

For the vectorized implementation of the Multi-Head MoE (MH-MoE) layer, please refer to the `MH-MoE.py` file.

### Lory

For the implementation of a fully differentiable MoE layer named Lory, navigate to the `MH_Lory.py` file. Set the number of heads to one to obtain the traditional Lory module.

### MH-Lory

In the `MH_Lory.py` file, increasing the number of heads will configure the Multi-Head Lory model, a novel architecture we propose.

### More Information

For more detailed information and mathematical formulations, please refer to the report section and the `main.tex` file where we describe our project in detail.