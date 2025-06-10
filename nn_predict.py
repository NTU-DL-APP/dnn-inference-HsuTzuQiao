import numpy as np
import json

# === Activation Functions ===
def relu(x):
    """
    Implement ReLU (Rectified Linear Unit) activation function
    ReLU(x) = max(0, x)
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Implement Softmax activation function
    Numerically stable version to avoid overflow
    """
    # Handle both 1D and multi-dimensional inputs
    if x.ndim == 1:
        # 1D case: subtract max value to avoid numerical overflow
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    else:
        # Multi-dimensional case: apply softmax along the last axis
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# === Flatten Layer ===
def flatten(x):
    """Flatten input tensor"""
    return x.reshape(x.shape[0], -1)

# === Dense Layer ===
def dense(x, W, b):
    """Dense layer operation: output = input @ weights + bias"""
    return x @ W + b

# Forward pass using numpy for TensorFlow h5 model
# Currently supports Dense, Flatten, relu, softmax layers only
def nn_forward_h5(model_arch, weights, data):
    """
    Forward inference using model architecture and weights
    
    Args:
        model_arch: Model architecture (loaded from JSON)
        weights: Model weights (loaded from NPZ)
        data: Input data
    
    Returns:
        Inference results
    """
    x = data
    
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
            
        elif ltype == "Dense":
            # Load weights and bias
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            
            # Apply linear transformation
            x = dense(x, W, b)
            
            # Apply activation function
            activation = cfg.get("activation", "linear")
            if activation == "relu":
                x = relu(x)
            elif activation == "softmax":
                x = softmax(x)
            # For "linear" or others, no activation is applied

    return x

# Main inference function
def nn_inference(model_arch, weights, data):
    """
    Neural network inference main function
    You can replace nn_forward_h5() with your own implementation
    """
    return nn_forward_h5(model_arch, weights, data)