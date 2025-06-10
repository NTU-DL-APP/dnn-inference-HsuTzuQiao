import numpy as np
import json

# === 激活函數 ===
def relu(x):
    """
    實現ReLU (Rectified Linear Unit) 激活函數
    ReLU(x) = max(0, x)
    """
    return np.maximum(0, x)

def softmax(x):
    """
    實現Softmax激活函數
    避免數值溢出的穩定版本
    """
    # 減去最大值以避免數值溢出
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# === 展平層 ===
def flatten(x):
    """展平輸入張量"""
    return x.reshape(x.shape[0], -1)

# === 全連接層 ===
def dense(x, W, b):
    """全連接層運算: output = input @ weights + bias"""
    return x @ W + b

# 使用numpy推理TensorFlow h5模型
# 目前只支援Dense、Flatten、relu、softmax層
def nn_forward_h5(model_arch, weights, data):
    """
    使用模型架構和權重對數據進行前向推理
    
    參數:
        model_arch: 模型架構（從JSON載入）
        weights: 模型權重（從NPZ載入）
        data: 輸入資料
    
    回傳:
        推理結果
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
            # 載入權重和偏移值
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            
            # 執行線性變換
            x = dense(x, W, b)
            
            # 應用激活函數
            activation = cfg.get("activation", "linear")
            if activation == "relu":
                x = relu(x)
            elif activation == "softmax":
                x = softmax(x)
            # 如果是 "linear" 或其他，不應用激活函數

    return x

# 主要推理函數
def nn_inference(model_arch, weights, data):
    """
    神經網路推理主函數
    您可以用自己的實現替換 nn_forward_h5()
    """
    return nn_forward_h5(model_arch, weights, data)