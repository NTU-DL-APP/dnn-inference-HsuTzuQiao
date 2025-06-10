import tensorflow as tf
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score

def create_simple_model():
    """Create simplified but efficient model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten'),
        tf.keras.layers.Dense(256, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(128, activation='relu', name='dense_2'),
        tf.keras.layers.Dense(64, activation='relu', name='dense_3'),
        tf.keras.layers.Dense(10, activation='softmax', name='dense_4')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_and_preprocess_data():
    """Load and preprocess Fashion-MNIST data"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)

def convert_model_to_custom_format(model, model_path_prefix):
    """Convert TensorFlow model to assignment required format"""
    
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    
    # 1. Save as .h5 format
    h5_path = f"{model_path_prefix}.h5"
    model.save(h5_path)
    print(f"Model saved as: {h5_path}")
    
    # 2. Extract model architecture and convert to custom format
    model_arch = []
    weights_dict = {}
    
    for i, layer in enumerate(model.layers):
        layer_info = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'config': {},
            'weights': []
        }
        
        if hasattr(layer, 'activation'):
            if layer.activation == tf.keras.activations.relu:
                layer_info['config']['activation'] = 'relu'
            elif layer.activation == tf.keras.activations.softmax:
                layer_info['config']['activation'] = 'softmax'
            elif layer.activation == tf.keras.activations.linear:
                layer_info['config']['activation'] = 'linear'
        
        # Handle weights
        if layer.get_weights():
            weights = layer.get_weights()
            if len(weights) >= 2:  # weights and bias
                weight_key = f"{layer.name}/kernel:0"
                bias_key = f"{layer.name}/bias:0"
                weights_dict[weight_key] = weights[0]
                weights_dict[bias_key] = weights[1]
                layer_info['weights'] = [weight_key, bias_key]
            elif len(weights) == 1:  # weights only
                weight_key = f"{layer.name}/kernel:0"
                weights_dict[weight_key] = weights[0]
                layer_info['weights'] = [weight_key]
        
        model_arch.append(layer_info)
    
    # 3. Save architecture as JSON
    json_path = f"{model_path_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(model_arch, f, indent=2)
    print(f"Architecture saved as: {json_path}")
    
    # 4. Save weights as NPZ
    npz_path = f"{model_path_prefix}.npz"
    np.savez(npz_path, **weights_dict)
    print(f"Weights saved as: {npz_path}")
    
    return model_arch, weights_dict

def main():
    """Main function"""
    print("Starting Fashion-MNIST model training...")
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create model
    model = create_simple_model()
    model.summary()
    
    # Train model (simple training without data augmentation)
    print("\nStarting training...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=50,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Save model
    convert_model_to_custom_format(model, "model/fashion_mnist")
    
    print("Training completed!")

if __name__ == "__main__":
    main()