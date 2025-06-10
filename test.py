import tensorflow as tf
import numpy as np
import json
import os

# === 建立輸出資料夾 ===
os.makedirs("model", exist_ok=True)

# === 載入資料集 ===
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# === 建立模型 ===
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
    tf.keras.layers.Dense(128, activation='relu', name="dense_1"),
    tf.keras.layers.Dense(10, activation='softmax', name="dense_2")
])

# === 編譯與訓練 ===
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# === 測試模型準確率 ===
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# === 儲存 .h5 模型（可選）===
model.save("model/fashion_mnist.h5")

# === 儲存模型架構為 .json ===
arch = []
for layer in model.layers:
    arch.append({
        "name": layer.name,
        "type": layer.__class__.__name__,
        "config": layer.get_config(),
        "weights": [w.name for w in layer.weights]
    })

with open("model/fashion_mnist.json", "w") as f:
    json.dump(arch, f, indent=2)
print("📐 Saved architecture to model/fashion_mnist.json")

# === 儲存權重為 .npz ===
weights_dict = {}
for layer in model.layers:
    for w in layer.weights:
        weights_dict[w.name] = w.numpy()

np.savez("model/fashion_mnist.npz", **weights_dict)
print("⚖️  Saved weights to model/fashion_mnist.npz")
