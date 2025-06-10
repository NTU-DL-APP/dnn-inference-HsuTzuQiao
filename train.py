import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_optimized_fashion_mnist_model():
    """建立並訓練一個優化的Fashion-MNIST DNN分類模型"""
    
    print("🔄 正在載入Fashion-MNIST資料集...")
    
    # 載入Fashion-MNIST資料集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # 進階資料預處理
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 資料標準化 (可選，但通常有助於DNN訓練)
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    print(f"訓練資料形狀: {x_train.shape}")
    print(f"測試資料形狀: {x_test.shape}")
    print(f"資料標準化 - 平均值: {mean:.4f}, 標準差: {std:.4f}")
    
    # 建立優化的模型架構
    print("\n🏗️ 正在建立優化模型...")
    model = tf.keras.Sequential([
        # 輸入層
        tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten'),
        
        # 第一隱藏層 - 更大的神經元數量
        tf.keras.layers.Dense(512, name='dense_1'),
        tf.keras.layers.BatchNormalization(name='bn_1'),
        tf.keras.layers.Activation('relu', name='relu_1'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),
        
        # 第二隱藏層
        tf.keras.layers.Dense(256, name='dense_2'),
        tf.keras.layers.BatchNormalization(name='bn_2'),
        tf.keras.layers.Activation('relu', name='relu_2'),
        tf.keras.layers.Dropout(0.4, name='dropout_2'),
        
        # 第三隱藏層
        tf.keras.layers.Dense(128, name='dense_3'),
        tf.keras.layers.BatchNormalization(name='bn_3'),
        tf.keras.layers.Activation('relu', name='relu_3'),
        tf.keras.layers.Dropout(0.4, name='dropout_3'),
        
        # 第四隱藏層
        tf.keras.layers.Dense(64, name='dense_4'),
        tf.keras.layers.BatchNormalization(name='bn_4'),
        tf.keras.layers.Activation('relu', name='relu_4'),
        tf.keras.layers.Dropout(0.3, name='dropout_4'),
        
        # 輸出層
        tf.keras.layers.Dense(10, activation='softmax', name='output',
                             kernel_regularizer=l2(0.001))
    ])
    
    # 使用優化的編譯參數
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 顯示模型架構
    print("\n📋 優化模型架構:")
    model.summary()
    
    # 設置回調函數
    callbacks = [
        # 早停：驗證準確率不再提升時停止訓練
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # 學習率調度：驗證損失不再下降時降低學習率
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        
        # 模型檢查點：儲存最佳模型
        ModelCheckpoint(
            'best_fashion_mnist.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 訓練模型
    print("\n🚀 開始訓練優化模型...")
    print("使用的優化技術:")
    print("- 更深的網路架構 (4個隱藏層)")
    print("- Batch Normalization")
    print("- Dropout 正則化")
    print("- L2 權重正則化")
    print("- 學習率調度")
    print("- 早停機制")
    print("- 資料標準化")
    
    history = model.fit(
        x_train, y_train,
        epochs=50,  # 增加最大訓練輪數，依賴早停機制
        batch_size=64,  # 較小的批次大小通常有助於泛化
        validation_split=0.15,  # 增加驗證集比例
        callbacks=callbacks,
        verbose=1
    )
    
    # 載入最佳模型進行評估
    model = tf.keras.models.load_model('best_fashion_mnist.h5')
    
    # 評估模型
    print("\n📊 評估最佳模型性能...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"最終測試準確率: {test_accuracy:.4f}")
    
    # 儲存最終模型
    model_path = 'fashion_mnist.h5'
    model.save(model_path)
    print(f"\n💾 模型已儲存至: {model_path}")
    
    # 顯示訓練歷史
    print("\n📈 訓練歷史摘要:")
    if len(history.history['accuracy']) > 0:
        max_train_acc = max(history.history['accuracy'])
        max_val_acc = max(history.history['val_accuracy'])
        print(f"最高訓練準確率: {max_train_acc:.4f}")
        print(f"最高驗證準確率: {max_val_acc:.4f}")
        print(f"最終測試準確率: {test_accuracy:.4f}")
    
    return model, test_accuracy, history

def create_alternative_model():
    """建立另一種架構的優化模型（如果第一種效果不佳）"""
    print("\n🔄 嘗試替代架構...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten'),
        
        # 使用更寬的網路
        tf.keras.layers.Dense(1024, activation='relu', name='dense_1',
                             kernel_regularizer=l2(0.0001)),
        tf.keras.layers.Dropout(0.5, name='dropout_1'),
        
        tf.keras.layers.Dense(512, activation='relu', name='dense_2',
                             kernel_regularizer=l2(0.0001)),
        tf.keras.layers.Dropout(0.5, name='dropout_2'),
        
        tf.keras.layers.Dense(256, activation='relu', name='dense_3',
                             kernel_regularizer=l2(0.0001)),
        tf.keras.layers.Dropout(0.4, name='dropout_3'),
        
        tf.keras.layers.Dense(10, activation='softmax', name='output')
    ])
    
    return model

def setup_data_folder():
    """設置資料夾結構"""
    os.makedirs('model', exist_ok=True)
    os.makedirs('data/fashion', exist_ok=True)
    print("📁 資料夾結構已建立")

if __name__ == "__main__":
    print("🎯 Fashion-MNIST 優化模型訓練程式")
    print("=" * 60)
    
    # 設置GPU記憶體成長（如果有GPU）
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("🔧 GPU記憶體成長已啟用")
        except RuntimeError as e:
            print(f"GPU設置警告: {e}")
    
    # 設置資料夾
    setup_data_folder()
    
    # 檢查是否已有模型
    if os.path.exists('fashion_mnist.h5'):
        choice = input("已存在 fashion_mnist.h5 檔案，要重新訓練嗎？(y/n): ").lower()
        if choice != 'y':
            print("使用現有模型檔案")
            exit()
    
    try:
        # 建立和訓練優化模型
        model, accuracy, history = create_optimized_fashion_mnist_model()
        
        print("\n✅ 優化模型訓練完成！")
        print(f"最終測試準確率: {accuracy:.4f}")
        
        # 如果準確率不夠高，給出建議
        if accuracy < 0.90:
            print("\n💡 準確率提升建議:")
            print("- 可以嘗試調整Dropout比率")
            print("- 增加或減少隱藏層神經元數量")
            print("- 調整學習率")
            print("- 增加訓練輪數")
        else:
            print(f"\n🎉 恭喜！達到了 {accuracy:.1%} 的高準確率！")
        
        print("\n📝 下一步:")
        print("1. 執行 'python convert_model.py' 轉換模型")
        print("2. 執行 'python model_test.py' 測試推理")
        
    except Exception as e:
        print(f"❌ 建立模型時發生錯誤: {e}")
        print("請檢查是否已安裝所需套件:")
        print("pip install tensorflow numpy matplotlib")