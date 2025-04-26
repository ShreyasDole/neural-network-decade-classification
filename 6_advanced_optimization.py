# 6_advanced_optimization.py

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense
from src.model_architectures import create_model_1
from src.data_preprocessing import preprocess_data, load_data
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess data
df = load_data('data/YearPredictionMSD.txt')
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)

# Test different optimizers
optimizers = [Adam(learning_rate=0.001), SGD(learning_rate=0.01, momentum=0.9)]
optimizer_names = ['Adam', 'SGD']

for optimizer, name in zip(optimizers, optimizer_names):
    model = create_model_1(X_train.shape[1], len(set(y_train)))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=15, batch_size=128, validation_data=(X_val, y_val), verbose=2)
    
    plt.plot(history.history['accuracy'], label=f'{name} Training Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{name} Validation Accuracy')

plt.title('Model Comparison: Adam vs SGD')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
