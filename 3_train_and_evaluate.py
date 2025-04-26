import numpy as np
import tensorflow as tf
import os
from src.model_architectures import create_model

# Load preprocessed data
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

# Model
model = create_model()
model.summary()

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=256
)

# Save model
if not os.path.exists('models'):
    os.makedirs('models')
model.save('models/best_model.h5')

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save training history for plotting
np.save('models/history.npy', history.history)
