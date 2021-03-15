import tensorflow.keras  # Tensorflow 2.0
import tensorflow as tf

# Testar o uso de GPU em Keras
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
