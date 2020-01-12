import tensorflow as tf

model = tf.keras.models.load_model("cyclist-CNN-4.model")

print(model.summary())
