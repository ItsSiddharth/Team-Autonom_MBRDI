import tensorflow as tf

model = tf.keras.models.load_model("cyclist-CNN_custom_architecture.model")

print(model.summary())
