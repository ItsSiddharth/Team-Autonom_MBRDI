import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("cyclist-CNN_custom_architecture.model")
WIDTH = 256
HEIGHT = 256
LR = 0.01
MODEL_NAME = "cyclist-CNN-4.model"

image = cv2.imread("tsinghuaDaimlerDataset_2014-11-20_074640_000000010_leftImg8bit.png")



image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image1 = cv2.resize(image1, (256, 256))
image1 = np.array(image1).reshape(-1, 256, 256, 1)
image1 = image1.astype(np.float32)
image1 = image1/255.0
prediction = model.predict(image1)
prediction[0][2]*=4
prediction[0][0]*=4
prediction[0][1]*=8
prediction[0][3]*=8
print(prediction[0])
label = prediction[0]
cv2.rectangle(image, (label[1], label[0]), (label[3], label[2]), (255,0,0), 2)
cv2.imshow('frame', image)
cv2.waitKey(0)

