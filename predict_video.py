import cv2
import tensorflow as tf
import numpy as np
import time

model = tf.keras.models.load_model("cyclist-CNN-4.model")
WIDTH = 256
HEIGHT = 256
LR = 0.01
MODEL_NAME = "cyclist-CNN-4.model"

cap = cv2.VideoCapture("WhatsApp Video 2020-01-05 at 00.30.56.mp4")

while True:
	_, image = cap.read()
	image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image1 = cv2.resize(image1, (256, 256))
	image1 = np.array(image1).reshape(-1, 256, 256, 1)
	image1 = image1.astype(np.float32)
	image1 = image1/255.0
	prediction = model.predict(image1)
	image = cv2.resize(image, (512,512))
	print(prediction[0])
	label = prediction[0]
	cv2.rectangle(image, (int(label[1]*2), int(label[0]*2)), (int(label[3]*2)+10, int(label[2]*2)), (255,0,0), 2)
	time.sleep(0.3)
	cv2.imshow('frame', image)
	if cv2.waitKey(1) &  0xFF == ord("q"):
		break
cap.release()
cv2.destroyAllWindows() 

