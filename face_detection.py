import cv2
import numpy as np 
import argparse

# анализируем аргументы
ap = argparse.ArgumentParser()
# --image - путь к входному изображению
ap.add_argument("-i", "--image", required = True, 
	help = "path to input image")
# --prototxt - путь к файлу прототипа Caffe
ap.add_argument("-p", "--prototxt", required = True, 
	help = "path to Caffe 'deploy' prototxt file")
# --model - путь к предварительно подготовленной модели Caffe
ap.add_argument("-m", "--model", required = True,
	help = "path to Caffe pre-trained model")
# --confidence - вероятность определения лица начинается с 0.5
ap.add_argument("-c", "--confidence", type = float, default = 0.5, 
	help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())

# загрузим нашу модель
print("[INFO] loading model....")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# загрузить входное изображение, построить входной blop для изображения,
# изменением его до 300*300 и его нормализацией
image = cv2.imread(args["image"])
(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image,(300, 300)), 1.0, 
	(300, 300),(104.0,177.0,123.0))

# передать blob в сеть и получить предсказание детектирования
print("[INFO] computing object detections....")
net.setInput(blob)
detections = net.forward()

# рисуем рамку вокруг лица
for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]

	# отфильтровывает слабые предсказания
	if confidence > args["confidence"]:
		# вычисляем x,y рамки
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# рисуем прямоугольник с текстом вероятности
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# показываем выходное изображение
cv2.imshow("Output", image)
cv2.waitKey(0)




