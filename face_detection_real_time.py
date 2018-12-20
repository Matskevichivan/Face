import cv2
import numpy as np 
import argparse
from imutils.video import VideoStream
import time
import imutils


# анализируем аргументы
ap = argparse.ArgumentParser()
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

# инициализировать видеопоток и дать прогреться камере
print("[INFO] starting video stream....")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# цикл над кадрами видеопотока
while True:
	# захватить кадр с видео потока и изменить ширину до 400 пикселей максимум
	frame = vs.read()
	frame = imutils.resize(frame, width = 400)

	# загрузить входное изображение, построить входной blop для изображения,
	# изменением его до 300*300 и его нормализацией
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300, 300)), 1.0, 
	(300, 300),(104.0,177.0,123.0))

	# передать blob в сеть и получить предсказание детектирования
	net.setInput(blob)
	detections = net.forward()

	# рисуем рамку вокруг лица
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# отфильтровывает слабые предсказания
		if confidence < args["confidence"]:
			continue

		# вычисляем x,y рамки
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# рисуем прямоугольник с текстом вероятности
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# показываем выходное изображение
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# если нажать "q", то выход из цикла
	if key == ord("q"):
		break
# чистка
cv2.destroyAllWindows()
vs.stop()







