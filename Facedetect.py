#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Patrick van der Leer <pat.vdleer@gmail.com>
# @license: This file is a part of py-opencv and thus GPL licensed
#
# @todo: comment everything to explain what happens
# @todo: make code less embarrassing

import sys
import cv2, cv
import os

class Main:
	cam = None
	storage = None
	
	cascadeFacePath = "haarcascade/haarcascade_frontalface_default.xml"
	cascadeMouthPath = "haarcascade/haarcascade_frontalmouth25x15.1.xml"
	cascadeNosePath = "haarcascade/haarcascade_nose25x15.1.xml"
	cascadeLeftEyePath = "haarcascade/haarcascade_LEye18x12.1.xml"
	cascadeRightEyePath = "haarcascade/haarcascade_REye18x12.1.xml"

	def __init__(self):
		self.cascadeFace = cv2.CascadeClassifier(os.path.join(os.getcwd(), self.cascadeFacePath))
		self.cascadeMouth = cv2.CascadeClassifier(os.path.join(os.getcwd(), self.cascadeMouthPath))
		self.cascadeNose = cv2.CascadeClassifier(os.path.join(os.getcwd(), self.cascadeNosePath))
		self.cascadeLeftEye = cv2.CascadeClassifier(os.path.join(os.getcwd(), self.cascadeLeftEyePath))
		self.cascadeRightEye = cv2.CascadeClassifier(os.path.join(os.getcwd(), self.cascadeRightEyePath))
		
		self.cam = cv2.VideoCapture(0)
		self.storage = cv.CreateMemStorage()  # @UndefinedVariable

	def search(self):	
		_, img = self.cam.read()

		grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		grayImg = cv2.equalizeHist(grayImg)

		faces =  self.detect(grayImg, self.cascadeFace, flags=cv.CV_HAAR_SCALE_IMAGE)  # @UndefinedVariable
		for (x,y,w,h) in faces:
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
			cropped = img[y:y+h, x:x+w]
			
			croppedLeftEyes = img[y:y+h/2, x:x+w/2]
			croppedRightEyes = img[y:y+h/2, x+w/2:x+w]
			eye = self.detect(croppedLeftEyes, self.cascadeLeftEye, minSize=(35,35))
			for (_x,_y,_w,_h) in eye:
				cv2.rectangle(img, (x+_x, y+_y), (x+_x+_w, y+_y+_h), (0, 255, 0), 2)
				break
			eye = self.detect(croppedRightEyes, self.cascadeRightEye, minSize=(35,35))
			for (_x,_y,_w,_h) in eye:
				cv2.rectangle(img, (x+_x+w/2, y+_y), (x+_x+w/2+_w, y+_y+_h), (0, 255, 0), 2)
				break
				
			nose = self.detect(cropped, self.cascadeNose, minSize=(20,20))
			for (_x,_y,_w,_h) in nose:
				cv2.rectangle(img, (x+_x,y+_y), (x+_x+_w,y+_y+_h), (128, 128, 128), 2)
				break
			
			croppedMouth = img[y+h/2:y+h, x:x+w]
			mouth = self.detect(croppedMouth, self.cascadeMouth, minSize=(30,30))
			for (_x,_y,_w,_h) in mouth:
				cv2.rectangle(img, (x+_x,y+_y+h/2), (x+_x+_w,y+_y+h/2+_h), (255, 0, 0), 2)
				break
		
		cv2.imshow('img',img)

	def detect(self, img, cascade, scaleFactor=1.2, minNeighbors=1, minSize=(100, 100), flags=cv.CV_HAAR_DO_CANNY_PRUNING):  # @UndefinedVariable
		return cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, flags=flags)



if __name__ == "__main__":
	main = Main()
	while True:
		main.search()
		if cv2.waitKey(10)==27:
			cv2.destroyAllWindows()
			sys.exit()
	
