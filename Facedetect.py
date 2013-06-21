#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Patrick van der Leer <pat.vdleer@gmail.com>
# @license: This file is a part of py-opencv and thus GPL licensed
#
# @todo: make code less embarrassing

import sys
import cv2, cv
import os

class Main:
	# camera stream
	cam = None
	# memory storage stream
	storage = None
	
	# paths to haarcascades
	cascadeFacePath = "haarcascade/haarcascade_frontalface_default.xml"
	cascadeMouthPath = "haarcascade/haarcascade_frontalmouth25x15.1.xml"
	cascadeNosePath = "haarcascade/haarcascade_nose25x15.1.xml"
	cascadeLeftEyePath = "haarcascade/haarcascade_LEye18x12.1.xml"
	cascadeRightEyePath = "haarcascade/haarcascade_REye18x12.1.xml"

	def __init__(self):
		# get path of this script, os.getcwd()
		# join the current location with the path to the cascade xml
		# load them  
		# set the loaded cascade to self
		self.cascadeFace = cv2.CascadeClassifier(os.path.join(os.getcwd(), self.cascadeFacePath))
		self.cascadeMouth = cv2.CascadeClassifier(os.path.join(os.getcwd(), self.cascadeMouthPath))
		self.cascadeNose = cv2.CascadeClassifier(os.path.join(os.getcwd(), self.cascadeNosePath))
		self.cascadeLeftEye = cv2.CascadeClassifier(os.path.join(os.getcwd(), self.cascadeLeftEyePath))
		self.cascadeRightEye = cv2.CascadeClassifier(os.path.join(os.getcwd(), self.cascadeRightEyePath))
		
		# setup videostream from camera
		self.cam = cv2.VideoCapture(0)
		
		# create memory storage
		self.storage = cv.CreateMemStorage()  # @UndefinedVariable

	def search(self):	
		# get the image from cam, this return a tuple, we don't need the first value
		_, img = self.cam.read()
		
		# convert the image to grayscale, we need to measure the contrast difference so color is overraded
		grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# reset the historygram from beter results
		grayImg = cv2.equalizeHist(grayImg)

		# detect the actual face from the frame taken from the cam
		faces =  self.detect(grayImg, self.cascadeFace, flags=cv.CV_HAAR_SCALE_IMAGE)  # @UndefinedVariable
		for (x,y,w,h) in faces:
			# the measurements we get back are x,y measured from the left top of the image
			# and the w and h are width and height (DUHHH)
			
			# draw a pretty rectangle on the original image (the one in color) (variable img)
			# so from left top point x,y to point x + width and y+height
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
			
			# create a separate variable with a specific part of the grayed image, namely the face part :)
			# we don't need to look for facial parts in places where there's no face...
			# this will prevent false results and gain performance since we check a smaller area for parts 
			cropped = grayImg[y:y+h, x:x+w]
			
			# same as above but an even smaller part
			# last time I checked the eyes are in the upper part of the face
			# the left in the left upper corner and right the vice versa
			croppedLeftEyes = grayImg[y:y+h/2, x:x+w/2]
			croppedRightEyes = grayImg[y:y+h/2, x+w/2:x+w]
			
			# detect actual (left) eye
			eye = self.detect(croppedLeftEyes, self.cascadeLeftEye, minSize=(35,35))
			for (_x,_y,_w,_h) in eye:
				# again draw a rectangle on the original image
				cv2.rectangle(img, (x+_x, y+_y), (x+_x+_w, y+_y+_h), (0, 255, 0), 2)
				break
			
			# detect actual (right) eye
			eye = self.detect(croppedRightEyes, self.cascadeRightEye, minSize=(35,35))
			for (_x,_y,_w,_h) in eye:
				# same as above but!!!
				# taking into account that the we have cropped the image of the face even more
				# this means we have to add up half of the face s an offset
				# otherwise the right eye rectangle would be displayed on top of the left one  
				cv2.rectangle(img, (x+_x+w/2, y+_y), (x+_x+w/2+_w, y+_y+_h), (0, 255, 0), 2)
				break
			
			# the nose is in the middle of the image so no real easy way to cut that part out
			# in other words, fuck it! just search in the complete face
			# btw, known to work kinda crappy
			nose = self.detect(cropped, self.cascadeNose, minSize=(20,20))
			for (_x,_y,_w,_h) in nose:
				cv2.rectangle(img, (x+_x,y+_y), (x+_x+_w,y+_y+_h), (128, 128, 128), 2)
				break
			
			# mouth (should be) in the bottom part of the face
			# so make sure to search there by creating a range starting from 
			# half of the face downwards
			croppedMouth = grayImg[y+h/2:y+h, x:x+w]
			mouth = self.detect(croppedMouth, self.cascadeMouth, minSize=(30,30))
			for (_x,_y,_w,_h) in mouth:
				cv2.rectangle(img, (x+_x,y+_y+h/2), (x+_x+_w,y+_y+h/2+_h), (255, 0, 0), 2)
				break
		
		# let openCv show the edited image
		cv2.imshow('img',img)

	def detect(self, img, cascade, scaleFactor=1.2, minNeighbors=1, minSize=(100, 100), flags=cv.CV_HAAR_DO_CANNY_PRUNING):  # @UndefinedVariable
		return cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, flags=flags)



if __name__ == "__main__":
	# initialize class
	main = Main()
	# while true is true
	while True:
		# search for the face including parts :)
		main.search()
		# wait for 10 milliseconds and check if we press key 27
		# key 27 == esc
		if cv2.waitKey(10)==27:
			# if we did press esc, read below :)
			cv2.destroyAllWindows()
			sys.exit()