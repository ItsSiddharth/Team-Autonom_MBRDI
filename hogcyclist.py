import cv2
import numpy as np 
import imutils 
import argparse
import os
winStride= (4,4)
meanShift= False
padding= (0,0)
scale= 1.05

parser= argparse.ArgumentParser(description= 'Finds bounding box on all thin cross section objects')
parser.add_argument('filename',nargs=1, help='a filename for video/image to be tested')
args= parser.parse_args()

fn, file_ext = os.path.splitext(args.filename[0])
if(file_ext=='.jpeg' or file_ext=='png' or file_ext=='jpg'):
        frame= cv2.imread(args.filename[0])
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        frame = imutils.resize(frame, 750, 1200)
        (rects, weights) = hog.detectMultiScale(frame, winStride=winStride,padding=padding, scale=scale, useMeanshiftGrouping=meanShift)
        for(x,y,w,h) in rects: 
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.imshow('fr', frame)
        cv2.waitKey(0)
else:
        cap = cv2.VideoCapture(args.filename[0])
        while(cap.isOpened()):
                ret,frame = cap.read()
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                frame = imutils.resize(frame, 750, 1200)
                (rects, weights) = hog.detectMultiScale(frame, winStride=winStride,
                        padding=padding, scale=scale, useMeanshiftGrouping=meanShift)
                for(x,y,w,h) in rects: 
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.imshow('fr', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
        cap.release() 
        cv2.destroyAllWindows()   
        
