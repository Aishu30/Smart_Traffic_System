import numpy as np
import cv2
import os
import pandas as pd


#Load Traffic cam video

fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
pts = np.array([[(220,30),(305,30),(320,240),(150,240), (140,100)]], dtype = np.int32)

rootdir = 'video/'
traffic = []

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filename = os.path.join(subdir, file)
        vehicleCount = []
        crowdDensity = []
        count = 0
        cap = cv2.VideoCapture(filename)
        ret, frame = cap.read()
        previous = frame
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:

                # Load our cascade classifier from cars3.xml
                car_cascade = cv2.CascadeClassifier('cars3.xml')
                car_cascade1 = cv2.CascadeClassifier('cars.xml')
        
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_p = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

                cv2.polylines(frame,pts,True,(0,255,255))

                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, pts, (255,255,255))
                #mask = cv2.bitwise_not(gray)
                masked_image1 = cv2.bitwise_and(gray, mask)
                cv2.imshow('mask',masked_image1)

                mask = np.zeros(gray_p.shape, dtype=np.uint8)
                cv2.fillPoly(mask, pts, (255,255,255))
                #mask = cv2.bitwise_not(gray)
                masked_image2 = cv2.bitwise_and(gray_p, mask)
                cv2.imshow('mask',masked_image2)

                fgmask = fgbg.apply(masked_image1)
        
                cars = car_cascade.detectMultiScale(masked_image1, 1.05, 1)
                count = len(cars)
                cars1 = car_cascade1.detectMultiScale(masked_image1, 1.05, 1)
                draw = frame.copy()
                for (x,y,w,h) in cars:
                    cv2.rectangle(draw,(x,y),(x+w,y+h),(255,0,0),2)
                for (x,y,w,h) in cars1:
                    if (x,y,w,h) not in cars:
                        cv2.rectangle(draw,(x,y),(x+w,y+h),(255,0,0),2)
                        count += 1

                frame_diff = cv2.absdiff(masked_image1,masked_image2)
                cv2.imshow('framediff', frame_diff)
                white = cv2.countNonZero(frame_diff)
                black = cv2.countNonZero(masked_image1)
                #print(len(cars)+len(cars1), white/float(black))
                vehicleCount.append(count)
                crowdDensity.append(white/float(black))
                cv2.imshow('frame', frame)
                cv2.imshow('gray',gray)
                cv2.imshow('fgmask',fgmask)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                previous = frame.copy()

            else:
                cap.release()
                cv2.destroyAllWindows()
                traffic.append([file, max(vehicleCount), np.mean(crowdDensity)])
                print(file, 'processed...')
                print('Max. Vehicle Count : ',max(vehicleCount),', Avg. Crowd Density : ',np.mean(crowdDensity))
                break

df = pd.DataFrame(traffic)
df.to_csv('traffic.csv')
