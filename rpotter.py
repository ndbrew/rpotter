#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
  _\
  \
O O-O
 O O
  O
  
Raspberry Potter
Ollivander - Version 0.2 
Use your own wand or your interactive Harry Potter wands to control the IoT.  
Copyright (c) 2016 Sean O'Brien.  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import random
import numpy as np
import cv2
from cv2 import *
import picamera #only needed if using RPi camera module
import threading
import sys
import math
import time
import warnings
import os
from paho.mqtt import client as mqtt_client
from from dotenv import load_dotenv

# MQTT Parameters
FIRST_RECONNECT_DELAY = 1
RECONNECT_RATE = 2
MAX_RECONNECT_COUNT = 12
MAX_RECONNECT_DELAY = 60

# Image Parameters
lk_params = dict( winSize  = (20,20),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# blur_params = (4,4)
dilation_params = (5, 5)
movment_threshold = 10

# start capturing
CAM_W_RES=640
CAM_H_RES=480

ig = [[0] for x in range(20)]

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

def connect_mqtt(broker, port):
    print("MQTT Connecting")
    client = mqtt_client.Client()
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def on_disconnect(client, userdata, rc):
    logging.info("Disconnected with result code: %s", rc)
    reconnect_count, reconnect_delay = 0, FIRST_RECONNECT_DELAY
    while reconnect_count < MAX_RECONNECT_COUNT:
        logging.info("Reconnecting in %d seconds...", reconnect_delay)
        time.sleep(reconnect_delay)

        try:
            client.reconnect()
            logging.info("Reconnected successfully!")
            return
        except Exception as err:
            logging.error("%s. Reconnect failed. Retrying...", err)

        reconnect_delay *= RECONNECT_RATE
        reconnect_delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
        reconnect_count += 1
    logging.info("Reconnect failed after %s attempts. Exiting...", reconnect_count)

def publish(client, topic, spell):
    result = client.publish(topic, spell)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{spell}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")

def decrease_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 0 + value
    v[v <= lim ] = 0
    v[v > lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def Spell(spell):    
    #clear all checks
    ig = [[0] for x in range(15)] 
    #Invoke IoT (or any other) actions here
    cv2.putText(mask, spell, (5, 25),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
    #renamed spell Aguamenti but kept Incendio naming - fix later
    print("CAST: %s" %spell)
    publish(client,spell)
    cv2.putText(frame, spell, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

def IsGesture(a,b,c,d,i):
    print("point: %s" % i)
    #look for basic movements - TODO: trained gestures
    if ((a<(c-5))&(abs(b-d)<2)):
        ig[i].append("left")
    elif ((c<(a-5))&(abs(b-d)<2)):
        ig[i].append("right")
    elif ((b<(d-5))&(abs(a-c)<5)):
        ig[i].append("up")
    elif ((d<(b-5))&(abs(a-c)<5)):
        ig[i].append("down")
    #check for gesture patterns in array
    astr = ''.join(map(str, ig[i]))
    if "rightrightupup" in astr:
        ig[i]=[]
        Spell("Lumos")
    elif "rightrightdowndown" in astr:
        ig[i]=[]
        Spell("Nox")
	#Colovaria spell removed
    #elif "leftdown" in astr:
    #    Spell("Colovaria")
    # elif "leftup" in astr:
    #     Spell("Aguamenti")
    if len(ig[i]) > 10:
        ig[i]=[]    
    print(astr)
        
def capture_frame():
    _, frame = cam.read()
    cv2.flip(frame,1,frame)
    # Mask detection area if dmask is defined
#    mask = np.zeros_like(frame)
#    cv2.rectangle(dmask,(300,200),(680,300),255,cv2.FILLED)
#    frame_gray = cv2.bitwise_and(frame,dmask)
    #frame_gray = decrease_brightness(frame_gray, value=30)
    frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)
    equalizeHist(frame_gray)
    frame_gray = GaussianBlur(frame_gray,(9,9),1.5)
    dilate_kernel = np.ones(dilation_params, np.uint8)
    frame_gray = cv2.dilate(frame_gray, dilate_kernel, iterations=1)
    frame_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    frame_gray = frame_clahe.apply(frame_gray)
    return frame,frame_gray

def FindWand(cam):
    try:
        frame, frame_gray = capture_frame(cam)
        p0 = cv2.HoughCircles(frame_gray,cv2.HOUGH_GRADIENT,3,50,param1=240,param2=8,minRadius=3,maxRadius=15)
        if p0 is not None:
            p0.shape = (p0.shape[1], 1, p0.shape[2])
            p0 = p0[:,:,0:2]
            # Generate empty mask
            mask = np.zeros_like(old_frame)
        print("finding...")
        cv2.imshow("FindWand",frame)
        cv2.imshow("FindWandGray",frame_gray)
        threading.Timer(5, FindWand).start()
        return p0, mask
    except:
        e = sys.exc_info()[1]
        print("Error: %s" % e) 
        exit

def TrackWand(cam):
        frame, frame_gray = capture_fram(cam)

        while True:
            try:
                frame, frame_gray = capture_frame(cam)
                if p0 is not None:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                    # p2 = cv2.HoughCircles(old_gray,cv2.HOUGH_GRADIENT,3,50,param1=240,param2=30,minRadius=4,maxRadius=10)
                    # p2 = p2[:,:,0:2]
                    # p2.shape = (p2.shape[1], 1, p2.shape[2])

                    # for i,circles in enumerate(p2):
                    #     a,b = circles.ravel()
                    #     cv2.circle(frame,(a,b),5,color,-1)
                    #     cv2.putText(frame, str(i), (a,b), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
                    # cv2.imshow("Raspberry Potter", frame)

                    
                    # Select good points
                    good_new = p1[st==1]
                    good_old = p0[st==1]

                    # draw the tracks
                    for i,(new,old) in enumerate(zip(good_new,good_old)):
                        a,b = new.ravel()
                        c,d = old.ravel()
                        # only try to detect gesture on highly-rated points (below 10)
                        if (i<10):
                            IsGesture(a,b,c,d,i)
                        dist = math.hypot(a - c, b - d)
                        if (dist<movment_threshold):
                            cv2.line(mask, (a,b),(c,d),(0,255,0), 2)
                        cv2.circle(frame,(a,b),5,color,-1)
                        cv2.putText(frame, str(i), (a,b), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255)) 
                    img = cv2.add(frame,mask)
                    cv2.putText(img, "Press ESC to close.", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                    cv2.imshow("img",img)
                #cv2.imshow("Raspberry Potter", frame)
                cv2.imshow("Frame Gray", frame_gray)

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
            except IndexError:
                print("Index error - Tracking")  
            except:
                e = sys.exc_info()[0]
                #print("Tracking Error: %s" % e)

            key = cv2.waitKey(20)
            if key in [27, ord('Q'), ord('q')]: # exit on ESC
                # cv2.destroyAllWindows()
                cam.release()
                break

try:
    load_dotenv()
    topic = os.getenv('TOPIC')
    broker = os.getenv('BROKER')
    port = os.getenv('PORT')

    print("Initializing mqtt client")
    client = connect_mqtt(broker, port)
    client.on_disconnect = on_disconnect

    print("Initializing Camera")
    cam = cv2.VideoCapture(-1)
    cam.set(3, CAM_W_RES)
    cam.set(4, CAM_H_RES)
    
    print("Initializing point tracking")
    FindWand(cam)
    #TrackWand(cam)  
finally:   
    cam.release()
 

