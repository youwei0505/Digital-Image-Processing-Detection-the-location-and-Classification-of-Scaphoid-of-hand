import sys
import csv
import cv2
import numpy as np
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtWidgets import QComboBox
# 路徑檔案
import os
from os import listdir
from os.path import isfile, isdir, join
# Image
from PIL import Image, ImageQt, ImageEnhance
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
# classify
from yolov5.classifier import *
# detect 
from yolov5.detect import *
# cal iou
from yolov5.cal_iou import *
import json


#def crop(filename):
    # img = cv2.imread("./images/" + filename + ".bmp")
    # # Crop the Scaphoid
    # annot = read_json_annotation("./annotations/" + filename + ".json")
    # dict = {}
    # dict.update(annot[0])
    # coordinates = dict["bbox"]
    # w = float(coordinates[2]) - float(coordinates[0])
    # h = float(coordinates[3]) - float(coordinates[1])
    # w = h = 224
    # crop = img[int(coordinates[1]):int(coordinates[1])+int(h), int(coordinates[0]):int(coordinates[0])+int(w)]
    # crop_frac = crop.copy()
    # # Draw the bbox of fracture part
    # data = []
    # with open("./Fracture_Coordinate/"+filename+".csv", newline='') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         data.append(row)
    # center_x = int(data[1][0])
    # center_y = int(data[1][1])
    # width = int(data[1][2])
    # height = int(data[1][3])
    # angle = int(data[1][4])
    # rect = ((center_x, center_y), (width, height), angle)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(crop_frac, [box], 0, (80, 80, 255), 2)
    # return crop, crop_frac



def p_r_f1():
    global dic
    # path  : source img has scrophoid
    # path2 : img has fracture
    # scaphoid_crop : cropped 

    # Append 1 for all images, since they all have Scaphoid
    # Source  Noraml
    for file in os.listdir(path):
        name = os.path.splitext(file)[0]
        dic[name] = [1]
    # Source Fracture 
    for file in os.listdir(path2):
        name = os.path.splitext(file)[0]
        dic[name] = [1]
    # Append 1 for images which detected with Scaphoid, else 0
    # Cropped Normal
    for file in os.listdir(path):
        name = os.path.splitext(file)[0]
        if not os.path.exists(scaphoid_crop + name + ".jpg"):
            x = dic.get(name)
            x.append(0)
            dic[name] = x
        else:
            x = dic.get(name)
            x.append(1)
            dic[name] = x
    # Cropped Normal
    for file in os.listdir(path2):
        name = os.path.splitext(file)[0]
        if not os.path.exists(scaphoid_crop2 + name + ".jpg"):
            x = dic.get(name)
            x.append(0)
            dic[name] = x
        else:
            x = dic.get(name)
            x.append(1)
            dic[name] = x
    # Append 0 for all normal images, 1 for all fracture images
    for file in os.listdir(path):
        name = os.path.splitext(file)[0]
        x = dic.get(name)
        x.append(0)
        dic[name] = x
    for file in os.listdir(path2):
        name = os.path.splitext(file)[0]
        x = dic.get(name)
        x.append(1)
        dic[name] = x
    # Same for Fracture
    for file in os.listdir(path):
        name = os.path.splitext(file)[0]
        if name in set(fracORnorm):
            x = dic.get(name)
            x.append(1)
            dic[name] = x
        else:
            x = dic.get(name)
            x.append(0)
            dic[name] = x
    for file in os.listdir(path2):
        name = os.path.splitext(file)[0]
        if name in set(fracORnorm):
            x = dic.get(name)
            x.append(1)
            dic[name] = x
        else:
            x = dic.get(name)
            x.append(0)
            dic[name] = x

def calculate_metrics():
    global dic, tp, fn, fp, tn, tp2, fn2, fp2, tn2
    for x in dic.values():
        # Scaphoid
        if x[0] == 1 and x[1] == 1:
            tp += 1
        if x[0] == 0 and x[1] == 0:
            tn += 1
        if x[0] == 1 and x[1] == 0:
            fn += 1
        if x[0] == 0 and x[1] == 1:
            fp += 1
        # Fracture
        if x[2] == 1 and x[3] == 1:
            tp2 += 1
        if x[2] == 0 and x[3] == 0:
            tn2 += 1
        if x[2] == 1 and x[3] == 0:
            fn2 += 1
        if x[2] == 0 and x[3] == 1:
            fp2 += 1

    scaphoid_precision = tp / (tp+fp)
    scaphoid_recall = tp / (tp+fn)
    scaphoid_f1 = (2*scaphoid_precision*scaphoid_recall) / (scaphoid_precision+scaphoid_recall)
    fracture_precision = tp2 / (tp2 + fp2)
    fracture_recall = tp2 / (tp2 + fn2)
    fracture_f1 = (2 * fracture_precision * fracture_recall) / (fracture_precision + fracture_recall)
    return scaphoid_precision, scaphoid_recall, scaphoid_f1, fracture_precision, fracture_recall, fracture_f1

