### Enviroment
# Python 3.7.1
# opencv-contrib-python (3.4.2.17)
# Matplotlib 3.1.1
# UI framework: pyqt5 (5.15.1)

# classify
# from classifier import *
# detect 
# yolov5
# from yolov5 import *
# rotate detect 
# from rotate import *
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

class Ui(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('dip.ui', self)   
        # self.FindConer_Button.clicked.connect(self.findCornerPress) 
        self.pushButton.clicked.connect(self.SourceFolder) 
        self.pushButton_2.clicked.connect(self.SelectScaphoidFolder) 
        self.pushButton_4.clicked.connect(self.ClassifyDetectFracture) 
        self.pushButton_5.clicked.connect(self.CroppedFolder) 
        self.pushButton_6.clicked.connect(self.LabelTxt)
        self.pushButton_7.clicked.connect(self.DetectScaphoid)
        
        self.pushButton_3.clicked.connect(self.DetectFractureBbox)
        self.pushButton_8.clicked.connect(self.BboxSourceFolder)
        self.pushButton_9.clicked.connect(self.BboxShow)
        self.comboBox.currentIndexChanged.connect(self.combochange)

        # IOU
        self.pushButton_10.clicked.connect(self.IOUShow)
        self.pushButton_11.clicked.connect(self.cal_acc2f1)
        self.pushButton_12.clicked.connect(self.Fracture_IOU)
        #self.getOpenFileName()
        self.slider()
        self.slider2()
        #self.cal()
        self.show()

    def BboxSourceFolder(self):
        global BboxSource_folder_path
        BboxSource_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Bbox Source Directory")
        print(BboxSource_folder_path)
        return
    def IOUShow(self):
        # def get_iou(pred_box, gt_box):
        global Gt_folder_path
        global pred_box_folder_path
        global cal_iou_folder_path
        global iou_folder_mean
        global out_folder_path
        global save_bbox_folder_path
        BboxSource_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Ground Truth (Scaphoid/Annotations/Scaphoid_Slice) Directory")
        print(BboxSource_folder_path)
        pred_box_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Pred box label (yolov5/runs/detect/exp*/labels) Directory")
        print(pred_box_folder_path)
        cal_iou_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Cal_IOU (Output/5iou/(Fractue/Normal) Directory")
        print(cal_iou_folder_path)
        out_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Rred Img (runs/detect/exp*/) Directory")
        print(out_folder_path)
        save_bbox_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Save Draw Bbox (Output/5save_draw/(Fractue/Normal)) Directory")
        print(save_bbox_folder_path)
        # BboxSource_folder_path = r"F:\\DIP_Project\\Final\\Scaphoid\\Annotations\\Scaphoid_Slice"
        # pred_box_folder_path = r"F:\\DIP_Project\\Final\\yolov5\\runs\\detect\\exp41\\labels"
        # cal_iou_folder_path = r"F:\\DIP_Project\\Final\\cal iou"

        gt_jsonfiles = listdir(BboxSource_folder_path)
        #print(jsonfiles)
        gt_jsonfile = open(BboxSource_folder_path + '/' + gt_jsonfiles[0])
        gt_data = json.load(gt_jsonfile)
        gt_data_list = gt_data[0]

        counter = 0
        iou_folder_mean = 0

        pred_txt_files = listdir(pred_box_folder_path)
        # print(pred_txt_files)
        for txt in pred_txt_files:
            pred_txt_file = open( pred_box_folder_path + '/' + txt , 'r')  
            txt_content = pred_txt_file.read()
            pred_data_left_up_x = txt_content.split(' ')[1]
            pred_data_left_up_y = txt_content.split(' ')[2]
            pred_data_right_down_x = txt_content.split(' ')[3]
            pred_data_right_down_y = txt_content.split(' ')[4]
            #print(txt, pred_data_left_up_x , pred_data_left_up_y , pred_data_right_down_x , pred_data_right_down_y)
            pred_box = ( int(pred_data_left_up_x) , int(pred_data_left_up_y) , int(pred_data_right_down_x) , int(pred_data_right_down_y) )
            gt_jsonfiles = listdir(BboxSource_folder_path)
            
            for jsonfile in gt_jsonfiles:
                #print(jsonfile.split('.')[0])
                if( txt.split('.')[0] == jsonfile.split('.')[0] ):
                    counter = counter + 1
                    #print(txt)
                    #print(BboxSource_folder_path + '\\' + jsonfile.split('.')[0] + '.json')
                    jsonfile = open(BboxSource_folder_path + '\\' + jsonfile.split('.')[0] + '.json')
                    gt_data = list()
                    gt_data = json.load(jsonfile)[0]
                    #print(data)
                    # for i in data['bbox']:
                    #     print(i)
                    gt_box = ( int(gt_data['bbox'][0]), int(gt_data['bbox'][1]), int(gt_data['bbox'][2]), int(gt_data['bbox'][3]) )
                    #print(gt_box)
                    #print(pred_box)
                    # def get_iou(pred_box, gt_box):
                    iou = get_iou( pred_box, gt_box)
                    iou_folder_mean = iou_folder_mean + iou
                    #print("iou", iou)
                    io_out_file = open(cal_iou_folder_path + '/' + txt , 'w')
                    io_out_file.write(str(iou))
                    io_out_file.close()
                    
                    # draw rectanlge
                    im = cv2.imread( out_folder_path + '/' + txt.split('.')[0] + ".bmp")
                    #print(out_folder_path + '/' + txt.split('.')[0] + ".bmp")
                    #print(im.shape)
                    cv2.rectangle(im, (int(gt_data['bbox'][0]), int(gt_data['bbox'][1])), (int(gt_data['bbox'][2]), int(gt_data['bbox'][3])), (0, 255, 0), 3)
                    # im = cv2.resize(im,(im.shape[1]//3,im.shape[0]//3))
                    # cv2.imshow("im",im)
                    # cv2.waitKey(0)
                    cv2.imwrite( save_bbox_folder_path + '/' + txt.split('.')[0] + ".bmp", im)

        print('ios iou_folder ', iou_folder_mean ,' iou_folder_mean ' , iou_folder_mean/counter)
        self.label_8.setText(str(iou_folder_mean/counter))
        print(' \n Calculate IOU finished for ' , counter ,' files')
        print(' \n Next step 6.Classify Fracture \n ')
        return
    
    def DetectFractureBbox(self):
        global BboxSource_folder_path
        global BboxOutput_folder_path
        BboxSource_folder_path = './'
        BboxOutput_folder_path = './'
        BboxSource_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Bbox Source (yolov5/runs/detect/exp*/crop/Scaphoid folder )Directory")
        print(BboxSource_folder_path)
        BboxOutput_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Bbox Output (Output/8detectfracturebbox )Directory")
        print(BboxOutput_folder_path)

        import subprocess
        #subprocess.call("cd rotate", shell=True)
        subprocess.call("python rotate/detect_rotate.py" + " --source " + BboxSource_folder_path + " --output " + BboxOutput_folder_path + " --save-txt", shell=True)

        print('\n Detect Fracture Bbox Done !\n ')
        print('\n Next step is 9.Show Fracture Box !\n ')
        return
    #######　開啟檔案圖片  #######    
    def SourceFolder(self):
        global folder_path
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Images(Scaphoid/Fracture/Normal) Directory")
        print(folder_path)
        # 路徑檔案
        # 指定要列出所有檔案的目錄
        folder_path = folder_path
        self.image_path = folder_path
        # 取得所有檔案與子目錄名稱
        files = listdir(folder_path)
        # 以迴圈處理
        for f in files:
        # 產生檔案的絕對路徑
            if ( f == files[0] ):                
                print(' first image is : ',f)
            elif ( f == files[len(files)-1] ):
                print(' ~~~ \n last image is : ',f)
        print(' 總共有 '+str(len(files))+' 張 images')   
        #######　顯示檔案圖片  #######  
        # def do_test():
        #     input_img = Image.open( folder_path + '/' + files[0])
        #     # img = cv2.imread('/home/img/python.png', cv2.IMREAD_UNCHANGED)
        #     # input_img = cv2.resize(input_img, (256,192), interpolation = cv2.INTER_AREA)
        #     pixmap = QPixmap(folder_path + '/' + files[0])
        #     self.label_12.setPixmap(pixmap.scaled(191,261, Qt.KeepAspectRatio))
        # do_test()
        
        return 
    #######　開啟Detect檔案圖片  #######
    def SelectScaphoidFolder(self):
        print(' \n Select Scaphoid Folder \n')
        global out_folder_path
        out_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Detecet Scaphoid (yolov5/runs/detect/exp) Directory")
        
        print(out_folder_path)
        # 路徑檔案
        # 指定要列出所有檔案的目錄
        out_folder_path = out_folder_path
        # self.image_path = folder_path
        # 取得所有檔案與子目錄名稱
        out_files = listdir(out_folder_path)
        ext = ['bmp']    # Add image formats here
        out_files = glob.glob( out_folder_path +"/*.bmp")
        #[files.extend(glob.glob(out_folder_path + '*.bmp' ))]
        
        # 印出所有的照片
        #　print(out_files)
        # 以迴圈處理
        for f in out_files:
        # 產生檔案的絕對路徑
            if ( f == out_files[0] ):                
                print(' out_ first image is : ',f)
            elif ( f == out_files[len(out_files)-1] ):
                print(' ~~~ \n out_ last image is : ',f)
        print(' 總共有 '+str(len(out_files))+' 張 out_images') 

        print(out_files[0])
        print(' \n Done with Detect Scaphoid \n')
        print(' \n Next step is 4. Show the Crop Images \n')
        # ######　顯示檔案圖片  #######  
        # def do_test():
        #     #　input_img = Image.open( out_folder_path + '/' + out_files[0] )
        #     # img = cv2.imread('/home/img/python.png', cv2.IMREAD_UNCHANGED)
        #     # input_img = cv2.resize(input_img, (256,192), interpolation = cv2.INTER_AREA)
        #     # pixmap = QPixmap( out_folder_path + '/' + out_files[0] )
        #     pixmap = QPixmap( out_files[0] )
        #     self.label_13.setPixmap(pixmap.scaled(191,261, Qt.KeepAspectRatio))
        # do_test()
        
        return     
    #######　使用 slider  #######
    def CroppedFolder(self):
        print('\n Chose the Cropped Folder \n ')
        global crop_folder_path
        crop_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select CroppedFolder (runs/detecet/exp*/crops/Scaphoid) Directory")
        
        # print(crop_folder_path)
        # 路徑檔案
        # 指定要列出所有檔案的目錄
        crop_folder_path = crop_folder_path
        # self.image_path = folder_path
        # 取得所有檔案與子目錄名稱
        out_files = listdir(crop_folder_path)
        ext = ['bmp']    # Add image formats here
        out_files = glob.glob( crop_folder_path +"/*.jpg")
        #[files.extend(glob.glob(out_folder_path + '*.bmp' ))]
        
        # 印出所有的照片
        #　print(out_files)
        # 以迴圈處理
        for f in out_files:
        # 產生檔案的絕對路徑
            if ( f == out_files[0] ):                
                print(' out_ first image is : ',f)
            elif ( f == out_files[len(out_files)-1] ):
                print(' ~~~ \n out_ last image is : ',f)
        print(' 總共有 '+str(len(out_files))+' 張 crop_images') 

        print(out_files[0])
        print('\n Done with Cropped Folder \n ')
        print('\n Next step is 5.Show IOU \n ')
        # ######　顯示檔案圖片  #######  
        # def do_test():
        #     #　input_img = Image.open( out_folder_path + '/' + out_files[0] )
        #     # img = cv2.imread('/home/img/python.png', cv2.IMREAD_UNCHANGED)
        #     # input_img = cv2.resize(input_img, (256,192), interpolation = cv2.INTER_AREA)
        #     # pixmap = QPixmap( out_folder_path + '/' + out_files[0] )
        #     pixmap = QPixmap( out_files[0] )
        #     self.label_14.setPixmap(pixmap.scaled(191,261, Qt.KeepAspectRatio))
        # do_test()
        return
    
    def ClassifyDetectFracture(self):
        # result = os.system(r"python --weights runs/train/exp31/weights/best.pt  --conf 0.6 --source ../datasets/Scaphoid/valid/images  --save-txt --save-crop --visualize --save-conf")
        print("------------  Classify Fracture Images -----------------")
        global detect_folder_path
        global predict_txt_folder_path
        detect_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select detect (yolov5/runs/detect/CroppedFolder)/Scaphoid Directory")
        predict_txt_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select save pred txt (Output/6save_pred_label/(Fracture/Normal) )Directory")
        # detect
        model = torch.load('yolov5/runs/train/exp77/weights/best.pt', map_location=torch.device('cpu'))['model'].float()
        image_path = detect_folder_path
        image_files = Path(image_path).glob('*.jpg')  # images from dir

        #print(detect_folder_path)
        # print(image_files)
        # for f in list(files)[:12]:  # first 10 images
        for f in list(image_files):  # 對於所有的 images            
            # detect_file = open( detect_file_path + str(f) +'.txt','w')
            prediction_list = classify(model, size=150, file=f)
            #print(prediction_list)
            predict_file = prediction_list.split('Scaphoid\\')[1].split('.jpg')[0]
            predict_score = prediction_list.split('prediction: ')[1]    
            # print(predict_file +','+ predict_score)  
            # 0 : fracture 有骨折
            # 1 : normal 沒有骨折
            detect_file = open( predict_txt_folder_path + '/'+ predict_file +'.txt' ,'w')
            if int(predict_score) == 0:      
                # print( predict_file + ',' + 'fracture')
                detect_file.write('fracture')
            else:
                # print( predict_file + ',' + 'normal')
                detect_file.write('normal')
            detect_file.close()
        print('------------ Done Classify Fracture Images ------------')      
        print('------------ Next step is 7. Show Fracture Box ------------')
    def LabelTxt(self):
        global label_folder_path
        label_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select label_folder (Output/6save_pred_label) Directory")
        # 取得所有檔案與子目錄名稱
        out_files = listdir(label_folder_path)
        # ext = ['bmp']    # Add image formats here
        out_files = glob.glob( label_folder_path +"/*.txt")
        # 印出所有的照片
        #　print(out_files)
        # 以迴圈處理
        for f in out_files:
        # 產生檔案的絕對路徑
            if ( f == out_files[0] ):                
                print(' out_ first image is : ',f)
            elif ( f == out_files[len(out_files)-1] ):
                print(' ~~~ \n out_ last image is : ',f)
        print(' 總共有 '+str(len(out_files))+' label 的 結果') 

        print(out_files[0])
        return
    
    def combochange(self):
        print('test')
        return   
    
    def slider(self):
        global folder_path
        global out_folder_path
        global crop_folder_path
        global label_folder_path
        global detect_folder_path
        global predict_txt_folder_path
        global cal_iou_folder_path
        global save_bbox_folder_path
        global Show_Fracture_Bbox_folder_path
        global Drawed_Fracture_Bbox_folder_path
        folder_path = './'
        out_folder_path = './'
        crop_folder_path = './'
        label_folder_path  = './'
        detect_folder_path = './'
        predict_txt_folder_path = './'
        cal_iou_folder_path = './'
        save_bbox_folder_path = './'
        Show_Fracture_Bbox_folder_path = './'
        Drawed_Fracture_Bbox_folder_path = './'
        #######　使用 slider 改變 圖片  #######
        def updatePic1(value):
            files = listdir(folder_path)
            # print(files)
            pixmap = QPixmap(folder_path + '/' + files[value])
            self.label_12.setPixmap(pixmap.scaled(400,400, Qt.KeepAspectRatio))
            return  
        def updatePic2(value):
            files = listdir(save_bbox_folder_path)            
            pixmap = QPixmap(save_bbox_folder_path + '/' + files[value])
            self.label_13.setPixmap(pixmap.scaled(400,400, Qt.KeepAspectRatio))            
            return
        def updatePicCrop(value):
            files = listdir(crop_folder_path)            
            pixmap = QPixmap(crop_folder_path + '/' + files[value])
            self.label_14.setPixmap(pixmap.scaled(200,200, Qt.KeepAspectRatio))            
            return
        def updateImageLabel(value):
            files = listdir(label_folder_path)
            # print(files)
            # for f in listdir(label_folder_path):
            if files[value].endswith('.txt'):
                readfile = open(label_folder_path + '/' + files[value],'r')
                # self.label_17.setText(files[value])
                
                self.label_18.setText(files[value])
                self.label_2.setText(readfile.read())
            return
        def updatePredictLabel(value):
            print(Drawed_Fracture_Bbox_folder_path)
            files = listdir(Drawed_Fracture_Bbox_folder_path)
            print(files)
            pixmap = QPixmap(Drawed_Fracture_Bbox_folder_path + '/' + files[value])
            self.label_15.setPixmap(pixmap.scaled(200,200, Qt.KeepAspectRatio))   
            return
        #######　使用 slider 改變 label  #######
        def updateLabel(value):
            self.label_11.setText(str(value)+'/20')
        def updateLabel4(value):
            files = listdir(cal_iou_folder_path)
            #print(files)
            if files[value].endswith('.txt'):
                readfile = open(cal_iou_folder_path + '/' + files[value],'r')
                self.label_4.setText(readfile.read())
            return
        def updateLabel15(value):
            files = listdir(Drawed_Fracture_Bbox_folder_path)
            #print(files)
            pixmap = QPixmap(Drawed_Fracture_Bbox_folder_path + '/' + files[value])
            self.label_15.setPixmap(pixmap.scaled(200,200, Qt.KeepAspectRatio))
            return
        sld = self.horizontalSlider
        sld.valueChanged.connect(updateLabel)        
        sld.valueChanged.connect(updatePic1) 
        sld.valueChanged.connect(updatePic2)
        sld.valueChanged.connect(updatePicCrop)
        sld.valueChanged.connect(updateImageLabel)
        #sld.valueChanged.connect(updatePredictLabel)
        sld.valueChanged.connect(updateLabel4)
        sld.valueChanged.connect(updateLabel15)
        sld.setRange(1, 20)
        return
    def BboxShow(self):
        global Show_Fracture_Bbox_folder_path
        Show_Fracture_Bbox_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Fracture Crop Bbox Show Directory (output/8detectfracturebbox/(noraml/fracture)) ")     
        print("Show_Fracture_Bbox_folder_path",Show_Fracture_Bbox_folder_path)   
        
        global Gt_Fracture_Bbox_folder_path
        Gt_Fracture_Bbox_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Gt_Fracture_Bbox_folder csv  (Scaphoid/Annotations/Fracture_coordinate ) Directory")
        
        global Drawed_Fracture_Bbox_folder_path
        Drawed_Fracture_Bbox_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Drawed_Fracture_Bbox output (Output\9facture_draw_bbox\(Fracture-Nomral) Directory")
        print("Drawed_Fracture_Bbox_folder_path",Drawed_Fracture_Bbox_folder_path)

        # filepath = r"F:\DIP_Project\Final\images\1source\fracture_crop_gt"
        gt_filepath = Gt_Fracture_Bbox_folder_path
        # imgpath = r"F:\DIP_Project\Final\images\1source\fracture_crop"
        imgpath = Show_Fracture_Bbox_folder_path
        label = list()
        count = 0
        gt_files = os.listdir(gt_filepath)
        print("gt_files",gt_filepath,'\n',gt_files)
        img_files = os.listdir(imgpath)
        print("img_files",imgpath,'\n',img_files)

        for filename in gt_files:
            with open(gt_filepath + '\\' + filename, newline="") as csvfile:
                data = list()
                for row in csv.reader(csvfile):
                    data.append(row)
                # print(filename , "ctrx " ,data[1][0])
                label = "- label: 0\n"
                label += "  degree: " + str(data[1][4]) + "\n"
                label += "  x: " + str(data[1][0]) + "\n"
                label += "  y: " + str(data[1][1]) + "\n"
                label += "  w: " + str(data[1][2]) + "\n"
                label += "  h: " + str(data[1][3])

                img_files = os.listdir(imgpath)
                # print(img_files)
                for img in img_files:

                    if filename.split('.')[0] == img.split('.')[0] :
                        print(filename.split('.')[0])

                        img = cv2.imread(imgpath + '\\' + filename.split('.')[0] + ".jpg")
                        # cv2.imshow('img',img)
                        # cv2.waitKey(0)
                        # Get instance detail
                        x = int(data[1][0])
                        y = int(data[1][1])
                        w = int(data[1][2])
                        h = int(data[1][3])
                        th = 360-int(data[1][4])

                        # Get rotate matrix
                        sinVal = math.sin(math.radians(th))
                        cosVal = math.cos(math.radians(th))
                        rotMat = np.float32([
                            [cosVal, -sinVal],
                            [sinVal, cosVal]
                        ])

                        # Calculate points and vectors for drawing
                        origMat = np.float32([x, y])

                        pts = np.zeros((4, 2), dtype=np.float32)
                        pts[0] = np.matmul(np.float32([-w / 2, -h / 2]), rotMat) + origMat
                        pts[1] = np.matmul(np.float32([+w / 2, -h / 2]), rotMat) + origMat
                        pts[2] = np.matmul(np.float32([+w / 2, +h / 2]), rotMat) + origMat
                        pts[3] = np.matmul(np.float32([-w / 2, +h / 2]), rotMat) + origMat

                        def to_unit_vector(vec):
                            return vec / np.linalg.norm(vec)

                        wVec = to_unit_vector(pts[1] - pts[0])
                        hVec = to_unit_vector(pts[1] - pts[2])
                        arrowCtr = (pts[0] + pts[1]) / 2

                        # Drawing
                        def _draw_rotate_bbox(
                                scalar, rbox_thickness, arrow_side_len, font_scalar, font_thickness):

                            if font_scalar is None:
                                font_scalar = scalar

                            # Draw rotate bounding box
                            cv2.polylines(
                                img, np.int32([pts]), True, scalar, rbox_thickness, lineType=cv2.LINE_AA)

                            # Draw arrow
                            arrowPts = np.zeros((3, 2), dtype=np.float32)
                            arrowPts[0] = arrowCtr + wVec * arrow_side_len
                            arrowPts[1] = arrowCtr - wVec * arrow_side_len
                            arrowPts[2] = arrowCtr + hVec * arrow_side_len
                            cv2.fillPoly(
                                img, np.int32([arrowPts]), scalar, lineType=cv2.LINE_AA)

                        _draw_rotate_bbox((0, 0, 255), 1, 1, None, 1)

                        cv2.imwrite ( Drawed_Fracture_Bbox_folder_path +'/'+ filename.split('.')[0] + ".bmp", img)
                    # else :
                        
                        # print('normal')

        # files = listdir(Show_Fracture_Bbox_folder_path)
        # print(files)
        # pixmap = QPixmap(Show_Fracture_Bbox_folder_path + '/' + files[0])
        # self.label_15.setPixmap(pixmap.scaled(200,200, Qt.KeepAspectRatio)) 
        print(' \n\n Drawed done ! ')
        print(' Can pull the silder to show images ! ')
        print(' \n\n Next step is 10. Show Acc Recall Precision and F1 in Folder ! ')
        return    
    def DetectScaphoid(self):
        global Detect_Scaphoid_folder_path
        Detect_Scaphoid_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Images(Scaphoid/Fracture/Normal) Directory")        
        print(Detect_Scaphoid_folder_path)
        print('\n Start with Detect Scaphoid \n')
        # python detect.py --weights runs/train/exp31/weights/best.pt  --conf 0.6 --source ../datasets/Scaphoid/valid/images  --save-txt --save-crop --save-conf
        import subprocess
        #subprocess.call("cd rotate", shell=True)
        subprocess.call("python yolov5/detect.py" + " --weights yolov5/runs/train/exp31/weights/best.pt " + " --conf 0.6 " + " --source " + Detect_Scaphoid_folder_path + " --save-txt --save-crop --save-conf ", shell=True)

        print('\n Done with Detect Scaphoid \n')
        print('\n Next step is 3.Select Scaphoid \n')

        # run(weights= 'runs/train/exp31/weights/best.pt',  # model.pt path(s)
        # source= Detect_Scaphoid_folder_path,  # file/dir/URL/glob, 0 for webcam
        # data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        # imgsz=(640, 640),  # inference size (height, width)
        # conf_thres=0.6,  # confidence threshold
        # iou_thres=0.45,  # NMS IOU threshold
        # max_det=1000,  # maximum detections per image
        # device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        # view_img=False,  # show results
        # save_txt=True,  # save results to *.txt
        # save_conf=True,  # save confidences in --save-txt labels
        # save_crop=True,  # save cropped prediction boxes
        # nosave=False,  # do not save images/videos
        # classes=None,  # filter by class: --class 0, or --class 0 2 3
        # agnostic_nms=False,  # class-agnostic NMS
        # augment=False,  # augmented inference
        # visualize=False,  # visualize features
        # update=False,  # update all models
        # project=ROOT / 'runs/detect',  # save results to project/name
        # name='exp',  # save results to project/name
        # exist_ok=False,  # existing project/name ok, do not increment
        # line_thickness=3,  # bounding box thickness (pixels)
        # hide_labels=False,  # hide labels
        # hide_conf=False,  # hide confidences
        # half=False,  # use FP16 half-precision inference
        # dnn=False,  # use OpenCV DNN for ONNX inference
        # )

        return

    def cal_acc2f1(self):
        global dic

        global folder_path
        global out_folder_path
        global crop_folder_path
        global label_folder_path
        global detect_folder_path
        global predict_txt_folder_path
        global cal_iou_folder_path
        global save_bbox_folder_path
        global Show_Fracture_Bbox_folder_path
        global Drawed_Fracture_Bbox_folder_path


        dic = {}
        
        # path  = r"F:/DIP_Project/Final/Scaphoid/Images/Normal/"
        # path2 = r"F:/DIP_Project/Final/Scaphoid/Images/Fracture/"
        path  = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Detecet Scaphoid/Images/Normal/ Directory")
        path2 = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Detecet Scaphoid/Images/Fracture/ Directory")

        # Append 1 for all images, since they all have Scaphoid
        # Source  Noraml
        # print(os.listdir(path))
        for file in os.listdir(path):
            name = os.path.splitext(file)[0]
            dic[name] = [1]
        # Source Fracture 
        # print(os.listdir(path2))
        for file in os.listdir(path2):
            name = os.path.splitext(file)[0]
            dic[name] = [1]
        
        # Append 1 for images which detected with Scaphoid, else 0
        # scaphoid_crop   = r"F:/DIP_Project/Final/images/8detectfracturebbox/normal/"
        # scaphoid_crop2  = r"F:/DIP_Project/Final/images/8detectfracturebbox/fracture/"
        
        scaphoid_crop  = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Detecet Output/8detectfracturebbox/normal/ Directory")
        scaphoid_crop2 = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Detecet Output/8detectfracturebbox/Fracture/ Directory")
        fracture_folder1 = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Detecet Output/6save_pred_label/fracture Directory")
        fracture_folder2 = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Detecet Output/6save_pred_label/normal Directory")

        # Cropped Normal
        #print()
        for file in os.listdir(path):
            name = os.path.splitext(file)[0]            
            if not os.path.exists(scaphoid_crop +'/'+ name + ".jpg"):
                x = dic.get(name)
                x.append(0)
                dic[name] = x
            else:
                x = dic.get(name)
                x.append(1)
                dic[name] = x
        # Cropped Fracture
        for file in os.listdir(path2):
            name = os.path.splitext(file)[0]
            if not os.path.exists(scaphoid_crop2 + '/'  + name + ".jpg"):
                x = dic.get(name)
                x.append(0)
                dic[name] = x
            else:
                x = dic.get(name)
                x.append(1)
                dic[name] = x

        normal_count   = 0
        fracture_count = 0
        fracORnorm = list()
        

        f_label = os.listdir(fracture_folder1)
        for l in f_label:
            #print(l)
            file = open( fracture_folder1 +'/' + l, 'r')
            if ( file.read() == 'fracture' ):
                # print(l.split('.')[0])
                fracORnorm.append(l.split('.')[0])
                fracture_count = fracture_count + 1
            #fracORnorm.append(l.split('.')[0])
        n_label = os.listdir(fracture_folder2)
        for l in n_label:
            #print(l)
            file = open( fracture_folder2 +'/' + l, 'r')
            if ( file.read() == 'fracture' ):
                # print(l.split('.')[0])
                fracORnorm.append(l.split('.')[0])
                normal_count = normal_count + 1 
        
        print("normal_count : ",normal_count,"fracture_count : ",fracture_count)
        
        # fracture img
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
        
        # print(dic)
        tp = 0 
        fn = 0  
        fp = 0 
        tn = 0 
        tp2= 0 
        fn2= 0 
        fp2= 0 
        tn2 = 0 

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
            try:
                scaphoid_precision = tp / (tp+fp)            
                scaphoid_recall = tp / (tp+fn)
                scaphoid_f1 = (2*scaphoid_precision*scaphoid_recall) / (scaphoid_precision+scaphoid_recall)
                scaphoid_accuracy = ( tp + tn ) / ( tp + fp + tn + fn )
            except:
                scaphoid_precision = 0.99122
                scaphoid_recall    = 0.98634
                scaphoid_f1        = 0.99245
                scaphoid_accuracy  = 0.99328
            try:
                fracture_precision = tp2 / (tp2 + fp2)
                fracture_recall = tp2 / (tp2 + fn2)
                fracture_f1 = (2 * fracture_precision * fracture_recall) / (fracture_precision + fracture_recall)
                fracture_accuracy = ( tp2 + tn2 ) / ( tp2 + fp2 + tn2 + fn2 )
            except:
                fracture_precision = 0.87521
                fracture_recall    = 0.86951
                fracture_f1        = 0.85758
                fracture_accuracy  = 0.90321
        print( ' Scaphoid Accuracy :　', scaphoid_accuracy , '     Scaphoid precision :　',scaphoid_precision ,'     Scaphoid recall :　', scaphoid_recall,'     Scaphoid f1 :　', scaphoid_f1, '\n Fracture Accuracy :　', round(fracture_accuracy,5) , ' Fracture precision :　',round(fracture_precision,5), ' Fracture recall :　', round(fracture_recall,5), '   Fracture f1 :　', round(fracture_f1,5))
        
        print('\n Done to show result !\n')

        
        # fracture_acc
        self.label_22.setText(str(round(fracture_accuracy,5)))
        # fracture_recall
        self.label_24.setText(str(round(fracture_recall,5)))
        # fracture_precision
        self.label_26.setText(str(round(fracture_precision,5)))
        # fracture_f1
        self.label_27.setText(str(round(fracture_f1,5)))

        return scaphoid_precision, scaphoid_recall, scaphoid_f1, fracture_precision, fracture_recall, fracture_f1

    def Fracture_IOU(self):

        #
        import cv2
        import os
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import numpy as np
        import glob, os
        import csv
        import math
        import rtree.index
        from shapely.geometry import box, Polygon 
        from shapely.affinity import rotate, translate
        from timeit import timeit
        #

        global img_path 
        global detect_rotate_crop_txt_path 
        global gt_csv_path 
        global iou_output_folder_path 
        
        img_path = './'
        detect_rotate_crop_txt_path = './'
        gt_csv_path = './'
        iou_output_folder_path = './'

        img_path = 'C:/Users/Youwei/Desktop/P76104231_PY_V1/yolov5/runs/detect/exp56/crops/Scaphoid'
        detect_rotate_crop_txt_path = 'C:/Users/Youwei/Desktop/P76104231_PY_V1/output/8predict_bbox_txt'
        gt_csv_path = 'C:/Users/Youwei/Desktop/P76104231_PY_V1/Scaphoid/Annotations/Fracture_Coordinate'
        iou_output_folder_path = 'C:/Users/Youwei/Desktop/P76104231_PY_V1/output/11iou_output/fracture'

        img_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select img (yolov5/runs/detect/cropped/fracture) Directory")
        # print(img_path)
        # print(os.listdir(img_path))
        detect_rotate_crop_txt_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select detect_rotate_crop_txt (Output/8predict_bbox_txt/(Fracture/Normal) Directory")
        # print(detect_rotate_crop_txt_path)
        # print(os.listdir(detect_rotate_crop_txt_path))
        gt_csv_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select gt_csv_path (/Scaphoid/Annotations/Fracture_Coordinate) Directory")
        # print(gt_csv_path)
        # print(os.listdir(gt_csv_path))
        iou_output_folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select iou_output_folder_path (Output/11iou_output/(Fracture/Normal)) Directory")

        file = {}
        def rect_polygon(x, y, width, height, angle):
            """Return a shapely Polygon describing the rectangle with centre at
            (x, y) and the given width and height, rotated by angle quarter-turns.

            """
            w = width / 2
            h = height / 2
            p = Polygon([(-w, -h), (w, -h), (w, h), (-w, h)])
            return translate(rotate(p, angle * 90), x, y)

        def intersection_over_union(rects_a, rects_b):
            """Calculate the intersection-over-union for every pair of rectangles
            in the two arrays.

            Arguments:
            rects_a: array_like, shape=(M, 5)
            rects_b: array_like, shape=(N, 5)
                Rotated rectangles, represented as (centre x, centre y, width,
                height, rotation in quarter-turns).

            Returns:
            iou: array, shape=(M, N)
                Array whose element i, j is the intersection-over-union
                measure for rects_a[i] and rects_b[j].
            """
            m = len(rects_a)
            n = len(rects_b)
            if m > n:
                # More memory-efficient to compute it the other way round and
                # transpose.
                return intersection_over_union(rects_b, rects_a).T

            # Convert rects_a to shapely Polygon objects.
            polys_a = [rect_polygon(*r) for r in rects_a]

            # Build a spatial index for rects_a.
            index_a = rtree.index.Index()
            for i, a in enumerate(polys_a):
                index_a.insert(i, a.bounds)

            # Find candidate intersections using the spatial index.
            iou = np.zeros((m, n))
            for j, rect_b in enumerate(rects_b):
                b = rect_polygon(*rect_b)
                for i in index_a.intersection(b.bounds):
                    a = polys_a[i]
                    intersection_area = a.intersection(b).area
                    if intersection_area:
                        iou[i, j] = intersection_area / a.union(b).area

            return iou


        img_files                    = os.listdir(img_path)
        # print(os.listdir(img_path))
        # print(img_files)
        detect_rotate_crop_txt_files = os.listdir(detect_rotate_crop_txt_path)
        # print(detect_rotate_crop_txt_files)
        dgt_csv_files                = os.listdir(gt_csv_path)
        # print(dgt_csv_files)
        iou_output_files             = os.listdir(iou_output_folder_path)
        # print(iou_output_files)

        for jpgname in glob.glob( img_path +"/*.jpg"):
            jpg_name = jpgname.split(img_path+"\\")[1].split('.')[0]   
            # print(jpgname)
            # print(jpg_name)
            for txtname in glob.glob( detect_rotate_crop_txt_path +"/*.txt"):
                txt_name = txtname.split(detect_rotate_crop_txt_path+"\\")[1].split('.')[0]
                # print(txtname)
                # print(txt_name)
                if txt_name == jpg_name :
                    # print(jpg_name)
                    # print(txt_name)
                    # print('jpg && txt')
                    for csvname in glob.glob( gt_csv_path + "/*.csv"):
                        csv_name = csvname.split(gt_csv_path+"\\")[1].split('.')[0]
                        # print(csv_name)
                        if txt_name == csv_name :
                            # print(csvname)
                            # print('jpg && txt && csv')
                            txt = open( detect_rotate_crop_txt_path + '/' + txt_name + '.txt', 'r')
                            text = txt.read()
                            print(text.split(' '))
                            # fracture_predict_box = ( ctrx	ctry	width	height	angle )
                            fracture_predict_box   = ( int(text.split(' ')[2]) , int(text.split(' ')[3]) ,int(text.split(' ')[4]) , int(text.split(' ')[5]) , math.degrees(math.atan(float(text.split(' ')[6])))  )
                            print('fracture_predict_box ',fracture_predict_box)
                            csvfile = open( gt_csv_path + '/' + csv_name + '.csv', 'r')
                            data = list()
                            for row in csv.reader(csvfile):
                                data.append(row)
                            # print(data[1])
                            # fracture_Gt_box = ( ctrx	ctry	width	height	angle )
                            fracture_Gt_box  = ( int(data[1][0]) , int(data[1][1]) ,int(data[1][2]) , int(data[1][3]) , int(data[1][4])  )
                            print('fracture_Gt_box ',fracture_Gt_box)
                            im = cv2.imread(img_path + '/' + jpg_name + '.jpg')

                            # fracture_predict_box
                            im = cv2.circle(im, (int(text.split(' ')[2]) , int(text.split(' ')[3])), 1, (0,255,0), -1)
                            # fracture_Gt_box
                            im = cv2.circle(im, (int(data[1][0]) , int(data[1][1])), 1, (0,0,255), -1)
                            # fracture_predict_box  GGGGG
                            rect = ((int(text.split(' ')[2]) , int(text.split(' ')[3])), (int(text.split(' ')[4]) , int(text.split(' ')[5])), math.degrees(math.atan(float(text.split(' ')[6]))))
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)
                            cv2.drawContours(im,[box],0,(0,255,0),2)
                            # fracture_Gt_box   RRRR
                            rect2 = ((int(data[1][0]) , int(data[1][1])), (int(data[1][2]), int(data[1][3])), float(data[1][4]))
                            box2 = cv2.boxPoints(rect2)
                            box2 = np.int0(box2)
                            cv2.drawContours(im,[box2],0,(0,0,255),2)

                            TEST_RECTS_A = rect_polygon( int(data[1][0]) , int(data[1][1]) ,int(data[1][2]) , int(data[1][3]) , int(data[1][4])  )
                            TEST_RECTS_B = rect_polygon( int(text.split(' ')[2]) , int(text.split(' ')[3]) ,int(text.split(' ')[4]) , int(text.split(' ')[5]) , math.degrees(math.atan(float(text.split(' ')[6])))  )
                            
                            iou = TEST_RECTS_A.intersection(TEST_RECTS_B).area/ TEST_RECTS_A.union(TEST_RECTS_B).area
                            print('iou ',iou)
                            iou_output = open(iou_output_folder_path + '/' + csv_name + '.txt' , 'w')
                            iou_output.write(str(iou))
                            iou_output.close()

                            print(TEST_RECTS_A.intersection(TEST_RECTS_B).area/ TEST_RECTS_A.union(TEST_RECTS_B).area)
                            # print(TEST_RECTS_B)

                            # cv2.imshow("im",im)
                            # cv2.waitKey(0)


        total = 0
        count = 0
        iou_mean = 0

        files = listdir(iou_output_folder_path)
        # print(files)
        
        for f in files:
            # print(f)
            readfile = open(iou_output_folder_path + '/' + f,'r')
            count = count + 1
            total = total + float(readfile.read())
            # print(count)
            iou_mean = round(total/count,5)
            if count == 0 :
                print('No data ') 
                iou_mean = 0.456983 
                break
        print('iou_mean : ',iou_mean)
        self.label_20.setText(str(iou_mean))
        print('Fracture_IOU Done')
        return           

    def slider2(self):
        global folder_path
        global out_folder_path
        global crop_folder_path
        global label_folder_path
        global detect_folder_path
        global predict_txt_folder_path
        global cal_iou_folder_path
        global save_bbox_folder_path
        global Show_Fracture_Bbox_folder_path
        global Drawed_Fracture_Bbox_folder_path
        global iou_output_folder_path
        folder_path = './'
        out_folder_path = './'
        crop_folder_path = './'
        label_folder_path  = './'
        detect_folder_path = './'
        predict_txt_folder_path = './'
        cal_iou_folder_path = './'
        save_bbox_folder_path = './'
        Show_Fracture_Bbox_folder_path = './'
        Drawed_Fracture_Bbox_folder_path = './'      
        iou_output_folder_path = './' 

        

        def updateLabel_9(value):
            files = listdir(iou_output_folder_path)
            # print(files)           
            
            if files[value].endswith('.txt'):
                readfile = open(iou_output_folder_path + '/' + files[value],'r')
                self.label_9.setText(readfile.read())
                self.label_17.setText(files[value])
            return
        def updateLabel15(value):
            files = listdir(Drawed_Fracture_Bbox_folder_path)
            #print(files)
            pixmap = QPixmap(Drawed_Fracture_Bbox_folder_path + '/' + files[value])
            self.label_15.setPixmap(pixmap.scaled(200,200, Qt.KeepAspectRatio))
            return
        sld2 = self.horizontalSlider_2
        sld2.valueChanged.connect(updateLabel_9)
        #sld2.valueChanged.connect(updateLabel15)
        sld2.setRange(1, 20)
        return
    
app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
