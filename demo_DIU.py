import streamlit as st
import cv2
import numpy as np
import requests
from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
from mmdet.utils.contextmanagers import concurrent
from pprint import pprint
from PIL import Image
import datetime

classes = ('caption', 'table', 'firgure')
colors = ((255,0,0), (0,255,0), (0,0,255), (255,255,0),	(0,255,255), (255,0,255), (192,192,192), (128,128,128), (128,0,0), (128,128,0), (0,128,0))
threshold = 0.3

def file():
    inputimg = st.file_uploader("Upload your image")
    if inputimg is not None:
        inputimg = Image.open(inputimg)
        inputimg = np.array(inputimg)
        inputimg = cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB)
        cv2.imwrite('demo_file.jpg', inputimg)
        return inputimg

def detect(model):
    config_file = f'./configs/{model}.py'
    checkpoint_file = f'./models/{model}.pth'
    
    # Build model from config file and checkpoint file

    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    # get img from input
    img = 'demo_file.jpg'

    start = datetime.datetime.now()
    result = inference_detector(model, img)
    end = datetime.datetime.now()

    time = end - start 

    time_mcs = time.microseconds

    # num_class = 11
    
    model.show_result(img, result, out_file='result.jpg')
    outputimg = visualize_detect(img, result)
    
    return outputimg

def visualize_detect(inputimg, result):
    img = cv2.imread(inputimg)
    for i in range(len(classes)):
        for bbox in result[i]:
            if bbox[4] > threshold:
                x = bbox[0]
                y = bbox[1]
                w = bbox[2]
                h = bbox[3]
                conf_score = bbox[4]
                class_name = classes[i]
                
                x1 = x
                y1 = y
                x2 = w - 1
                y2 = h - 1

                x1 = round(x1)
                y1 = round(y1)
                x2 = round(x2)
                y2 = round(y2)

                color = colors[i]
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color , 2)
                txt = class_name + " " + str(conf_score)
                img = cv2.putText(img, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img

def file():
    inputimg = st.file_uploader("Upload your image")
    if inputimg is not None:
        inputimg = Image.open(inputimg)
        inputimg = np.array(inputimg)
        inputimg = cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB)
        cv2.imwrite('demo_file.jpg', inputimg)
        return inputimg

st.title("Demo phát hiện đối tượng trong ảnh tài liệu")
st.write("Nguyễn Nhật Trường - 20522087")
st.write("Lê Trương Ngọc Hải - 20520481")

inputimg_file = file()
if inputimg_file is not None: 
    st.image(cv2.cvtColor(inputimg_file, cv2.COLOR_BGR2RGB))
    outputimg = detect('truongDIU')
    st.image(cv2.cvtColor(outputimg, cv2.COLOR_BGR2RGB))

