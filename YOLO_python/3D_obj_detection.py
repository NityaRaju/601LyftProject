import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import lyz

import matplotlib as mpl
import matplotlib.pyplot as plt
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points
import numpy
# Initialize the parameters
confThreshold = 0.5  #Confidence threshold  置信度阈值
nmsThreshold = 0.4   #Non-maximum suppression threshold 设置非极大值抑制
inpWidth = 416       #Width of network's input image 输入到神经网络中的图像宽度
inpHeight = 416      #Height of network's input image输入到神经网络中的图像的高度

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
#读取分类名 Load names of classes
classesFile = "coco.names"; #将分类名称导入，分类名称是一个专用的.Name文件，本质上是文本文件
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# 从外部导入模型结构文件.cfg和权重配置文件.weights
modelConfiguration = "yolov3.cfg";
modelWeights = "yolov3.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights) #use the function from opencv to build an objecr:net
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# 获取输出层Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 画图函数
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # 获取标签和置信度Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #标签显示Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    #保存高置信度框
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs: #推测outs中，每个out是一个物体
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    ##################################################################################
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    result_boxes=[]
    result_confidences=[]
    result_classIds=[]
    for i in indices:
        i = i[0]
        result_boxes.append(boxes[i])
        result_classIds.append(classIds[i])
        result_confidences.append(confidences[i])
    length=len(result_boxes)
    for j in range(length):
        box =boxes[j]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[j], confidences[j], left, top, left + width, top + height)
    ##################################################################################
    result=lyz.yolo_result(result_classIds,result_confidences,result_boxes)
    return result
        

#主程序开始 main()


# 输入处理
                                                   #建立一个新的窗口

level5data = LyftDataset(data_path='C:/Boston_University/3d-object-detection-for-autonomous-vehicles/', json_path='C:/Boston_University/3d-object-detection-for-autonomous-vehicles/train_data', verbose=True)
winName = 'result'
cv.namedWindow(winName, cv.WINDOW_NORMAL)    
#level5data.list_scenes()            #get all of the sence
my_scene = level5data.scene[0]      #chosse [0]for analysing
my_sample_token = my_scene["first_sample_token"] #choose token [0] for analyse
my_sample = level5data.get('sample', my_sample_token)
KEY=200


while 1:
    if KEY==13:
        break
    img_token=level5data.get('sample_data', my_sample['data']['CAM_FRONT'])
    img_filepath=level5data.get_sample_data_path(my_sample['data']['CAM_FRONT'])
    cap = cv.VideoCapture(str(img_filepath))
    lidar_filepath = level5data.get_sample_data_path(my_sample['data']['LIDAR_TOP'])
    lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
    hasFrame, frame = cap.read()#显示图像读取状态并获取图像帧

    # network
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))

    # 后期处理
    cam_result=postprocess(frame, outs)
    cam_result.get_box_area()
    cam_result.get_anglelist()
    filted_frame=cam_result.show_box_area_picture(frame)
    lid_result=lyz.point_result(lidar_pointcloud)
    lid_result.get_mapped_points(level5data,my_sample,'LIDAR_TOP','CAM_FRONT')
    
    if KEY==ord('a'):
        points_frame=lid_result.show_points()
    else:
        points_frame=lyz.lyz_show(lid_result.cam_points)
    
    htitch= np.hstack((frame, points_frame))
    cv.imshow(winName, htitch)
    KEY=cv.waitKey()
    my_sample = level5data.get('sample', my_sample['next'])
    if my_sample['next']=="":
        break

cap.release()
cv.destroyAllWindows()

