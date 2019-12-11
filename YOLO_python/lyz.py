import cv2 as cv
import argparse
import sys
import numpy as np
import os.path


import matplotlib as mpl
import matplotlib.pyplot as plt
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, box_in_image, view_points
class yolo_result():
    def __init__ (self,id,confidence,boxes):
        self.id=id
        self.confidence=confidence
        self.boxes=boxes
        self.boxes_area=[]
        self.anglelist=[]
    def get_box_area(self):
        size=len(self.boxes)
        for i in range(size):
            box=self.boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            right=left+width
            bottom=top+height
            area=[top,bottom,left,right]
            self.boxes_area.append(area)
            

        
    def show_box_area_picture(self,frame):
        temp=np.zeros(frame.shape,dtype=np.uint8)
        for item in self.boxes_area:
            temp[item[0]:item[1],item[2]:item[3]]=frame[item[0]:item[1],item[2]:item[3]]
        return temp
              
class point_result():
    def __init__(self,cloud,level5data,my_sample,ldName,caName):
        self.size=len(cloud.points[0])
        self.points=[]  
        self.sp_points=[]
        self.test_points=[]
        self.origin=cloud
        self.level5data=level5data
        self.my_sample=my_sample
        self.ldName=ldName
        self.caName=caName
        for i in range(self.size):
                point=[cloud.points[0][i],cloud.points[1][i],cloud.points[2][i],cloud.points[3][i]]
                self.points.append(point)
                    
    def get_mapped_points(self):
 ###############################################################################
        pointsensor = self.level5data.get('sample_data', self.my_sample['data'][self.ldName])
        cam = self.level5data.get("sample_data", self.my_sample['data'][self.caName])
        if 1>0:
            pc=self.origin
     # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
            cs_record = self.level5data.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])                             #需要判
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform to the global frame.
            poserecord =self.level5data.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform into the ego vehicle frame for the timestamp of the image.
            poserecord =self.level5data.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform into the camera.
            cs_record = self.level5data.get("calibrated_sensor", cam["calibrated_sensor_token"])
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)
            depths = pc.points[2, :]

            # Retrieve the color from the depth.
            coloring = depths

            # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
            points = view_points(pc.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

            # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > 0)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < 1920 - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < 1080 - 1)
            #points[:, mask]
            temp_points = points
            coloring = coloring[mask]
            self.cam_points=[]
            for i in range(temp_points[0].size):
                point=[temp_points[0][i],temp_points[1][i],temp_points[2][i]]
                self.cam_points.append(point)
 
    def show_testpoints(self):
        frame=np.zeros((1080,1920,3),dtype=np.uint8)
        length=len(self.sp_points)
        for i in range(length):
            x0=int(self.sp_points[i][0]*12)
            y0=int(self.sp_points[i][1]*12)
            x=810+x0
            y=540-y0
            cv.circle(frame,(x,y),0,(255,255,255), -1)
        return frame
    def show_points(self):
        frame=np.zeros((1080,1920,3),dtype=np.uint8)
        length=len(self.points)
        for i in range(length):
            x0=int(self.points[i][0]*12)
            y0=int(self.points[i][1]*12)
            x=810+x0
            y=540-y0
            cv.circle(frame,(x,y),1,(255,255,255), -1)
        cv.line(frame,(810,540),(0,1080),(0,0,255),3)
        cv.line(frame,(810,540),(0,0),(0,0,255),3)
        return frame
    
    def show_sp_points(self):
        frame=np.zeros((1000,1000,3),dtype=np.uint8)
        size=len(self.sp_points)
        for i in range(size-1):
            size1=len(self.sp_points[i])
            for j in range(size1):
                x0=int(self.sp_points[i][j][0]*10)
                y0=int(self.sp_points[i][j][1]*10)
                x=500+x0
                y=500-y0
                cv.circle(frame,(x,y),1,point_result.color(i), -1)
        for point in self.sp_points[size-1] :
            x0=int(point[0]*10)
            y0=int(point[1]*10)
            x=500+x0
            y=500-y0
            cv.circle(frame,(x,y),1,(255,255,255), -1)
        return frame
  
    def color(num):
        if(num%6==0):
            return (255,0,0)
        elif (num%6==1):
            return  (0,255,0)
        elif (num%6==2):
            return  (0,0,255)
        elif (num%6==3):
            return  (255,255,0)
        elif (num%6==4):
            return  (255,0,255)
        elif (num%6==5):
            return  (0,255,255)


    def get_sp_points(self,box_area):
        pointsensor = self.level5data.get('sample_data', self.my_sample['data'][self.ldName])
        cam = self.level5data.get("sample_data", self.my_sample['data'][self.caName])
        pc=self.origin
        temp=pc.points[:,:]
        cs_record0 = self.level5data.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
        cs_record1 = self.level5data.get("calibrated_sensor", cam["calibrated_sensor_token"])
        poserecord0 =self.level5data.get("ego_pose", pointsensor["ego_pose_token"])
        poserecord1 =self.level5data.get("ego_pose", cam["ego_pose_token"])
        for i in range(self.size):
            pc.points=temp[:4,i:i+1]
     # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.                          #需要判
            pc.rotate(Quaternion(cs_record0["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record0["translation"]))
            # Second step: transform to the global frame.           
            pc.rotate(Quaternion(poserecord0["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord0["translation"]))
            # Third step: transform into the ego vehicle frame for the timestamp of the image.
            pc.translate(-np.array(poserecord1["translation"]))
            pc.rotate(Quaternion(poserecord1["rotation"]).rotation_matrix.T)
            # Fourth step: transform into the camera.
            pc.translate(-np.array(cs_record1["translation"]))
            pc.rotate(Quaternion(cs_record1["rotation"]).rotation_matrix.T)

            # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
            points = view_points(pc.points[:3, :], np.array(cs_record1["camera_intrinsic"]), normalize=True)
            #self.test_points.append(points)
            if points[0]<box_area[3] and points[0]>box_area[2] and points[1]<box_area[1] and points[1]>box_area[0] and points[2]>0:
                if self.points[i][0]<0:
                    self.sp_points.append(self.points[i])

def lyz_show(cloud):
    size=len(cloud)
    frame=np.zeros((1080,1920,3),dtype=np.uint8)    
    for i in range(size):
        x=int(cloud[i][0])
        y=int(cloud[i][1])
        cv.circle(frame,(x,y),1,(255,255,255), -1)

    return frame

