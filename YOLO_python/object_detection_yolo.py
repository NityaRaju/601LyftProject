import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
import numpy
import lyz
import cv2 as cv
level5data = LyftDataset(data_path='C:/Boston_University/3d-object-detection-for-autonomous-vehicles/', json_path='C:/Boston_University/3d-object-detection-for-autonomous-vehicles/train_data', verbose=True)

#level5data.list_scenes()            #get all of the sence
my_scene = level5data.scene[0]      #chosse [0]for analysing
my_sample_token = my_scene["first_sample_token"] #choose token [0] for analyse
my_sample = level5data.get('sample', my_sample_token)
lidar_data = level5data.get('sample_data', my_sample['data']['LIDAR_TOP'])
lidar_filepath = level5data.get_sample_data_path(my_sample['data']['LIDAR_TOP'])
lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
result=lyz.point_result(lidar_pointcloud.points)
frame=result.show_points()
0=int(2*10)
y0=int(10*10)
x=500+x0
y=500+y0
cv.circle(frame,(x,y),20,(255,0,255), -1)
cv.imshow("hello",frame)

cv.waitKey()
cv.destroyAllWindows()
#print(lidar_pointcloud)
#print(lidar_pointcloud.points)
#print(level5data.calibrated_sensor)
#print(level5data.calibrated_sensor[0])

