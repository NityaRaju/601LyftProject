# 601LyftProject

### 1. Product Mission
   To improve the quality of life by optimizing self-driving technology, which can lead to fewer accidents and easier commutes. 3D object detection over semantic maps is a primary problem in the development of autonomous driving technologies, so the product mission is to use machine learning techniques to improve this object detection. We do this by predicting the bounding volume of all the objects in a given scene/picture and classifying them into different classes. 

### 1.1 Target User(s)
   This product in particular is being developed for a specific client/target user- Lyft. Apart from Lyft, any company which is looking to manufacture, sell or use automated self-driving cars, such as Tesla, Uber, Toyota etc. would benefit from this product. 
### 1.2 User Stories
  3D object detection over sematic maps is used by many different people in different fields. 
  
  I, Lyft, would like a software to improve autonomous driving capabilities by reducing the number of accidents.
  
  I, Lyft, would like to use only the raw camera sensor inputs in order to facilitate autonomous driving, without any other sensors. 
  
  I, the car manufacturer, would like a software that can be used in a range of autonomous vehicles, with little modifications.
  
  I, Lyft, would like a software that can differentiate between different types of vehicles, living and non living objects, and traffic     signs. 
  
  I, Lyft, would like a software that can accurately tell me the bounding volume, center, width, length and height of a given object in     the picture. 
  
  I, Lyft, would like a software that can accurately classify all the given objects in a picture into different categories. 
  
  I, a robotics developer, would like a software that can be used by a robot to build a semantically annotated 3D map in real time, to       improve robot-human interactions
  
  I, the researcher, would like a software that can automatically annotate/label a picture so I may classify objects of interest to me. 

  
### 1.3 MVP
   An algorithm which can perform 3D object detection (by returning the volume of the object) over semantic maps (by returning the classification of the object) 
### 1.4 User Interface Design for main user story if required
   Since this is an algorithm that is to be developed, there is no user interface because it is not a product. This algorithm would be programmed into an autonomous car, and the car would act as the user interface between our algorithm and the driver.     

# 2.Product survey
### 2.1Existing similar products:
   Existing similar products: There are a large number of project/companies working in the automated car industry. Here is a list of several companies which have been developing and implementing autonomous technologies, similar to the one we hope to develop. 
    
   Autopilot Projects: Waymo(Google), Apple, Tesla
   Automatic Taxi Services:  Uber, Zoox, Voyage, Renovo.auto
   Automated Buses/Trucks: TuSimple, Navya, Bauer's
   Autopilot Technology: NIO,Faraday Future, PlusAi, Aimotive, Pony.AI, Aurora Innovation, GM Cruise, Delphi
   Autopilot Artificial Intelligence: Drive.ai, AutoX
   Autopilot Softwares:  Vector.ai, nuTonomy, Horizon, Oxbotica, Argo.ai
    
   Out of these, we look more closely at Uber and Waymo, because they are ride-sharing services and Lyfts direct competitors.Uber, Lyft and Waymo are the 3 main companies looking to dominate the autonomous ride-sharing space, and are in constant competition to develop their autonomous systems- in fact, this project itself is a Lyft kaggle competition meant to outsource optimization of Lyft’s algorithms. All the companies are involved in this sort of R&D to optimize their technologies. 
   There is alot of proprietary research and development being conducted on all these algorithms since it is a race to get vehicles on the road as quickly as possible. Our product is simply an algorithm which can perform 3D object classification with semantic mapping, and we can assume that our algorithm will be very similar to the other companies, because they will all require a similar algorithm in order to automate driving. Without 3D mapping and classification it would literally be impossible to automate cars. However, most of these algorithms are heavily protected trade secrets in order to maintain an upper hand. So it is difficult for us to get a clear idea of the competitor space, at least in terms of the exact underlying algorithm that each company is using. 
    However, if we look at the final system into which our algorithm will be implemented- we can see more clearly the differences in driving quality. In the current market, Waymo has begun autonomous ride-sharing in Arizona whereas Uber has had a major setback due to a fatal collision with one of its cars. Lyft has purchased Blue Vision, a 3D mapping company, and partnered with Aptiv to bring self-driving cars to Las Vegas. 

### 2.2 Patent Analysis
   By checking lots of patents, like google autopilot patents and huawei lidar detection system we can find out that the image and lidar file process is the core of the whole autopilot system. And all of these companies choose machine learning to their tool to process. And in the patent of Lyft named Identifying Objects for Display in a Situational-Awareness View of an Autonomous-Vehicle Environment mentioned that Lyft used both video camera and lidar as the eyes of autonomous system. Also, in this patent it mentioned that: when a possible object is detected, machine learning is applied to correlate the sensed possible object data to a specific object and a set of object attributes. It revealed how Lyft try to process the data from the sensors. And this is going to be our main work in this project.
  
   We need to build a solid machine learning model to process the data accurately and quickly enough.
  
   Solid object detection system using laser and radar sensor fusion
   https://patents.google.com/patent/US9097800B1/en
  
   Object detection using radar and machine learning
   https://patents.google.com/patent/WO2017181643A1/en
  
   Evaluating and Presenting Pick-Up and Drop-Off Locations in a Situational-Awareness View of an Autonomous Vehicle
   https://patents.google.com/patent/US20180136656A1/en?oq=20180136656

   Identifying Objects for Display in a Situational-Awareness View of an Autonomous-Vehicle Environment
   https://patents.google.com/patent/US20180136000A1/en?oq=20180136000

# 3.System design
### 3.1 Technologies Used
   We will use 
   Opencv libraries 
   Machine learning(tensorflow) algorithms、
   point cloud processing、
   YOLO implementation、
   3D rebuild  and  
   camera calibration.
  
   Open cv provides us with lots of functions such as Kalman filter、Canny、Roberts、Sobel、Prewitt operator and so on. It is an open source library so we can use it directly. We choose python because we don’t need to take care of the data type(int、long int）and the allocation of memory. During imagine processing we need to use many large matrices to store imagine, it is difficult to manage the memory.
### 3.2 Components map
![Image description](https://github.com/NityaRaju/601LyftProject/blob/master/Tech.JPG)

<img src= "https://github.com/NityaRaju/601LyftProject/blob/master/Screen%20Shot%202019-12-12%20at%2011.55.50%20AM.png">
### 3.3 Test program
      import cv2 as cv
      import numpy as np
      import math

      imgBus=cv.imread("Bus.jpg",1)
      imgDes=cv.imread("Desktop.jpg",1)
      cv.imshow("111",imgBus)
      imgtest=cv.imread("Bus.jpg",0)
      imgtest=cv.Canny(imgtest,30,200)
      contours, hierarchy=cv.findContours(imgtest,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
      cv.drawContours(imgtest,contours,0,(0,0,255),3)
      cv.imshow("222",imgtest)
      cv.waitKey()
      cv.destroyAllWindows()
 ![Image description](https://github.com/NityaRaju/601LyftProject/blob/master/Test%20bus0.JPG)t
                  (I photo this picture with my phone in China)
 ![Image description](https://github.com/NityaRaju/601LyftProject/blob/master/Test%20bus1.JPG)
 
 # 4.Results
 ### 4.1 Preliminary results
 Before working on the huge Lyft dataset, we tested our algorithms for object classification on images we took of ourselves. For this initial testibg, we used teh COCO(common objects in context) dataset to train our neural network so that it was able to classify objects other than those commonly found on the street. 
 
 <img src=" ">
 
 Our team photo!
 
 ### 4.2 Final Results
 We then worked on actual images taken by a camera in the dashboard for a self-driving car. The Lyft training data was used in conjunction with OpenCV and YOLO implementation in order to classify all the 3D objects in a certain image, including the probablility that the clasification is correct. This can help the car make decisions later on. 
 
 <img src=" ">
 
 
 ### Final Poster 
 <img src="https://github.com/NityaRaju/601LyftProject/blob/master/Poster.png">
 
