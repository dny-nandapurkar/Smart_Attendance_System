import cv2
import pickle   # To store face data with their name in pickle file
import numpy as np  # To convert face data into numpy array
import os

video = cv2.VideoCapture(0)    # to open your in-built web cam
faceDetect = cv2.CascadeClassifier('D:\Smart Attendance System\Data\haarcascade_frontalface_default.xml')    # detects faces using haarcascade_frontalface.xml file
 
faces_data = []  # Stores crop images data into list
i=0
name = input("Enter Your Name : ")

while True:
    ret, frame = video.read()   # read video ives two values ret(boolean value wich tells video is ok or not) and another is frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # To detect faces we need to create convert faces into grayscale as cascade need it in grayscale and opencv has it in bgr format
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)   # two parameters grayscale and threshold value.  This will give coordinate values from faces, X, Y and W, H - width and height of images
    for (x, y, w, h) in faces:   # Checking wheather faces are detected or not by creating rectangle around faces
        crop_img = frame[y:y+h, x:x+w, :]  # Crop image to store its value in pickle file
        resized_img = cv2.resize(crop_img, (50, 50)) # To ensure every image is in same format
        if len(faces_data) <= 100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50,255) ,1) # 7 parameters to see how many picture have been detected/taken = original frame, text=len of face data, coordinate, font , font scale, color, thickness
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 1)  # 5 parameters = original frame, cordinate value, width and height of channel ,rectangle color=red, thickness 
    cv2.imshow("Frame", frame)  # To show video frame
    k = cv2.waitKey(1)  # key binding function to break infinite loop
    if k==ord("q") or len(faces_data)==100:
        break
video.release()   # Release frame
cv2.destroyAllWindows()  # Close frame

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)  # Converted face data into numpy array

if 'names.pkl' not in os.listdir('D:\Smart Attendance System\Data'):    # Creating .pkl file if not available
    names = [name]*100
    with open('data/names.pkl', 'wb') as f:     
        pickle.dump(names, f)   # names of 100 img into .pkl file
else:
    with open('data/names.pkl', 'rb') as f:     
        pickle.load(f)   
    names = names + [name]*100
    with open('data/names.pkl', 'wb') as f:     
        pickle.dump(names, f)  

if 'faces_data.pkl' not in os.listdir('D:\Smart Attendance System\Data'):    # Creating .pkl file if not available
    with open('data/faces_data.pkl', 'wb') as f:     
        pickle.dump(faces_data, f)   # names of 100 img into .pkl file
else:
    with open('data/faces_data.pkl', 'rb') as f:     
        faces = pickle.load(f)   
    faces = np.append(faces, faces_data, axis = 0)
    with open('data/faces_data.pkl', 'wb') as f:     
        pickle.dump(names, f)    