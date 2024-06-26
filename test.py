from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle   # To store face data with their name in pickle file
import numpy as np  # To convert face data into numpy array
import os
import csv
import time
from datetime import datetime

from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)


video = cv2.VideoCapture(0)    # to open your in-built web cam
faceDetect = cv2.CascadeClassifier('D:\Smart Attendance System\Data\haarcascade_frontalface_default.xml')    # detects faces using haarcascade_frontalface.xml file
 
with open('data/names.pkl', 'rb') as f:     
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:     
    FACES = pickle.load(f)   

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Define background image path
background_image_path = r"D:\Smart Attendance System\background.png"
img_background = cv2.imread(background_image_path)

COL_NAMES = ['Name', 'TIME']

# Define the region to place the video frame within the background image
# Ensure the region fits within the dimensions of the cropped background image
frame_width, frame_height = 640, 480
bg_height, bg_width, _ = img_background.shape
x_offset = 55
y_offset = 162

if x_offset + frame_width > bg_width or y_offset + frame_height > bg_height:
    print("Error: The video frame does not fit within the background image. Adjust the offsets or dimensions.")
    exit()

while True:
    ret, frame = video.read()   # read video ives two values ret(boolean value wich tells video is ok or not) and another is frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # To detect faces we need to create convert faces into grayscale as cascade need it in grayscale and opencv has it in bgr format
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)   # two parameters grayscale and threshold value.  This will give coordinate values from faces, X, Y and W, H - width and height of images
    for (x, y, w, h) in faces:   # Checking wheather faces are detected or not by creating rectangle around faces
        crop_img = frame[y:y+h, x:x+w, :]  # Crop image to store its value in pickle file
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1) # To ensure every image is in same format
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("D:\Smart Attendance System\Attendance\Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)   # # rectangle line code to draw name rectangle
        cv2.rectangle(frame, (x, y-40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 1)  # 5 parameters = original frame, cordinate value, width and height of channel ,rectangle color=red, thickness 
        attendance = [str(output[0]), str(timestamp)]
    img_background[y_offset:y_offset + frame_height, x_offset:x_offset + frame_width] = frame
    cv2.imshow("Frame", img_background)  # To show video frame
    k = cv2.waitKey(1)  # key binding function to break infinite loop
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        if exist:
            with open("D:\Smart Attendance System\Attendance\Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("D:\Smart Attendance System\Attendance\Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
    if k==ord("q"):
        break
video.release()   # Release frame
cv2.destroyAllWindows()  # Close frame

