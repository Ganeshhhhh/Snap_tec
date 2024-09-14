import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tkinter import filedialog
import tkinter as tk

# Initialize a Tkinter window to ask the user for the image file
root = tk.Tk()
root.withdraw()  # Hide the main Tkinter window

# Ask the user to select an image file
file_path = filedialog.askopenfilename()

# Load the image
img = cv2.imread(file_path)
imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')

        # Check if the name is not in the list and mark attendance
        if name not in nameList:
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Detect faces in the loaded image with a larger scale parameter
facesCurFrame = face_recognition.face_locations(imgS, model='cnn')
encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

for faceLoc in facesCurFrame:
    y1, x2, y2, x1 = faceLoc
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

    matchIndex = np.argmin(faceDis)

    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        markAttendance(name)

# Display the image with all detected faces
cv2.imshow('image', img)
cv2.waitKey(0)
