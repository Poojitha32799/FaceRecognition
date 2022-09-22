import os
import cv2 as cv
import numpy as np
import face_recognition as fr
from datetime import datetime
path = 'CSE-2A'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    currimg = cv.imread(f'{path}/{cl}')
    currimg = cv.resize(currimg,(500,500),interpolation=cv.INTER_CUBIC)
    images.append(currimg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findencodings(images):
    encodelist=[]
    for img in images:
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        facesimg = fr.face_locations(img)
        encodeimg = fr.face_encodings(img,facesimg)[0]
        encodelist.append(encodeimg)
    return encodelist

def markAttendence(name):
    namelist = []
    with open('attendence.csv','r+') as f:
        mydatalist= f.readlines()
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now= datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtstring}')


encodelistKnown = findencodings(images)
print('encoding complete')

cap = cv.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS= cv.resize(img,(0,0),None,0.25,0.25)
    imgS = cv.cvtColor(imgS,cv.COLOR_BGR2RGB)

    facesCurrFrame = fr.face_locations(imgS)
    encodeCurrFrame = fr.face_encodings(imgS,facesCurrFrame)

    for encodeFace,faceLoc in zip(encodeCurrFrame,facesCurrFrame):
        matches = fr.compare_faces(encodelistKnown,encodeFace)
        facedis = fr.face_distance(encodelistKnown,encodeFace)
        print(facedis)
        matchindex = np.argmin(facedis)
        if facedis[matchindex] > 0.4:
            print("try again")
        elif matches[matchindex]:
            name = classNames[matchindex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1= y1*4,x2*4,y2*4,x1*4
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv.rectangle(img,(x1,y2+25),(x2,y2),(0,255,0),cv.FILLED)
            cv.putText(img,name,(x1+25,y2+25),cv.FONT_HERSHEY_COMPLEX,1,(255,255.255),2)
            markAttendence(name)

    cv.imshow('webcam',img)
    cv.waitKey(1)