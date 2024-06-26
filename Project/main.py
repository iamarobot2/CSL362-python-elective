import os
import pickle
import cv2
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,
                              {
                                  'databaseURL':"https://faceattendancepython-8f98b-default-rtdb.firebaseio.com/",
                                  'storageBucket':"faceattendancepython-8f98b.appspot.com"
                              })
bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

imgBackground = cv2.imread("Resources/background.png")

#importing different mode backgrounds.
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

#loading the imgEncodeFile
file = open("imgEncodeFile.p",'rb')
encodeListKnownWithId = pickle.load(file)
file.close()
encodeListKnown,studentId = encodeListKnownWithId
print(studentId)

modeType=0
counter=0
id=-1
studentImg = []
flag = True

while flag:
    success,img=cap.read()

    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
            match = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            #print("matches",match)
            #print("faceDis",faceDis)

            matchIndex = np.argmin(faceDis)
            #print("Match Index: ",matchIndex)

            if match[matchIndex]:
                print("Student Identified with id ",studentId[matchIndex])
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                bbox = 55+x1,162+y1,x2-x1,y2-y1
                imgBackground = cvzone.cornerRect(imgBackground,bbox,rt=0)
                id = studentId[matchIndex]
                studentInfo = db.reference(f'Students/{id}').get()
                studentName = studentInfo['name']

                if counter==0:
                    cvzone.putTextRect(imgBackground,"Loading",(274,350))
                    cv2.imshow("Face Reco. based attendance system",imgBackground)
                    cv2.waitKey(1)
                    counter=1
                    modeType=1

        if counter!=0:
            ref = db.reference(f'Students/{id}')
            if counter==1:
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)
                blob = bucket.get_blob(f'Images/{id}.png')
                array = np.frombuffer(blob.download_as_string(),np.uint8)
                studentImg = cv2.imdecode(array,cv2.COLOR_BGRA2BGR)
                # Resize studentImg to 216x216
                studentImg = cv2.resize(studentImg, (216, 216))

                #update attendance
                datetimeObject = datetime.strptime(studentInfo['last_attendance'],"%Y-%m-%d %H:%M:%S")
                timeElapsed = (datetime.now()-datetimeObject).total_seconds()
                print(timeElapsed)
                if timeElapsed>86400:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['attendance'] +=1
                    ref.child('attendance').set(studentInfo['attendance'])
                    ref.child('last_attendance').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType=3
                    counter=0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType!=3:

                if 10<counter<40:
                    modeType=2
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if counter<=10:
                    attendance = (studentInfo['attendance']/100) * 100
                    cv2.putText(imgBackground,str(attendance)+'%',(861,125),cv2.QT_FONT_NORMAL,0.6,(255,255,255),1)
                    cv2.putText(imgBackground,str(studentInfo['major']),(1006,550),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1)
                    cv2.putText(imgBackground,str(id),(1006,493),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),1)
                    cv2.putText(imgBackground,str(studentInfo['batch']),(910,625),cv2.FONT_HERSHEY_COMPLEX,0.5,(100,100,100),1)
                    cv2.putText(imgBackground,str(studentInfo['semester']),(1025,625),cv2.FONT_HERSHEY_COMPLEX,0.6,(100,100,100),1)
                    cv2.putText(imgBackground,str(studentInfo['starting_year']),(1125,625),cv2.FONT_HERSHEY_COMPLEX,0.6,(100,100,100),1)

                    (w,h), _ =cv2.getTextSize(studentInfo['name'],cv2.FONT_HERSHEY_COMPLEX,1,1)
                    offset = (414-w)//2
                    cv2.putText(imgBackground,str(studentInfo['name']),(808+offset,445),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,50),1)

                    imgBackground[175:175+216,909:909+216] = studentImg
                counter+=1

                if counter>=40:
                    counter=0
                    modeType=0
                    studentInfo= []
                    studentImg = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        modeType=0
        counter=0


    cv2.imshow("Face Reco. based attendance system",imgBackground)
    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty("Face Reco. based attendance system", cv2.WND_PROP_VISIBLE) < 1: 
        flag = False
cap.release()
cv2.destroyAllWindows()       