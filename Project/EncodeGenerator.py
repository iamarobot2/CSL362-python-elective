import pickle
import cv2
import face_recognition
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,
                              {
                                  'databaseURL':"https://faceattendancepython-8f98b-default-rtdb.firebaseio.com/",
                                  'storageBucket':"faceattendancepython-8f98b.appspot.com"
                              })

#importing student images.
folderPath = 'Images'
imgPathList = os.listdir(folderPath)
imgList = []
studentId = []
for path in imgPathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    studentId.append(os.path.splitext(path)[0])
    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

def findEncodings(imageList):
    encodeList = []
    for img in imageList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithId = [encodeListKnown,studentId]
print("Encoding Complete")

file = open("imgEncodeFile.p",'wb')
pickle.dump(encodeListKnownWithId,file)
file.close()
print("Endodefile saved")