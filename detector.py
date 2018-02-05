import cv2,os
import numpy as np
from PIL import Image
import pickle
import sqlite3


class Detector:
    def main(self):
        # recognizer = cv2.face.LBPHFaceRecognizer_create();                # LBPH Face Recognizer
        recognizer=cv2.face.EigenFaceRecognizer_create();                 # Eigen Face Recognizer
        # recognizer = cv2.face.FisherFaceRecognizer_create()                  # Fisher Face Recognizer
        recognizer.read("trainer/trainer.yml")
        faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
        path = 'dataSet'

        def getProfile(id):
            conn=sqlite3.connect("FaceBase.db")
            cmd="SELECT * FROM People WHERE ID="+str(id)
            profile=None
            """if id<0:
                return profile
            else:"""
            cursor = conn.execute(cmd)
            for row in cursor:
                profile = row
            conn.close()
            return profile

        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        profiles={}
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
            for(x,y,w,h) in faces:
                id,conf=recognizer.predict(cv2.resize(gray[y:y+h,x:x+w], (70,70)))
                profile = None
                cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
                print (conf)
                if conf>=80:
                    profile=None
                else :
                    profile=getProfile(id)

                if(profile!=None):
                    cv2.putText(img,str(profile[0]),(x,y+h+30),font,2,255)
                    cv2.putText(img,str(profile[1]),(x,y+h+60),font,2,255)
                else:
                    cv2.putText(img, "Unknown", (x, y + h + 30), font, 2, 255)
            cv2.imshow('img',img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = Detector()
    detector.main()
