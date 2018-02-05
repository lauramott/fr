import numpy as np
import cv2
import sqlite3
import subprocess
import os
from django import template

register = template.Library()


class DatasetGenerator:
    def main(request):
        detector = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml");

        def insertOrUpdate(Id,Name):
            conn=sqlite3.connect("FaceBase.db")
            cmd="SELECT * FROM People WHERE ID="+str(Id)
            cursor=conn.execute(cmd)
            isRecordExist=0
            for row in cursor:
                isRecordExist=1
            if(isRecordExist==1):
                cmq="UPDATE People SET Name="+str(Name)+" WHERE ID="+str(Id)
            else:
                cmd="INSERT INTO People(ID,Name) Values("+str(Id)+","+str(Name)+")"
                conn.execute(cmd)
                conn.commit()
                conn.close()

        cam = cv2.VideoCapture(0)
        """identifier: user enter name from console"""

        id=input('enter user id')
        name=input('enter user name')
        insertOrUpdate(id,name)
        sampleNum=0
        while(True):
            factor = 0.75
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5)
            for (x,y,w,h) in faces:
                sampleNum=sampleNum+1
                cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg", cv2.resize(gray[y:y+h,x:x+w], (70,70)))
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            cv2.imshow('frame', img)
            cv2.waitKey(100)
            if(sampleNum>10):
                break
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    datasetGenerator = DatasetGenerator()
    datasetGenerator.main()
