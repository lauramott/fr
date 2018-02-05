import os
import cv2
import numpy as np
from PIL import Image


class Train:
    def main(self):
        # recognizer = cv2.face.LBPHFaceRecognizer_create();                # LBPH Face Recognizer
        recognizer=cv2.face.EigenFaceRecognizer_create();                 # Eigen Face Recognizer
        # recognizer = cv2.face.FisherFaceRecognizer_create()                  # Fisher Face Recognizer
        path = 'dataSet'

        def getImagesWithID(path):
            imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
            faces=[]
            IDs=[]
            for imagePath in imagePaths:
                faceImg=Image.open(imagePath).convert('L');
                faceNp = np.array(faceImg, 'uint8')
                ID=int(os.path.split(imagePath)[-1].split('.')[1])
                faces.append(faceNp)
                print (ID)
                IDs.append(ID)
                cv2.imshow("training",faceNp)
                cv2.waitKey(10)
            return IDs, faces

        Ids,faces=getImagesWithID(path)
        recognizer.train(faces, np.array(Ids))
        recognizer.save('trainer/trainer.yml')
        cv2.destroyAllWindows()


if __name__ == '__main__':
    train = Train()
    train.main()
