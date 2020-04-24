import cv2
import numpy as np  #for mathematical calculations
from os import listdir  #used to fetch data from any directory
from os.path import isfile, join  #libraries are called

data_path = '/Users/himanimense/Downloads/faces/'

#function to obtain those files from above path
#loop through listdir
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_data, Labels = [], []

for i, files in enumerate(onlyfiles):   #enumerate provides the number of iterations
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)   #by default it gives gray scale images
    Training_data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_data), np.asarray(Labels))

print("Model Training Complete!!!!")

face_classifier = cv2.CascadeClassifier('/Users/himanimense/PycharmProjects/main_project/venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    #converting image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return  img,[]

#for loop for all parameters of ROI-Region Of Interest
    #to draw rectangle around faces
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi,(200,200))

    return img,roi

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500 :   #this value should fall under confidence for the face to be locked means assured that it's the user's face
            confidence = int(100*(1-(result[1])/300))   #to avoid getting float confidence values and shows how much percentage of the face matches
            display_string = str(confidence)+'% Confidence that it is the user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX, 1, (250,120,255),2)


        if confidence > 75:
            cv2.putText(image, "Unlocked",(250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),2)
            cv2.imshow('Face Cropper',image)

        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:
        cv2.putText(image, "Face not present", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        #cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  #both messages get printed
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()







