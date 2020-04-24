import cv2
import numpy as np

#create an object of cascade classifier
face_classifier = cv2.CascadeClassifier('/Users/himanimense/PycharmProjects/main_project/venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')

#functions to extract face features
#we shall work with gray scale images rather than RGB since they are easy to operate on
def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)  #third param is the number of neighbors of which value should be between 3-6 for better accuracy

    if faces == ():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

        return cropped_face


#configure camera
cap = cv2.VideoCapture(0)

#variable used to count the number of images taken
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame), (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)   #to convert the resized image to gray

        file_name_path = '/Users/himanimense/Downloads/faces/user'+str(count)+'.jpg' #location of file to save the face values
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper', face)
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1)==13 or count==100:  #allows you to collect just 100 samples
        break

cap.release()
cv2.destroyAllWindows()
print('Face Sample collection complete!!')


#img = cv2.imread("/Users/himanimense/PycharmProjects/helloWorld/nature.jpg", 1)#Macintosh HD

#print(img)
#cv2.imshow('image',img)