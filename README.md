# FaceDetection
ALGORITHM:
Step 1: Create a cascade classifier. It will contain the features of the face. (OpenCV already contains many pre-trained classifiers for face, eyes, smile etc. Those XML files are stored in/data/haarcascades/ folder.)

Step 2: OpenCV reads the image and features file. Also, search for the row and column values of the face numpy array (i.e. The face rectangle co-ordinates using detectMultiScale function)

Step 3: Display the image of the face with a rectangle around it. (using cv2.rectangle function)
