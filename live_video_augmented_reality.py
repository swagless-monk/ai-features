import cv2
import numpy as np
from fer import FER
from facenet_pytorch import MTCNN

vid = cv2.VideoCapture(0)

overlay = cv2.imread('./images/mk51.jpg')
size = 75
overlay = cv2.resize(overlay, (size, size))

image_to_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(image_to_gray, 1, 255, cv2.THRESH_BINARY)

id_model_path = 'C:/Users/Keith Family/OneDrive/Documents/Custom Office Templates/helmet/id_models/'
face_detect = cv2.CascadeClassifier(id_model_path + 'haarcascade_frontalface_alt.xml')
face_profile_detect = cv2.CascadeClassifier(id_model_path + 'haarcascade_profileface.xml')

while True:
    ret, frame = vid.read()

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detecting objects
    (humans, _) = hog.detectMultiScale(frame, winStride=(10, 10),
                                       padding=(32, 32), scale=1.1)
    faces = face_detect.detectMultiScale(frame)
    face_profiles = face_profile_detect.detectMultiScale(frame)

    # Add outline detected objects on screen
    for(x, y, w, h) in faces:
        pad_w, pad_h = int(0.15 * w), int(0.01 * h)
        cv2.rectangle(frame, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)
        cv2.putText(frame, text="Hello User (front)!", org=(x + (w-6), y+10), 
                    fontScale=1, color=(0, 0, 0), fontFace=1, lineType= cv2.LINE_AA)
    for(x, y, w, h) in face_profiles:
        pad_w, pad_h = int(0.15 * w), int(0.01 * h)
        cv2.rectangle(frame, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (255, 0, 0), 2)
        cv2.putText(frame, text="Hello User (profile)!", org=(x + (w-6), y+10), 
                    fontScale=1, color=(0, 0, 0), fontFace=1, lineType= cv2.LINE_AA)
    for(x, y, w, h) in humans:
        pad_w, pad_h = int(0.15 * w), int(0.01 * h)
        cv2.rectangle(frame, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)
    
    image_region = frame[-size-5: -5, -size-5: -5]
    image_region[np.where(mask)] = 0
    image_region += overlay

    cv2.imshow('Web Camera Live Feed', frame)

    if cv2.waitKey(1) * 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()