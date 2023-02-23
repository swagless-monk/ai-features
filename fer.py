from warnings import filterwarnings
from os import listdir
from fer import FER
from facenet_pytorch import MTCNN
from cv2 import imread, imshow

image_path = './images/faces/'

for image in listdir(image_path):
    print(image)
    image = imread(image_path + image)

    emotion_detector = FER(mtcnn=True)
    emotions = emotion_detector.detect_emotions(image)

    emo_dict = emotions[0]['emotions']

    emo = 0
    for val in emo_dict.values():
        if val > emo:
            emo = val

    facial_emotion = list(emo_dict.keys())[list(emo_dict.values()).index(emo)]
    print(f'EMOTION: {facial_emotion.capitalize()} ({emo*100} %)', end='\n\n')