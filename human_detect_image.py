import cv2
from os import listdir

image_path = './images/'

for image in listdir(image_path):
    print(image)
    image = cv2.imread(image_path + image)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (humans, _) = hog.detectMultiScale(image, winStride=(10, 10),
                                       padding=(32, 32), scale=1.1)
    
    print(f'Human Detected: {len(humans)}', end="\n\n")

    for(x, y, w, h) in humans:
        pad_w, pad_h = int(0.15 * w), int(0.01 * h)
        cv2.rectangle(image, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)
        cv2.putText(image, text="Hello User!", org=(x + (w-6), y+10), 
                    fontScale=1, color=(0, 255, 0), fontFace=1, lineType= cv2.LINE_AA)

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()