import cv2
import numpy as np
from facenet_pytorch import MTCNN
from warnings import filterwarnings
from deepface import DeepFace

filterwarnings(action='ignore')

def live_video() -> None:
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    overlay = cv2.imread('../images/mk51.jpg')
    size = 1000
    overlay = cv2.resize(overlay, (size, size))

    #image_to_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    #ret, mask = cv2.threshold(image_to_gray, 1, 255, cv2.THRESH_BINARY)

    id_model_path = 'C:/Users/Keith Family/OneDrive/Documents/Custom Office Templates/helmet/id_models/'
    face_detect = cv2.CascadeClassifier(id_model_path + 'haarcascade_frontalface_alt.xml')
    face_profile_detect = cv2.CascadeClassifier(id_model_path + 'haarcascade_profileface.xml')


    while True:
        ret, frame = vid.read()

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        try:
            df = DeepFace.analyze(img_path=frame)
            profile = {
                'Age': df[0]['age'],
                'Gender': df[0]['dominant_gender'],
                'Race': df[0]['dominant_race']
            }
        except ValueError:
            continue

        # Detecting objects
        (humans, _) = hog.detectMultiScale(frame, winStride=(10, 10),
                                        padding=(32, 32), scale=1.1)
        faces = face_detect.detectMultiScale(frame)

        """try:
            for face in faces[0]:
                cv2.putText(frame, text=profile, org=(14, 14), 
                            fontScale=1, color=(0, 0, 0), fontFace=1, lineType= cv2.LINE_AA)
        except ValueError:
            break"""

        # Add outline detected objects on screen
        for(x, y, w, h) in faces:
            pad_w, pad_h = int(0.15 * w), int(0.01 * h)
            cv2.rectangle(frame, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)
            cv2.putText(frame, text="Hello User (front)!", org=(x + (w-6), y+10), 
                        fontScale=1, color=(0, 0, 0), fontFace=1, lineType= cv2.LINE_AA)
            try:
                cv2.putText(frame, text=f'Age: {str(profile["Age"])}', org=(x + (w-6), y+24), 
                                fontScale=1, color=(0, 0, 0), fontFace=1, lineType= cv2.LINE_AA)
                cv2.putText(frame, text=f'Gender: {str(profile["Gender"]).capitalize()}', org=(x + (w-6), y+36), 
                                fontScale=1, color=(0, 0, 0), fontFace=1, lineType= cv2.LINE_AA)
                cv2.putText(frame, text=f'Race: {str(profile["Race"]).capitalize()}', org=(x + (w-6), y+48), 
                                fontScale=1, color=(0, 0, 0), fontFace=1, lineType= cv2.LINE_AA)
            except NameError:
                cv2.putText(frame, text=f'I am unable to clearly make out target face', org=(x + (w-6), y+24), 
                                fontScale=1, color=(0, 0, 0), fontFace=1, lineType= cv2.LINE_AA)
            except KeyError:
                cv2.putText(frame, text=f'I am unable to clearly make out target face', org=(x + (w-6), y+24), 
                                fontScale=1, color=(0, 0, 0), fontFace=1, lineType= cv2.LINE_AA)

        for(x, y, w, h) in humans:
            pad_w, pad_h = int(0.15 * w), int(0.01 * h)
            cv2.rectangle(frame, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)

        #image_region = frame[-size-5: -5, -size-5: -5]
        #image_region[np.where(mask)] = 0
        #image_region += overlay

        window_name = 'Web Camera Live Feed'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) * 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

live_video()