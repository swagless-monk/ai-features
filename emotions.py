from fer_pytorch.fer import FER
from warnings import filterwarnings

def emotions() -> None:
    filterwarnings(action='ignore')
    fer = FER()
    fer.get_pretrained_model("resnet34")
    fer.run_webcam()

    #loop.close()