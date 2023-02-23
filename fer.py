from fer_pytorch.fer import FER

fer = FER()
fer.get_pretrained_model("resnet34")
fer.run_webcam()
