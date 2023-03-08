import multiprocessing
from time import time, sleep

def run(funtion_name, timeout:int=0):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=funtion_name, args=())
    p.start()

    start = time()

    while True:
        sleep(0.1)
        end = time()

        if q.empty():
            if end-start > timeout:
                p.terminate()
                return None
        else:
            return q.get()