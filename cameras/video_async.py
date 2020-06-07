import threading

class VideoCaptureAsync:
    def __init__(self,camera):
        self.camera = camera
        self.grabbed, self.frame = self.camera.read()
        self.started = False
        self.read_lock = threading.Lock()

    
    def startReading(self):
        if self.started:
            print('Started video capture async')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args = ())
        self.thread.start()
    
    def update(self):
        while self.started:
            grabbed, frame = self.camera.read()
            with self.read_lock:
                self.frame = frame
                self.grabbed = grabbed
    
    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self):
        self.camera.stop()