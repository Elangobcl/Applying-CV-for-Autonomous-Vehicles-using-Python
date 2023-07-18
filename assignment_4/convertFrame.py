from threading import Thread
import cv2, time

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4)) 
        self.size = (frame_width, frame_height)
        self.result = cv2.VideoWriter('Lane_detect.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20,self.size)
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)
            
    def show_frame(self):
        self.result.write(self.frame)
        cv2.imshow('frame', self.frame)
        cv2.waitKey(self.FPS_MS)

cap = cv2.VideoCapture("Lane_video.mp4")
# Write video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height)
#result = cv2.VideoWriter('Lane_detect.avi', cv2.VideoWriter_fourcc(*'MJPG'), 35,size)
src = "Lane_video.mp4"
threaded_camera = ThreadedCamera(src)
while(1):
    try:
        threaded_camera.show_frame()
    except AttributeError:
        pass
    """
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    cv2.imshow("ImageRegion", frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #result.write(frame)
    """