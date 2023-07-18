from ultralytics import YOLO
#import ultralytics as ut
#import cv2
#ut.checks()
# model = YOLO('yolov8n.pt') # downloads yolo weights
model = YOLO('yoloWeights/yolov8n.pt')
results = model("Images/3.png", show=True, verbose = True)  # show = True displays image after processing
# cv2.waitKey(0)
