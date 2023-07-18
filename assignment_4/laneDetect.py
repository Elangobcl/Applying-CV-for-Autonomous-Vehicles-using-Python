
import numpy as np
import cv2
import time
import numba as nb # for fast looping through image array

def read_img(img,chn=3):
    img_read = cv2.imread(img,chn)
    return img_read

def canny_edge_detection(img):
    G_Blur_img = cv2.GaussianBlur(img,(3,3), 0.6)
    canny_edges = cv2.Canny(G_Blur_img, 30, 120)
    return canny_edges

def region_of_interest(img,polyPoints):
    mask_img = np.zeros_like(img)
    height,width = img.shape
    roi = cv2.fillPoly(mask_img,np.array([polyPoints]), (250,0,0))
    roi_merged_img = cv2.bitwise_and(img,roi)
    #roi_plot = cv2.polylines(roi_merged_img, np.int32([polyPoints]), True, (255,255,255), 3)
    return roi_merged_img

def display_lines(img,lines):
    mask_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line
            cv2.line(mask_img, (x1,y1), (x2,y2), (255,0,0), 10)
    
    merged_img = cv2.addWeighted(img,0.8,mask_img,1,1)
    return merged_img

def make_line(img, line_coefficient):

    slope = line_coefficient[0]
    intercept = line_coefficient[1]    
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    line = np.array([x1,y1,x2,y2])
    return line

def img_cvt2gray(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray_img

def average_slope_intercept(img,lines):
    left_line = []
    right_line = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4) # convert 2d-array into 1-d array    
            slope,intercept = np.polyfit((x1,x2),(y1,y2),1)
          
            if slope < 0:
                left_line.append((slope,intercept))
            else:
                right_line.append((slope,intercept))

        left_fit_avg = np.average(left_line,axis=0)
        right_fit_avg = np.average(right_line,axis=0)

        left_line = make_line(img, left_fit_avg)    
        right_line = make_line(img, right_fit_avg)
    return(np.array([left_line,right_line]))

@nb.njit()
def remove_nonLaneLine(T, image_data):
    # grab the image dimensions
    maskImg = np.copy(image_data)
    for y in range(0,image_data.shape[0]):
        for x in range(0,image_data.shape[1]):
            # threshold the pixel
            if maskImg[y, x] >= T:
                maskImg[y, x] = 0 
    return maskImg
cap = cv2.VideoCapture('Lane_video.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height)
# save video in .avi format at 35fps
result = cv2.VideoWriter('Lane_detect.avi', cv2.VideoWriter_fourcc(*'MJPG'), 35,size)
while(1): 
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    gray_img = img_cvt2gray(frame)
    imgInv = cv2.bitwise_not(gray_img) #invert the image
    ThresImg = remove_nonLaneLine(70, imgInv) #mask non-lane center line by thresholding
    expImg_Inv = cv2.bitwise_and(imgInv,ThresImg) # remove non-lane center line by and with invertedImg
    canny_img = canny_edge_detection(expImg_Inv) # detect canny edges
    Lines = [(400,300),# Top-left corner 
              (0, 500), # Bottom-left corner            
              (900,540), # Bottom-right corner
              (580,275)] # Top-right corner
    height,width = frame.shape[:2]
    lines = [(0,height),(920,height),(400,250)]
    roi_img = region_of_interest(canny_img,Lines)# crops roi from lane image
    houghLines = cv2.HoughLinesP(roi_img, 3, np.pi/180, 40,None, 35, 35) #returs list of line in the frame
    avg_lines = average_slope_intercept (frame,houghLines) # Avg all the lines in the frame
    lines_img = display_lines(frame,avg_lines) # plots lane tracking line on the frame
    result.write(lines_img)  # save frame
    cv2.imshow("ImageRegion", lines_img)     
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()