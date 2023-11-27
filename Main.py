import cv2
import numpy as np
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import accuracy
import tensorflow as tf
import imutils

from ultralytics import YOLO
Model=YOLO("yolov8n.pt")


#Called by cv2 createTrackbar internally
def setVal(x):
    print("")

#Creates Window
cv2.namedWindow("Color Match")
#Creates Trackbar
cv2.resizeWindow("Color Match", 500, 500)
cv2.createTrackbar("Max Hue", "Color Match", 50, 180, setVal)
cv2.createTrackbar("Max Saturation", "Color Match", 255, 255, setVal)
cv2.createTrackbar("Max Value", "Color Match", 255, 255, setVal)
cv2.createTrackbar("Min Hue", "Color Match", 30, 180, setVal)
cv2.createTrackbar("Min Saturation", "Color Match", 72, 255, setVal)
cv2.createTrackbar("Min Value", "Color Match", 49, 255, setVal)


#List to store points
bpoints = [deque(maxlen=1024)]

#Index used to mark the points in bpoints
index = 0

#Colour of Line
colors = (0, 0, 0)

#For Canvas setup
paintWindow = np.zeros((471,636,3)) + 255

#kernel later used for dilation/eroding
kernel = np.ones((5,5),np.uint8)

def matchVal(val):
    Alpha = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'D', 13: 'E', 14: 'G', 15: 'I', 16: 'J', 17: 'K', 18: 'L', 19: 'M', 20: 'N', 21: 'P', 22: 'R', 23: 'S', 24: 'T', 25: 'U', 26: 'V', 27: 'X', 28: 'samadengan'}
    return (Alpha[val])

#Used for resizing(cv2 resize) captured frame
def get_frame(cam, factor):
    check, frame = cam.read()
    #Is a frame captured by camera?
    if(not check):
        return None

    #INTER_AREA is for scaling down the frame for better quality
    frame = cv2.resize(frame, None, fx=factor, fy=factor, interpolation = cv2.INTER_AREA)
    return frame

def save_image(filename, image):
    size = 128
    image = imutils.resize(image, width=size)
    cv2.imwrite(filename, image)
    im = Image.open("result.png")
    mergedImage = Image.new("RGB",(416, 416))
    mergedImage.paste(im, (0, 0))
    mergedImage.save("Test//detected.png")

def load_image():
    SEGMENTATION_ENGINE = YOLO("best.pt")
    RESULT=SEGMENTATION_ENGINE.predict(source="Test",save=True,save_txt=True, project="Test",name="ab")
    with open("Test//ab//labels//detected.txt","r") as file:
        for line in file:
            val = int(line[:2])
            print(matchVal(val))

#Is this being called by main function?
if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    factor = 0.8
    
    while True:
        frame = get_frame(cam, factor)
        #Mirror Frame
        frame = cv2.flip(frame, 1)
        #Rgb to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)        
        #Get value of trackbars
        max_hue = cv2.getTrackbarPos("Max Hue", "Color Match")
        max_sat = cv2.getTrackbarPos("Max Saturation", "Color Match")
        max_val = cv2.getTrackbarPos("Max Value", "Color Match")
        min_hue = cv2.getTrackbarPos("Min Hue", "Color Match")
        min_sat= cv2.getTrackbarPos("Min Saturation", "Color Match")
        min_val = cv2.getTrackbarPos("Min Value", "Color Match")

        #Creating array from those values
        max_hsv = np.array([max_hue, max_sat, max_val])
        min_hsv = np.array([min_hue, min_sat, min_val])

        #Creating Rectangular Box and Text on Live Frame
        frame = cv2.rectangle(frame, (200,1), (300,65), (122,122,122), -1)
        cv2.putText(frame, "CLEAR ALL", (210, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        #Creating mask
        mask = cv2.inRange(hsv, min_hsv, max_hsv)
        #Erode outer surface of subject
        mask = cv2.erode(mask, kernel, iterations=1)
        #Creates Outline by erode and dilation subtraction
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #Makes boundary bold(kind of)
        mask = cv2.dilate(mask, kernel, iterations=1)

        #Finding contours(external) and approximating vertical & horizontal points
        #_ used as we don't need to store heirarchy returned
        cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(cnts)>0:
            #Sort contours in descending wrt their Area 
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            #Get radius, x & y values
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            #Create circle
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 3)

            #Capture contour movements to find center
            M = cv2.moments(cnt)
            #M['m10'] is for x coordinates summation, M['m01'] for y coordinates summation
            #& M['m00'] for total area
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            
            #Checking if the user clicks on the clear button
            if center[1] <= 65:
                if 200 <= center[0] <= 300:                    
                    #Clear all points and paintWindow, reset all values
                    bpoints = [deque(maxlen=512)]
                    index = 0
                    #Clear paintWindow to white
                    paintWindow[67:, :] = 255
                    
            #Storing the points if tracker isn't in Clear All Button Range
            else :
                bpoints[index].appendleft(center)
                
        #When countour length is 0, then append some values to stop errors
        else:
            bpoints.append(deque(maxlen=512))
            index += 1
        
        #Draw lines on canvas and frame 
        points = [bpoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors, 10)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors, 10)

            
        #Bitwise AND
        final = cv2.bitwise_and(frame, frame, mask=mask)
        #Adding Blur
        final = cv2.medianBlur(final, 5)

        #Show results
        #cv2.imshow("Final Image", final)
        #cv2.moveWindow("Final Image", int(1366/2), int(768/2))
        cv2.imshow("Original Image", frame)
        #cv2.moveWindow("Original Image", 0, 0)
        cv2.imshow("Paint", paintWindow)
        
        #Get user key input, if Esc then break
        c = cv2.waitKey(100)
        if(c==83 | c==115):
            save_image("result.png", paintWindow)
            load_image()
        if(c==27):
            break
    cv2.destroyAllWindows()
