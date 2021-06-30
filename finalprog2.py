import cv2
import numpy as np
from matplotlib import pyplot as plt

def unit_vector(v):
    return v / np.linalg.norm(v)
def distance(p1, p2, p3, p4):
    global l1, l2, l3, l4, a ,b 
    l1 = int(((p4[0] - p3[0])**2 + (p4[1] - p3[1])**2)**0.5)
    l2 = int(((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5)
    l3 = int(((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)**0.5)
    l4 = int(((p4[0] - p2[0])**2 + (p4[1] - p2[1])**2)**0.5)
    a = l1 == l2
    b = l3 == l4
    return l1, l2, l3, l4, a, b

def get_corner_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p1[0] - p3[0], p1[1] - p3[1]])
    v1_unit = unit_vector(v1)
    v2_unit = unit_vector(v2)
    radians = np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1, 1))
    return np.degrees(radians)
    
def get_corner_angle_opp(p4, p2, p3):
    v3 = np.array([p4[0] - p2[0], p4[1] - p2[1]])
    v4 = np.array([p4[0] - p3[0], p4[1] - p3[1]])
    v3_unit = unit_vector(v3)
    v4_unit = unit_vector(v4)
    radians = np.arccos(np.clip(np.dot(v3_unit, v4_unit), -1, 1))
    return np.degrees(radians)  

# reading image
smallimg = cv2.imread('shapes.jpg')

width=int(smallimg.shape[1]*200/100)
height=int(smallimg.shape[0]*200/100)
dsize=(width,height)
img=cv2.resize(smallimg,dsize)
# converting image into grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.blur(gray,(10,10)) 
# setting threshold of gray image
_, threshold = cv2.threshold(blurred,230,255,cv2.THRESH_BINARY)
  
# using a findContours() function
contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
i = 0
  
# list for storing names of shapes
for contour in contours:
  
    # here we are ignoring first counter because 
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue
  
    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(
        contour, 0.012 * cv2.arcLength(contour, True), True)
      
    # using drawContours() function
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)
  
    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
    cv2.putText(img, str(len(approx)), (x-10, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # putting shape name at center of each shape
    if len(approx) == 3:
        cv2.putText(img, 'Triangle', (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
  
    elif len(approx) == 4:
        p1 = approx[0][0]
        p2 = approx[1][0]
        p3 = approx[-1][0]
        p4 = approx[2][0]
        (degrees)=get_corner_angle(p1, p2, p3)
        (degrees_opp) = get_corner_angle_opp(p4, p2, p3)
        dist1 = distance(p1, p2, p3, p4)

        if ((88 <= int(degrees) <= 92) and (88 <= int(degrees_opp) <= 92)):
            if(a == b and (l2-l3) in range(-30,30)):
                #print(l2-l3)
                cv2.putText(img, 'Square', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                #print(l2,l3);
                cv2.putText(img, 'Rectangle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                 
        else:cv2.putText(img, 'Trapezium', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
  
    elif len(approx) == 5:
        cv2.putText(img, 'Pentagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
  
    elif len(approx) == 6:
        cv2.putText(img, 'Hexagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    elif len(approx) == 7:
        cv2.putText(img, 'Heptagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    elif len(approx) == 8:
        cv2.putText(img, 'Octagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
  
    else:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

            # convert the (x, y) coordinates and radius of the circles to integers
        circle = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
        for (aaa, bbb, r) in circle:
                #print(circle)
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                #cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.putText(img, 'circle', (aaa, bbb),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            if(aaa!=x and bbb!=y):
                cv2.putText(img, 'ellipse', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            #if ellipse is not None:



# displaying the image after drawing contours
cv2.imshow('shapes', img)
  
cv2.waitKey(0)
cv2.destroyAllWindows()
