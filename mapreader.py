#!/usr/bin/env python3
# python 3.8.19
# Run in terminal python3 mapreader.py develop/develop-001.png
# Calling required Imports
import cv2 #version 4.8.0
import sys
import numpy as np #version 1.22.3

# ----------------------------------------------------------------------------
# Loading original Image
original_image = cv2.imread(sys.argv[1])
# ----------------------------------------------------------------------------
# Function for:
# Segmenting the map from the blue background in the original image
# and extracting it into a separate image then
# returning the converted image to hsvWarped space

def blue_background(original_image):

    bgrTohsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    # Setting the mask using values found using xv /2
    mask = cv2.inRange(bgrTohsv, (90, 80, 5), (110, 255, 255))
    # inverting the mask
    notMask = cv2.bitwise_not(mask)
    # using bitwise_and to apply the inverted mask
    blueRemoved = cv2.bitwise_and(bgrTohsv, bgrTohsv, mask=notMask)

    # Returning image with blue background removed and replaced with black color
    return blueRemoved

bl = blue_background(original_image)
# ----------------------------------------------------------------------------
# This function converts the image in:
# BGR and Gray Space
def imagecon(bl):

    # Converting hsv blueRemoved to bgr
    hsvTobgr = cv2.cvtColor(bl, cv2.COLOR_HSV2BGR)
    # convert from bgr to gray
    bgrToGray = cv2.cvtColor(hsvTobgr, cv2.COLOR_BGR2GRAY)

    # Returning the images
    return hsvTobgr, bgrToGray

hsvTobgr, bgrToGray = imagecon(bl)
Gray = imagecon(bl)
# ----------------------------------------------------------------------------
# This function uses the output of 'imagecon' to do:
# Binary Thresholding to isolate the map
# Then drawing a contour around the map

def thresCont(Gray):

    # Creating a binary thresh to isolate the map
    original, binary_thresh = cv2.threshold(bgrToGray, 0, 255,
                                            cv2.THRESH_BINARY)
    # Get contours of threshold
    contours, hierarchy = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original_image, contours, -1, (0, 255, 0), 3)

    # Used the website below to obtain the best min area around the map
    # Code has been adapted from link mentioned below.
    # https://theailearner.com/2020/11/03/opencv-minimum-area-rectangle/
    # Using minAreaRect by passing contours to obtain 4 cords of the map
    dimensions = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(dimensions)
    # int0 uses the system architecture uses either int32or64
    box = np.int0(box)
    # creating an anti-aliased line to plot around the map
    cv2.drawContours(hsvTobgr, [box], -1, (0, 255, 0), 1, cv2.LINE_AA)
    # returning the calculated box numbers
    return box

box = thresCont(Gray)
# ----------------------------------------------------------------------------
# This function first Organises the points in the order of:
# top(left, right)(tl,tr), bottom(right, left)(br,bl)
# The dimensions of the image are then calculated
# And a transformation multiplier is created which is the return of the func
# Code below is adapted from:
# https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

def warpMulti(box):

    # Initializing a list to pass the cords to
    point_s = np.zeros((4, 2), dtype="float32")

    # Finding the sum between the points
    boxsum = box.sum(axis=1)
    # Top left: this will have the lowest sum
    point_s[0] = box[np.argmin(boxsum)]
    # Bottom right: largest sum
    point_s[2] = box[np.argmax(boxsum)]

    # Finding the difference between the points
    diffr = np.diff(box, axis=1)
    # Top right: Smallest Difference
    point_s[1] = box[np.argmin(diffr)]
    # Bottom left: largest Difference
    point_s[3] = box[np.argmax(diffr)]

    # Ordering the points and putting them in a list
    (tl, tr, br, bl) = point_s

    # width of the image calculation: max distance from br and bl points
    wA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    wB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # Calculating the max width of the image
    maximumWidth = max(int(wA), int(wB))

    # height of image calculation: max distance tr and br points
    hA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    hB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # Calculating the max height of the image
    maximumHeight = max(int(hA), int(hB))

    # Putting the dimensions of the image via the cords
    # this results in a top-down view of the image
    # sorted by: tl, tr, br, bl
    multiplier = np.array([
        [0, 0],
        [maximumWidth, 0],
        [maximumWidth, maximumHeight],
        [0, maximumHeight]], dtype="float32")

    return point_s, maximumWidth, maximumHeight, multiplier

point_s, maximumWidth, maximumHeight, multiplier = warpMulti(box)
nums = warpMulti(box)
# ----------------------------------------------------------------------------
# Function for applying perspective transform matrix and applying it

def warptransform(nums):
    PerTform = cv2.getPerspectiveTransform(point_s, multiplier)
    warped = cv2.warpPerspective(hsvTobgr, PerTform,
                                 (maximumWidth, maximumHeight))
    # returning the warped image which is just the map by itself
    return warped

segmentedmap = warptransform(nums)

# ----------------------------------------------------------------------------
# Segmented map convert to hsv to find Red mask to find triangle
# Code below is adapted from:
# https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/

def isolatetri(segmentedmap):
    #  original_image to hsvWarped
    hsvWarped = cv2.cvtColor(segmentedmap, cv2.COLOR_BGR2HSV)

    # Setting ranges and limits for the HSV value of triangle
    # Hue(0-180), [saturation, value: 255]

    # Hue (0-30)
    lower1 = np.array([0, 55, 0])
    upper1 = np.array(([30, 255, 255]))
    # Hue (160-179)
    lower2 = np.array([160, 50, 0])
    upper2 = np.array([179, 255, 255])

    lower_mask = cv2.inRange(hsvWarped, lower1, upper1)
    upper_mask = cv2.inRange(hsvWarped, lower2, upper2)
    full_mask = cv2.bitwise_or(lower_mask, upper_mask)

    # Isolating triangle
    maskedIM = cv2.bitwise_and(hsvWarped, hsvWarped, mask=full_mask)

    # Converting from HSV to BGR space for final isolated triangle
    triangleBGR = cv2.cvtColor(maskedIM, cv2.COLOR_HSV2BGR)

    return triangleBGR

iso = isolatetri(segmentedmap)
# ----------------------------------------------------------------------------
# Function for drawing a counter around the triangle and then
# finding the 3 points
# Code below has been adapted from:
#https://stackoverflow.com/questions/11424466/how-to-detect-triangle-edge-in-opencv-or-emgu-cv


def pointstri(iso):
    #Using Canny to obtain a outline of the triangle

    tri_edges = cv2.Canny(iso, 100, 179)  # 100, 179

    # Calculating contours
    contours, hir = cv2.findContours(tri_edges, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Applying cv2 function to find the best triangle
    area, triangle = cv2.minEnclosingTriangle(contours[0])
    # Passing the 3 points found to a variable, this will be returned
    triangle = np.int0(triangle)
    # Drawing contours around the triangle
    cv2.drawContours(iso, triangle, -1, (255, 0, 0), 3)  # -1 #255,0,0

    return triangle

triangle = pointstri(iso)

# ----------------------------------------------------------------------------
# This function finds the tip and the base points
# It first finds the distance between the points: D1,2,3
# The distances are then sorted by the

def sortpoints(triangle):
    # Setting the cords of the triangle to a variable to be used in calculation
    A = triangle[0]
    B = triangle[1]
    C = triangle[2]
    # Distance between points calculation
    D1 = ((B[0][0] - A[0][0]) ** 2 + (B[0][1] - A[0][1]) ** 2) ** 0.5
    D2 = ((C[0][0] - B[0][0]) ** 2 + (C[0][1] - B[0][1]) ** 2) ** 0.5
    D3 = ((C[0][0] - A[0][0]) ** 2 + (B[0][1] - C[0][1]) ** 2) ** 0.5

    # Creating a list which then will sort the cords
    # The list is sorted by finding the tip which has the longest distance
    # The remaining two points are the base points
    ls = {'C': D1, 'A': D2, 'B': D3}
    sortedls = sorted(ls.items(), key=lambda x: x[1])

    # If statement which places the points in the relevant place
    if (sortedls[0][0] == 'A'):
        tip = A
        Base1 = B
        Base2 = C

    elif (sortedls[0][0] == 'B'):
        tip = B
        Base1 = A
        Base2 = C

    elif (sortedls[0][0] == 'C'):
        tip = C
        Base1 = A
        Base2 = B

    # Returning the tip and base points
    return tip, Base1, Base2

tip, Base1, Base2 = sortpoints(triangle)
cords = sortpoints(triangle)
# ----------------------------------------------------------------------------
# This function normalises the cords to the ratio of the map
# calculated by dividing the cords by the calculated width and height
# the middle points of the triangle is also found here
def normalisation(cords):
    tipx = tip[0][0]
    # origin at the bottom left-hand (south-west) corner
    tipy = maximumHeight - tip[0][1]
    xnorm = tipx / maximumWidth
    ynorm = tipy / maximumHeight

    BaseMP = ((Base1[0] + Base2[0]) / 2)
    return xnorm, ynorm, BaseMP

xnorm, ynorm, BaseMP = normalisation(cords)

cords = normalisation(cords)
# ----------------------------------------------------------------------------
# Function that appends the coordinates
# that will be used to calculate the bearing

def bearing(cords):
    PointA = (tip[0][0], tip[0][1])
    PointB = (BaseMP[0], BaseMP[1])

    return PointA, PointB

PointA, PointB = bearing(cords)
# ----------------------------------------------------------------------------
# Function to find the bearing
# Code adapted from:
#https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points

def angle_between_points(A, B):
    distance1 = B[0] - A[0]
    distance2 = B[1] - A[1]
    if distance1 == 0:
        if distance2 == 0:  # The same points are detected
            ang = 0 # The angle is 0
        else:
            ang = 0 if A[1] > B[1] else 180
    elif distance2 == 0:
        ang = 90 if A[0] < B[0] else 270
    else:
        ang = np.arctan(distance2 / distance1) / np.pi * 180
        flip = A[1] < B[1]
        if (flip and ang < 0) or (not flip and ang > 0):
            ang += 270
        else:
            ang += 90
    return ang
# Passing the bearing from Point B which is the middle points to the tip
bearing = angle_between_points(PointB, PointA)
# ----------------------------------------------------------------------------
# Main program.

# Ensure we were invoked with a single argument.

if len(sys.argv) != 2:
    print("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    exit(1)

print("The filename to work on is %s." % sys.argv[1])
xpos = xnorm
ypos = ynorm
hdg = bearing
# xpos = 0.5
# ypos = 0.5
# hdg = 45.1

# Output the position and bearing in the form required by the test harness.
print("POSITION %.3f %.3f" % (xpos, ypos))
print("BEARING %.1f" % hdg)

# ----------------------------------------------------------------------------