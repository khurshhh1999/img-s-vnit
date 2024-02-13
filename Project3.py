import numpy as np
import cv2
from harris_detector import *
import imutils
import os

def stitching(addresses, threshold1, threshold2, typeStitch, detectorType = 'sift'):

    address1, address2 = addresses
    image1 = cv2.imread(address1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    image2 = cv2.imread(address2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    if typeStitch == 1:
        image1 = cv2.flip(image1, 1)
        image2 = cv2.flip(image2, 1)
        image1, image2 = image2, image1

        threshold1, threshold2 = threshold2, threshold1
        
        gray1 = cv2.flip(gray1, 1)
        gray2 = cv2.flip(gray2, 1)
        gray1, gray2 = gray2, gray1

    _, kp1 = harris_corner_detection(gray1, threshold1)
    _, kp2 = harris_corner_detection(gray2, threshold2)


    if detectorType == 'sift':
        featureDetector = cv2.xfeatures2d.SIFT_create()
        bMatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    elif detectorType == 'brisk':
        featureDetector = cv2.BRISK_create()
        bMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif detectorType == 'orb':
        featureDetector = cv2.ORB_create()
        bMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


    kp1, desc1 = featureDetector.compute(gray1, kp1)
    kp2, desc2 = featureDetector.compute(gray2, kp2)

    matches = bMatcher.match(desc1,desc2)
    matches = sorted(matches, key = lambda m:m.distance)

    print(len(matches), len(desc1), len(desc2))

    matchesIm = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(matchesIm)
    plt.show()

    pts1 = np.zeros((len(matches), 1, 2), dtype=np.float32)
    pts2 = np.zeros((len(matches), 1, 2), dtype=np.float32)

    for i in range(0,len(matches)):
        pts1[i] = kp1[matches[i].queryIdx].pt
        pts2[i] = kp2[matches[i].trainIdx].pt

    H, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4)
    print(H)
    print(len(matches), len(kp1))

    width = image2.shape[1] + image1.shape[1]
    height = image2.shape[0] + image1.shape[0]

    result = cv2.warpPerspective(image2, H, (width, height))
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if result[i,j].all() == 0:
                result[i,j] = image1[i,j]
    

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    maxContour = max(contours, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(maxContour)
    result = result[y:y + h, x:x + w]
    if typeStitch == 1:
        result = cv2.flip(result, 1)
    plt.imshow(result)
    plt.show()

    return result, H

def panorama(folder):
    files = os.listdir(folder)
    images = [folder + '/' + i for i in files]
    addresses = [images[0]]
    threshold2 = threshold1 = 10000000000
    for i in range(1,len(images)):
        addresses.append(images[i])
        result, H = stitching(addresses, threshold1, threshold2, i%2)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        print(i)
        filename = 'Results/AltPan/result'+str(i)+'.jpg'
        cv2.imwrite(filename, result)
        addresses = [filename]
        threshold1 = threshold1*0.2
    return None

def div_conq(folder):
    files = os.listdir(folder)
    images = [folder + '/' + i for i in files]
    addresses = []
    outputs = []
    threshold2 = threshold1 = 8000000000
    j = 0
    while len(outputs) > 1 or len(outputs) == 0:
        for i in range(int(len(images)/2)):
            addresses.append(images.pop(0))
            if len(images) > 0:
                addresses.append(images.pop(0))
            else:
                outputs.append(addresses.pop(0))
            print(addresses)
            result, H = stitching(addresses, threshold1, threshold2, j%2)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            filename = 'Results/DivConq/result'+str(j)+str(i)+'.jpg'
            cv2.imwrite(filename, result)
            outputs.append(filename)
            addresses = []
        j += 1
        images = outputs
        outputs = []
        threshold1 = threshold1*0.5
        threshold2 = threshold2*0.5


panorama('imagesSet3')