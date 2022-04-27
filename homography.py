import cv2
import numpy as np
from time import time 

img = cv2.imread("image2.jpeg", cv2.IMREAD_GRAYSCALE)  # queryiamge
tiger = cv2.imread("tiger.png")
tiger = cv2.resize(tiger, (img.shape[1]//3, img.shape[0]//3))

img = cv2.resize(img, (img.shape[1]//3, img.shape[0]//3))
cap = cv2.VideoCapture("http://192.168.1.5:8081/video")
# Features
orb = cv2.ORB_create(1000)

kp_image, desc_image = orb.detectAndCompute(img, None)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
p_dst = np.zeros((4,1,2))

while True:
    try:
        _, frame = cap.read()
        t1 = time()
        
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
        
        kp_grayframe, desc_grayframe = orb.detectAndCompute(grayframe, None)

        
        
        matches = matcher.match(desc_image, desc_grayframe, None)
        
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        matches = matches[:12]

        points1 = np.zeros((len(matches), 2), dtype=np.float32) 
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        
        for i, match in enumerate(matches):
            points1[i, :] =  kp_image[match.queryIdx].pt   
            points2[i, :] = kp_grayframe[match.trainIdx].pt

        img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, matches, grayframe)

        # Homography
        if len(matches) > 10:
            
            matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
            
            # Perspective transform
            h, w = img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        
            dst = cv2.perspectiveTransform(pts, matrix)
            
            M = cv2.getPerspectiveTransform(pts, dst)
            warp = cv2.warpPerspective(tiger, M, (500, 500))
            
            dst = p_dst*0.5 + dst*0.5
            p_dst = dst
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Homography", grayframe)


        cv2.imshow("Image", warp)
        cv2.imshow("grayFrame", grayframe)
        # cv2.imshow("img3", img3)
        t2 = time()
        # print((t2-t1)*1000)
    except:
        print("No feature")

    
    
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()