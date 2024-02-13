import numpy as np
import cv2
import matplotlib.pyplot as plt

def harris_corner_detection(image, threshold = 50000000000):
    Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = Ix*Iy

    duplicate_image = np.copy(image)
    kps = []
    shape = np.shape(image)
    k1 = 3
    k2 = 3
    i = j = 0
    m = n = 1

    M = np.zeros((2, 2))
    out = np.zeros(np.shape(Ix))

    while(i + k1 <= shape[0]):
        while(j + k2 <= shape[1]):
            M[0][0] = np.sum(Ixx[i:i+k1, j:j+k2])
            M[1][0] = M[0][1] = np.sum(Ixy[i:i+k1, j:j+k2])
            M[1][1] = np.sum(Iyy[i:i+k1, j:j+k2])
            out[m][n] = np.linalg.det(M) - 0.06*(np.trace(M)**2)
            if out[m][n] > threshold:
                cv2.circle(duplicate_image, (n, m), 3, (0, 0, 255), -1)
                kp = cv2.KeyPoint(n, m, 5, _class_id=0)
                kps.append(kp)
            j += 1
            n += 1
        i += 1
        m += 1
        j = 0
        n = 0

    plt.imshow(duplicate_image)
    plt.show()
    im1 = cv2.drawKeypoints(image, kps, None)
    plt.imshow(im1)
    result = duplicate_image
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    filename = 'Results/Harris/result.jpg'
    cv2.imwrite(filename, result)
    plt.show()
    return result, kps