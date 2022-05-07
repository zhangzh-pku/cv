from operator import index
from cv2 import grabCut
import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img
from math import pi
from queue import Queue


def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = ((x_grad / 2)**2 + (y_grad / 2)**2)**0.5
    direction_grad = np.arctan2(y_grad, x_grad)
    return magnitude_grad, direction_grad


def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """
    h, l = grad_mag.shape
    grad_x0y0 = np.copy(grad_mag)

    grad_x1y0 = np.zeros((h, l))
    grad_x1y0[0:h, 1:l] = grad_x0y0[0:h, 0:l - 1]

    grad_x0y1 = np.zeros((h, l))
    grad_x0y1[1:h, 0:l] = grad_x0y0[0:h - 1, 0:l]

    grad_x1y1 = np.zeros((h, l))
    grad_x1y1[1:h, 1:l] = grad_x0y0[0:h - 1, 0:l - 1]

    x = np.cos(grad_dir)
    y = np.sin(grad_dir)

    posi = grad_x0y0 * (1 - x) * (1 - y) + grad_x1y0 * (
        1 - y) * x + grad_x1y1 * x * y + grad_x0y1 * (1 - x) * y

    dir = grad_dir + pi
    x = np.cos(grad_dir)
    y = np.sin(grad_dir)

    nega = grad_x0y0 * (1 - x) * (1 - y) + grad_x1y0 * (
        1 - y) * x + grad_x1y1 * x * y + grad_x0y1 * (1 - x) * y

    NMS_output = grad_mag.copy()
    NMS_output[(grad_mag < posi) | (grad_mag < nega)] = 0

    return NMS_output


def hysteresis_thresholding(img):
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """

    #you can adjust the parameters to fit your own implementation
    low_ratio = 0.10
    high_ratio = 0.30
    strong = 1.0
    week = 0.8
    output = np.zeros(img.shape)
    vis = np.zeros(img.shape, dtype='int32')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            q = Queue(maxsize=0)
            if (img[i, j] > high_ratio):
                output[i, j] = strong
                q.put((i, j))
                while q.empty() == False:
                    tem = q.get()
                    if output[tem[0], tem[1]] != strong:
                        output[tem[0], tem[1]] = week
                    for _i in range(-1, 2):
                        for _j in range(-1, 2):
                            if _i == 0 and _j == 0: continue
                            x = tem[0] + _i
                            y = tem[1] + _j
                            if x >= img.shape[0] or y >= img.shape[1]: continue
                            if vis[x, y] == 1: continue
                            else: vis[x, y] = 1
                            if img[x, y] > low_ratio and img[x,
                                                             y] < high_ratio:
                                q.put((x, y))

    return output


if __name__ == "__main__":

    #Load the input images
    input_img = read_img("lenna.png") / 255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(
        x_grad, y_grad)
    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)

    write_img("result/HM1_Canny_result.png", output_img * 255)
