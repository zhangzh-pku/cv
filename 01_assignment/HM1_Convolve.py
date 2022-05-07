from turtle import shape
import numpy as np
from utils import read_img, write_img
from numpy.lib.stride_tricks import as_strided
from scipy.linalg import toeplitz


def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """
    p = padding_size
    img_h, img_l = img.shape
    h = img.shape[0] + 2 * p
    l = img.shape[1] + 2 * p
    padding_img = np.zeros((h, l))
    # main info
    #print(img.shape)
    padding_img[p:p + img_h, p:p + img_l] = img
    if type == "zeroPadding":
        return padding_img
    elif type == "replicatePadding":
        # corner
        img_corner = np.ones((padding_size, padding_size))
        padding_img[0:p, 0:p] = img[0, 0] * img_corner
        padding_img[0:p, p + img_l:l] = img[0, img_l - 1] * img_corner
        padding_img[p + img_h:h, 0:p] = img[img_h - 1, 0] * img_corner
        padding_img[p + img_h:h,
                    img_l:l] = img[img_h - 1, img_l - 1] * img_corner
        # edge
        padding_img[0:p, p:p + img_l] = np.ones(
            (p, 1)) @ (img[0].reshape(1, img_l))
        padding_img[p + img_h:h, p:p + img_l] = np.ones(
            (p, 1)) @ (img[img_h - 1].reshape((1, img_l)))
        padding_img[p:p + img_h,
                    0:p] = (img[..., 0].reshape(img_h, 1)) @ np.ones((1, p))
        padding_img[p:p + img_h, p + img_l:l] = (
            img[..., img_l - 1].reshape(img_h, 1) @ np.ones((1, p)))
        return padding_img
    elif type == "reflectionpadding":
        # todo

        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    h, l = img.shape
    kernel_size = kernel.shape[0]

    #zero padding
    padding_img = padding(img, 1, "zeroPadding")

    #build the Toeplitz matrix and compute convolution
    _mat = np.zeros((kernel_size, l + 2), dtype=kernel.dtype)
    _mat[:, 0:kernel_size] = kernel
    kernel_flatten = _mat.reshape(-1)[0:((kernel_size - 1) * (l + 2) +
                                         kernel_size)]

    _idx_y = np.arange(h * l)
    _idx_x = _idx_y // (l + 2 - kernel_size +
                        1) * (l + 2) + _idx_y % (l + 2 - kernel_size + 1)
    idxs_x = np.tile(np.arange(kernel_flatten.shape[0]),
                     (h * l, 1)) + _idx_x.reshape(-1, 1)

    Toeplitz_matrix = np.zeros((h * l, (h + 2) * (l + 2)), dtype=kernel.dtype)
    Toeplitz_matrix[_idx_y.reshape(-1, 1), idxs_x] = kernel_flatten

    output = Toeplitz_matrix @ padding_img.reshape(-1)
    output = output.reshape(img.shape)
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """

    # build the sliding-window convolution here
    # the img is padded
    kernel_d1 = kernel.ravel()
    h = img.shape[0] - kernel.shape[0] + 1
    l = img.shape[1] - kernel.shape[1] + 1

    #4 dim is easy to understand
    shape = (h, l, kernel.shape[0], kernel.shape[1])
    stride = (img.strides[0], img.strides[1], img.strides[0], img.strides[1])
    is_strided = np.lib.stride_tricks.as_strided(img, shape, stride)
    conv = is_strided.reshape(h, l, 1, -1)
    output = (conv @ kernel_d1).reshape(h, l)
    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8],
                                [1 / 16, 1 / 8, 1 / 16]])
    output = convolve(padding_img, gaussian_kernel)
    return output


def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output


def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output


if __name__ == "__main__":

    np.random.seed(111)
    input_array = np.random.rand(6, 6)
    input_kernel = np.random.rand(3, 3)

    # task1: padding
    zero_pad = padding(input_array, 1, "zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt", zero_pad)

    replicate_pad = padding(input_array, 1, "replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt", replicate_pad)

    # task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    # task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    # task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png") / 255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x * 255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y * 255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur * 255)
