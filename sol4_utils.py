from imageio import imread
from skimage.color import rgb2grey
import numpy as np
from scipy.ndimage.filters import convolve1d
from scipy.signal import convolve2d

GRAYSCALE = 1
RGB = 2
BASE_FILTER = np.array([1, 1])


def read_image(filename, representation):
    """
    this function read an image.
    :param filename: image name or path to read.
    :param representation: 1 for RGB, 2 for GRAYSCALE
    :return: an image in float64 format.
    """
    if representation == GRAYSCALE:
        image = imread(filename)
        if image.ndim == 3:
            return rgb2grey(image)
        return image / 255
    elif representation == RGB:
        image = imread(filename)
        return image / 255
    else:
        exit()


def build_filter(size):
    """
    this function buils the gaussian filter vector.
    :param size: the size of the gaussian filter
    :return: the normalized gaussian vector of size size
    """
    gaus = np.array(BASE_FILTER).astype(np.uint64)
    for i in range(size - 2):
        gaus = np.convolve(gaus, BASE_FILTER)
    print(gaus.shape)
    return gaus * (2 ** -(size - 1))


def reduce(im, filter):
    """
    reduce the image size by 2.
    :param im: image to reduce
    :param filter: size of blur filter to use
    :return: reduced image
    """
    im = convolve1d(im, filter, mode='constant')
    im = convolve1d(im.T, filter, mode='constant')
    return im.T[::2, ::2]


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    this method builds a gaussian pyramid of an input grayscale image.
    :param im: input grayscale image.
    :param max_levels: maximum number of levels in the pyramid.
    :param filter_size: size of gaussian blur to use.
    :return: (the gaussian pyramid as a list, filter vector (1,filter_size))
    """
    gaussian_pyramid = list()
    gaussian_pyramid.append(im)
    filter = build_filter(filter_size)
    for i in range(max_levels - 1):
        x, y = im.shape
        if x < 16 or y < 16:
            break
        im = reduce(im, filter)
        gaussian_pyramid.append(im)
    return gaussian_pyramid, filter.reshape((1, filter_size))


def create_kernel(size):
    """
    creates a gaussian kernel
    :param size: the size of the kernel N*N
    :return: the gaussian kernel
    """
    res = np.array([1, 1]).astype(np.double)
    for i in range(size - 2):
        res = np.convolve(res, np.array([1, 1]))
    res = res * np.transpose(res.reshape((1, res.shape[0])))

    return res / np.sum(res)


def blur_spatial(im, kernel_size):
    """
    performs the blur in the spatial domain using convolution
    :param im: image to blur
    :param kernel_size:
    :return:
    """
    if kernel_size <= 1:
        return im
    return convolve2d(im, create_kernel(kernel_size), 'same')


def expand(im, filter):
    """
    expand the image size by 2.
    :param im: image to expand
    :param filter: size of blur filter to use
    :return: expanded image
    """
    x, y = im.shape
    im = np.insert(im, np.arange(1, y + 1, 1), 0, axis=1)
    im = np.insert(im, np.arange(1, x + 1, 1), 0, axis=0)
    im = convolve1d(im, 2 * filter, mode='constant')
    im = convolve1d(im.T, 2 * filter, mode='constant')
    return im.T


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    this method builds a laplacia pyramid of an input grayscale image.
    :param im: input grayscale image.
    :param max_levels: maximum number of levels in the pyramid.
    :param filter_size: size of gaussian blur to use.
    :return: (the laplacian pyramid as a list, filter vector (1,filter_size))
    """
    gaussian_pyramid, filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_pyramid = list()
    for i in range(len(gaussian_pyramid) - 1):
        laplacian = gaussian_pyramid[i] - expand(gaussian_pyramid[i + 1], filter.reshape((filter_size,)))
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid, filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    this function constructs laplacian pyramid back to an image.
    :param lpyr: the laplacian pyramid
    :param filter_vec: the vector returned by the function to create the pyramid.
    :param coeff: a list of coefficients specifying weight of each level of the pyramid
    :return: reconstructed image.
    """
    x, y = filter_vec.shape
    while len(lpyr) > 1:
        lpyr[-2] = (lpyr[-2] + coeff[-1] * expand(lpyr[-1], filter_vec.reshape((y,))))
        lpyr = lpyr[:-1]
        coeff = coeff[:-1]
    return lpyr[0]


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    this method blends 2 different GRAYSACLE images using a given mask image.
    :param im1: first image to blend
    :param im2: second image to blend
    :param mask: mask to use
    :param max_levels: maximum number of levels when building the pyramid
    :param filter_size_im: filter size to use on images when building the pyrmaid
    :param filter_size_mask: filter size to use on mask when building the pyrmaid
    :return: the blended rendered image
    """
    lap1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gaus, gaus_filter = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    lap_out = []
    for i in range(len(lap1)):
        lap_out.append(gaus[i] * lap1[i] + (1 - gaus[i]) * lap2[i])
    coef = [1 for i in range(len(lap_out))]
    return laplacian_to_image(lap_out, filter1, coef)
