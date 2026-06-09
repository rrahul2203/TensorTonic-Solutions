import numpy as np
def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    kernel = np.array(kernel)
    image = np.array(image)
    padded_image = np.pad(image, (padding, padding), mode='constant', constant_values = 0)
    height, width = padded_image.shape
    f_height, f_width = kernel.shape
    output_height = (height - f_height) // stride + 1
    output_width = (width - f_width) //stride + 1

    output_image = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            region = padded_image[i*stride: i*stride + f_height, j*stride: j*stride + f_width]
            output_image[i][j] = np.sum(region * kernel)

    return output_image.tolist()