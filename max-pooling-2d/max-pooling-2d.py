import numpy as np
def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    # Write code here
    x = np.array(X)
    input_height, input_width = x.shape
    
    output_height = input_height // pool_size
    output_width = input_width // pool_size

    pooled_image = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            res = np.max(x[i*pool_size : (i+1)*pool_size, j*pool_size : (j+1)*pool_size])
            pooled_image[i][j] = res

    return pooled_image.tolist()