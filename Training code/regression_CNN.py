'''
# Reshape the input data
input_data = preprocessing.ReshapeInputData(input_data, image_data_format,
                                            height = num_of_cells * (num_of_CUEs + num_of_D2Ds),
                                            width = num_of_cells * (1 + num_of_D2Ds),
                                            depth = 1)
'''

"""
# Convolutional 1st layer
model.add(Conv2D(filters = 64, kernel_size = (2, 2),
                 padding = 'same', data_format = image_data_format,
                 activation = 'relu', input_shape = input_shape))

# Max pooling 1st layer
model.add(MaxPooling2D(pool_size = (2, 2), padding = 'valid',
                       data_format = image_data_format))

# Fully connected 1st layer
model.add(Flatten(data_format = image_data_format))
"""