from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras import optimizers, losses
from prep import preprocessing, loader, saver  
from sim import simulation, metrics
from spp.layers import SpatialPyramidPooling

# Parameters setting
num_of_cells = 2
num_of_CUEs = {2, 3, 4}
num_of_D2Ds = {2, 3, 4}
batch_size = 64
epochs = 10

# Get the image data format which Keras follows
image_data_format = loader.get_image_data_format()

# Get the input data and target data
input_data_list = [loader.load_input_data(num_of_cells, i, j, {2000, 8000, 10000}, image_data_format) for i in num_of_CUEs for j in num_of_D2Ds]
target_data_list = [loader.load_target_data(num_of_cells, i, j, {2000, 8000, 10000}) for i in num_of_CUEs for j in num_of_D2Ds]

# Reshape the input data
for index, input_data in enumerate(input_data_list):
    rows, cols, channels = preprocessing.get_input_shape(input_data)
    input_data_list[index] = preprocessing.reshape_input_data_3D(input_data, image_data_format, rows, cols * channels, 1)

# Get the maximum length of the target data in the target data list
max_length = preprocessing.get_max_length(target_data_list)

# Zero padding
for index, target_data in enumerate(target_data_list):
    target_data_list[index] = preprocessing.zero_padding(target_data, max_length)

# Split the datadset into the training set and testing set
x_train_list, y_train_list, x_test_list, y_test_list = [[None] * len(input_data_list) for _ in range(4)]

for index, (input_data, target_data) in enumerate(zip(input_data_list, target_data_list)):
    (x_train_list[index], y_train_list[index]), (x_test_list[index], y_test_list[index]) = preprocessing.split_dataset(input_data, target_data, proportion = 0.8, shuffle = False)

# Build the model
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (2, 2), data_format = image_data_format, activation = 'relu', input_shape = (None, None, 1)))

model.add(Conv2D(filters = 16, kernel_size = (2, 2), data_format = image_data_format, activation = 'relu'))

model.add(SpatialPyramidPooling([1, 2, 4]))

model.add(Dense(units = 512, activation = 'relu'))

model.add(Dense(units = 512, activation = 'relu'))

model.add(Dense(units = max_length, activation = 'linear'))

# Summary
model.summary()

# Configures the model for training
adam = optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)                 
model.compile(loss = losses.mean_squared_error, optimizer = adam)

# Train the model for a fixed number of epochs (iterations on dataset)
for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
    model.fit(x_train, y_train, batch_size, epochs, verbose = 1, validation_data = (x_test, y_test))

# Save the model in HDF5 format
saver.save_model(model, "CNN-SPP", num_of_cells)