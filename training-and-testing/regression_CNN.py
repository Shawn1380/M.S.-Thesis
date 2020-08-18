from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras import optimizers, losses
from prep import preprocessing, loader, saver  
from sim import simulation, metrics

# Parameters setting
num_of_cells = 2
num_of_CUEs = 2
num_of_D2Ds = 2
batch_size = 64
epochs = 10

# Get the image data format which Keras follows
image_data_format = loader.get_image_data_format()

# Get the input data and target data
input_data = loader.load_input_data(num_of_cells, num_of_CUEs, num_of_D2Ds, {2000, 8000, 10000}, image_data_format)
target_data = loader.load_target_data(num_of_cells, num_of_CUEs, num_of_D2Ds, {2000, 8000, 10000})

# Reshape the input data
rows, cols, channels = preprocessing.get_input_shape(input_data)
reshaped_input_data = preprocessing.reshape_input_data_3D(input_data, image_data_format, rows, cols * channels, 1)

# Split the datadset into the training set and testing set
(x_train, y_train), (x_test, y_test) = preprocessing.split_dataset(reshaped_input_data, target_data, proportion = 0.8, shuffle = False)

# Get the input shape of input data and the output shape of target data 
input_shape = preprocessing.get_input_shape(x_train)
target_shape = preprocessing.get_target_shape(y_train)

# Build the model
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (2, 2), data_format = image_data_format, activation = 'relu', input_shape = input_shape))

model.add(Conv2D(filters = 16, kernel_size = (2, 2), data_format = image_data_format, activation = 'relu'))

model.add(Flatten())

model.add(Dense(units = 512, activation = 'relu'))

model.add(Dense(units = 512, activation = 'relu'))

model.add(Dense(units = target_shape, activation = 'linear'))

# Summary
model.summary()

# Configures the model for training
adam = optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)                 
model.compile(loss = losses.mean_squared_error, optimizer = adam)

# Train the model for a fixed number of epochs (iterations on dataset)
history = model.fit(x_train, y_train, batch_size, epochs, verbose = 1, validation_data = (x_test, y_test))

# Save the model in HDF5 format
saver.save_model(model, "CNN", num_of_cells, num_of_CUEs, num_of_D2Ds)