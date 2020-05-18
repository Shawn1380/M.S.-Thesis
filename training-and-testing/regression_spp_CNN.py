from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import optimizers, losses, metrics
from preprocessing import Preprocessing
from simulation import Simulation
from spp import SpatialPyramidPooling, R2
import numpy as np
import matplotlib.pyplot as plt

# Parameters setting
num_of_cells = 2
num_of_CUEs = 2
num_of_D2Ds = (2, 3)
batch_size = 64
epochs = 10

# Get the image data format which Keras follows
image_data_format = Preprocessing.GetImageDataFormat()

# Get the input data and target data
input_data_list = [Preprocessing.GetInputData(num_of_cells, num_of_CUEs, i, (2000, 8000, 10000), image_data_format) for i in num_of_D2Ds]
target_data_list = [Preprocessing.GetTargetData(num_of_cells, num_of_CUEs, i, (2000, 8000, 10000)) for i in num_of_D2Ds]

# Reshape the input data
for index, input_data in enumerate(input_data_list):
    rows, cols, channels = Preprocessing.GetInputShape(input_data)
    input_data_list[index] = Preprocessing.ReshapeInputData3D(input_data, image_data_format, rows, cols * channels, 1)

# Get the maximum length of the target data in the target data list
max_length = Preprocessing.GetMaxLength(target_data_list)

# Zero padding
for index, target_data in enumerate(target_data_list):
    target_data_list[index] = Preprocessing.ZeroPadding(target_data, max_length)

# Split the datadset into the training set and testing set
x_train_list, y_train_list, x_test_list, y_test_list = [[None] * len(input_data_list) for _ in range(4)]

for index, (input_data, target_data) in enumerate(zip(input_data_list, target_data_list)):
    (x_train_list[index], y_train_list[index]), (x_test_list[index], y_test_list[index]) = Preprocessing.SplitDataset(input_data, target_data, proportion = 0.8, shuffle = False)

# Build the model
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (2, 2), data_format = image_data_format, activation = 'relu', input_shape = (None, None, 1)))

model.add(Conv2D(filters = 16, kernel_size = (2, 2), data_format = image_data_format, activation = 'relu'))

model.add(SpatialPyramidPooling.SpatialPyramidPooling([1, 2, 4]))

model.add(Dense(units = 512, activation = 'relu'))

model.add(Dense(units = 512, activation = 'relu'))

model.add(Dense(units = max_length, activation = 'linear'))

# Summary
model.summary()

# Configures the model for training
adam = optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)                 
model.compile(loss = losses.mean_squared_error, optimizer = adam, metrics = [R2.R2_score])

# Train the model fro a fixed number of epochs (iterations on dataset)
for x_train, y_train, x_test, y_test in zip(x_train_list, y_train_list, x_test_list, y_test_list):
    model.fit(x_train, y_train, batch_size, epochs, verbose = 1, validation_data = (x_test, y_test))

# Simulation
for x_test, y_test, num_of_D2Ds in zip(x_test_list, y_test_list, num_of_D2Ds):

    channel_gain_matrix = Simulation.GetChannelGainMatrix(x_test, num_of_cells, num_of_CUEs, num_of_D2Ds)
    QoS_of_CUE = Simulation.GetQoSofCUE(channel_gain_matrix, num_of_cells, num_of_CUEs, rate_proportion = 0.05)

    y_test = Preprocessing.RemoveRedundantZero(y_test, num_of_cells, num_of_CUEs, num_of_D2Ds) 
    opt_CUE_power, opt_D2D_power = Simulation.GetPowerAllocation(y_test, num_of_cells, num_of_CUEs, num_of_D2Ds)
    opt_CUE_rate, opt_D2D_rate = Simulation.GetDataRate(channel_gain_matrix, opt_CUE_power, opt_D2D_power)

    opt_system_sum_rate, opt_CUE_sum_rate, opt_D2D_sum_rate = Simulation.GetSumRate(opt_CUE_rate, opt_D2D_rate)
    opt_system_power_consumption, opt_CUE_power_consumption, opt_D2D_power_consumption = Simulation.GetPowerConsumption(opt_CUE_power, opt_D2D_power)
    opt_system_EE, opt_CUE_EE, opt_D2D_EE = Simulation.GetEnergyEfficiency(opt_system_sum_rate, opt_CUE_sum_rate, opt_D2D_sum_rate, opt_system_power_consumption, opt_CUE_power_consumption, opt_D2D_power_consumption)
    opt_system_UIR, opt_CUE_UIR, opt_D2D_UIR = Simulation.GetUIR(opt_CUE_rate, opt_D2D_rate, opt_CUE_power, opt_D2D_power, QoS_of_CUE)
    opt_system_RIR, opt_CUE_RIR, opt_D2D_RIR = Simulation.GetRIR(opt_CUE_rate, opt_D2D_rate, opt_CUE_power, opt_D2D_power, QoS_of_CUE)

    opt_avg_system_sum_rate, opt_avg_CUE_sum_rate, opt_avg_D2D_sum_rate = Simulation.GetAvgSumRate(opt_system_sum_rate, opt_CUE_sum_rate, opt_D2D_sum_rate)
    print(f"\nOptimal average system sum rate: {opt_avg_system_sum_rate}")
    print(f"Optimal average CUE sum rate: {opt_avg_CUE_sum_rate}")
    print(f"Optimal average D2D sum rate: {opt_avg_D2D_sum_rate}")

    opt_avg_system_power_consumption, opt_avg_CUE_power_consumption, opt_avg_D2D_power_consumption = Simulation.GetAvgPowerConsumption(opt_system_power_consumption, opt_CUE_power_consumption, opt_D2D_power_consumption)
    print(f"Optimal average system power consumption: {opt_avg_system_power_consumption}")
    print(f"Optimal average CUE power consumption: {opt_avg_CUE_power_consumption}")
    print(f"Optimal average D2D power consumption: {opt_avg_D2D_power_consumption}")

    opt_avg_system_EE, opt_avg_CUE_EE, opt_avg_D2D_EE = Simulation.GetAvgEnergyEfficiency(opt_system_EE, opt_CUE_EE, opt_D2D_EE)
    print(f"Optimal average system energy efficiency: {opt_avg_system_EE}")
    print(f"Optimal average CUE energy efficiency: {opt_avg_CUE_EE}")
    print(f"Optimal average D2D energy efficiency: {opt_avg_D2D_EE}")

    opt_avg_system_UIR, opt_avg_CUE_UIR, opt_avg_D2D_UIR = Simulation.GetAvgRIR(opt_system_UIR, opt_CUE_UIR, opt_D2D_UIR)
    print(f"Optimal average system infeasible rate (per user): {opt_avg_system_UIR}")
    print(f"Optimal average CUE infeasible rate (per user): {opt_avg_CUE_UIR}")
    print(f"Optimal average D2D infeasible rate (per user): {opt_avg_D2D_UIR}")

    opt_avg_system_RIR, opt_avg_CUE_RIR, opt_avg_D2D_RIR = Simulation.GetAvgUIR(opt_system_RIR, opt_CUE_RIR, opt_D2D_RIR)
    print(f"Optimal average system infeasible rate (per realization): {opt_avg_system_RIR}")
    print(f"Optimal average CUE infeasible rate (per realization): {opt_avg_CUE_RIR}")
    print(f"Optimal average D2D infeasible rate (per userealization): {opt_avg_D2D_RIR}")

    pred_y_test = model.predict(x_test)
    pred_y_test = Preprocessing.RemoveRedundantZero(pred_y_test, num_of_cells, num_of_CUEs, num_of_D2Ds)

    pred_CUE_power, pred_D2D_power = Simulation.GetPowerAllocation(pred_y_test, num_of_cells, num_of_CUEs, num_of_D2Ds)
    pred_CUE_rate, pred_D2D_rate = Simulation.GetDataRate(channel_gain_matrix, pred_CUE_power, pred_D2D_power)

    pred_system_sum_rate, pred_CUE_sum_rate, pred_D2D_sum_rate = Simulation.GetSumRate(pred_CUE_rate, pred_D2D_rate)
    pred_system_power_consumption, pred_CUE_power_consumption, pred_D2D_power_consumption = Simulation.GetPowerConsumption(pred_CUE_power, pred_D2D_power)
    pred_system_EE, pred_CUE_EE, pred_D2D_EE = Simulation.GetEnergyEfficiency(pred_system_sum_rate, pred_CUE_sum_rate, pred_D2D_sum_rate, pred_system_power_consumption, pred_CUE_power_consumption, pred_D2D_power_consumption)
    pred_system_UIR, pred_CUE_UIR, pred_D2D_UIR = Simulation.GetUIR(pred_CUE_rate, pred_D2D_rate, pred_CUE_power, pred_D2D_power, QoS_of_CUE)
    pred_system_RIR, pred_CUE_RIR, pred_D2D_RIR = Simulation.GetRIR(pred_CUE_rate, pred_D2D_rate, pred_CUE_power, pred_D2D_power, QoS_of_CUE)

    pred_avg_system_sum_rate, pred_avg_CUE_sum_rate, pred_avg_D2D_sum_rate = Simulation.GetAvgSumRate(pred_system_sum_rate, pred_CUE_sum_rate, pred_D2D_sum_rate)
    print(f"\nPredicted average system sum rate: {pred_avg_system_sum_rate}")
    print(f"Predicted average CUE sum rate: {pred_avg_CUE_sum_rate}")
    print(f"Predicted average D2D sum rate: {pred_avg_D2D_sum_rate}")

    pred_avg_system_power_consumption, pred_avg_CUE_power_consumption, pred_avg_D2D_power_consumption = Simulation.GetAvgPowerConsumption(pred_system_power_consumption, pred_CUE_power_consumption, pred_D2D_power_consumption)
    print(f"Predicted average system power consumption: {pred_avg_system_power_consumption}")
    print(f"Predicted average CUE power consumption: {pred_avg_CUE_power_consumption}")
    print(f"Predicted average D2D power consumption: {pred_avg_D2D_power_consumption}")

    pred_avg_system_EE, pred_avg_CUE_EE, pred_avg_D2D_EE = Simulation.GetAvgEnergyEfficiency(pred_system_EE, pred_CUE_EE, pred_D2D_EE)
    print(f"Predicted average system energy efficiency: {pred_avg_system_EE}")
    print(f"Predicted average CUE energy efficiency: {pred_avg_CUE_EE}")
    print(f"Predicted average D2D energy efficiency: {pred_avg_D2D_EE}")

    pred_avg_system_UIR, pred_avg_CUE_UIR, pred_avg_D2D_UIR = Simulation.GetAvgRIR(pred_system_UIR, pred_CUE_UIR, pred_D2D_UIR)
    print(f"Predicted average system infeasible rate (per user): {pred_avg_system_UIR}")
    print(f"Predicted average CUE infeasible rate (per user): {pred_avg_CUE_UIR}")
    print(f"Predicted average D2D infeasible rate (per user): {pred_avg_D2D_UIR}")

    pred_avg_system_RIR, pred_avg_CUE_RIR, pred_avg_D2D_RIR = Simulation.GetAvgUIR(pred_system_RIR, pred_CUE_RIR, pred_D2D_RIR)
    print(f"Predicted average system infeasible rate (per realization): {pred_avg_system_RIR}")
    print(f"Predicted average CUE infeasible rate (per realization): {pred_avg_CUE_RIR}")
    print(f"Predicted average D2D infeasible rate (per userealization): {pred_avg_D2D_RIR}")