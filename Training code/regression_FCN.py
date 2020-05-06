import preprocessing
import simulation
import R2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import optimizers, losses, metrics

# Parameters setting
num_of_cells = 2
num_of_CUEs = 2
num_of_D2Ds = 2
batch_size = 64
epochs = 10

# Get the image data format which Keras follows
image_data_format = preprocessing.GetImageDataFormat()

# Get the input data and target data
input_data = preprocessing.GetInputData(num_of_cells, num_of_CUEs, num_of_D2Ds, (2000, 8000, 10000), image_data_format)
target_data = preprocessing.GetTargetData(num_of_cells, num_of_CUEs, num_of_D2Ds,(2000, 8000, 10000))

# Reshape the input data
reshaped_input_data = preprocessing.ReshapeInputData1D(input_data)

# Split the datadset into the training set and testing set
(x_train, y_train), (x_test, y_test) = preprocessing.SplitDataset(reshaped_input_data, target_data, proportion = 0.8, shuffle = True)

# Get the input shape of input data and the output shape of target data 
input_shape = preprocessing.GetInputShape(reshaped_input_data)
target_shape = preprocessing.GetTargetShape(target_data)                                                                  

# Build the model
model = Sequential()

model.add(Dense(units = 512, input_shape = input_shape,
                activation = 'relu'))

model.add(Dense(units = 512,
                activation = 'relu'))

model.add(Dense(units = 512,
                activation = 'relu'))

model.add(Dense(units = 512,
                activation = 'relu'))

model.add(Dense(units = 512,
                activation = 'relu'))

model.add(Dense(units = 512,
                activation = 'relu'))

model.add(Dense(units = target_shape,
                activation = 'linear'))

# Summary
model.summary()

# Configures the model for training
adam = optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)                 
model.compile(loss = losses.mean_squared_error, optimizer = adam, metrics = [R2.R2_score])

# Train the model fro a fixed number of epochs (iterations on dataset)
history = model.fit(x_train, y_train, batch_size, epochs, verbose = 1, validation_data = (x_test, y_test))

# Simulation
channel_gain_matrix = simulation.GetChannelGainMatrix(x_test, num_of_cells, num_of_CUEs, num_of_D2Ds)
QoS_of_CUE = simulation.GetQoSofCUE(channel_gain_matrix, num_of_cells, num_of_CUEs)

opt_CUE_power, opt_D2D_power = simulation.GetPowerAllocation(y_test, num_of_cells, num_of_CUEs, num_of_D2Ds)
opt_CUE_rate, opt_D2D_rate = simulation.GetDataRate(channel_gain_matrix, opt_CUE_power, opt_D2D_power)

opt_system_sum_rate, opt_CUE_sum_rate, opt_D2D_sum_rate = simulation.GetSumRate(opt_CUE_rate, opt_D2D_rate)
opt_system_power_consumption, opt_CUE_power_consumption, opt_D2D_power_consumption = simulation.GetPowerConsumption(opt_CUE_power, opt_D2D_power)
opt_system_EE, opt_CUE_EE, opt_D2D_EE = simulation.GetEnergyEfficiency(opt_system_sum_rate, opt_CUE_sum_rate, opt_D2D_sum_rate, opt_system_power_consumption, opt_CUE_power_consumption, opt_D2D_power_consumption)
opt_system_UIR, opt_CUE_UIR, opt_D2D_UIR = simulation.GetUIR(opt_CUE_rate, opt_D2D_rate, opt_CUE_power, opt_D2D_power, QoS_of_CUE)
opt_system_RIR, opt_CUE_RIR, opt_D2D_RIR = simulation.GetRIR(opt_CUE_rate, opt_D2D_rate, opt_CUE_power, opt_D2D_power, QoS_of_CUE)

opt_avg_system_sum_rate, opt_avg_CUE_sum_rate, opt_avg_D2D_sum_rate = simulation.GetAvgSumRate(opt_system_sum_rate, opt_CUE_sum_rate, opt_D2D_sum_rate)
print(f"\nOptimal average system sum rate: {opt_avg_system_sum_rate}")
print(f"Optimal average CUE sum rate: {opt_avg_CUE_sum_rate}")
print(f"Optimal average D2D sum rate: {opt_avg_D2D_sum_rate}")

opt_avg_system_power_consumption, opt_avg_CUE_power_consumption, opt_avg_D2D_power_consumption = simulation.GetAvgPowerConsumption(opt_system_power_consumption, opt_CUE_power_consumption, opt_D2D_power_consumption)
print(f"Optimal average system power consumption: {opt_avg_system_power_consumption}")
print(f"Optimal average CUE power consumption: {opt_avg_CUE_power_consumption}")
print(f"Optimal average D2D power consumption: {opt_avg_D2D_power_consumption}")

opt_avg_system_EE, opt_avg_CUE_EE, opt_avg_D2D_EE = simulation.GetAvgEnergyEfficiency(opt_system_EE, opt_CUE_EE, opt_D2D_EE)
print(f"Optimal average system energy efficiency: {opt_avg_system_EE}")
print(f"Optimal average CUE energy efficiency: {opt_avg_CUE_EE}")
print(f"Optimal average D2D energy efficiency: {opt_avg_D2D_EE}")

opt_avg_system_UIR, opt_avg_CUE_UIR, opt_avg_D2D_UIR = simulation.GetAvgRIR(opt_system_UIR, opt_CUE_UIR, opt_D2D_UIR)
print(f"Optimal average system infeasible rate (per user): {opt_avg_system_UIR}")
print(f"Optimal average CUE infeasible rate (per user): {opt_avg_CUE_UIR}")
print(f"Optimal average D2D infeasible rate (per user): {opt_avg_D2D_UIR}")

opt_avg_system_RIR, opt_avg_CUE_RIR, opt_avg_D2D_RIR = simulation.GetAvgUIR(opt_system_RIR, opt_CUE_RIR, opt_D2D_RIR)
print(f"Optimal average system infeasible rate (per realization): {opt_avg_system_RIR}")
print(f"Optimal average CUE infeasible rate (per realization): {opt_avg_CUE_RIR}")
print(f"Optimal average D2D infeasible rate (per userealization): {opt_avg_D2D_RIR}")

pred_y_test = model.predict(x_test)

pred_CUE_power, pred_D2D_power = simulation.GetPowerAllocation(pred_y_test, num_of_cells, num_of_CUEs, num_of_D2Ds)
pred_CUE_rate, pred_D2D_rate = simulation.GetDataRate(channel_gain_matrix, pred_CUE_power, pred_D2D_power)

pred_system_sum_rate, pred_CUE_sum_rate, pred_D2D_sum_rate = simulation.GetSumRate(pred_CUE_rate, pred_D2D_rate)
pred_system_power_consumption, pred_CUE_power_consumption, pred_D2D_power_consumption = simulation.GetPowerConsumption(pred_CUE_power, pred_D2D_power)
pred_system_EE, pred_CUE_EE, pred_D2D_EE = simulation.GetEnergyEfficiency(pred_system_sum_rate, pred_CUE_sum_rate, pred_D2D_sum_rate, pred_system_power_consumption, pred_CUE_power_consumption, pred_D2D_power_consumption)
pred_system_UIR, pred_CUE_UIR, pred_D2D_UIR = simulation.GetUIR(pred_CUE_rate, pred_D2D_rate, pred_CUE_power, pred_D2D_power, QoS_of_CUE)
pred_system_RIR, pred_CUE_RIR, pred_D2D_RIR = simulation.GetRIR(pred_CUE_rate, pred_D2D_rate, pred_CUE_power, pred_D2D_power, QoS_of_CUE)

pred_avg_system_sum_rate, pred_avg_CUE_sum_rate, pred_avg_D2D_sum_rate = simulation.GetAvgSumRate(pred_system_sum_rate, pred_CUE_sum_rate, pred_D2D_sum_rate)
print(f"\nPredicted average system sum rate: {pred_avg_system_sum_rate}")
print(f"Predicted average CUE sum rate: {pred_avg_CUE_sum_rate}")
print(f"Predicted average D2D sum rate: {pred_avg_D2D_sum_rate}")

pred_avg_system_power_consumption, pred_avg_CUE_power_consumption, pred_avg_D2D_power_consumption = simulation.GetAvgPowerConsumption(pred_system_power_consumption, pred_CUE_power_consumption, pred_D2D_power_consumption)
print(f"Predicted average system power consumption: {pred_avg_system_power_consumption}")
print(f"Predicted average CUE power consumption: {pred_avg_CUE_power_consumption}")
print(f"Predicted average D2D power consumption: {pred_avg_D2D_power_consumption}")

pred_avg_system_EE, pred_avg_CUE_EE, pred_avg_D2D_EE = simulation.GetAvgEnergyEfficiency(pred_system_EE, pred_CUE_EE, pred_D2D_EE)
print(f"Predicted average system energy efficiency: {pred_avg_system_EE}")
print(f"Predicted average CUE energy efficiency: {pred_avg_CUE_EE}")
print(f"Predicted average D2D energy efficiency: {pred_avg_D2D_EE}")

pred_avg_system_UIR, pred_avg_CUE_UIR, pred_avg_D2D_UIR = simulation.GetAvgRIR(pred_system_UIR, pred_CUE_UIR, pred_D2D_UIR)
print(f"Predicted average system infeasible rate (per user): {pred_avg_system_UIR}")
print(f"Predicted average CUE infeasible rate (per user): {pred_avg_CUE_UIR}")
print(f"Predicted average D2D infeasible rate (per user): {pred_avg_D2D_UIR}")

pred_avg_system_RIR, pred_avg_CUE_RIR, pred_avg_D2D_RIR = simulation.GetAvgUIR(pred_system_RIR, pred_CUE_RIR, pred_D2D_RIR)
print(f"Predicted average system infeasible rate (per realization): {pred_avg_system_RIR}")
print(f"Predicted average CUE infeasible rate (per realization): {pred_avg_CUE_RIR}")
print(f"Predicted average D2D infeasible rate (per userealization): {pred_avg_D2D_RIR}")

# Plot training & validation accuracy 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss: MSE')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'upper right')
plt.show()

plt.plot(history.history['R2_score'])
plt.plot(history.history['val_R2_score'])
plt.title('Metric: R2 score')
plt.ylabel('Metric')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'lower right')
plt.show()