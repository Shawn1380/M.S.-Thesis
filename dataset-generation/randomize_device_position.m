function [devices_position] = randomize_device_position(num_of_cells, radius, cenX, cenY, num_of_devices)
% Input:
% num_of_cells: Number of the cells in the system
% radius: The radius of the cell (meter)
% cenX: The X-coordinate of the center of the cells
% cenY: The Y-coordinate of the center of the cells
% num_of_devices: Number of the devices in each cell
%
% Output:
% devices_position: The position (X and Y coordinate) of all devices

devices_position = zeros(num_of_devices, num_of_cells * 2);

for i = 1 : num_of_cells
    theta = rand(num_of_devices, 1) * (2 * pi);
    r = sqrt(rand(num_of_devices, 1)) * radius;
    posX = cenX(i) + r .* cos(theta);
    posY = cenY(i) + r .* sin(theta);
    devices_position(:, 2 * i - 1) = posX;
    devices_position(:, 2 * i) = posY;
end