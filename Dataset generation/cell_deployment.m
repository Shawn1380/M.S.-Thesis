function [cenX, cenY] = cell_deployment(num_of_cells, radius)
% Input:
% num_of_cells: Number of the cells in the system
% radius: The radius of the cell (meter)
%
% Output:
% cenX: The X-coordinate of the center of the cells
% cenY: The Y-coordinate of the center of the cells

if num_of_cells == 2
    cenX = [0, 2 * radius];
    cenY = [0, 0];
elseif num_of_cells == 3
    cenX = [0, 2 * radius, radius];
    cenY = [0, 0, sqrt(3) * radius];
elseif num_of_cells == 4
    cenX = [0, 2 * radius, radius, 3 * radius];
    cenY = [0, 0, sqrt(3) * radius, sqrt(3) * radius];
else
    % Deploy the cells like this pattern:
    % https://imgur.com/NMkyZ3I
end