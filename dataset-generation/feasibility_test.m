function [feasible, particular_sol] = feasibility_test(num_of_CUEs, num_of_D2Ds, matrix, Pmax, QoS_of_CUE, QoS_of_D2D)
% Input:
% num_of_CUEs: Number of the CUEs in each cell
% num_of_D2Ds: Number of the D2D pairs in each cell
% matrix: The channel gain matrix that takes account for path-loss
% Pmax: Maximum transmit power of all devices (Watt)
% QoS_of_CUE: Minimum data rate requirement of all CUEs (bps/Hz)
% QoS_of_D2D: Minimum data rate requirement of all D2D pairs (bps/Hz)
%
% Output:
% feasible: A binary indicator which indicates whether the feasibility test is passed or not
% particular_sol: Calculated initial transmit power

noise = 7.161e-16; % Noise power (Watt)
factor = 1;  

N = length(matrix);
h = matrix';

feasible = false;
particular_sol = zeros(1, N);

CUE_SINR_requirement = 2 ^ QoS_of_CUE - 1;
D2D_SINR_requirement = (2 ^ (QoS_of_D2D / num_of_CUEs) - 1) * ones(1, num_of_D2Ds);
gamma = [CUE_SINR_requirement * factor, D2D_SINR_requirement * factor];

CUE_Pmax = Pmax;
D2D_Pmax = (Pmax / num_of_CUEs) * ones(1, num_of_D2Ds);
maximum_power = [CUE_Pmax, D2D_Pmax];

% Use the feasibility test introduced in the reference:
% M. Klugel and W. Kellerer,¡§Determining frequency reuse feasibility in device-to-device cellular networks,¡¨ in IEEE PIMRC, Hong Kong, Aug.-Sep. 2015

% u is the vector of normalized noise power
u = gamma' * noise ./ diag(h);

% Compute the Foschini matrix F with (i,j) entry:
% if i ~= j, then F_{i,j} = gamma_i * h_{ij} / h_{ii}
% if i = j, then F_{i,j} = 0
F = diag(gamma' ./ diag(h)) * ones(N, N) .* h;
F(1 : 1 + N : end) = 0;	% Set all diagnoal elements to zero

eigenvalues = eig(F);
max_modulus_eigenvalue = max(abs(eigenvalues));

if max_modulus_eigenvalue <= 1
    Pp = (eye(N) - F) \ u;
    particular_sol = Pp';
    
    if sum(particular_sol > maximum_power) == 0
       feasible = true; 
    end
end
