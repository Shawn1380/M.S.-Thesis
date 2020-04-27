function print_transmit_power(num_of_cells, num_of_CUEs, num_of_D2Ds, Pmax, CUE_power, D2D_power, mode)
% Input:
% num_of_cells: Number of the cells in the system
% num_of_CUEs: Number of the CUEs in each cell
% num_of_D2Ds: Number of the D2D pairs in each cell
% Pmax: Maximum transmit power of all devices (Watt)
% CUE_power: The transmit power of all CUEs 
% D2D_power: The transmit power of all D2D pairs
% mode: Determine what kind of message should be print

num_of_RBs = num_of_CUEs;

if mode == 0
    str = 'Initialization';   
elseif mode == 1
    str = 'Optimal solution';
elseif mode == 2
    str = 'CVX summary';
else
    % Define your mode right here
end

for cell_index = 1 : num_of_cells
    
    cprintf('Blue', '%s: Cell %d (transmit power)\n', str, cell_index);
    fprintf('Maximum transmit power: %2.1f (Watt)\n\n', Pmax);
    
    % Print the CUE's transmit power
    fprintf('\t\tRB');
    fprintf('\t\t\t\t\tLimitation\n');
    for CUE_index = 1 : num_of_CUEs
        fprintf('CUE %d\t%12.10f\t\t', CUE_index, CUE_power(CUE_index, 1, cell_index));
        fprintf('%12.10f\n', Pmax);
    end
    fprintf('\n');
    
    % Print the D2D pair's transmit power
    fprintf('\t\t');
    for RB_index = 1 : num_of_RBs
        fprintf('RB %d\t\t\t\t', RB_index);    
    end
    fprintf('Total\t\t\t\t');
    fprintf('Limitation\n');
    
    for D2D_index = 1 : num_of_D2Ds
        fprintf('D2D %d\t', D2D_index);
        for RB_index = 1 : num_of_RBs
            fprintf('%12.10f\t\t', D2D_power(D2D_index, RB_index, cell_index)) 
        end
        fprintf('%12.10f\t\t', sum(D2D_power(D2D_index, :, cell_index)));
        fprintf('%12.10f\n', Pmax);
    end
    
    fprintf('\n');
end
    