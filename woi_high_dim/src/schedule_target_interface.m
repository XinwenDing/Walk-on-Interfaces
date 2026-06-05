function [interface_schedule, sign_schedule, schedule_size] = schedule_target_interface(interface_weight, MC_tree_height, N, schedule_size)
    % if MC tree is of height 4, then the interface_schedule is of structure
    %[ 1, 2, 1, 1;  <- this is one branch of MC tree
    %  2, 2, 1, 2; 
    %     ....
    %  1, 1, 2, 2]
    %interface_schedule = zeros(schedule_size, MC_tree_height);
    interface_weight = gpuArray(interface_weight);
    schedule_size = floor(schedule_size / N) * N;
    first_interface_schedule =  kron(gpuArray(1 : N).', ones(floor(schedule_size / N), 1, 'gpuArray'));
    %first_interface_schedule = [ones(schedule_size / N, 1); 2 * ones(schedule_size / N, 1)];
    %first_interface_schedule = randi([1, N], schedule_size, 1);
    
    abs_interface_weight = abs(interface_weight);
    target_interface_cdf = cumsum(abs_interface_weight / sum(abs_interface_weight));
    
    %later_interface_schedule = zeros(schedule_size, MC_tree_height - 1);
    uniform_sample = rand(schedule_size, MC_tree_height - 1, 'gpuArray');
    out_val = interp1([0; target_interface_cdf(:)], 0 : 1/N : 1, uniform_sample(:));
    later_interface_schedule = reshape(ceil(out_val * N), size(uniform_sample));
    interface_schedule = [first_interface_schedule, later_interface_schedule];
    
    target_interface_sign = sign(interface_weight);
    sign_schedule = cumprod(reshape(target_interface_sign(later_interface_schedule), size(later_interface_schedule)), 2);
    sign_schedule = [ones(schedule_size, 1, 'gpuArray'), sign_schedule];
    %sign_schedule = gpuArray(sign_schedule);
    
    % Note: try not to clear variables in the middle of the functions.
    % the following line takes 2 seconds
    %clear interface_weight first_interface_schedule later_interface_schedule
end