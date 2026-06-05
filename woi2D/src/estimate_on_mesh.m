function [ux, grad_ux] = estimate_on_mesh(x, b, D, hierarchy_matrix, interface_weight, Markov_tree_height, schedule_per_batch, batch_num)
    dim = D.dim;
    N = D.N;
    query_num = size(x, 1);
    x = permute(x, [3, 2, 1]);
    [Phi, grad_Phi] = fundamental_solution(dim);
    NP_Operator_kernel = define_NP_Operator_kernel(dim);
    exp_weight_lst = sum(abs(interface_weight)).^(0 : (Markov_tree_height-1));
    schedule_per_batch = ceil(schedule_per_batch / N) * N;

    batch_ux = 0; batch_grad_ux = 0;
    for batch_idx = 1 : batch_num
        [batch_interface_schedule, batch_sign_schedule, ~] = schedule_target_interface(interface_weight, Markov_tree_height, N, schedule_per_batch);
        step_ux = zeros(1, 1, query_num, 'gpuArray'); step_grad_ux = zeros(1, dim, query_num, 'gpuArray');
        depart_schedule = -1; prev_y = -1; prev_y_edge_idx = -1; prev_ny = -1;
        Q = ones(schedule_per_batch, 1, 'gpuArray');
        T = ones(schedule_per_batch, 1, 'gpuArray');
        Markov_chain_weight = 1;
        for h = 1 : Markov_tree_height
            %fprintf("h = %d \n", h);
            if h == Markov_tree_height
                Markov_chain_weight = 0.5;
            end
            arrive_schedule = batch_interface_schedule(:, h);
            arrive_sign = batch_sign_schedule(:, h);
            exp_weight = exp_weight_lst(h);
            [step_h_ux, step_h_grad_ux, y, y_edge_idx, ny, Q, T] = woi_on_mesh_uniform(x, b, D, hierarchy_matrix, Phi, grad_Phi, NP_Operator_kernel, ...
                                                                                    exp_weight, arrive_schedule, arrive_sign, depart_schedule, ...
                                                                                    prev_y, prev_y_edge_idx, prev_ny, Q, T, Markov_chain_weight);
            step_ux = step_ux + sum(step_h_ux, 1);
            step_grad_ux = step_grad_ux + sum(step_h_grad_ux, 1);
            
            prev_y = y;
            prev_y_edge_idx = y_edge_idx;
            prev_ny = ny;
            depart_schedule = arrive_schedule;
        end
        batch_ux = batch_ux + step_ux;
        batch_grad_ux = batch_grad_ux + step_grad_ux;
    end
    ux = batch_ux ./ (schedule_per_batch * batch_num);
    grad_ux = batch_grad_ux ./ (schedule_per_batch * batch_num);

    ux = N * ux;
    grad_ux = N * grad_ux;
    ux = permute(ux, [3, 2, 1]);
    grad_ux = permute(grad_ux, [3, 2, 1]);
end