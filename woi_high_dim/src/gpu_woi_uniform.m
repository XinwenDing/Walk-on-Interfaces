function [step_h_ux, step_h_grad_ux, y, Q] = woi_uniform(x, b, D, hierarchy_matrix, Phi, grad_Phi, NP_Operator_kernel, init_pdf, exp_weight, ...
                                                                 arrive_schedule, arrive_sign, depart_schedule, ...
                                                                 prev_y, Q, schedule_per_batch, Markov_chain_weight)
    % MC_chain_length is the index m in chapter 1 Eq 1.19 of Sabelfeld and Simonov
    % transition_weight = K*(Y_{i+1}, Y_i) / p(Y_i, Y_{i+1})
    cowalker_num = 1;
    N = D.N; dim = D.dim; uniform_pdf = D.uniform_pdf;
    y = zeros(schedule_per_batch, dim, 'gpuArray');
    %ny = zeros(batch_size, dim, 'gpuArray');
    if depart_schedule == -1 % first step leaving from x
        for i = 1 : N
            omega_i = D.omega(i);
            interface_i_idx = find(arrive_schedule == i);
            total_length_i = length(interface_i_idx);
            bi = b{i};
            p0 = init_pdf(i);
            [yi, n_yi] = gpu_uniformly_sample_ellipsoid(omega_i, dim, total_length_i, cowalker_num);
            y(interface_i_idx,:) = yi;
            %ny(interface_i_idx, :) = n_yi;
            Q(interface_i_idx) = Q(interface_i_idx) .* bi(yi, n_yi) / p0;
        end
        Q = single(Q); eval_y = single(y);
        %eval_y = y;
        step_h_ux = Markov_chain_weight * arrive_sign .* Q .* Phi(x, eval_y);
        step_h_grad_ux = Markov_chain_weight * arrive_sign .* Q .* grad_Phi(x, eval_y);
    else
        for i = 1 : N
            omega_i = D.omega(i);
            
            arrive_interface_i_idx = find(arrive_schedule == i);
            depart_interface = depart_schedule(arrive_interface_i_idx);
            prev_yi = prev_y(arrive_interface_i_idx, :);
            hierarchy_matrix_slice = hierarchy_matrix(depart_interface, i);
            
            walk_to_ancestor_idx = find(hierarchy_matrix_slice == 1);
            walk_to_self_idx = find(hierarchy_matrix_slice == 0);
            walk_to_descendent_idx = find(hierarchy_matrix_slice == -1);
            
            ancestor_num = length(walk_to_ancestor_idx);
            self_num = length(walk_to_self_idx);
            descendent_num = length(walk_to_descendent_idx);
            total_length_i = ancestor_num + self_num + descendent_num;
            
            %y_i_ancestor = get_sphere_intersection(dim, ancestor_num, prev_yi(walk_to_ancestor_idx,:), center_i, radius_i);
            %y_i_self = get_sphere_intersection(dim, self_num, prev_yi(walk_to_self_idx,:), center_i, radius_i);
            [yi_ancestor, ~] = gpu_get_ellipsoid_intersection(prev_yi(walk_to_ancestor_idx,:), omega_i, cowalker_num);
            [yi_self, ~] = gpu_get_ellipsoid_intersection(prev_yi(walk_to_self_idx,:), omega_i, cowalker_num);
            [yi_descendent, n_yi_descendent] = gpu_uniformly_sample_ellipsoid(omega_i, dim, descendent_num, cowalker_num);
            adj_K = NP_Operator_kernel(yi_descendent, prev_yi(walk_to_descendent_idx,:), n_yi_descendent);
            
            transition_weight = ones(total_length_i, 1, 'gpuArray');
            transition_weight(walk_to_self_idx) = 0.5;
            transition_weight(walk_to_descendent_idx) = adj_K / uniform_pdf(i);
            Q(arrive_interface_i_idx, :) = Q(arrive_interface_i_idx, :) .* transition_weight;
            
            yi = zeros(total_length_i, dim, 'gpuArray');
            yi(walk_to_ancestor_idx, :) = yi_ancestor;
            yi(walk_to_self_idx, :) = yi_self;
            yi(walk_to_descendent_idx, :) = yi_descendent;
            y(arrive_interface_i_idx, :) = yi;

            %{
            n_yi = zeros(total_length_i, dim, 'gpuArray');
            n_yi(walk_to_ancestor_idx, :) = n_yi_ancestor;
            n_yi(walk_to_self_idx, :) = n_yi_self;
            n_yi(walk_to_descendent_idx, :) = n_yi_descendent;
            ny(arrive_interface_i_idx, :) = n_yi;
            %}
        end
        Q = single(Q); eval_y = single(y);
        %eval_y = y;
        step_h_ux = Markov_chain_weight * exp_weight * arrive_sign .* Q .* Phi(x, eval_y);
        step_h_grad_ux = Markov_chain_weight * exp_weight * arrive_sign .* Q .* grad_Phi(x, eval_y);
    end
end