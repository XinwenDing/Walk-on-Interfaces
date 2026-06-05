function [step_h_ux, step_h_grad_ux, y, y_tri_idx, ny, Q, T] = woi_on_mesh(x, b, D, hierarchy_matrix, Phi, grad_Phi, NP_Operator_kernel, exp_weight, ...
                                                                                    arrive_schedule, arrive_sign, depart_schedule, ...
                                                                                    prev_y, prev_y_tri_idx, prev_ny, Q, T, Markov_chain_weight)
    % MC_chain_length is the index m in chapter 1 Eq 1.19 of Sabelfeld and Simonov
    % transition_weight = K*(Y_{i+1}, Y_i) / p(Y_i, Y_{i+1})
    cowalker_num = 1;
    batch_size = length(Q);
    N = D.N; dim = D.dim; uniform_pdf = D.uniform_pdf;
    y = zeros(batch_size, dim, 'gpuArray');
    y_tri_idx = zeros(batch_size, 1, 'gpuArray');
    ny = zeros(batch_size, dim, 'gpuArray');
    if depart_schedule == -1 % first step leaving from x
        for i = 1 : N
            %fprintf("i = %d \n", i);
            omega_i = D.omega(i);
            interface_i_idx = find(arrive_schedule == i);
            total_length_i = length(interface_i_idx);
            bi = b{i};
            p0 = uniform_pdf(i);
            if (~isempty(omega_i.radius) && ~isempty(omega_i.center)) || (~isfield(omega_i, "radius") && ~isfield(omega_i.radius))
                [yi, n_yi] = D.uniform_sphere_sampler(dim, total_length_i, cowalker_num, omega_i.center, omega_i.radius);
            else
                [yi, yi_tri_idx] = D.uniform_mesh_sampler(omega_i, total_length_i, dim, cowalker_num);
                n_yi = point_outward_normal(omega_i, yi_tri_idx, total_length_i, dim, cowalker_num);
                y_tri_idx(interface_i_idx, :) = yi_tri_idx;
            end
            y(interface_i_idx, :) = yi;
            ny(interface_i_idx, :) = n_yi;
            %transition_weight = bi(y_i) / p0;
            Q(interface_i_idx) = Q(interface_i_idx) .* bi(yi, n_yi) / p0;
        end
        Q = single(Q); eval_y = single(y);
        step_h_ux = Markov_chain_weight * arrive_sign .* Q .* Phi(x, eval_y);
        step_h_grad_ux = Markov_chain_weight * arrive_sign .* Q .* grad_Phi(x, eval_y);
    else
        for i = 1 : N
            %fprintf("i = %d \n", i);
            omega_i = D.omega(i);
            arrive_interface_i_idx = find(arrive_schedule == i);
            depart_interface = depart_schedule(arrive_interface_i_idx);
            prev_yi = prev_y(arrive_interface_i_idx, :);
            prev_yi_tri_idx = prev_y_tri_idx(arrive_interface_i_idx, :);
            prev_n_yi = prev_ny(arrive_interface_i_idx, :);
            hierarchy_matrix_slice = hierarchy_matrix(depart_interface, i);
            
            walk_to_ancestor_idx = find(hierarchy_matrix_slice == 1);
            walk_to_self_idx = find(hierarchy_matrix_slice == 0);
            walk_to_descendent_idx = find(hierarchy_matrix_slice == -1);
            
            ancestor_num = length(walk_to_ancestor_idx);
            self_num = length(walk_to_self_idx);
            descendent_num = length(walk_to_descendent_idx);
            total_length_i = ancestor_num + self_num + descendent_num;

            if ~isempty(omega_i.radius) && ~isempty(omega_i.center)
                center_i = omega_i.center; radius_i = omega_i.radius;
                [yi_ancestor, n_yi_ancestor] = get_sphere_intersection(prev_yi(walk_to_ancestor_idx,:), center_i, radius_i);
                [yi_self, n_yi_self] = get_sphere_intersection(prev_yi(walk_to_self_idx,:), center_i, radius_i);
                [yi_descendent, n_yi_descendent] = D.uniform_sphere_sampler(dim, descendent_num, cowalker_num, center_i, radius_i);
                adj_K = NP_Operator_kernel(yi_descendent, prev_yi(walk_to_descendent_idx,:), n_yi_descendent);
            else
                ray_origin_to_ancestor = prev_yi(walk_to_ancestor_idx,:);
                ray_dir_to_ancestor = sample_directions_to_ancestor(ray_origin_to_ancestor);
                ancestor_ray_tri_uv = zeros(0, 4, 'gpuArray'); ancestor_intersection_count = zeros(ancestor_num, 1, 'gpuArray');
                [ancestor_ray_tri_uv, ancestor_intersection_count] = omega_i.aabbtree.intersectAncestor(ancestor_ray_tri_uv, ancestor_intersection_count, ...
                                                                        gpuArray((1:ancestor_num)'), ray_origin_to_ancestor, ray_dir_to_ancestor, omega_i.V);
                [yi_ancestor, yi_ancestor_triIdx, n_yi_ancestor] = find_random_intersection(omega_i, ancestor_ray_tri_uv, ancestor_intersection_count,...
                                                                                            ray_origin_to_ancestor, ray_dir_to_ancestor);

                ray_origin_to_self = prev_yi(walk_to_self_idx,:);
                ray_origin_outward_normal = prev_n_yi(walk_to_self_idx, :);
                to_self_tri_idx = prev_yi_tri_idx(walk_to_self_idx, :);
                ray_dir_to_self = sample_directions_to_self(ray_origin_to_self, ray_origin_outward_normal, dim);
                self_ray_tri_uv = zeros(0, 4, 'gpuArray'); self_intersection_count = zeros(self_num, 1, 'gpuArray');
                [self_ray_tri_uv, self_intersection_count] = omega_i.aabbtree.intersectSelf(self_ray_tri_uv, self_intersection_count, gpuArray((1:self_num)'), ...
                                                                                        ray_origin_to_self, ray_dir_to_self, to_self_tri_idx, omega_i.V);
                if any(self_intersection_count == 0)
                    fprintf("exists ray failed to intersect self, resample ray \n");
                    fail_idx = find(self_intersection_count == 0);
                    fail_num = length(fail_idx);
                    new_self_ray_tri_uv = zeros(0, 4, 'gpuArray'); new_self_intersection_count = zeros(fail_num, 1, 'gpuArray');
                    while any(new_self_intersection_count == 0)
                        failed_origins_to_self = ray_origin_to_self(fail_idx, :);
                        failed_origin_outward_normal = ray_origin_outward_normal(fail_idx, :);
                        failed_to_self_tri_idx = to_self_tri_idx(fail_idx, :);
                        new_dir_to_self = sample_directions_to_self(failed_origins_to_self, failed_origin_outward_normal, dim);
                        [new_self_ray_tri_uv, new_self_intersection_count] = omega_i.aabbtree.intersectSelf(new_self_ray_tri_uv, new_self_intersection_count, ...
                                                                            gpuArray((1 : fail_num)'), failed_origins_to_self, new_dir_to_self, failed_to_self_tri_idx, omega_i.V);
                    end
                    ray_dir_to_self(fail_idx, :) = new_dir_to_self;
                    self_ray_tri_uv = [self_ray_tri_uv; new_self_ray_tri_uv];
                    self_intersection_count(fail_idx, :) = new_self_intersection_count;
                end
                [yi_self, yi_self_triIdx, n_yi_self] = find_random_intersection(omega_i, self_ray_tri_uv, self_intersection_count, ...
                                                                                ray_origin_to_self, ray_dir_to_self);
                
                [yi_descendent, yi_descendent_triIdx] = D.uniform_mesh_sampler(omega_i, descendent_num, dim, cowalker_num);
                n_yi_descendent = point_outward_normal(omega_i, yi_descendent_triIdx, descendent_num, dim, cowalker_num);
                adj_K = NP_Operator_kernel(yi_descendent, prev_yi(walk_to_descendent_idx,:), n_yi_descendent);
                
                adjust_p = ones(total_length_i, 1, 'gpuArray'); % no need to adjust probability distribution if we go to descendents (uniform)
                adjust_p(walk_to_ancestor_idx) = ancestor_intersection_count .* sign(sum((yi_ancestor - ray_origin_to_ancestor) .* n_yi_ancestor, 2));
                adjust_p(walk_to_self_idx) = self_intersection_count .* sign(sum((yi_self - ray_origin_to_self) .* n_yi_self, 2));
                T(arrive_interface_i_idx, :) = T(arrive_interface_i_idx, :) .* adjust_p;

                yi_tri_idx = zeros(total_length_i, 1, 'gpuArray');
                yi_tri_idx(walk_to_ancestor_idx) = yi_ancestor_triIdx;
                yi_tri_idx(walk_to_self_idx) = yi_self_triIdx;
                yi_tri_idx(walk_to_descendent_idx) = yi_descendent_triIdx;
                y_tri_idx(arrive_interface_i_idx) = yi_tri_idx;
            end

            transition_weight = ones(total_length_i, 1, 'gpuArray'); % transition weight to parent is 1
            transition_weight(walk_to_self_idx) = 0.5;
            transition_weight(walk_to_descendent_idx) = adj_K / uniform_pdf(i);
            Q(arrive_interface_i_idx, :) = Q(arrive_interface_i_idx, :) .* transition_weight;


            yi = zeros(total_length_i, dim, 'gpuArray');
            yi(walk_to_ancestor_idx, :) = yi_ancestor;
            yi(walk_to_self_idx, :) = yi_self;
            yi(walk_to_descendent_idx, :) = yi_descendent;
            y(arrive_interface_i_idx, :) = yi;

            n_yi = zeros(total_length_i, dim, 'gpuArray');
            n_yi(walk_to_ancestor_idx, :) = n_yi_ancestor;
            n_yi(walk_to_self_idx, :) = n_yi_self;
            n_yi(walk_to_descendent_idx, :) = n_yi_descendent;
            ny(arrive_interface_i_idx, :) = n_yi;
        end
        Q = single(Q); T = single(T); eval_y = single(y);
        step_h_ux = Markov_chain_weight * exp_weight * arrive_sign .* Q .* T .* Phi(x, eval_y);
        step_h_grad_ux = Markov_chain_weight * exp_weight * arrive_sign .* Q .* T .* grad_Phi(x, eval_y);
    end
end