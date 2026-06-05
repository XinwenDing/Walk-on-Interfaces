function sampler_fnc = uniform_mesh_sampler(dim)
    if dim == 2
        sampler_fnc = @uniformly_sample_2Dmesh;
    end
end
function [y, y_edge_idx] = uniformly_sample_2Dmesh(omega, random_point_num, dim, cowalker_num)
    % surface measure is:
    %       edge length: if mesh is in 2D 
    %       surface area: if mesh is in 3D
    V = omega.V; 
    F = omega.F; 
    surface_measure_cdf = omega.surface_measure_cdf;

    %y = NaN(random_point_num, 1);
    len_surface_measure_cdf = length(surface_measure_cdf);
    uniform_sample = rand(random_point_num, cowalker_num, 'gpuArray');
    out_val = interp1([0; surface_measure_cdf(:)], 0 : 1/len_surface_measure_cdf : 1, uniform_sample(:));
    y_edge_idx = reshape(ceil(out_val * len_surface_measure_cdf), size(uniform_sample));
    target_edge = F(y_edge_idx, :);

    target_edge_start = V(target_edge(:,1), :);    % (random_point_num * cowalker_num, dim)
    target_edge_end = V(target_edge(:,2), :);      % (random_point_num * cowalker_num, dim)
    target_edge_start = permute(reshape(target_edge_start', dim, random_point_num, cowalker_num), [2, 1, 3]); % (random_point_num, dim, cowalker_num)
    target_edge_end = permute(reshape(target_edge_end', dim, random_point_num, cowalker_num), [2, 1, 3]);     % (random_point_num, dim, cowalker_num)
    lambda = rand(random_point_num, 1, cowalker_num, 'gpuArray');
    y = lambda .* target_edge_start + (1 - lambda) .* target_edge_end;
end