function sampler_fnc = uniform_mesh_sampler(dim)
    if dim == 2
        sampler_fnc = @uniformly_sample_2Dmesh;
    elseif dim == 3
        sampler_fnc = @uniformly_sample_3Dmesh;
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

function [y, y_tri_idx] = uniformly_sample_3Dmesh(omega, random_point_num, dim, cowalker_num)
    V = omega.V; 
    F = omega.F;
    % surface measure is:
    %       edge length: if mesh is in 2D 
    %       surface area: if mesh is in 3D
    surface_measure_cdf = omega.surface_measure_cdf;

    %y = NaN(random_point_num, 1);
    len_surface_measure_cdf = length(surface_measure_cdf);
    uniform_sample = rand(random_point_num, cowalker_num, 'gpuArray');
    out_val = interp1([0; surface_measure_cdf(:)], 0 : 1/len_surface_measure_cdf : 1, uniform_sample(:));
    y_tri_idx = reshape(ceil(out_val * len_surface_measure_cdf), size(uniform_sample));
    target_triangle = F(y_tri_idx, :);

    target_triangle_v0 = V(target_triangle(:,1), :);    % (random_point_num * cowalker_num, dim)
    target_triangle_v1 = V(target_triangle(:,2), :);      % (random_point_num * cowalker_num, dim)
    target_triangle_v2 = V(target_triangle(:,3), :);      % (random_point_num * cowalker_num, dim)
    target_triangle_v0 = pagetranspose(reshape(target_triangle_v0', dim, random_point_num, cowalker_num)); % (random_point_num, dim, cowalker_num)
    target_triangle_v1 = pagetranspose(reshape(target_triangle_v1', dim, random_point_num, cowalker_num));     % (random_point_num, dim, cowalker_num)
    target_triangle_v2 = pagetranspose(reshape(target_triangle_v2', dim, random_point_num, cowalker_num));     % (random_point_num, dim, cowalker_num)
    r1 = rand(random_point_num, 1, cowalker_num, 'gpuArray');
    r2 = rand(random_point_num, 1, cowalker_num, 'gpuArray');
    sqrt_r1 = sqrt(r1);
    y = (1 - sqrt_r1) .* target_triangle_v1 + (sqrt_r1 .* (1 - r2)) .* target_triangle_v2 + (sqrt_r1 .* r2) .* target_triangle_v0;
end