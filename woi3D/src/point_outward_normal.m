function n = point_outward_normal(omega, tri_idx, point_num, dim, third_dim)
    % this function returns the outward normal on a given mesh after we
    % walk to the next point on interface
    %n = permute(reshape(omega.outward_normal(tri_idx, :)', dim, third_dim, point_num), [3, 1, 2]);
    n = permute(reshape(omega.outward_normal(tri_idx, :)', dim, point_num, third_dim), [2, 1, 3]);
end