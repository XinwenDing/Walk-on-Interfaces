function [y, w] = uniform_sphere_sampler(dim, point_num, cowalker_num, center, radius)
    w = randn(point_num, dim, cowalker_num, 'gpuArray');
    w = w ./ vecnorm(w, 2, 2);
    y = radius * w + center;
end