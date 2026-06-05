function [y, ny] = uniform_ellipsoid_sampler(omega, dim, point_num, cowalker_num)
    % A point on ellipsoid should satisfy (x - c)^T A (x - c) = 1.
    % Cholesky: A = L L'
    % (x - c)^T L L' (x - c) = 1.
    c = omega.center';           % dim x 1
    w = randn(dim, point_num, cowalker_num, 'gpuArray');
    w = w ./ vecnorm(w, 2, 1);  % Normalize each column to unit length
    %{
    y = pagemldivide(repmat(omega.U, 1, 1, cowalker_num), w) + omega.center;   %  dim * point_num * cowalker
    ny = pagemtimes(repmat(omega.A, 1, 1, cowalker_num), y - omega.center);    %  dim * point_num * cowalker

    y = permute(y, [2,1,3]);    %  point_num * dim, cowalker
    ny = permute(ny, [2,1,3]);  %  point_num * dim, cowalker
    %}
    y = pagemldivide(repmat(omega.U, 1, 1, cowalker_num), w) + c;
    ny = pagemtimes(repmat(omega.A, 1, 1, cowalker_num), y - c);

    y = pagetranspose(y);
    ny = pagetranspose(ny);
    ny = ny ./ vecnorm(ny, 2, 2);
end
