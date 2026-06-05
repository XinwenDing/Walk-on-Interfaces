function query_points = uniformly_sample_query_points(dim, omega, N)
    % Uniformly sample N points inside the ellipsoid (x - center)' A (x - center) <= 1 as query points
    % Step 1: sample unit vectors (on sphere)
    Z = randn(dim, N);
    Z = Z ./ vecnorm(Z);  % Normalize columns to unit norm
    
    % Step 2: sample radius from unit ball (Marsaglia method)
    r = rand(1, N).^(1/dim);
    Z = Z .* r;  % Scale unit vectors by r^(1/n)
    
    % Step 3: transform to ellipsoid interior
    query_points = (omega.U \ Z)' + omega.center;
end