function [y, n_yi] = get_sphere_intersection(x, center, radius)
%get_sphere_intersection(dim, point_num, x, center, radius)
    % x: current point
    % w: random direction
    % Domain: unit ball in R^2 or R^3
    % Goal: find t, such that y = x + tw is on the boundary of the domain (sphere).
    % A random normal distribution of coordinates gives uniform distribution of directions.
    x = x - center; % shift to origin
    normx = vecnorm(x, 2, 2);
    w = randn(size(x), 'gpuArray');
    %theta = 2 * pi * rand(1);
    %w = [cos(theta), sin(theta)];
    w = w ./ vecnorm(w, 2, 2);
    dot_xw = sum(w .* x, 2);
    %fprintf("dot_xw = %.20f \n", dot_xw);
    % Find t by solving ||x + tw||^2 = r^2
    % ||x||^2 + 2t dot(x,w) + t^2 ||w||^2 = r^2 => ||w||^2 t^2 + 2t dot(x,w) + ||x||^2 - r^2 = 0
    %close_to_bdr_cond = (abs(normx - radius) < 1e-12);
    dist_to_bdr = radius - normx;
    on_bdr_cond = (abs(dist_to_bdr) < 1e-12);
    interior_cond = (dist_to_bdr > 1e-12);
    t = -2 * dot_xw .* on_bdr_cond + (-dot_xw + sqrt(max(dot_xw.^2 - normx.^2 + radius^2, 0))) .* (interior_cond);
    if sum(normx - radius > 1e-12)
        fprintf("radius = %f \n", radius);
        %disp(x(normx - radius > 1e-12, :))
        fprintf("x = [%f, %f], norm(x) = %f \n", x(1), x(2), norm(x));
        msg = "x is outside the sphere. \n";
        error(msg)
    end
    y = x + t .* w; 
    n_yi = y ./ vecnorm(y, 2, 2);
    %fprintf("both condition counts: %d\n", sum(abs(normx - radius) < 1e-12) + sum(normx < radius))
    %disp(vecnorm(y, 2, 2)- radius);
    %fprintf("t = %f \n", t);
    if sum(vecnorm(y, 2, 2) - radius >= 1e-12)
        fprintf("y = [%f, %f] \n", y(1), y(2));
        fprintf("norm(y) = %f \n", norm(y));
        fprintf("t = %f \n", t);
        msg = "y is outside the sphere. \n";
        error(msg)
    end
    y = y + center;
    %fprintf("y = [%f, %f], norm(y) = %.20f \n", y(1), y(2), norm(y));
    %disp(roots([norm(w)^2, 2 * dot(x, w), norm(x)^2 - radius^2]))
    %fprintf("w = [%f, %f], t = %f, y = [%f, %f] \n", w(1), w(2), t, y(1), y(2));
end