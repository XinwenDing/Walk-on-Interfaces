function [y, ny] = get_ellipsoid_intersection(prev_y, omega, cowalker_num)
    % x: current point
    % w: random direction
    % Domain: unit ball in R^2 or R^3
    % Goal: find t, such that y = x + tw is on the boundary of the domain (sphere).
    % A random normal distribution of coordinates gives uniform distribution of directions.
    % Computes the unique intersection of a ray x = p + t*w with an ellipsoid
    % (x - c)' * A * (x - c) = 1, where p is on or inside the ellipsoid and t > 0.
    % substitude in ray: (wt + p - c)' * A * (wt + p - c) = 1. 
    % Let q = p - c. Then (wt + q)' * A * (wt + q) = 1. 
    % Hence, we need to solve (w'Aw) t^2 + (2 w'Aw) t  + q'Aq - 1 = 0.
    % Let a = w'Aw, b = w'Aq, c = q'Aq - 1. The quadratic function is at^2 + 2bt + c = 0.
    % t1 = [-2b + sqrt(4b^2 - 4ac)] / 2a = [-b + sqrt(b^2 - ac)] / a. We'll take this, since t1 > 0.
    % t2 = [-2b - sqrt(4b^2 - 4ac)] / 2a = [-b - sqrt(b^2 - ac)] / a.
    w = pagetranspose(randn(size(prev_y), 'gpuArray'));  % prev_y: point_num x dim x cowalker_num
    %w = randn(dim, point_num, cowalker_num, 'gpuArray');
    % Shifted vector
    center = omega.center;
    %disp(size(prev_y));
    %disp(size(center));
    q = prev_y - center;         % point_num * dim * cowalker_num, prev_y is p
    page_transpose_q = pagetranspose(q);

    page_A = repmat(omega.A, 1, 1, cowalker_num);
    % Quadratic coefficients
    pageA_times_w = pagemtimes(page_A, w);
    a = sum(w .* pageA_times_w, 1);
    b = sum(page_transpose_q .* pageA_times_w, 1);
    c = sum(page_transpose_q .* pagemtimes(page_A, page_transpose_q), 1) - 1;

    % Tolerance for boundary check
    tol = 1e-12;
    on_bdr_cond = (abs(c) < tol);    %(x - c)' * A * (x - c) - 1 = 0
    interior_cond = (c < -tol);    %(x - c)' * A * (x - c) - 1 < 0

    if any(c > tol)
        disp(c(c > tol));
        error("exists point outside the ellipsoid.")
    end
    
    discriminant = b.^2 - a .* c;
    t = on_bdr_cond .* (-2 * b ./ a) + interior_cond .* (-b + sqrt(max(discriminant, 0))) ./ a;
    
    if any(discriminant < -tol)
        disp(discriminant(discriminant < -tol))
        error('No real intersection found — possibly numerical error.');
    end

    y = prev_y + pagetranspose(t .* w);
    ny = pagemtimes(y - center, page_A);
    ny = ny ./ vecnorm(ny, 2, 2);


    % =========================================
    %{
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
    t = -2 * dot_xw .* on_bdr_cond + ...
        (-dot_xw + sqrt(max(dot_xw.^2 - normx.^2 + radius^2, 0))) .* (interior_cond);
    if sum(normx - radius > 1e-12)
        fprintf("x = [%f, %f], norm(x) = %f \n", x(1), x(2), norm(x));
        msg = "x is outside the sphere. \n";
        error(msg)
    end
    y = x + t .* w; 
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
    %}
end