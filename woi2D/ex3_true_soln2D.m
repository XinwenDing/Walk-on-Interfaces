function retval = ex3_true_soln2D(x, A, B, C, m, amp, alpha)
    r = sqrt(x(:, 1, :).^2 + x(:, 2, :).^2);
    theta = atan2(x(:, 2, :), x(:, 1, :));

    retval = zeros(size(r));
    
    % Handle r = 0 separately to avoid singularity
    retval(r == 0) = 0;
    
    % r <= alpha
    idx1 = (r > 0) & (r <= alpha);
    retval(idx1) = A .* (r(idx1)).^m .* sin(m .* theta(idx1));
    
    % alpha < r <= 1
    idx2 = (r > alpha) & (r <= 1);
    retval(idx2) = (B .* (r(idx2)).^m + C .* (r(idx2)).^(-m)) .* sin(m .* theta(idx2));

    retval = retval .* amp;
end