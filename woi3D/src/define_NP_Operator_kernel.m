function NP_Operator_kernel = define_NP_Operator_kernel(dim)
    % function handle of kernel of Neumann-Poincare operator
    % K_star = -del G / del n(x) = dot(y - x, n(x)) / m * alpha(m) * |x - y|^m.
    % This function defines adj_K = dot(x - y, n(x)) / m * alpha(m) * |x - y|^m,
    % where alpha(m) = pi^{m/2} / gamma(1 + m/2) is the volumn of unit ball in dim m.
    % x is a 1 x dim vector, y and nx are (some number) x dim matrix
    half_dim = dim / 2;
    nsphere_vol = (pi^half_dim) / gamma(half_dim + 1);
    nsphere_surface = dim * nsphere_vol;
    NP_Operator_kernel = @(x, y, nx) sum((x - y) .* nx, 2) ./ (nsphere_surface * vecnorm(x - y, 2, 2).^dim);
end