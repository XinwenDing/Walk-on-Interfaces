function retval = ex3_true_soln3D(x, a, sigma_ratio, amp)
    coefficientsM1L1 = constructCoefficientsForTrueSolution(a, sigma_ratio);
    r = vecnorm(x, 2, 2);
    retval = zeros(size(r));

    idx1 = (r > 0) & (r <= a);
    retval(idx1) = coefficientsM1L1(1)*(sqrt(3/(4*pi)) * x(idx1, 1, :));

    idx2 = (r > a) & (r <= 1 + 1e-5);
    retval(idx2) = (coefficientsM1L1(2) * r(idx2) + coefficientsM1L1(3) * r(idx2).^(-2)) .* (sqrt(3/(4*pi)) * x(idx2, 1, :) ./ r(idx2));

    retval = retval * amp;
end

function coefficientsM1L1 = constructCoefficientsForTrueSolution(a, sigma_ratio)
    % calculate coefficients of spherical harmonics
    ML = [  0,       1,        -2;
            a,       -a,    -a^(-2);
            sigma_ratio,  -1, 2*a^(-3);];
    % (m, l) = (1, 1)
    b11 = [sqrt(4*pi/3); 0; 0];

    coefficientsM1L1 = ML\b11;
end