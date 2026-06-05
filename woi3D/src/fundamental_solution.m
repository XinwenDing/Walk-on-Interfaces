function [Phi, grad_Phi] = fundamental_solution(dim)
    % note: gamma(z+1) = z gamma(z)
    if dim == 2
        % x = (x1, x2), y = (y1, y2) should be two points in R2
        Phi = @(x,y) -log(vecnorm(x - y, 2, 2)) / (2 * pi);
    else
        % x = (x1, x2, ... xn), y = (y1, y2, ...yn) should be two points in Rn
        half_dim = dim / 2;
        nsphere_vol = (pi^half_dim) / gamma(half_dim + 1);
        Phi = @(x, y) 1 ./ (dim * (dim - 2) * nsphere_vol * vecnorm(x - y, 2, 2).^(dim - 2));
        %Phi = @(x,y) 1 / (4 * pi * vecnorm(x - y, 2, 2));
    end
    grad_Phi = @(x, y) (y - x) ./ (dim * nsphere_vol * vecnorm(x - y, 2, 2).^dim);  
end