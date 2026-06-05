function [b, scaling_coeff] = scale_b(domain_layout, b, N, sigma)
    scaling_coeff = zeros(N,1);
    iterator = domain_layout.depthfirstiterator;
    for node = iterator
        if node == 1
            scaling_coeff(1) = 2 / sigma(1);
        else
            domain_idx = domain_layout.get(node);
            parent_node = domain_layout.getparent(node);
            parent_domain_idx = domain_layout.get(parent_node);
            scaling_coeff(domain_idx) = -2 / (sigma(domain_idx) + sigma(parent_domain_idx));
        end
    end
    for i = 1 : N
        bi = b{i};
        b{i} = @(x, n) scaling_coeff(i) * bi(x, n);
    end
end