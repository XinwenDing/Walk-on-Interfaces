function target_interface_cost = get_interface_weight(domain_layout, N, sigma)
    target_interface_cost = zeros(1, N);
    iterator = domain_layout.depthfirstiterator;
    for node = iterator
        if node == 1
            target_interface_cost(1) = 2;
        else
            domain_idx = domain_layout.get(node);
            parent_node = domain_layout.getparent(node);
            parent_domain_idx = domain_layout.get(parent_node);
            target_interface_cost(domain_idx) = 2 * (sigma(domain_idx) - sigma(parent_domain_idx)) / ...
                                                    (sigma(domain_idx) + sigma(parent_domain_idx));
        end
    end
end