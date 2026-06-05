function hierarchy_matrix = get_hierarchy_matrix(domain_layout, root_id, N)
    % hierarchy_matrix(depart_interface, arrive_interface)
    % hierarchy_matrix(i,j) = -1: node j is a descendant of node i 
    %                       (i.e. node i is an ancestor of node j)
    % hierarchy_matrix(i,j) = 0: node j is the same as node i
    % hierarchy_matrix(i,j) = 1: node j is an ancestor of node i
    hierarchy_matrix = -ones(N, N, 'gpuArray');
    %iterator = domain_layout.depthfirstiterator;
    % Iterate over all pairs of nodes
    for i = 1 : N
        for j = 1 : N
            % i is the id of node i, j is the id of node j
            if i ~= j
                % Check if j is an ancestor of i
                %fprintf("i = %d, j = %d \n", i, j);
                %disp(isAncestor(domain_layout, root_id, i, j))
                if isAncestor(domain_layout, root_id, i, j) % true if j is an ancestor of i
                    hierarchy_matrix(i, j) = 1;  % j is an ancestor of i
                    hierarchy_matrix(j, i) = -1;  % i is an ancestor of j
                end
            else
                hierarchy_matrix(i, j) = 0;  % i == j
            end
        end
    end
end

function result = isAncestor(tree, root_id, node_id, test_node_id)
    % Helper function to check if a test_node is an ancestor of node
    result = false;
    % Traverses up the tree to root
    current_id = node_id;
    parent_id = -1;
    while current_id ~= root_id && current_id > 0
        % ID = getparent(obj, ID)
        parent_id = getparent(tree, current_id);
        if parent_id == test_node_id
            result = true;
            return;
        end
        % Move up to the parent
        current_id = parent_id;
    end
end