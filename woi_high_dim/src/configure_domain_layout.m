function [T, root_id] = configure_domain_layout()
    % T is a tree implemented by https://github.com/tinevez/matlab-tree
    [T, root_id] = tree(1);
    T = T.addnode(1, 2);
    T = T.addnode(2, 3);
    disp(T.tostring)
end