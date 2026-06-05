function [T, root_id] = configure_domain_layout()
    % T is a tree implemented by https://github.com/tinevez/matlab-tree
    [T, root_id] = tree(1);
    T = T.addnode(1, 2);
    T = T.addnode(1, 3);
    T = T.addnode(1, 4);
    T = T.addnode(1, 5);
    T = T.addnode(1, 6);
    T = T.addnode(1, 7);
    disp(T.tostring)
end