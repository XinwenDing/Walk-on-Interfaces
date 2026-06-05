function mesh_outward_normal = define_mesh_outward_normal(dim)
    if dim == 2
        mesh_outward_normal = @mesh_outward_normal2D;
    end
end

function outward_normal = mesh_outward_normal2D(V, F, dim)
    if dim == 2
        edge_starts = V(F(:, 1), 1 : dim);
        edge_ends = V(F(:,2), 1 : dim);
        edge_dir = edge_ends - edge_starts;
        edge_dir = edge_dir ./ vecnorm(edge_dir, 2, 2);
        outward_normal = [-edge_dir(:, 2), edge_dir(:, 1)];
    end
end