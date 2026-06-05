function mesh_surface_measure = define_mesh_surface_measure(dim)
    if dim == 2
        mesh_surface_measure = @mesh_surface_measure2D;
    end
end

function [mesh_uniform_pdf, surface_measure_cdf] = mesh_surface_measure2D(V, F, dim)
    % surface measure is:
    %       edge length: if mesh is in 2D 
    %       surface area: if mesh is in 3D
    if dim == 2
        edge_length = vecnorm(V(F(:,1), 1 : dim) - V(F(:,2), 1 : dim), 2, 2);
        perimeter = sum(edge_length);
        surface_measure_cdf = cumsum(edge_length / perimeter);
        mesh_uniform_pdf = 1 / perimeter;
    end
end