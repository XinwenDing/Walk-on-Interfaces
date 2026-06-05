function mesh_surface_measure = define_mesh_surface_measure(dim)
    if dim == 2
        mesh_surface_measure = @mesh_surface_measure2D;
    elseif dim == 3
        mesh_surface_measure = @mesh_surface_measure3D;
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

function [mesh_uniform_pdf, surface_measure_cdf] = mesh_surface_measure3D(V, F, dim)
    % surface measure is:
    %       edge length: if mesh is in 2D 
    %       surface area: if mesh is in 3D
    if dim == 3
        v0 = V(F(:,1), :);
        v1 = V(F(:,2), :);
        v2 = V(F(:,3), :);
        
        e1 = v1 - v0;
        e2 = v2 - v0;
        
        cross_prod = cross(e1, e2, 2); % Cross product along rows
        tri_area = 0.5 * vecnorm(cross_prod, 2, 2); % area = 0.5 * ||(v1 - v0) x (v2 - v0)||
        surface_area = sum(tri_area);
        surface_measure_cdf = cumsum(tri_area / surface_area);
        mesh_uniform_pdf = 1 / surface_area;
    end
end