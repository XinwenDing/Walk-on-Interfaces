function [omega, D] = define_domain(V, F, dim, N, conductivity_const, mesh_outward_normal, mesh_surface_measure) 
    for i = 1 : N
        Vi = gpuArray(V{i}(:, 1:dim));
        Fi = gpuArray(F{i}(:, 1:dim));
        omega(i).V = Vi;
        omega(i).F = Fi;
        omega(i).center = []; omega(i).radius = [];
        omega(i).outward_normal = mesh_outward_normal(Vi, Fi, dim);
        [omega(i).uniform_pdf, omega(i).surface_measure_cdf] = mesh_surface_measure(Vi, Fi, dim);
        omega(i).sigma = conductivity_const(i);
        omega(i).aabbtree = AABBTree3D(Vi, Fi, (1:size(Fi, 1))');
        [Vi, Fi] = gather(Vi, Fi);
    end

    %omega(1).center = [0, 0, 0]; % center and radius will only be used to find proper query points
    %omega(1).radius = 2;

    D.omega = omega;
    %D.center = reshape([omega.center], dim, N).';
    %D.radius = reshape([omega.radius], N, 1);
    D.uniform_pdf = reshape([omega.uniform_pdf], [], 1);
    D.sigma = gpuArray(conductivity_const);
    D.dim = dim; D.N = N;
    D.uniform_mesh_sampler = uniform_mesh_sampler(dim);
    D.uniform_sphere_sampler = @uniform_sphere_sampler;
    D.mesh_outward_normal = mesh_outward_normal;
    D.mesh_surface_measure = mesh_surface_measure;
end