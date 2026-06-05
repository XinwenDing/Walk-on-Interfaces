function [omega, D] = define_domain(V, F, dim, N, conductivity_const, mesh_outward_normal, mesh_surface_measure) 
    for i = 1 : N
        Vi = gpuArray(V{i}(:, 1:dim));
        Fi = gpuArray(F{i}(:, 1:dim));
        omega(i).V = Vi;
        omega(i).F = Fi;
        omega(i).outward_normal = mesh_outward_normal(Vi, Fi, dim);
        [omega(i).uniform_pdf, omega(i).surface_measure_cdf] = mesh_surface_measure(Vi, Fi, dim);
        omega(i).sigma = conductivity_const(i);
        omega(i).aabbtree = AABBTree2D(Vi, Fi);
        omega(i).id = i;
        [Vi, Fi] = gather(Vi, Fi);
    end

    %{ 
    % This is the domain for Ex1 in paper
    omega(1).center = [0.5, -0.5]; % center and radius will only be used to find proper query points
    omega(1).radius = 2;
    omega(2).center = [0, -1.2]; % center and radius will only be used to find proper query points
    omega(2).radius = 0.6;
    %}
    
    %{
    % This is the test domain for bunny, which does not work in WoI4Dirichlet
    omega(2).center = [-1.4, 0.7];
    omega(2).radius = 0.2;
    %}
    D.omega = omega;
    D.uniform_pdf = reshape([omega.uniform_pdf], [], 1);
    D.sigma = conductivity_const;
    D.dim = dim; D.N = N;
    D.uniform_mesh_sampler = uniform_mesh_sampler(dim);
    D.uniform_sphere_sampler = @uniform_sphere_sampler;
    D.mesh_outward_normal = mesh_outward_normal;
    D.mesh_surface_measure = mesh_surface_measure;
end