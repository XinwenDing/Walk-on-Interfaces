%% define domain
close all; clear;
fprintf("Device: %s\n", gpuDevice().Name);

currentFolder = pwd;
addpath(genpath(fullfile(currentFolder, "src")));

dim = 5;
N = 3;

% diagnal entries of A are 1 / a_i^2, where a_i is the length of the ith axis
omega(1).center = [1.3, zeros(1, dim-1)];
omega(1).A = [1, 0, 0, 0, 0; 0, 0.9, 0, 0, 0; 0, 0, 1.2, 0, 0; 0, 0, 0, 1.2, 0; 0, 0, 0, 0, 1.2].^2;
omega(1).U = chol(omega(1).A);      % A = U' U
omega(1).uniform_pdf = get_ellipsoid_uniform_pdf(dim, omega(1).A);
omega(1).sigma = 1.1;

omega(2).center = [1.4, zeros(1, dim-1)];
omega(2).A = [1.5, -1, 0, 0, 0; -1, 3.2, 0, 0, 0; 0, 0, 2, 0, 0; 0, 0, 0, 2, 0; 0, 0, 0, 0, 2].^2;
omega(2).U = chol(omega(2).A);      % A = U' U
omega(2).uniform_pdf = get_ellipsoid_uniform_pdf(dim, omega(2).A);
omega(2).sigma = 1.3;

omega(3).center = [1.6, zeros(1, dim-1)];
omega(3).A = [8, -2, 0, 0, 0; -2, 6, 0, 0, 0; 0, 0, 4, 0, 0; 0, 0, 0, 4, 0; 0, 0, 0, 0, 4].^2;
omega(3).U = chol(omega(3).A);
omega(3).uniform_pdf = get_ellipsoid_uniform_pdf(dim, omega(3).A);
omega(3).sigma = 0.9;

D.omega = omega;
D.uniform_pdf = reshape([omega.uniform_pdf], [], 1);
D.sigma = reshape([omega.sigma], [], 1);
D.dim = dim; D.N = N;
D.uniform_ellipsoid_sampler = @uniform_ellipsoid_sampler;

[domain_layout, root_id] = configure_domain_layout();
hierarchy_matrix = get_hierarchy_matrix(domain_layout, root_id, N);
interface_weight = get_interface_weight(domain_layout, N, D.sigma);
fprintf("Spherical domain defined.\n")

%% Define true solution (a harmonic function) and boundary/interface condition's
true_soln = @(x) vecnorm(x, 2, 2).^(2 - dim);
fprintf("True solution(a harmonic function) defined.\n")
% grad_u = [3x^2 - 3y^2, -6xy], n = ([x, y] - center) / radius
b1 = @(x, n) omega(1).sigma * (2 - dim) * sum(x .* n, 2) ./ (vecnorm(x, 2, 2).^dim);
b2 = @(x, n) (omega(1).sigma - omega(2).sigma) * (2 - dim) * sum(x .* n, 2) ./ (vecnorm(x, 2, 2).^dim);
b3 = @(x, n) (omega(2).sigma - omega(3).sigma) * (2 - dim) * sum(x .* n, 2) ./ (vecnorm(x, 2, 2).^dim);
b = {b1; b2; b3};
[b, scaling_coeff] = scale_b(domain_layout, b, D.N, D.sigma);

%% define query points
query_point_num = 1e4;
query_points = uniformly_sample_query_points(dim, D.omega(1), query_point_num-1);
query_points = [query_points; D.omega(1).center];
ground_truth = true_soln(query_points);

%% WoI estimation
Markov_tree_height = 7;
schedule_per_batch = 1e7;
batch_num = 4;
hierarchy_matrix = gpuArray(hierarchy_matrix);

repeat_exp_time = 1;
query_point_num_per_batch = 10; % use 6 for uniform
total_query_point_batch_idx = ceil(query_point_num / query_point_num_per_batch);
shuffled_query_point_idx = randperm(query_point_num);
ux_checkpoint_log = zeros(query_point_num, 1);
grad_ux_checkpoint_log = zeros(query_point_num, dim);

for exp_time = 1 : repeat_exp_time
    fprintf("experiment: %d \n", exp_time);
    for query_point_batch_idx = 1 : total_query_point_batch_idx
        tstart = tic;
        x_start_idx = (query_point_batch_idx - 1) * query_point_num_per_batch + 1;
        x_end_idx = min(query_point_batch_idx * query_point_num_per_batch, query_point_num);
        x_idx = shuffled_query_point_idx(x_start_idx : x_end_idx);
        fprintf("estimating the %d th - %d th shuffled query points\n", x_start_idx, x_end_idx);
        x = query_points(x_idx, :);
        % ux: query_point_num_per_batch * 1 * checkpoint_num | grad_ux: query_point_num_per_batch * 2 * checkpoint_num
        [ux, grad_ux] = estimate_on_high_dim(x, b, D, hierarchy_matrix, interface_weight, Markov_tree_height, schedule_per_batch, batch_num);
        disp("error (take diff since this is a Neumann problem -- we need a reference point):");
        disp(abs(diff(ux(:, :, end) - ground_truth(x_idx))))
        ux_checkpoint_log(x_idx, :) = ux;
        grad_ux_checkpoint_log(x_idx, :) = grad_ux;
        tend = toc(tstart);
        fprintf("estimation for experiment %d finishes within %f seconds \n", exp_time, tend);
    end
end