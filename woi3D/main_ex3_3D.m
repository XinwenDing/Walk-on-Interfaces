%% define domain
close all; clear;
fprintf("Device: %s\n", gpuDevice().Name);

currentFolder = pwd;
addpath(genpath(fullfile(currentFolder, "src")));
addpath(genpath(fullfile(currentFolder, "lib")));

dim = 3;
N = 2;

omega(1).center = [0, 0, 0];
omega(1).radius = 1;
omega(1).uniform_pdf = get_sphere_uniform_pdf(dim, omega(1).radius);
omega(1).sigma = 1;


omega(2).center = [0, 0, 0];
omega(2).radius = 0.4;
omega(2).uniform_pdf = get_sphere_uniform_pdf(dim, omega(2).radius);
omega(2).sigma = 0.5;

D.uniform_pdf = reshape([omega.uniform_pdf], [N, 1]);
D.sigma = reshape([omega.sigma], [N, 1]);
D.omega = omega;
D.dim = dim; 
D.N = N;
D.uniform_sphere_sampler = @uniform_sphere_sampler;

[domain_layout, root_id] = configure_domain_layout();
hierarchy_matrix = get_hierarchy_matrix(domain_layout, root_id, N);
interface_weight = get_interface_weight(domain_layout, N, D.sigma);
fprintf("Spherical domain defined.\n")

%% Define true solution (a harmonic function) and boundary/interface condition's
amp = 5;
sigma_ratio = D.sigma(2) / D.sigma(1);
true_soln = @(x) ex3_true_soln3D(x, omega(2).radius, sigma_ratio, amp);
fprintf("True solution(a harmonic function) and BC defined.\n")
% later we will switch to the following bi(x, n),  where x is the coordinate, n is the unit outward normal.
b1 = @(x, n) amp * cos(atan2(x(:,2,:), x(:,1,:))) .* sin(acos(x(:, 3, :) ./ vecnorm(x, 2, 2)));
b2 = @(x, n) zeros(size(x, 1), 1, size(x, 3));
b = {b1; b2};
[b, scaling_coeff] = scale_b(domain_layout, b, D.N, D.sigma);

%% define boundary query points
query_point_num = 7000;
dir = randn(query_point_num, dim);
dir = dir ./ vecnorm(dir, 2, 2);
query_points = omega(1).radius * rand(query_point_num, 1).^(1/dim) .* dir;       % Uniform volume
query_points = gpuArray(query_points);

ground_truth = true_soln(query_points);
fprintf("query points and ground truth defined \n")

%% WoI estimation
Markov_tree_height = 6;
schedule_per_batch = 1e7;
batch_num = 1;
hierarchy_matrix = gpuArray(hierarchy_matrix);

repeat_exp_time = 1;
query_point_num_per_batch = 10;
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
        [ux, grad_ux] = estimate_on_mesh(x, b, D, hierarchy_matrix, interface_weight, Markov_tree_height, schedule_per_batch, batch_num);
        disp("error (take diff since this is a Neumann problem -- we need a reference point):");
        disp(abs(diff(ux(:, :, end) - ground_truth(x_idx))));
        ux_checkpoint_log(x_idx, :) = ux;
        grad_ux_checkpoint_log(x_idx, :) = grad_ux;
        tend = toc(tstart);
        fprintf("estimation for the batch finishes within %f seconds \n", tend);
    end
end