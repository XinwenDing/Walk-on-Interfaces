%% define domain
close all; clear;
fprintf("Device: %s\n", gpuDevice().Name);

currentFolder = pwd;
addpath(genpath(fullfile(currentFolder, "src")));
addpath(genpath(fullfile(currentFolder, "lib")));

dim = 3;
DOMAIN_NAME = "example5";
DOMAIN_PATH = "domains/" + DOMAIN_NAME;

mesh_outward_normal = define_mesh_outward_normal(dim);
mesh_surface_measure = define_mesh_surface_measure(dim);

cd(DOMAIN_PATH);
[V, F, domain_layout, root_id] = load_domain_info(DOMAIN_NAME);
cd(currentFolder);

N = length(V);
% unit: cm/s
conductivity_const = [0.2, 0.05, 0.03, 0.03, 0.03, 0.03, 0.005];
[omega, D] = define_domain(V, F, dim, N, conductivity_const, mesh_outward_normal, mesh_surface_measure);
hierarchy_matrix = get_hierarchy_matrix(domain_layout, root_id, N);
interface_weight = get_interface_weight(domain_layout, N, D.sigma);

%% define BC and interface condition
k1 = 1 / 3;
k2 = k1;
a1 = 1.2;
a2 = -a1;
amp = 1;
[xmin, xmax] = deal(-2, 2);
[ymin, ymax] = deal(-1.5, 1.5);
zmin = -2;
height = @(x) (0 <= x & x <= a1) .* (k1 .* x) + (x > a1) .* (k1 .* a1) + (x < 0 & x >= a2) .* (k2 .* x) + (x < a2) .* (k2 .* a2);
b1 = @(x, n) ( abs(x(:, 3, :) - height(x(:, 1, :))) < 1e-9 ) .* ( amp * sin(pi * x(:, 1, :) / xmax) .* cos(pi * x(:, 2, :) / ymax / 2) );
b2 = @(x, n) zeros(size(x, 1), 1, size(x, 3));
b3 = @(x, n) zeros(size(x, 1), 1, size(x, 3));
b4 = @(x, n) zeros(size(x, 1), 1, size(x, 3));
b5 = @(x, n) zeros(size(x, 1), 1, size(x, 3));
b6 = @(x, n) zeros(size(x, 1), 1, size(x, 3));
b7 = @(x, n) zeros(size(x, 1), 1, size(x, 3));

b = {b1; b2; b3; b4; b5; b6; b7};
[b, scaling_coeff] = scale_b(domain_layout, b, D.N, D.sigma);
fprintf("Boundary and interface condition defined.\n")

%% define mesh for the cutting plane of interest
query_points = load("query_points/example5_query_points.mat").query_points;
query_point_num = size(query_points, 1);

query_points = gpuArray(query_points);
fprintf("query points defined \n")
disp("plane query point num"); disp(query_point_num);

%% WoI estimation
Markov_tree_height = 5;
schedule_per_batch = 1e7;
batch_num = 4; 
hierarchy_matrix = gpuArray(hierarchy_matrix);

repeat_exp_time = 1;
query_point_num_per_batch = 12; % use 6 for uniform
total_query_point_batch_idx = ceil(query_point_num / query_point_num_per_batch);
shuffled_query_point_idx = randperm(query_point_num);
ux_checkpoint_log = zeros(query_point_num, 1);
grad_ux_checkpoint_log = zeros(query_point_num, dim);

interface_schedule = []; sign_schedule = [];
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

        ux_checkpoint_log(x_idx, :) = ux;
        grad_ux_checkpoint_log(x_idx, :) = grad_ux;
        tend = toc(tstart);
        fprintf("estimation for the batch finishes within %f seconds \n", tend);
    end
end