%% define domain
close all; clear;
fprintf("Device: %s\n", gpuDevice().Name);

currentFolder = pwd;
addpath(genpath(fullfile(currentFolder, "src")));
addpath(genpath(fullfile(currentFolder, "lib")));

dim = 3;
DOMAIN_NAME = "example4";
DOMAIN_PATH = "domains/" + DOMAIN_NAME;

mesh_outward_normal = define_mesh_outward_normal(dim);
mesh_surface_measure = define_mesh_surface_measure(dim);

cd(DOMAIN_PATH);
[V, F, domain_layout, root_id] = load_domain_info(DOMAIN_NAME);
cd(currentFolder);

N = length(V);
conductivity_const = [1.2; 0];
[omega, D] = define_domain(V, F, dim, N, conductivity_const, mesh_outward_normal, mesh_surface_measure);
D.omega(1).center = [0, 0, 0];
D.omega(1).radius = 2;
hierarchy_matrix = get_hierarchy_matrix(domain_layout, root_id, N);
interface_weight = get_interface_weight(domain_layout, N, D.sigma);

%% Define true solution (a harmonic function) and boundary/interface condition's
biem_soln = load("numerical_soln/example4_biem.mat").true_soln;
% grad_u = [2x, 2y, -4z], n = ([x, y, z] - center) / radius
amp = 10;
b1 = @(x, n) amp * cos(atan2(x(:, 2, :), x(:, 1, :))).^9 .* sin(acos(x(:, 3, :) ./ vecnorm(x, 2, 2))).^9;
b2 = @(x, n) zeros(size(x, 1), 1, size(x, 3));
b = {b1; b2};
[b, scaling_coeff] = scale_b(domain_layout, b, D.N, D.sigma);
fprintf("True solution(a harmonic function) defined.\n")

%% define query points
query_points = load("query_points/example4_query_points.mat").query_points;
query_point_num = size(query_points, 1);
[test_x_start, test_x_end] = deal(1, length(query_points));
x = query_points(test_x_start:test_x_end, :);   
x = gpuArray(x);
true_ux = biem_soln(test_x_start:test_x_end);
fprintf("query points defined \n")

%% WoI estimation
Markov_tree_height = 6;
schedule_per_batch = 1e7;
batch_num = 1; 
hierarchy_matrix = gpuArray(hierarchy_matrix);

repeat_exp_time = 1;
query_point_num_per_batch = 12;
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
        disp(abs(diff(ux(:, :, end) - biem_soln(x_idx))));
        ux_checkpoint_log(x_idx, :) = ux;
        grad_ux_checkpoint_log(x_idx, :) = grad_ux;
        tend = toc(tstart);
        fprintf("estimation for the batch finishes within %f seconds \n", tend);
    end
end