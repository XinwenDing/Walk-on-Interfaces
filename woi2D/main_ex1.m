%% define domain
close all; clear;
fprintf("Device: %s\n", gpuDevice().Name);

currentFolder = pwd;
addpath(genpath(fullfile(currentFolder, "src")));
addpath(genpath(fullfile(currentFolder, "lib")));

dim = 2;
DOMIAN_NAME = "example1"; % a circle containing a circle and a curved star
DOMIAN_PATH = "domains/" + DOMIAN_NAME;

mesh_outward_normal = define_mesh_outward_normal(dim);
mesh_surface_measure = define_mesh_surface_measure(dim);

cd(DOMIAN_PATH);
[V, F, domain_layout, root_id] = load_domain_info(DOMIAN_NAME);
cd(currentFolder);

N = length(V);
conductivity_const = [1.5; 0.5; 1.1];
[omega, D] = define_domain(V, F, dim, N, conductivity_const, mesh_outward_normal, mesh_surface_measure);
hierarchy_matrix = get_hierarchy_matrix(domain_layout, root_id, N);
interface_weight = get_interface_weight(domain_layout, N, D.sigma);
% specify center and radius for circular domain if you want. This will make the code faster.
D.omega(1).center = [0.5, -0.5]; % center and radius will only be used to find proper query points
D.omega(1).radius = 2;
D.omega(2).center = [0, -1.2]; % center and radius will only be used to find proper query points
D.omega(2).radius = 0.6;

%% Define true solution (a harmonic function) and boundary/interface condition's
true_soln = @(x) x(:, 1, :).^3 - 3 * x(:, 1, :) .* x(:, 2, :).^2;
fprintf("True solution(a harmonic function) and BC defined.\n")
% grad_u = [3x^2 - 3y^2, -6xy], n = ([x, y] - center) / radius
b1 = @(x, n) D.omega(1).sigma * sum([3 * (x(:,1,:).^2 - x(:,2,:).^2), - 6 * x(:,1,:) .* x(:,2,:)] .* n, 2);
b2 = @(x, n) (D.omega(1).sigma - D.omega(2).sigma) * sum([3 * (x(:,1,:).^2 - x(:,2,:).^2), - 6 * x(:,1,:) .* x(:,2,:)] .* n, 2);
b3 = @(x, n) (D.omega(1).sigma - D.omega(3).sigma) * sum([3 * (x(:,1,:).^2 - x(:,2,:).^2), - 6 * x(:,1,:) .* x(:,2,:)] .* n, 2);
b = {b1; b2; b3};
[b, scaling_coeff] = scale_b(domain_layout, b, D.N, D.sigma);

%% define mesh
nx = 100; dxx = 2 * D.omega(1).radius / nx;
ny = 100; dyy = 2 * D.omega(1).radius / ny;
xx = -1.5 : dxx : 2.5;
yy = -2.5 : dyy : 1.5;
[X, Y] = meshgrid(xx, yy);

% Step 2: Compute mask for points inside the unit circle
mask = (X - D.omega(1).center(1)).^2 + (Y - D.omega(1).center(2)).^2 <= D.omega(1).radius^2;

% Step 3: Extract coordinates of valid points
x_inside = X(mask);
y_inside = Y(mask);
query_points = [x_inside(:), y_inside(:)]; % N_points × 2 matrix
query_point_num = size(query_points, 1);
ground_truth = true_soln(query_points);
disp(query_point_num)

%% WoI estimation
Markov_tree_height = 4;
schedule_per_batch = 1e7;
batch_num = 1;
hierarchy_matrix = gpuArray(hierarchy_matrix);

repeat_exp_time = 1;
query_point_num_per_batch = 5;
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
        disp(abs(diff(ux(:, :, end)) - diff(ground_truth(x_idx))));
        ux_checkpoint_log(x_idx, :) = ux;
        grad_ux_checkpoint_log(x_idx, :) = grad_ux;
        tend = toc(tstart);
        fprintf("estimation for experiment %d finishes within %f seconds \n", exp_time, tend);
    end
end
