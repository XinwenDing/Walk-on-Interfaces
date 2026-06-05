function [rand_intersections, rand_triIdx, intersection_normal] = find_random_intersection(omega, rayidx_triIdx_u_v, intersection_count, origins, directions)
    V = omega.V;
    F = omega.F;
    [~, sort_idx] = sort(rayidx_triIdx_u_v(:,1));
    sorted_triIdx_u_v = rayidx_triIdx_u_v(sort_idx, 2 : 4);
    %sorted_edgeidx = sorted_edgeidx_t(:,1);
    %sorted_t = sorted_edgeidx_t(:,2);
    if any(intersection_count == 0)%~all(intersection_count == 1)
        disp("trouble shooting starts")
        fail_idx = find(intersection_count == 0);
        %fprintf("# of intersections: %d, source idx = %d target idx = %d, t1 = %f, t2 = %f, mask = %d \n", intersection_count(fail_idx), rayidx_edgeidx_t(fail_idx, 1), ...
        %    rayidx_edgeidx_t(fail_idx, 2), rayidx_edgeidx_t(fail_idx, 4), rayidx_edgeidx_t(fail_idx, 3), rayidx_edgeidx_t(fail_idx, 5));
        %load("troubleshooting_data/fail_origins.mat", "fail_origins");
        %load("troubleshooting_data/fail_directions.mat", "fail_directions"); 
        fail_origins = gather(origins(fail_idx, :));
        fail_directions = gather(directions(fail_idx, :));
        %disp("fail origin"); disp(origins(fail_idx, :));
        %disp("fail direction"); disp(directions(fail_idx, :));
        disp(omega)
        save("troubleshooting_data/fail_origins.mat", "fail_origins");
        save("troubleshooting_data/fail_directions.mat", "fail_directions");
    end
    rand_intersection_idx = arrayfun(@(x) randi(x), intersection_count);
    rand_intersection_idx = [0; cumsum(intersection_count(1:end-1))] + rand_intersection_idx;
    rand_triIdx_u_v_mat = sorted_triIdx_u_v(rand_intersection_idx, :);
    rand_triIdx = rand_triIdx_u_v_mat(:, 1);
    rand_u = rand_triIdx_u_v_mat(:, 2);
    rand_v = rand_triIdx_u_v_mat(:, 3);
    % edge_starts = V(F(:, 1), :), edge_ends = V(F(:, 2), :)
    % intersections = edge_starts + (edge_ends - edge_starts) * t2 = (1 - t2) * edge_starts + t2 * edge_ends
    rand_intersections = rand_u .* V(F(rand_triIdx, 2), :) + rand_v .* V(F(rand_triIdx, 3), :) + (1 - rand_u - rand_v) .* V(F(rand_triIdx, 1), :);
    % start from here
    intersection_normal = omega.outward_normal(rand_triIdx, :);
end