function [rand_intersections, rand_edgeidx, intersection_normal] = find_random_intersection(omega, rayidx_edgeidx_t, intersection_count, origins, directions)
    V = omega.V;
    F = omega.F;
    [~, sort_idx] = sort(rayidx_edgeidx_t(:,1));
    sorted_edgeidx_t = rayidx_edgeidx_t(sort_idx, 2 : end);
    %sorted_edgeidx = sorted_edgeidx_t(:,1);
    %sorted_t = sorted_edgeidx_t(:,2);
    if any(intersection_count == 0)
        disp("intersection_count")
        fail_idx = find(intersection_count == 0);
        fprintf("omega id = %d \n", omega.id);
        fail_origins = []; fail_directions = []; 
        fail_origins = [fail_origins; origins(fail_idx(1:10), :)]; 
        fail_directions = [fail_directions; directions(fail_idx(1:10), :)];
        %disp("fail origin"); disp(origins(fail_idx, :));
        %disp("fail direction"); disp(directions(fail_idx, :));
        save("troubleshooting_data/sign_fail_origins.mat", "fail_origins");
        save("troubleshooting_data/sign_fail_directions.mat", "fail_directions");
        error('Error in counting intersections. Check it! \n');
    end
    rand_intersection_idx = arrayfun(@(x) randi(x), intersection_count);
    rand_intersection_idx = [0; cumsum(intersection_count(1:end-1))] + rand_intersection_idx;
    rand_edgeidx_t_mat = sorted_edgeidx_t(rand_intersection_idx, :);
    rand_edgeidx = rand_edgeidx_t_mat(:,1);
    rand_t = rand_edgeidx_t_mat(:,2);
    % edge_starts = V(F(:, 1), :), edge_ends = V(F(:, 2), :)
    % intersections = edge_starts + (edge_ends - edge_starts) * t2 = (1 - t2) * edge_starts + t2 * edge_ends
    rand_intersections = rand_t .* V(F(rand_edgeidx, 2), :) + (1 - rand_t) .* V(F(rand_edgeidx, 1), :);
    intersection_normal = omega.outward_normal(rand_edgeidx, :);
end