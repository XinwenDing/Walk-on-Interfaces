function directions = sample_directions_to_self(origins, origins_outward_normal, dim)
    directions = randn(size(origins), 'gpuArray');
    directions = directions ./ vecnorm(directions, 2, 2);
    %%{
    parallel_to_face_idx = find(abs(sum(directions .* origins_outward_normal, 2)) < 1e-9);
    parallel_outward_normal = origins_outward_normal(parallel_to_face_idx, :);
    while ~isempty(parallel_to_face_idx)
        loop_count = 1;
        fprintf("while loop count %d: ray directions parallel to face \n", loop_count);
        new_directions = randn(length(parallel_to_face_idx), dim, 'gpuArray');
        directions(parallel_to_face_idx, :) = new_directions ./ vecnorm(new_directions, 2, 2);

        parallel_to_face = find(abs(sum(new_directions .* parallel_outward_normal, 2)) < 1e-9);
        parallel_to_face_idx = parallel_to_face_idx(parallel_to_face);
        parallel_outward_normal = parallel_outward_normal(parallel_to_face, :);
        loop_count = loop_count + 1;
    end
    %}
    %directions = -sign(sum(directions .* origins_outward_normal, 2)) .* directions;
    %directions_norm = vecnorm(directions, 2, 2);
    %disp(directions_norm(directions_norm == 0));
end