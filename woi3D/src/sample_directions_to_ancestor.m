function directions = sample_directions_to_ancestor(origins)
    directions = randn(size(origins), 'gpuArray');
    directions = directions ./ vecnorm(directions, 2, 2);
    
    %directions = -sign(sum(directions .* origins_outward_normal, 2)) .* directions;
    directions_norm = vecnorm(directions, 2, 2);
    disp(directions_norm(directions_norm == 0));
end