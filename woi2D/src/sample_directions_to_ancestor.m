function directions = sample_directions_to_ancestor(origins)
    directions = randn(size(origins), 'gpuArray');
    %{
    directions_norm = vecnorm(directions, 2, 2);
    if ~isnumeric(directions_norm)
        fprintf("there exists zero vector \n")
    end
    %}
    directions = directions ./ vecnorm(directions, 2, 2);
end