function uniform_pdf = get_ellipsoid_uniform_pdf(dim, A)
    sample_batch_lst = [1, 1, 1, 5, 10, 20];
    sample_num = 1e7;
    [V, D] = eig(A);
    lambda = diag(D);        % lambda = 1 / axis_length^2
    %axis_length_sqr = 1 ./ lambda;
    %axis_length = 1 ./ sqrt(lambda);
    coeff = dim * gamma(dim / 2) / gamma((dim+1) / 2);
    unit_ball_volume = pi^(dim / 2) / gamma(dim/2 + 1);
    %vol = unit_ball_volume * prod(axis_length);
    vol = unit_ball_volume / sqrt(det(A));

    counter = 1;
    total_sum_fx = 0;
    while counter <= sample_batch_lst(dim)
        X = randn(sample_num, dim, 'gpuArray') / sqrt(2);
        rotated_X = (V' * X')';
        fx = sqrt(rotated_X.^2 * lambda);
        total_sum_fx = total_sum_fx + sum(fx, 1);
        mean_fx = total_sum_fx / counter / sample_num;

        surface_measure = vol * coeff * mean_fx;
        stderr_fx = std(fx) / sqrt(sample_num);
        stderr = vol * coeff * stderr_fx;
        rel_err = stderr / surface_measure;

        counter = counter + 1;
        %fprintf("counter = %d, rel_err = %f \n", counter, rel_err);
    end

    %surface_measure = vol * coeff * mean_fx;
    uniform_pdf = 1 / surface_measure;
    disp("surface measure"); disp(surface_measure);
    X = gather(X);
end