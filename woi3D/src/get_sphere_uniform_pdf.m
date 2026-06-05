function uniform_pdf = get_sphere_uniform_pdf(dim, radius)
    half_dim = dim / 2;
    surface_measure = (radius ^ (dim - 1)) * (2 * pi ^ (half_dim) / gamma(half_dim));
    uniform_pdf = 1 / surface_measure;
end