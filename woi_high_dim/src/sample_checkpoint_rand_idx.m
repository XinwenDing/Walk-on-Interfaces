function checkpoint_rand_idx = sample_checkpoint_rand_idx(convergence_checkpoint, schedule_size)
    checkpoint_rand_idx = {};
    for i = 1 : length(convergence_checkpoint)
        checkpoint_rand_idx{i}  = randperm(schedule_size, convergence_checkpoint(i));
    end
end