% settings: the xlsx name of settings
% num_domain: the number of domains
% num_class: the number of class, fixed at 3, plz
% prior_set: the settings of prior probability, see Readme
% sigma_set: the settings of the standard deviations, see Readme
% base_samples_size: the sample size for a class, default 50

function XY_cell = data_generate(settings, num_domain, num_class, prior_set, sigma_set, base_samples_size)
switch nargin
    case 2
        prior_set = 'aaaa'; sigma_set = [1, 1, 1, 1]; base_samples_size = 200;
    case 3
        sigma_set = [1, 1, 1, 1]; base_samples_size = 200;
    case 4
        base_samples_size = 200;
end

sets = xlsread(settings);
k = 1;
for s = 1:num_domain
    switch prior_set(s)
        case 'a'
            sample_sizes = round([1/3, 1/3, 1/3] * base_samples_size * num_class);
        case 'b'
            sample_sizes = round([1/6, 1/3, 1/2] * base_samples_size * num_class);
        case 'c'
            sample_sizes = round([1/2, 1/3, 1/6] * base_samples_size * num_class);
        case 'd'
            sample_sizes = round([1/6, 2/3, 1/6] * base_samples_size * num_class);
        case 'e'
            sample_sizes = round([2/5, 1/5, 2/5] * base_samples_size * num_class);
    end
    X = []; Y = [];
    for i = 1:num_class
        switch sigma_set(s)
            case 1
                X = [X; mvnrnd(sets(k, 3:4), diag(sets(k, 5:6)).^2 ,sample_sizes(i))];
            case 2
                X = [X; mvnrnd(sets(k, 3:4), diag(sets(k, 7:8)).^2 ,sample_sizes(i))];
        end
        Y = [Y; repmat(i, [sample_sizes(i),1])];
        k = k + 1;
    end
    XY_cell{s} = [X, Y];
end

end