addpath(genpath('modules/'))
warning off
verbose_flag = 1; % set 0 for no verbose
%% Data Generation
num_domain = 4; % number of domain, including train and test and val(if have)
num_class = 3; % number of class, set 3 plz

prior_set = 'aaaa'; % see readme
sigma_set = [1, 1, 1, 1]; % see readme
base_samples_size = 200; %
settings = 'data_settings_4domains.xlsx';
XY_cell = data_generate(settings, num_domain, num_class, prior_set, sigma_set, base_samples_size);

%% Data Split
% target  / source domains/  valid domain
tgt_dm = [4];
src_dm = [1 2 3];

data_cell = XY_cell;
X_t = data_cell{tgt_dm(1)}(:, 1:2);
Y_t = data_cell{tgt_dm(1)}(:, 3);

% ----- prepare training data & validation data
X_s_cell = cell(1,length(src_dm));
Y_s_cell = cell(1,length(src_dm));
for idx = 1:length(src_dm)
    cu_dm = src_dm(1, idx);
    X_s_cell{idx} = data_cell{cu_dm}(:, 1:2);
    Y_s_cell{idx} = data_cell{cu_dm}(:, 3);
end
valid_p = 0.3; % take 30% for validation 

%% DFDG, EigenAdjusted, SVM
options = struct('verbose', verbose_flag, 'valid_p', valid_p);
options.params = struct('gamma', [0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100], ...
    'bdw', [0.1, 0.5, 1, 5, 10, 50, 100], ...
    'q', [2 3 4 5], ...
    'cost', [0.1, 0.5, 1, 2, 5, 10, 20, 50], 'opt', [1], 'classifier', [2]); % opt = 1 means EigenAdjusted, classifier = 2 means SVM
tic
[Mdl, acc, pre_labels, Zs, Ys, Ds, Zt] = cv_train('DFDG', X_s_cell, Y_s_cell, X_t, Y_t, options);
toc
%% DFDG, EigenAdjusted, 1nn
rng(2023)
options = struct('verbose', verbose_flag, 'valid_p', valid_p);
options.params = struct('gamma', [0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100], ...
    'bdw', [1], ...
    'q', [2, 3, 4, 5], ...
    'cost', [0.1, 0.5, 1, 2, 5, 10, 20, 50], 'opt', [1], 'classifier', [1]); % opt = 1 means EigenAdjusted, classifier = 1 means 1NN
tic
[Mdl, acc, pre_labels, Zs, Ys, Ds, Zt] = cv_train('DFDG', X_s_cell, Y_s_cell, X_t, Y_t, options);
toc
%% DFDG, CovFilter, SVM
rng(2023)
options = struct('verbose', verbose_flag, 'valid_p', valid_p);
options.params = struct('gamma', [0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100], ...
    'bdw', [0.1, 0.5, 1, 5, 10, 50, 100], ...
    'q', [2 3 4 5], ...
    'cost', [0.1, 0.5, 1, 2, 5, 10, 20, 50], 'opt', [2], 'classifier', [2]); % opt = 2 means CovFilter, classifier = 2 means SVM
tic
[Mdl, acc, pre_labels, Zs, Ys, Ds, Zt] = cv_train('DFDG', X_s_cell, Y_s_cell, X_t, Y_t, options);
toc

%% DFDG, CovFilter, 1NN
rng(2023)
options = struct('verbose', verbose_flag, 'valid_p', valid_p);
options.params = struct('gamma', [0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100], ...
    'bdw', [1], ...
    'q', [2,3,4,5], ...
    'cost', [0.1, 0.5, 1, 2, 5, 10, 20, 50], 'opt', [2], 'classifier', [1]); % opt = 2 means CovFilter, classifier = 2 means 1NN
tic
[Mdl, acc, pre_labels, Zs, Ys, Ds, Zt] = cv_train('DFDG', X_s_cell, Y_s_cell, X_t, Y_t, options);
toc

%% plot
Xs = cat(1, X_s_cell{:});
plotFeatures(Xs, Zs, Ys, Ds, X_t, Zt, Y_t)
