%
% cv_train(method, XY_train_cell, XY_t, center, options)
% Implementation of different DG methods and tuning with crossvalidation
%
% INPUT:
%   X_s_cell          - cell of (n_s*d) matrices, each matrix corresponds to the instance features of a source domain
%   Y_s_cell          - cell of (n_s*1) matrices, each matrix corresponds to the instance labels of a source domain
%   X_t               - (n_t*d) matrix, rows correspond to instances and columns correspond to features
%   Y_t               - (n_t*1) matrix, each row is the class label of corresponding instances in X_t
%   options (optional)- options.center:  if centeralize the features in each domain
%                       options.params:  tuning paramters for a certan method, see SM
%                       options.verbose: if true, show the validation accuracy of each parameter setting
%                       options.X_v:     validation X data, matrix
%                       options.Y_v:     validation Y data, matrix
%                       options.valida_p:the ratio of validation set, confilict with X.v and Y.v
%
% OUTPUT:
%   Mdl               - Model
%   test_accuracy     - test accuracy of target instances
%   predicted_labels  - predicted labels of target instances
%   Zs                - Projected features of source domain
%   Ys                - class vecto
%   Ds                - domain vector
%   Zt                - Projected features of target domain


function [Mdl, Acc, pre_labels, Zs, Ys, Ds, Zt] = cv_train(method, X_s_cell, Y_s_cell, X_t, Y_t, options)

%% default parameters
switch nargin
    case 5
        options = struct('center', 0, 'verbose', 0);
    case 6
        if ~isfield(options, 'center')  options.center = 0;  end
        if ~isfield(options, 'verbose') options.verbose = 0; end
end
num_source = length(X_s_cell); % the number of source domain
%% Set Parameters
inner_scale = 1;
inner_options = struct();
inner_verbose = [];
switch method
    case 'DFDG' 
        if ~isfield(options, 'params')
            options.params = struct('gamma', [1e-3 1e-2 1e-1 1 1e1 1e2 1e3 1e4 1e5 1e6], ...
                'bdw', [0.1, 0.5, 1, 2, 5, 10], ...
                 'opt', [1], 'classifier', [1], ...,
                 'q', [2, 3, 5, 8, 10 , 20], 'cost', [1000,400,300,200,100,75,50,10,1,0.1]);
        end
        [G, B, O, C] = ndgrid(options.params.gamma, options.params.bdw, options.params.opt, options.params.classifier);
        params_grid = [G(:), B(:), O(:), C(:)];
        verbose_l = 'gamma: %f, bdw: %f, '; cut_off = 2;
        if options.params.classifier == 1
            inner_options = struct('q', options.params.q,  'cost', []); inner_verbose = 'q: %f, ';
            inner_scale = length(inner_options.q);
        else
            inner_options = struct('q', options.params.q, 'cost', options.params.cost); inner_verbose = 'q: %f, cost: %f, ';
            inner_scale = length(inner_options.q) * length(inner_options.cost);
        end
end

%% Centeralize

if options.center
    for i = 1:num_source
        X_s_cell{i} = X_s_cell{i} - mean(X_s_cell{i});
    end
    X_t = X_t - mean(X_t);
    if isfield(options, 'X_v')
        options.X_v = options.X_v - mean(options.X_v);
    end
end

if isfield(options, 'valid_p')
    valid_p = options.valid_p;
    X_v = [];
    Y_v = [];
    D_v = [];
    X_o_cell = X_s_cell;
    Y_o_cell = Y_s_cell;
    for idx = 1:length(X_s_cell)
        n_s = length(X_s_cell{idx});
        [~, n_idx] = datasample(Y_s_cell{idx}(:, end), round(valid_p * n_s));
        X_s_cell{idx} = X_o_cell{idx}(setdiff(1:n_s, n_idx), :);
        Y_s_cell{idx} = Y_o_cell{idx}(setdiff(1:n_s, n_idx));
        X_v = [X_v; X_o_cell{idx}(n_idx, :)];
        Y_v = [Y_v; Y_o_cell{idx}(n_idx)];
        D_v = [D_v; idx * ones(length(n_idx), 1)];
    end
    options.X_v = X_v;
    options.Y_v = Y_v;
end


%% Tunning

if options.verbose
    fprintf(['Method: ', method, ' start CV searching! \n'] );
end

[pL, num_para] = size(params_grid);

ttn = length(cat(1, Y_s_cell{:}));
cv_accuracy = zeros(pL,1);

% cv_p initial
if isfield(options, 'X_v')
    [Mdl, ~, ~, ~, ~, inner_output] = trainModel(method, X_s_cell, Y_s_cell, options.X_v, options.Y_v, params_grid(1,:), inner_options, [], D_v);
else
    Mdl = [];
end

inner_parameter = repmat(inner_output, [pL, 1]);

parfor ip = 1:pL
    if isfield(options, 'X_v') % use validation set
        [~, acc, ~, ~, ~, inner_output] = trainModel(method, X_s_cell, Y_s_cell, options.X_v, options.Y_v, params_grid(ip,:), inner_options, Mdl, D_v);
        cv_accuracy(ip) = acc;
        inner_parameter(ip) = inner_output;
    else    % cv, take one as validation, the rest for traning
        acc_n = 0;
        for idxs = 1:num_source
            tr_idx = setdiff([1:num_source], idxs); % idx for training domain
            cur_X_s_cell = X_s_cell(tr_idx); cur_Y_s_cell = Y_s_cell(tr_idx); % train
            cur_X_v      = X_s_cell{idxs};   cur_Y_v      = Y_s_cell{idxs};   % valid
            [~, acc] = trainModel(method, cur_X_s_cell, cur_Y_s_cell, cur_X_v, cur_Y_v, params_grid(ip,:), inner_options, Mdl, D_v);
            acc_n = acc_n + acc * length(cur_Y_v);
        end
        cv_accuracy(ip) = acc_n/ttn;
    end
    
    if options.verbose
        fprintf('\t %d of %d tuning, acc: %f \n', [inner_scale * ip, inner_scale * pL, cv_accuracy(ip)]);
        % fprintf(['\t' verbose_l 'acc: %f \n'], [params_grid(ip,:) cv_accuracy(ip)]);
    end
    
end

if isfield(options, 'valid_p')
   X_s_cell = X_o_cell;
   Y_s_cell = Y_o_cell;
end

[max_acc, max_idx] = max(cv_accuracy);
best_para = params_grid(max_idx, :);
[Mdl, acc, pre_labels, Zs, Zt, ~, bst_parameter] = trainModel(method, X_s_cell, Y_s_cell, X_t, Y_t, best_para, inner_parameter(max_idx), [], ones(size(Y_t)));

Acc = [acc, max_acc];
Ys = cat(1, Y_s_cell{:});
Ds = [];
for i = 1:num_source
    Ds = [Ds; i * ones(length(Y_s_cell{i}), 1)];
end
fprintf(['Method: ', method, '. best parameters:' verbose_l inner_verbose 'CV acc: %f, acc on target: %f \n'], [best_para(1:cut_off) bst_parameter max_acc  acc]);

end