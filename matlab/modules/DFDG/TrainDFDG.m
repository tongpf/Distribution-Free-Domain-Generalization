function [Mdl, acc, pre_labels, Zs, Zt] = TrainDFDG(cur_X_s_cell, cur_Y_s_cell, X_v, Y_v, cur_params, q_array, cost_array, Mdl, D_v)

cellpara = num2cell(cur_params);
[cur_gamma, cur_bdw, opt, classifier] = cellpara{:}; % opt = 1 for EigenAdj, opt = 2 for CovFilter;
% classifier = 1 for knn, = 2 for svm

n_domain = length(cur_X_s_cell); 
D_s = [];
for s = 1:n_domain
    D_s = [D_s; repmat(s, [length(cur_X_s_cell{s}), 1])];
end

% reshape the data
X_s = cat(1, cur_X_s_cell{:});
Y_s = cat(1, cur_Y_s_cell{:});

% ----- ----- distance matrix
if ~isempty(Mdl)
    dist_s_s = Mdl.dist_s_s;
    sgm_s = Mdl.sgm_s;
    dist_s_v = Mdl.dist_s_v;
    sgm_v = Mdl.sgm_v;
    dist_v_v = Mdl.dist_v_v;
    sgm_vv = Mdl.sgm_vv;
    md_array = Mdl.md_array;
else
    dist_s_s = pdist2(X_s, X_s);
    dist_s_s = dist_s_s.^2;
    sgm_s = compute_width(dist_s_s);
    
    dist_s_v = pdist2(X_s, X_v);
    dist_s_v = dist_s_v.^2;
    sgm_v = compute_width(dist_s_v);
    
    dist_v_v = pdist2(X_v, X_v);
    dist_v_v = dist_v_v.^2;
    sgm_vv = compute_width(dist_v_v);
    md_array = [];
end
n_s = size(X_s, 1);
n_v = size(X_v, 1);

% ----- ----- kernel matrix
K_s_s = exp(-dist_s_s./(cur_bdw * sgm_s * sgm_s));
K_s_v = exp(-dist_s_v./(cur_bdw * sgm_v * sgm_v));
K_v_v = exp(-dist_v_v./(cur_bdw * sgm_vv * sgm_vv));
[K_s_s, K_s_v] = transfer_kernel(D_s, D_v, K_s_s, K_s_v, K_v_v);

Jns = (1/n_s) * ones(n_s, n_s);
Jnv = (1/n_v) * ones(n_v, n_v);

K_s_s_bar = K_s_s - Jns * K_s_s - K_s_s * Jns + Jns * K_s_s * Jns;
K_s_v_bar = K_s_v - Jns * K_s_v - K_s_v * Jnv + Jns * K_s_v * Jnv;



switch opt
    case 1
        [F, Q] = EigAdj(K_s_s_bar, cur_X_s_cell, cur_Y_s_cell, cur_gamma);
    case 2
        [F, Q] = CovFilter(K_s_s_bar, cur_X_s_cell, cur_Y_s_cell, cur_gamma);
end

[B, w] = DFDG_trans(F, Q, K_s_s, q_array);

if isempty(md_array)
    [acc, pre_labels, Zs, Zt, md_array, bst_parameter] = DFDG_test(B, w, K_s_s_bar, K_s_v_bar, Y_s, Y_v, classifier, q_array, cost_array);
else
    [acc, pre_labels, Zs, Zt, md_array, bst_parameter] = DFDG_test(B, w, K_s_s_bar, K_s_v_bar, Y_s, Y_v, classifier, q_array, cost_array, md_array);
end

Mdl = struct('dist_s_s', dist_s_s, 'dist_s_v', dist_s_v, 'sgm_s', sgm_s, 'sgm_v', sgm_v, 'md_array', md_array, 'bst_parameter', bst_parameter, ...
    'dist_v_v', dist_v_v, 'sgm_vv', sgm_vv);

end