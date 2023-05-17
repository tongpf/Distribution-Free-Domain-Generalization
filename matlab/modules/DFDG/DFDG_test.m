function [acc, pre_labels, Zs, Zt, md_array, bst_parameter] = DFDG_test(B, w, K_s, K_t, Y_s, Y_t, mdl, eig_ratio, cost_array, md_array)
%
%
% INPUT:
%   B           - transformation matrix
%   w           - eigenvalues
%   K_s         - (n*n) kernel matrix of all source instances
%   K_t         - (n*n_t) kernel matrix between source and target instances
%   Y_s         - (n*1) matrix of class labels of all source instances
%   Y_t         - (n_t*1) matrix of class labels of target instances
%   eig_ratio   - dimension of the transformed space
%   mdl         - classifier, 1 for 'knn'(default) or 2 for 'svm',
%   md_array    - (optional) the half distance
% OUTPUT:
%   acc         - test accuracy of target instances
%   pre_labels  - predicted labels of target instances
%   Zs          - projected all source instances
%   Zt          - projected target instances
%   md_array    - the half distance
%   bst_parameter
%

% ----------------------------------------------------------------------

if nargin == 9
    md_array = [];
end

[n_s, n_t] = size(K_t);

n_eigs = zeros(size(eig_ratio));
for i = 1:length(eig_ratio)
    if eig_ratio(i) < 1
        n_eigs(i) = max(1, floor(eig_ratio(i) * n_s));
    else
        n_eigs(i) = eig_ratio(i);
    end
end

acc = 0;

if mdl == 1
    for i = 1:length(n_eigs)
        Zt = K_t' * B(:, 1:n_eigs(i));
        Zs = K_s' * B(:, 1:n_eigs(i));
        Mdl = fitcknn(Zs, Y_s, 'NumNeighbors', 1);
        t_pre_labels = predict(Mdl, Zt);
        if acc < mean(t_pre_labels == Y_t)
            acc = mean(t_pre_labels == Y_t);
            pre_labels = t_pre_labels;
            bst_parameter = [eig_ratio(i)];
        end
    end
    md_array = [];
else
    %     Cost = 1 * ones(2) - 1 * eye(2);
    %     t = templateSVM('KernelFunction','linear', 'Cost', Cost);
    %     Mdl = fitcecoc(Zs, Y_s, 'Learners', t);
    if isempty(md_array)
        md_array = zeros(size(eig_ratio));

        for i = 1:length(n_eigs)
            Zt = K_t' * B(:, 1:n_eigs(i));
            Zs = K_s' * B(:, 1:n_eigs(i));
            dist_s_s = pdist2(Zs, Zs);
            dist_s_s = dist_s_s.^2;
            half_dist = dist_s_s-tril(dist_s_s);
            half_dist = reshape(half_dist, size(dist_s_s, 1)^2, 1);
            md_array(i) = sqrt(median(half_dist(half_dist>0)));
            for j = 1:length(cost_array)
                Cost = cost_array(j) * ones(2,2) - cost_array(j) * eye(2,2);
                t = templateSVM('KernelFunction', 'rbf', 'KernelScale', md_array(i), 'Cost', Cost);
                Mdl = fitcecoc(Zs, Y_s, 'Coding', 'onevsall', 'Learners', t);
                t_pre_labels = predict(Mdl, Zt);
                if acc < mean(t_pre_labels == Y_t)
                    acc = mean(t_pre_labels == Y_t);
                    bst_parameter = [eig_ratio(i), cost_array(j)];
                    pre_labels = t_pre_labels;
                end
            end
        end
    else
        for i = 1:length(n_eigs)
            Zt = K_t' * B(:, 1:n_eigs(i));
            Zs = K_s' * B(:, 1:n_eigs(i));
            for j = 1:length(cost_array)
                Cost = cost_array(j) * ones(2,2) - cost_array(j) * eye(2,2);
                t = templateSVM('KernelFunction', 'rbf', 'KernelScale', md_array(i), 'Cost', Cost);
                Mdl = fitcecoc(Zs, Y_s, 'Coding', 'onevsall', 'Learners', t);
                t_pre_labels = predict(Mdl, Zt);
                if acc < mean(t_pre_labels == Y_t)
                    acc = mean(t_pre_labels == Y_t);
                    bst_parameter = [eig_ratio(i), cost_array(j)];
                    pre_labels = t_pre_labels;
                end
            end
        end
    end
end


