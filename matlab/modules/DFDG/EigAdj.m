function [F, Q] = EigAdj(K, X, Y, gamma, eps)
% 
% INPUT
%     K      - kernel matrix of data of all source domains
%     X      - cell of (n_s*d) matrices, each matrix corresponds to the instance features of a source domain
%     Y      - cell of (n_s*1) matrices, each matrix corresponds to the instance labels of a source domain
%     gamma  - hyperparameter in constructing Q
%     eps    - small perturbation, default 1e-5
%
% OUTPUT
%     F      - F in cross class discrepancy, Eq.(21)
%     Q      - Q in cross domain discrepancy, Eq.(22)
%  ----------------------------------------------------------------------

if nargin == 4
   eps = 1e-5; 
end

% number of domain
n_domain = length(X);

% labels of all domains in a vector
Y_ALL = cat(1, Y{:});
n_total = size(Y_ALL, 1);            % number of instances in all source domains
n_class = length(unique(Y_ALL));   % number of classes mush begin from one

% save class and domain index of all instances into two row vectors 
class_index = zeros(1, n_total);
domain_index = zeros(1, n_total);
count = 1;
for s = 1:n_domain
    for i = 1:size(Y{s}, 1)
        temp_c = Y{s}(i);
        class_index(count) = temp_c;
        domain_index(count) = s;
        count = count + 1;
    end
end

% count and proportion matrix
% [cnt_mat]_{sj} is the number of instances in domain s class j
cnt_mat = zeros(n_domain, n_class);
for s = 1:n_domain
    idx = find(domain_index==s);
    for j = 1:n_class
        idx_2 = idx(class_index(idx)==j);
        cnt_mat(s, j) = length(idx_2);
    end
end

F = zeros(n_total,n_total);
for dm = 1:n_domain
    dm_idx = (domain_index == dm);
    for cls1 = 1:n_class
        cls1_idx = dm_idx & (class_index == cls1);
        for cls2 = cls1+1:n_class
            cls2_idx = dm_idx & (class_index == cls2);
            n1 = sum(cls1_idx); n2 = sum(cls2_idx);
            K_diff = mean(K(:, cls1_idx), 2) - mean(K(:, cls2_idx), 2);
            K12 = K(cls1_idx, cls2_idx);
            K12 = K12 - (ones(n1)/n1) * K12 - K12 * (ones(n2)/n2) + (ones(n1)/n1) * K12 * (ones(n2)/n2);
            nuc_norm = sum(svd(K12, 'econ'));
            F = F + (K_diff * K_diff')/nuc_norm * n1 * n2 / (n1 + n2);
        end
    end
end

Q = zeros(n_total,n_total);
for cls = 1:n_class
    cls_idx = (class_index == cls);
    for dm1 = 1:n_domain
        dm1_idx = cls_idx & (domain_index == dm1);
        for dm2 = dm1+1:n_domain
            dm2_idx = cls_idx & (domain_index == dm2);
            n1 = sum(dm1_idx); n2 = sum(dm2_idx);
            K_diff = mean(K(:, dm1_idx), 2) - mean(K(:, dm2_idx), 2);
            K12 = K(dm1_idx, dm2_idx);
            K12 = K12 - (ones(n1)/n1) * K12 - K12 * (ones(n2)/n2) + (ones(n1)/n1) * K12 * (ones(n2)/n2);
            nuc_norm = sum(svd(K12, 'econ'));
            Q = Q + (K_diff * K_diff')/nuc_norm * n1 * n2 / (n1 + n2);
        end
    end
end

Q = Q + gamma * K + eps * eye(n_total);
