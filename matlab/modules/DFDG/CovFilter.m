function [F, Q] = CovFilter(K, X, Y, gamma, eps)
% 
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

%% pcd
K_pcd = zeros(n_total, n_class * n_domain); 
for dm = 1:n_domain
    dm_idx = (domain_index == dm);
    for cls = 1:n_class
        cls_idx = dm_idx & (class_index == cls);
        K_pcd(:, (dm - 1) * n_class + cls) = mean(K(:, cls_idx), 2); 
    end
end
Gamma_pcd = zeros(n_class * n_domain, n_class * n_domain);
for dm = 1:n_domain
    dm_idx = (domain_index == dm);
    for cls = 1:n_class
        cls_idx = (class_index == cls);
        K_dm_cls = K_pcd(dm_idx & cls_idx, :);
        [n_dm_cls, ~] = size(K_dm_cls);
        Gamma_pcd = Gamma_pcd + (n_dm_cls - 1) /n_dm_cls * cov(K_dm_cls);
    end
end
Gamma_pcd = Gamma_pcd/(n_domain * n_class); 
% Gamma_pcd = cov(K_pcd); 
sqrt_Gamma_pcd = real(sqrtm(Gamma_pcd)); 
K_pcd_tilde = K_pcd * pinv(sqrt_Gamma_pcd); 
G_pcd = n_class * eye(n_class) - ones(n_class);
G_pcd = kron(eye(n_domain), G_pcd);
F = K_pcd_tilde * G_pcd * K_pcd_tilde'/( n_domain * n_class * (n_class - 1) / 2); 

%% pdd
K_pdd = zeros(n_total, n_class * n_domain); 
for cls = 1:n_class
    cls_idx = (class_index == cls);
    for dm = 1:n_domain
        dm_idx = cls_idx & (domain_index == dm);
        K_pdd(:, (cls - 1) * n_class + dm) = mean(K(:, dm_idx), 2);
    end
end

Gamma_pdd = zeros(n_class * n_domain, n_class * n_domain);
for cls = 1:n_class
    cls_idx = (class_index == cls);
    for dm = 1:n_domain
        dm_idx = (domain_index == dm);
        K_dm_cls = K_pdd(dm_idx & cls_idx, :);
        [n_dm_cls, ~] = size(K_dm_cls);
        Gamma_pdd = Gamma_pdd + (n_dm_cls - 1) /n_dm_cls * cov(K_dm_cls);
    end
end
Gamma_pdd = Gamma_pdd/(n_domain * n_class);
% Gamma_pdd = cov(K_pdd); 
sqrt_Gamma_pdd = real(sqrtm(Gamma_pdd)); 
K_pdd_tilde = K_pdd * pinv(sqrt_Gamma_pdd); 
G_pdd = n_domain * eye(n_domain) - ones(n_domain);
G_pdd = kron(eye(n_class), G_pdd);
Q = K_pdd_tilde * G_pdd * K_pdd_tilde'/( n_class * n_domain * (n_domain - 1) / 2); 
Q = Q + gamma * K + eps * eye(n_total);



