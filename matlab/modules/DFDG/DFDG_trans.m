function [B, w] = DFDG_trans(F, Q, K, eig_ratio)

% INPUT
%     F                   - F in cross class discrepancy, Eq.(21)
%     Q                   - Q in cross domain discrepancy, Eq.(22)
%     K                   - kernel matrix
%     eig_ratio           - dimension of the transformed space

%
% OUTPUT
%     B                   - matrix of projection
%     w                   - corresponding eigenvalues


% ----------------------------------------------------------------------

n_total = length(K);
n_eigs = zeros(size(eig_ratio));
for i = 1:length(eig_ratio)
    if eig_ratio(i) < 1
        n_eigs(i) = max(1, floor(eig_ratio(i) * n_total));
    else
        n_eigs(i) = eig_ratio(i);
    end
end

F_inv_Q = Q\F;
[B, A] = eigs(F_inv_Q, max(n_eigs));

B = real(B);
A = real(A);
eigvalues = diag(A);
[val, idx] = sort(eigvalues, 'descend');
B = B(:, idx);
w = val;

B_norm = diag(B' * K * B);
B = B ./ repmat(abs(B_norm.^(1/2)'), [n_total, 1]);
end