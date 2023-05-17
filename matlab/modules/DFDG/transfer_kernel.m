function [K_s_s, K_s_v] = transfer_kernel(D_s, D_v, K_s_s, K_s_v, K_v_v)
num_s = length(unique(D_s));
num_v = length(unique(D_v));

MMD_v = zeros(num_s, num_v);
for dm1 = 1:num_s
    dm1_idx = (D_s == dm1);
    for dm2 = 1:num_v
       	dm2_idx = (D_v == dm2);
        dst = mean(K_s_s(dm1_idx, dm1_idx), 'all') + mean(K_v_v(dm2_idx, dm2_idx), 'all') - 2 * mean(K_s_v(dm1_idx, dm2_idx), 'all');
        MMD_v(dm1, dm2) = exp(-dst/2);
        K_s_v(dm1_idx, dm2_idx) = MMD_v(dm1, dm2) * K_s_v(dm1_idx, dm2_idx); 
    end
end

MMD_s = zeros(num_s, num_s);
for dm1 = 1:num_s
    dm1_idx = (D_s == dm1);
    for dm2 = 1:num_s
       	dm2_idx = (D_s == dm2);
        dst = mean(K_s_s(dm1_idx, dm1_idx), 'all') + mean(K_s_s(dm2_idx, dm2_idx), 'all') - 2 * mean(K_s_s(dm1_idx, dm2_idx), 'all');
        MMD_s(dm1, dm2) = exp(-dst/2);
        K_s_s(dm1_idx, dm2_idx) = MMD_s(dm1, dm2) * K_s_s(dm1_idx, dm2_idx); 
    end
end
end