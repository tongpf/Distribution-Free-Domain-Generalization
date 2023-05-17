function plotFeatures(Xs, Zs, Ys, Ds, Xt, Zt, Yt)

n_class = max(Ys); n_domain = max(Ds);
% raw features
color = {'r', 'b', 'k', 'g'};
mark = {'+', 'o', 'd', 's', 'v', 'p'};

figure

subplot(2,2,1)
set(gca, 'FontName', 'times new roman');
title('raw training features')
hold on
for i = 1:n_domain
    idx_d = (Ds == i);
    for j = 1:n_class
        idx_c = (Ys == j);
        plot(Xs(idx_d & idx_c,1), Xs(idx_d & idx_c,2), [char(color(j)), char(mark(i))], 'MarkerSize', 3, 'LineWidth', 0.25);
    end
end

% kernel feature
subplot(2,2,3)
set(gca, 'FontName', 'times new roman');
title('kernel training features')
hold on
for i = 1:n_domain
    idx_d = (Ds == i);
    for j = 1:n_class
        idx_c = (Ys == j);
        plot(Zs(idx_d & idx_c,1), Zs(idx_d & idx_c,2), [char(color(j)), char(mark(i))], 'MarkerSize', 3, 'LineWidth', 0.25);
    end
end

% test feature
subplot(2,2,2)
set(gca, 'FontName', 'times new roman');
hold on
title('raw testing features')

for j = 1:n_class
    idx_c = (Yt == j);
    plot(Xt(idx_c,1), Xt(idx_c,2), [char(color(j)), 'x'], 'MarkerSize', 3,  'LineWidth', 0.25);
end

% test kernel feature
subplot(2,2,4)
set(gca, 'FontName', 'times new roman');
hold on
title('kernel testing features')

for j = 1:n_class
    idx_c = (Yt == j);
    plot(Zt(idx_c,1), Zt(idx_c,2), [char(color(j)), 'x'], 'MarkerSize', 3 ,'LineWidth', 0.25);
end


set(gcf, 'Position', [400, 100 ,800, 800]);

end