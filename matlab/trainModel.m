function [Mdl, acc, pre_labels, Zs, Zt, inner_output, bst_parameter] = trainModel(method, cur_X_s_cell, cur_Y_s_cell, X_v, Y_v, cur_params, inner_options, Mdl, Dv)

switch method
    case 'DFDG'
        [Mdl, acc, pre_labels, Zs, Zt] = TrainDFDG(cur_X_s_cell, cur_Y_s_cell, X_v, Y_v, cur_params, inner_options.q, inner_options.cost, Mdl, Dv);
        inner_options.q = Mdl.bst_parameter(1);
        if cur_params(4) == 2
            inner_options.cost = Mdl.bst_parameter(2);
        end
        bst_parameter = Mdl.bst_parameter;
        inner_output = inner_options;
end

end