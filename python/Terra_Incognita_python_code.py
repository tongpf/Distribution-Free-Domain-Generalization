import pandas as pd
import math
import sklearn
# from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from multiprocessing import Pool
from Utils.corefun import *


def stratified_sampling(training_feature, subsamplecoeff):
    training_idx = np.empty(0)
    for class_idx in np.unique(training_feature['label']):
        classrows = np.where(training_feature['label'] == class_idx)[0]

        training_idx1 = np.random.choice(classrows,
                                         math.ceil(max(len(classrows) * subsamplecoeff, min(5, len(classrows)))),
                                         replace=False)

        training_idx = np.concatenate([training_idx, training_idx1])
    train_set = np.sort(training_idx)
    training_feature = training_feature.loc[train_set]
    training_feature = training_feature.reset_index(drop=True)
    return training_feature


def multi_processing_DFDG(bandwidth, DGmethod, classifier,
                          performance_hyper_para, subsamplecoeff, input_feature_dict, target_domain):
    if classifier == 'SVM':
        Clist = [0.5, 1, 2, 5]#, 10, 20, 50, 100]
    elif classifier == 'KNN':
        Clist = [1]

    for domain in target_domain:
        # preprocessing and/or down sampling
        train_domain = list(input_feature_dict.keys())
        train_domain.remove(domain)
        test_domain = domain
        training_feature = copy.deepcopy(input_feature_dict[train_domain[0]])

        # stratified sampling training samples
        training_feature = stratified_sampling(training_feature, subsamplecoeff)
        training_feature['domain'] = train_domain[0]
        feature_names = training_feature.columns
        usedfeature_names = feature_names[np.arange(0, (len(feature_names) - 2))]

        # no need for standardization
        # std_scaler = StandardScaler(with_mean=False, with_std=False)
        # df_scaled = std_scaler.fit_transform(training_feature[usedfeature_names])
        # df_scaled = pd.DataFrame(df_scaled, columns=usedfeature_names)
        # training_feature[usedfeature_names] = df_scaled

        for domain_idx in train_domain[1:]:
            training_feature_temp = copy.deepcopy(input_feature_dict[domain_idx])
            training_feature_temp = stratified_sampling(training_feature_temp, subsamplecoeff)
            training_feature_temp['domain'] = domain_idx

            # std_scaler = StandardScaler(with_mean=False, with_std=False)
            # df_scaled = std_scaler.fit_transform(training_feature_temp[usedfeature_names])
            # df_scaled = pd.DataFrame(df_scaled, columns=usedfeature_names)
            # training_feature_temp[usedfeature_names] = df_scaled

            training_feature = pd.concat([training_feature, training_feature_temp],
                                                  ignore_index=True)

        training_feature = training_feature.reset_index(drop=True)

        # prepare testing data
        testing_feature = copy.deepcopy(input_feature_dict[test_domain])
        testing_feature['domain'] = test_domain
        # std_scaler = StandardScaler(with_mean=False, with_std=False)
        # df_scaled = std_scaler.fit_transform(testing_feature[usedfeature_names])
        # df_scaled = pd.DataFrame(df_scaled, columns=usedfeature_names)
        # testing_feature[usedfeature_names] = df_scaled

        training_feature_temp = copy.deepcopy(training_feature)
        testing_feature_temp = copy.deepcopy(testing_feature)

        for overfit in Clist:
            for gamma in [0.1, 0.5, 1, 2, 3, 5, 10, 20]:  # , 50, 100]:
                # if q >= 1, select the exact dimension q.
                # if q < 1, then the invariant feature dimension will be q * dimension of the raw features
                qlist = [1, 3, 5, 7, 10, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15]
                #qlist = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15]
                qdict = {}
                for subspacepercent in qlist:
                    qdict[subspacepercent] = {}

                # full training set
                X = training_feature_temp[usedfeature_names].to_numpy()  # feature
                Xlabel = training_feature_temp['domain'].to_numpy()  # domain
                y = training_feature_temp['label'].to_numpy()  # class

                # use the median heuristic to select the initial bandwidth
                dists_sq = sklearn.metrics.pairwise_distances(X, X)
                dists_sq = np.median(dists_sq)
                dists_sq = dists_sq ** 2

                # main procedure, calculate the mixing matrix B
                if DGmethod == 'DFDG-Eig':
                    Bmatrix, Kmatrix, eigenvalue = \
                        DFDG_Eig(X, y, Xlabel, qmax=int(X.shape[0] * 0.15), bandwidth=bandwidth * dists_sq,
                                 epsilon=1e-5, gamma=gamma)
                elif DGmethod == 'DFDG-Cov':
                    Bmatrix, Kmatrix, eigenvalue = \
                        DFDG_Cov(X, y, Xlabel, qmax=int(X.shape[0] * 0.15), bandwidth=bandwidth * dists_sq,
                                 epsilon=1e-5, gamma=gamma)
                # sort B matrix by the eigenvalue
                sortidx = (-eigenvalue).argsort()
                Bmatrix = Bmatrix[:, sortidx]
                eigenvalue = eigenvalue[sortidx]
                eigenvalue[eigenvalue < 0] = 0

                # one can comment out the following lines and use eigenvalue percentage to select the dimension of
                # invariant features q
                # cumsum_eigen = np.cumsum(eigenvalue)
                # cumsum_eigen = cumsum_eigen / cumsum_eigen[-1]

                Xtest = testing_feature_temp[usedfeature_names].to_numpy()
                Xlabeltest = testing_feature_temp['domain'].to_numpy()
                ytest = testing_feature_temp['label'].to_numpy()

                dists_sq = sklearn.metrics.pairwise_distances(Xtest, X)
                dists_sq = np.median(dists_sq)
                dists_sq = dists_sq ** 2

                Kmatrix_test = transfer_kernel(Xtest, X, Xlabeltest, Xlabel, bandwidth=bandwidth * dists_sq)

                # select validation sets from the training data
                cdlabel = training_feature_temp['label'].astype(str) + "_" + training_feature_temp['domain'].astype(str)
                validation_train, validation_test = train_test_split(training_feature_temp, train_size=0.7,
                                                                     stratify=cdlabel)

                X_val = validation_train[usedfeature_names].to_numpy()
                Xlabel_val = validation_train['domain'].to_numpy()
                y_val = validation_train['label'].to_numpy()

                dists_sq_val = sklearn.metrics.pairwise_distances(X_val, X_val)
                dists_sq_val = np.median(dists_sq_val)
                dists_sq_val = dists_sq_val ** 2

                if DGmethod == 'DFDG-Eig':
                    Bmatrix_val, Kmatrix_val, eigenvalue_val = \
                        DFDG_Eig(X_val, y_val, Xlabel_val, qmax=int(X_val.shape[0] * 0.15),
                                 bandwidth=bandwidth * dists_sq_val, epsilon=1e-5, gamma=gamma)
                elif DGmethod == 'DFDG-Cov':
                    Bmatrix_val, Kmatrix_val, eigenvalue_val = \
                        DFDG_Cov(X_val, y_val, Xlabel_val, qmax=int(X_val.shape[0] * 0.15),
                                 bandwidth=bandwidth * dists_sq_val, epsilon=1e-5, gamma=gamma)

                sortidx_val = (-eigenvalue_val).argsort()
                Bmatrix_val = Bmatrix_val[:, sortidx_val]
                eigenvalue_val = eigenvalue_val[sortidx_val]
                eigenvalue_val[eigenvalue_val < 0] = 0

                Xtest_val = validation_test[usedfeature_names].to_numpy()
                Xlabeltest_val = validation_test['domain'].to_numpy()
                ytest_val = validation_test['label'].to_numpy()

                dists_sq_val = sklearn.metrics.pairwise_distances(Xtest_val, X_val)
                dists_sq_val = np.median(dists_sq_val)
                dists_sq_val = dists_sq_val ** 2

                Kmatrix_test_val = transfer_kernel(Xtest_val, X_val, Xlabeltest_val, Xlabel_val,
                                                   bandwidth=bandwidth * dists_sq_val)

                for subspacepercent in qlist:
                    if subspacepercent < 1:
                        q_dim = max(int(subspacepercent * X.shape[0]), 1)
                        # q_dim = max(np.where(cumsum_eigen > subspacepercent)[0][0], 1)
                    else:
                        q_dim = subspacepercent

                    # subset the B matrix
                    Bmatrix_first_m = Bmatrix[:, :q_dim]

                    if classifier == 'KNN':
                        # a simple KNN classifier
                        myclassifier = KNeighborsClassifier(n_neighbors=10)
                        Z_train = Kmatrix.dot(Bmatrix_first_m)
                        myclassifier.fit(Z_train, y)
                        Z_test = Kmatrix_test.dot(Bmatrix_first_m)
                    elif classifier == 'SVM':
                        # SVM classifier
                        myclassifier = svm.SVC(kernel='precomputed', probability=True, C=overfit)
                        Z_train = Kmatrix.dot(Bmatrix_first_m).dot(Bmatrix_first_m.T).dot(Kmatrix.T)
                        myclassifier.fit(Z_train, y)
                        Z_test = Kmatrix_test.dot(Bmatrix_first_m).dot(Bmatrix_first_m.T).dot(Kmatrix.T)

                    prediction = myclassifier.predict(Z_test)
                    testacc = sklearn.metrics.accuracy_score(ytest, prediction)

                    prediction = myclassifier.predict(Z_train)
                    trainacc = sklearn.metrics.accuracy_score(y, prediction)

                    # save the validation accuracy for hyperparameter selection
                    Bmatrix_first_m = Bmatrix_val[:, :q_dim]
                    if classifier == 'KNN':
                        myclassifier = KNeighborsClassifier(n_neighbors=10)
                        Z_train = Kmatrix_val.dot(Bmatrix_first_m)
                        myclassifier.fit(Z_train, y_val)
                        Z_test = Kmatrix_test_val.dot(Bmatrix_first_m)
                    elif classifier == 'SVM':
                        myclassifier = svm.SVC(kernel='precomputed', probability=True, C=overfit)
                        Z_train = Kmatrix_val.dot(Bmatrix_first_m).dot(Bmatrix_first_m.T).dot(Kmatrix_val.T)
                        myclassifier.fit(Z_train, y_val)
                        Z_test = Kmatrix_test_val.dot(Bmatrix_first_m).dot(Bmatrix_first_m.T).dot(Kmatrix_val.T)

                    prediction = myclassifier.predict(Z_test)
                    validationacc = sklearn.metrics.accuracy_score(ytest_val, prediction)

                    qdict[subspacepercent][domain] = {'test': testacc, 'train': trainacc, 'validation': validationacc}

                for qvalue, q_performance in qdict.items():
                    print(bandwidth, overfit, gamma, qvalue)
                    print(q_performance)
                    new_row = [bandwidth, overfit, gamma, qvalue]

                    for i in q_performance.values():
                        new_row = new_row + list(i.values())
                    new_row = {x: y for x, y in zip(performance_hyper_para.columns, new_row)}
                    new_row['domain'] = domain
                    new_row = pd.Series(new_row)
                    performance_hyper_para = pd.concat([performance_hyper_para, new_row.to_frame().T], ignore_index=True)

    return performance_hyper_para

if __name__ == '__main__':
    # random.seed(2023)
    # read the Terra Incognita feature files
    input_feature_dict = {}
    datadir = './data/'

    # The ERM-adjusted ResNet 50 itself is a domain generalization method. In the feature extraction procedure,
    # we have to avoid the model seeing the target domain. Thus, if the target domain is L100, then L38, L43 and L46
    # were used to train the ERM model. After the training procedure, all the features are extracted from the
    # last hidden layer of the final ERM model.
    target_domain = 'L46'  # choose one from {'L38', 'L43', 'L46' and 'L100'}

    data = pd.read_csv(datadir + target_domain + '_location_38.csv', index_col=False)
    data = data.set_axis([*data.columns[:-1], 'label'], axis=1, inplace=False)
    input_feature_dict['L38'] = data

    data = pd.read_csv(datadir + target_domain + '_location_43.csv', index_col=False)
    data = data.set_axis([*data.columns[:-1], 'label'], axis=1, inplace=False)
    input_feature_dict['L43'] = data

    data = pd.read_csv(datadir + target_domain + '_location_46.csv', index_col=False)
    data = data.set_axis([*data.columns[:-1], 'label'], axis=1, inplace=False)
    input_feature_dict['L46'] = data

    data = pd.read_csv(datadir + target_domain + '_location_100.csv', index_col=False)
    data = data.set_axis([*data.columns[:-1], 'label'], axis=1, inplace=False)
    input_feature_dict['L100'] = data
    del data

    # classifier and performance
    classifier = 'KNN'  # choose 'SVM' or 'KNN'
    DGmethod = 'DFDG-Cov'  # choose 'DFDG-Cov' or 'DFDG-Eig'
    # increase subsamplecoeff to 1 for a better performance
    subsamplecoeff = 0.5  # down-sampling observations to save memory space

    performance_hyper_para = pd.DataFrame({'bandwidth': [], 'C': [], 'gamma': [], 'q': [],
                                           'test': [], 'train': [], 'validation': [], 'domain': []})

    process_pool = Pool(processes=4)

    performance_hyper_para_list = \
        process_pool.starmap(
            multi_processing_DFDG,
            [(0.5, DGmethod, classifier, performance_hyper_para, subsamplecoeff, input_feature_dict, [target_domain]),
             (1, DGmethod, classifier, performance_hyper_para, subsamplecoeff, input_feature_dict, [target_domain]),
             (2, DGmethod, classifier, performance_hyper_para, subsamplecoeff, input_feature_dict, [target_domain]),
             (5, DGmethod, classifier, performance_hyper_para, subsamplecoeff, input_feature_dict, [target_domain])])

    performance_hyper_para = pd.concat(performance_hyper_para_list, ignore_index=True)

    performance_hyper_para_best = \
        performance_hyper_para['test'][np.flatnonzero(performance_hyper_para['validation'] ==
                                                       performance_hyper_para['validation'].max())].max()

    print(performance_hyper_para_best)
    print(classifier)
    print(DGmethod)
    performance_hyper_para.to_csv('result.csv')

