from sklearn.metrics import r2_score
from importlib import reload
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import Classifiers
import Classifiers.src.NeuralNet as nn
reload(Classifiers.src.NeuralNet)
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.base import clone
import numpy as np
import pandas as pd
import Classifiers.src.utilities 
reload(Classifiers.src.utilities)
from Classifiers.src.utilities import divideTraining, bootstrapping
import itertools
import xgboost as xgb
from sklearn.metrics import accuracy_score


class clf(object):
    def __init__(self, matrix_class, testX=0, testy=0,
                 num_folds=10, grid_search_random=0.01,
                 coeffs=0, feature_df=0,
                 original_cols=0, param_space=5, iterations=1,
                 feature_selection=False, bst=False):
        """
        general classifier operations

        Parameters:
        ======================================

        matrix_class -> instance of Predictome The class container for the matrix built using the Predictome Class
    
        testX -> Sparse Matrix CSR: Optional if test / training split setup selected design matrix

        testy -> Numpy Array: Optional if test / training split setup selected response variable

        num_folds -> Integer: for cross valiation this determines the k-folds and bootstrapping the number 
                              of bootstrap events to execute (between 1-infinity)

        grid_search_random -> float: 0-1 what percent of models to choose from

        coeffs -> numpy array user-defined weights used for the plot_weights function
    
        feature_df -> a dataframe containing the name of the features in the 'chrm_pos_ref_alt' column
                      important for plot_weights() and gradient boosted decision trees

        original_cols -> Numpy Array: a vector containing the indicies original columns

        param_space -> Integer: How deep to make the grid for hyperparameter tuning

        iterations -> Integer: How many interations of the boostrap or cross-validation to perform

        feature_selection -> Boolean: Whether to turn on a fisher based feature selection method

        bst -> Boolean: Whether to perform boosting, cannot perform boosting and crossvalidation together

        Attributes:
        ======================================
        
        matrix_class -> instance of Predictome The class container for the matrix built using the Predictome Class
    
        testX -> Sparse Matrix CSR: Optional if test / training split setup selected design matrix

        testy -> Numpy Array: Optional if test / training split setup selected response variable

        num_folds -> Integer: for cross valiation this determines the k-folds and bootstrapping the number 
                              of bootstrap events to execute (between 1-infinity)

        grid_search_random -> float: 0-1 what percent of models to choose from

        coeffs -> numpy array user-defined weights used for the plot_weights function
    
        feature_df -> a dataframe containing the name of the features in the 'chrm_pos_ref_alt' column
                      important for plot_weights() and gradient boosted decision trees

        original_cols -> Numpy Array: a vector containing the indicies original columns

        param_space -> Integer: How deep to make the grid for hyperparameter tuning

        iterations -> Integer: How many interations of the boostrap or cross-validation to perform

        feature_selection -> Boolean: Whether to turn on a fisher based feature selection method

        bst -> Boolean: Whether to perform boosting, cannot perform boosting and crossvalidation together

        test_score -> Float: Accuracy of model against a test set

        conf_score -> Float: Confidence of model against a test set

        prediction_tracker -> panda df: keeps track of the scores as a function of boostraps

        gc -> Integer: Global counter used in bootstrapping
        
        Notes:
        ======================================

        This is the generic classifier class that can perform a variety of the base functions.
        Such as cross-validation and feature-selection (though slow and not reccomended).

        Examples:
        ======================================

        >>>  None
        """


        '''Data'''
        self.matrix_class = matrix_class
        self.trainX = matrix_class.gt_matrix[:, :-1]
        self.trainy = matrix_class.gt_matrix[:, -1].A.T[0]
        self.testX = testX
        self.testy = testy
        '''Parameters'''
        self.num_folds = num_folds
        self.grid_search_random = grid_search_random
        self.param_space = param_space
        self.iterations = iterations
        self.filter_method = 0
        self.saved_scores = []
        self.total_weights_list = []
        self.bst = bst
        '''Attributes'''
        self.coeffs = coeffs
        self.test_score = 0
        self.conf_score = 0
        index = np.arange(self.trainX.shape[0])
        self.prediction_tracker = pd.DataFrame(index=index, columns=np.arange(self.num_folds))
        self.gc = 0
        """matrix information"""
        self.feature_df = feature_df
        self.original_cols = original_cols

    def plot_weights(self):
        """
        A function that takes the weights and plots them
        either as a histogram, or if there are few enough
        we plot the weights as a bar graph with reliability 
        as a marker.
        """
        if len(np.where(self.coeffs != 0)[1]) is 0:
            print("No significant features found with coefficients")
            return
        num_feats = len(np.where(self.coeffs != 0)[1])
        weight_df = self.feature_df.loc[self.original_cols[np.where(self.coeffs != 0)[1]]]
        weight_df['weights'] = self.coeffs[np.where(self.coeffs != 0)]
        weight_df['new_columns'] = np.where(self.coeffs != 0)[1]
        if num_feats > 1000:
            weight_df['weights'].plot.hist()
        else:
            if 'reliability' in weight_df:
                weight_df.sort_values(by='reliability', inplace=True)
                alphas = weight_df['reliability'].values.copy()
                rgba_colors = np.zeros((len(alphas), 4))
                rgba_colors[:, 0] = 1.0
                rgba_colors[:, 3] = alphas
                weight_df['weights'].plot.bar(color=rgba_colors)
            else:
                weight_df.sort_values(by='weights', inplace=True)
                vals = weight_df['weights'].values.copy()
                rgba_colors = np.zeros((vals.shape[0], 4))
                rgba_colors[:, 0] = 1.0
                rgba_colors[:, 3] = np.abs(vals)
                weight_df['weights'].plot.bar(color=rgba_colors)
        return weight_df

    def random_search(self, models, threshold=0.01):
        """randomly choose a percent of the models to train"""
        num_models = len(models)
        indices = np.random.choice(num_models, size=int(threshold * num_models), replace=False)
        return [models[x] for x in indices], indices

    def train(self):
        """core function that organizes the order of training"""
        if self.grid_search_random < 1:
            r_models, indices = self.random_search(self.models, self.grid_search_random)
        else:
            r_models = self.models
            indices = np.arange(len(self.models))
        max_score_i, max_score, betas = self.cross_validation(r_models)
        self.current_model = r_models[max_score_i]
        self.max_index = indices[max_score_i]
        self.betas = betas
        self.current_score = max_score
        print("Highest score: %s" % max_score)

    def transform_features(self):
        method = self.filter_method[0]
        t = self.filter_method[1]
        self.matrix_class.gt_matrix = self.trainX
        self.trainX = self.matrix_class.filter_matrix(by=method, threshold=t)

    def feature_selection(self, filter_method, model=None):
        """
        Filter a matrix using a feature of choice
        then perform cross-validation to see how
        the model performs with the user-selected features
        """
        total_score_list = []
        i = 0
        total_iterations = self.num_folds * int(len(filter_method)) * self.iterations
        if isinstance(filter_method, list):
            j = 0
            for method, t in filter_method:
                index_pool = divideTraining(self.trainy,
                                            num_folds=self.num_folds)
                if model is None:
                    m = SGDClassifier(penalty='none')
                else:
                    m = model[j]
                avg_score_list = []
                for ii in range(self.iterations):
                    for fold in range(self.num_folds):
                        tcoords, vcoords = index_pool.choseIndex()
                        tcoords = [x[0] for x in tcoords]
                        vcoords = [x[0] for x in vcoords]
                        print(self.matrix_class.gt_matrix.shape)
                        self.matrix_class.tcoords = tcoords
                        trainX, pcols = self.matrix_class.filter_matrix(by=method, threshold=t)
                        print(trainX.shape)
                        if type(pcols).__module__ == np.__name__:
                            valX = self.trainX[vcoords]
                            score, betas = self.score_model(trainX[tcoords],
                                                        self.trainy[tcoords],
                                                        valX[:, pcols],
                                                        self.trainy[vcoords], model=m)
                            avg_score_list.append(score)
                            i += 1
                            n_feats = len(np.where(betas != 0)[0])
                            print("Current score=%f percent: %f n_feats: %d" % (score, float(i / total_iterations), n_feats), end="\r")
                        else:
                            print("model failed pval too small")
                total_score_list.append(np.mean(avg_score_list))
                self.saved_scores.append(avg_score_list)
                j += 1
            max_score_i = total_score_list.index(max(total_score_list))
            self.filter_method = filter_method[max_score_i]
            if model is not None:
                self.current_model = m[max_score_i]
            print(filter_method[max_score_i], max(total_score_list))
        else:
            print("Model is not in a list format")
            return

    def cross_validation(self, model):
        """
        Perform either k-fold cross-validation or bootstrapping 
        on a list of models to determine which model is best.
        """        
        total_score_list = []
        total_iterations = self.num_folds * int(len(model)) * self.iterations
        i = 0
        if isinstance(model, list):
            for m in model:
                avg_score_list = []
                fold_weights = []
                if self.bst is False:
                    index_pool = divideTraining(self.trainy,
                                                num_folds=self.num_folds)
                elif self.bst is True:
                    index_pool = bootstrapping(self.trainy)
                for ii in range(self.iterations):
                    for fold in range(self.num_folds):
                        tcoords, vcoords = index_pool.choseIndex()
                        tcoords = [x[0] for x in tcoords]
                        vcoords = [x[0] for x in vcoords]
                        self.vcoords = vcoords
                        score, betas = self.score_model(self.trainX[tcoords],
                                                        self.trainy[tcoords],
                                                        self.trainX[vcoords],
                                                        self.trainy[vcoords], model=m)
                        avg_score_list.append(score)
                        if betas is not 0:
                            fold_weights.append(betas)
                            print("n_feats=>%d" % len(np.where(betas != 0)[0]))
                        i += 1
                        print("Current score=%f percent: %f" % (score, float(i / total_iterations)))
                self.total_weights_list.append(fold_weights)
                total_score_list.append(np.mean(avg_score_list))
                self.saved_scores.append(avg_score_list)
            max_score_i = total_score_list.index(max(total_score_list))
            if betas is not 0:
                max_betas = self.total_weights_list[max_score_i]
                return max_score_i, max(total_score_list), max_betas
            else:
                return max_score_i, max(total_score_list), 0
        else:
            print("Model is not in a list format")
            return


class clf_svm(clf):
    def __init__(self, model_type=None, **kwargs):
        """
        Setup support vector machine classifier from scikit-learn
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC
        
        Parameters:
        ======================================
        model_type -> String: e.g. kernel or linear    
    
        Attributes:
        ======================================

        c -> [0-inf] a float that penalizes your error term

        kernel -> ['rbf', 'poly', 'sigmoid'] ways of transforming the data into new spaces

        gamma -> a regularization term for kernels

        degree -> the power by which a poly kernel takes e.g. X^2 X^3

        params -> parameters of the test model

        models -> a list of models/parameters that have yet to be trained on

        Notes:
        ======================================

        Support vector machines are suitable for high-dimensinoal datasets like exome data. Usually kernels
        are not going to help here since the data is already in a very high-dimensional space. Intead its
        better to use a linear SVM which will return coeff for each of the features that we can
        easily interpert. If kernels are chosen then you are unable to interpert the importance of the
        features with regard to the disease. Also, because the data is in high-dimensional space it is
        important to use a very strong regularization technique, otherwise we run the risk of overfitting. Thus L1 is perferred over L2.

        Examples:
        ======================================

        >>> svm = clf_svm(matrix_class=pme.custom, feature_df=pme.custom.columns_df,     
                                           grid_search_random=0.1, original_cols=pme.custom.original_cols,
                                           bst=True, num_folds=500)
        >>> svm.log_model_search()
        >>> svm.train()
        >>> svm.get_model_params()
        >>> {LinearSVM, C: 0.02, dual:True, penalty:l2, class_weight:balanced}
        >>> svm.linear_model_search()
        >>> svm.train()
        >>> svm.get_model_params()
        >>> {LinearSVM, C: 0.002, dual:True, penalty:l1, class_weight:balanced}
        >>> svm.test()
        >>> 0.831
        """        
        clf.__init__(self, **kwargs)
        self.c = 0
        self.kernel = ['rbf', 'poly', 'sigmoid']
        self.gamma = 0
        self.degree = [2, 3]
        self.params = 0
        self.alpha = 0
        self.l1_ratio = 0
        self.model_type = model_type

    def get_model_params(self):
        """A function just to view the model parameters"""
        print(self.current_model.get_params())

    def log_model_search(self):
        """Perform a grid search on the parameter space using a log range"""
        self.compute_hyperparameter_ranges()
        none_model = [SGDClassifier(penalty='none', alpha=x, loss='hinge',
                                    class_weight='balanced') for x in self.c]
        l2_model = [LinearSVC(penalty='l2', C=x, dual=True,
                              class_weight='balanced') for x in self.c]
        l1_model = [LinearSVC(penalty='l1', dual=False,
                              C=x, class_weight='balanced') for x in self.c]
        sgd_parameters = list(itertools.chain(itertools.product(self.alpha, self.l1_ratio)))
        en_model = [SGDClassifier(penalty='elasticnet', loss='hinge',
                                  alpha=alpha, l1_ratio=l1r, class_weight='balanced') for alpha, l1r in sgd_parameters]
        parameters = list(itertools.chain(itertools.product(self.c, self.gamma,
                                                            self.degree, self.kernel)))
        kernel_models = [SVC(C=C, degree=deg, gamma=gamma, kernel=k, class_weight='balanced') for C, gamma, deg, k in parameters]
        if self.model_type is 'none':
            self.models = none_model
        elif self.model_type is 'l1':
            self.models = l1_model
        elif self.model_type is 'l2':
            self.models = l2_model
        elif self.model_type is 'elasticnet':
            self.models = en_model
        elif self.model_type is 'kernel':
            self.models = kernel_models
        elif self.model_type is None:
            self.models = none_model + l2_model + l1_model + en_model + kernel_models

    def linear_model_search(self):
        """Perform a grid search on the parameter space using a linear range"""
        clf = self.models[self.max_index]
        self.models = []
        params = clf.get_params()
        new_params = [self.c, self.gamma, self.degree]
        if isinstance(clf, LinearSVC):
            new_params[0] = np.linspace(params['C'] / 2, params['C'] * 2, self.param_space)
            for param in new_params[0]:
                clf_copy = clone(clf)
                self.models.append(clf_copy.set_params(C=param))
        elif isinstance(clf, SGDClassifier):
            self.alpha = np.linspace(params['alpha'] / 2, params['alpha'] * 2, self.param_space)
            self.l1_ratio = np.linspace(params['l1_ratio'] / 2, params['l1_ratio'] * 2, self.param_space)
            sgd_parameters = list(itertools.chain(itertools.product(self.alpha, self.l1_ratio)))
            for alpha, l1r in sgd_parameters:
                clf_copy = clone(clf)
                self.models.append(clf_copy.set_params(alpha=alpha, l1_ratio=l1r))
        elif isinstance(clf, SVC):
            for i, k in enumerate(['C', 'gamma', 'degree']):
                if k in params:
                    new_params[i] = np.linspace(params[k] / 2, params[k] * 2, self.param_space)
            param_comb = list(itertools.chain(itertools.product(new_params[0], new_params[1],
                                                                new_params[2])))
            for C, gamma, deg, in param_comb:
                clf_copy = clone(clf)
                self.models.append(clf_copy.set_params(C=C, degree=deg, gamma=gamma))

    def score_model(self, trainX, trainy, valX, valy, model):
        """A function called by cross-validation. The score of the model is computed"""
        model.fit(trainX, trainy)
        score = model.score(valX, valy)
        if isinstance(model, LinearSVC) or isinstance(model, SGDClassifier):
            return score, model.coef_
        else:
            return score, 0

    def compute_hyperparameter_ranges(self):
        """Compute the hyperparameter space using some heurtistics"""
        exponent = np.floor(
            np.log10(np.abs(1 / self.trainX.shape[0]))).astype(int)
        self.gamma = np.logspace(exponent - 1, exponent + 4, self.param_space)
        self.c = np.logspace(exponent, 1, self.param_space)
        self.alpha = np.logspace(exponent, 1, self.param_space)
        self.l1_ratio = np.logspace(exponent, 0, self.param_space)

    def test(self):
        """
        If a test matrix exists, than one can perform a test on
        the best model selected.
        """
        self.params = self.current_model.get_params()
        self.test_model = self.current_model.fit(self.trainX, self.trainy)
        self.test_score = self.test_model.score(self.testX, self.testy)
        if 'kernel' not in self.params:
            self.coeffs = self.test_model.coef_
        self.conf_score = self.test_model.decision_function(self.testX)
        print("This is the test score:%s" % self.test_score, "Model params=%s" %
              self.test_model.get_params())


class clf_gb(clf):
    def __init__(self, model_type='binary', **kwargs):
        """
        Setup gradient boosted decision trees
        https://xgboost.readthedocs.io/en/latest/#

        Parameters:
        ======================================

        model_type -> String: The model can be binary or regression 

        Attributes:
        ======================================

        max_depth -> Integer how many splits the tree will make

        num_boost_round -> Integer how many rounds of boosting

        gamma -> Float  mininum loss for a split

        eta -> List of floats: The learning rates to try 

        lambda_r -> l2 regularization

        alpha -> l1 regularization

        subsample -> randomly sample row data

        keys -> List of strings: The different parameters in a format that xgboost understands

        objective -> String: The type of output given the number of classes

        Notes:
        ======================================

        XBGradient boosted decision trees are an extermely flexible algorithm for machine learning.
        The algorithm does not seem to deal well with high-dimensional data and therefore the dimensions
        should be reduced prior to training. Additionally we have found that GBDT are typically worse
        than SVM and nerual nets. 

        Examples:
        ======================================

        >>> gb = clf_gb(matrix_class=pme.custom, 
                                            feature_df=pme.custom.columns_df,     
                                            grid_search_random=0.1, original_cols=pme.custom.original_cols,
                                            bst=True, num_folds=500)
        
        >>> gb.log_model_search()
        >>> gb.train()
        >>> gb.get_model_params()
        >>> gb.linear_model_search()
        >>> gb.train()
        >>> gb.get_model_params()
        >>> gb.test()
        >>> 0.831
        """        
        clf.__init__(self, **kwargs)
        self.model_type = model_type
        self.max_depth = np.linspace(1, 10, self.param_space)
        self.num_boost_round = 1000
        self.gamma = 0  
        self.eta = np.logspace(-3, -1, self.param_space)
        self.lambda_r = 0
        self.alpha = 0
        self.subsample = np.linspace(0.1, 1, self.param_space)  
        self.keys = ['bst:gamma', 'bst:eta', 'bst:lambda', 'bst:alpha',
                     'bst:max_depth', 'num_boost_round', 'bst:subsample']

    def get_model_params(self):
        """A function just to view the model parameters"""
        print(self.models[self.max_index])

    def log_model_search(self):
        """Perform a grid search on the parameter space using a log range"""
        self.compute_gamma()
        self.compute_l1_l2_range()
        param_comb = list(itertools.chain(itertools.product(self.gamma, self.eta,
                                                            self.lambda_r, self.alpha,
                                                            self.max_depth, self.num_boost_round,
                                                            self.subsample)))
        self.models = []
        model_p = {}
        for i, value in enumerate(param_comb):
            para_value = list(zip(self.keys, value))
            model_p[i] = {key: value for (key, value) in para_value}
            model_p[i]['silent'] = 1
            if len(np.unique(self.trainy)) < 3:
                model_p[i]['objective'] = 'binary:logitraw'
                self.objective = 'binary:logitraw'
            else:
                model_p[i]['objective'] = 'multi:softmax'
                self.objective = 'multi:softmax'
                model_p[i]['num_class'] = len(np.unique(self.trainy))
            model_p[i]['eval_metric'] = 'error'
            self.models.append(model_p[i].items())

    def linear_model_search(self):
        """Perform a grid search on the parameter space using a linear range"""
        params = self.models[self.max_index]
        new_params = [self.gamma, self.eta, self.lambda_r, self.alpha,
                      self.max_depth, self.num_boost_round, self.subsample]
        for i, k in enumerate(self.keys):
            if k in params:
                new_params[i] = np.linspace(params[k] / 2, params[k] * 2, self.param_space)
        param_comb = list(itertools.chain(itertools.product(new_params[0], new_params[1],
                                                            new_params[2], new_params[3],
                                                            new_params[4], new_params[5],
                                                            new_params[6])))
        self.models = []
        model_p = {}
        for i, value in enumerate(param_comb):
            para_value = list(zip(self.keys, value))
            model_p[i] = {key: value for (key, value) in para_value}
            model_p[i]['silent'] = 1
            if len(np.unique(self.trainy)) < 3:
                model_p[i]['objective'] = 'binary:logistic'
            else:
                model_p[i]['objective'] = 'multi:softmax'
                model_p[i]['num_class'] = len(np.unique(self.trainy))
            model_p[i]['eval_metric'] = 'error'
            self.models.append(model_p[i].items())

    def score_model(self, trainX, trainy, valX, valy, model):
        """Function called by cross-validation in order to fit the model and then make predictions"""
        if self.feature_df is not None:
            dtrain = xgb.DMatrix(data=trainX, missing=-1,
                                 label=trainy, feature_names=self.feature_df['chrm_pos_ref_alt'].map(str) + "_" + self.feature_df['variant_call'].map(str))
            dval = xgb.DMatrix(data=valX, missing=-1,
                               label=valy, feature_names=self.feature_df['chrm_pos_ref_alt'].map(str) + "_" + self.feature_df['variant_call'].map(str))
        else:
            dtrain = xgb.DMatrix(data=trainX, missing=-1,
                                 label=trainy)
            dval = xgb.DMatrix(data=valX, missing=-1,
                               label=valy)
        fit_model = xgb.train(params=model, dtrain=dtrain, num_boost_round=self.num_boost_round)
        y_pred = fit_model.predict(dval)
        tr_pred = fit_model.predict(dtrain)
        
        if model['objective'] == 'multi:softmax':
            predictions = y_pred.argmax(axis=1)
        else:
            predictions = [1 if x >= 0 else 0 for x in y_pred]
        
        self.prediction_tracker.loc[self.vcoords, self.gc] = y_pred
        self.gc += 1
        if self.model_type == 'binary':
            accuracy = accuracy_score(valy, predictions)
            val_score = brier_score_loss(valy, self.sigmoid_transform(y_pred))
            trn_score = brier_score_loss(trainy, self.sigmoid_transform(tr_pred))
            print("brier_score %s, train_brier %s, accuracy %s" % (val_score, trn_score, accuracy))
            return val_score, 0
        elif self.model_type == 'reg':
            accuracy = r2_score(valy, predictions)
            print("R2" % (accuracy))
            return accuracy, 0

    def sigmoid_transform(self, item):
        """Function called by cross-validation in order to fit the model and then make predictions"""
        return (np.exp((-1 * item)) + 1) ** -1

    def compute_gamma(self):
        """Compute the hyperparameter space using some heurtistics"""
        exponent = np.floor(np.log10(np.abs(1 / self.trainX.shape[0]))).astype(int)
        self.gamma = np.logspace(exponent - 1, exponent + 4, self.param_space)

    def compute_l1_l2_range(self):
        """Compute the hyperparameter space using some heurtistics"""
        auto = np.log10(1 / self.trainX.shape[0]).astype(int)
        self.lambda_r = np.logspace(auto, 1, self.param_space)
        self.alpha = np.logspace(auto, 1, self.param_space)

    def test(self):
        """
        If a test matrix exists, than one can perform a test on
        the best model selected.
        """        
        dtest = xgb.DMatrix(data=self.testX, missing=-1,
                            label=self.testy, feature_names=self.feature_df['chrm_pos_ref_alt'])
        self.params = self.current_model
        self.test_model = self.fit(self.current_model, self.trainX, self.trainy)
        self.conf_score = self.test_model.predict(dtest)
        self.test_score = self.score(self.test_model, dtest, self.testy)
        print("This is the test score:%s" % self.test_score, "Model params=%s" % self.params)


class clf_log(clf):
    def __init__(self, model_type=None, **kwargs):
        """
        Setup logistic regression from scikit-learn
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        
        Parameters:
        ======================================
        
        model_type -> String: indicates which regularization penatly
                      to use e.g. l1, l2, elasticnet, None

        Attributes:
        ======================================

        c -> [0-inf] a float that penalizes your error term for l1 and l2

        alpha -> [0-inf] a float that is used for the mixing penalty of the elastic net

        params -> parameters of the test model

        l1_ratio -> [0-inf] a float that is used for the mixing penalty of the elastic net

        Notes:
        ======================================

        Regression methods are routinely applied in data analysis, especially in the
        context of epidemiological studies. Logistic regression is concerned with
        uncovering the relationship between a dichotomous response variable and one or
        more explanatory variables. The relationship is mathematically modeled and can
        be expressed generally, in terms of the expected value of the response variable,
        given the explanatory variables, x and some transformation of the values x. In the
        context of logistic regression, we use a logistic function, to transform the
        data such that it lies between the 0 and 1. 

        Examples:
        ======================================

        >>> lg = clf_log(matrix_class=pme.custom, 
                                             feature_df=pme.custom.columns_df,
                                             grid_search_random=0.1, original_cols=pme.custom.original_cols,
                                             bst=True, num_folds=500)
        >>> lg.log_model_search()
        >>> lg.train()
        >>> lg.linear_model_search()
        >>> lg.train()
        >>> lg.test()
        >>> 0.831
        """
        clf.__init__(self, **kwargs)
        self.c = 0
        self.params = 0
        self.alpha = 0
        self.l1_ratio = 0
        self.model_type = model_type

    def get_model_params(self):
        """A function just to view the model parameters"""
        print(self.current_model.get_params())

    def log_model_search(self):
        """Perform a grid search on the parameter space using a log range"""
        self.compute_hyperparameter_ranges()
        none_model = [SGDClassifier(penalty='none', loss='log', class_weight={1: 0.07, 0: 1 - 0.07})]
        l2_model = [LogisticRegression(penalty='l2', C=x, class_weight={1: 0.07, 0: 1 - 0.07}) for x in self.c]
        l1_model = [LogisticRegression(penalty='l1', C=x, class_weight={1: 0.07, 0: 1 - 0.07}) for x in self.c]
        sgd_parameters = list(itertools.chain(itertools.product(self.alpha, self.l1_ratio)))
        en_model = [SGDClassifier(penalty='elasticnet', loss='log',
                                  alpha=alpha, l1_ratio=l1r, class_weight={1: 0.07, 0: 1 - 0.07}) for alpha, l1r in sgd_parameters]
        if self.model_type is 'none':
            self.models = none_model
        elif self.model_type is 'l1':
            self.models = l1_model
        elif self.model_type is 'l2':
            self.models = l2_model
        elif self.model_type is 'elasticnet':
            self.models = en_model
        elif self.model_type is None:
            self.models = none_model + l2_model + l1_model + en_model

    def linear_model_search(self):
        """Perform a grid search on the parameter space using a linear range"""
        clf = self.models[self.max_index]
        self.models = []
        params = clf.get_params()
        new_params = [self.c]
        if isinstance(clf, LogisticRegression):
            new_params[0] = np.linspace(params['C'] / 2, params['C'] * 2, self.param_space)
            for param in new_params[0]:
                clf_copy = clone(clf)
                self.models.append(clf_copy.set_params(C=param))
        elif isinstance(clf, SGDClassifier):
            self.alpha = np.linspace(params['alpha'] / 2, params['alpha'] * 2, self.param_space)
            self.l1_ratio = np.linspace(params['l1_ratio'] / 2, params['l1_ratio'] * 2, self.param_space)
            sgd_parameters = list(itertools.chain(itertools.product(self.alpha, self.l1_ratio)))
            for alpha, l1r in sgd_parameters:
                clf_copy = clone(clf)
                self.models.append(clf_copy.set_params(alpha=alpha, l1_ratio=l1r))

    def score_model(self, trainX, trainy, valX, valy, model):
        """Function called by cross-validation in order to fit the model and then make predictions"""
        model.fit(trainX, trainy)
        train_pred = model.predict_proba(trainX)[:, 1]
        val_pred = model.predict_proba(valX)[:, 1]
        val_dec_fun = model.decision_function(valX)
        self.prediction_tracker.loc[self.vcoords, self.gc] = val_dec_fun
        self.gc += 1

        val_score = brier_score_loss(valy, val_pred)
        train_score = brier_score_loss(trainy, train_pred)
        score = train_score + val_score
        accuracy = model.score(valX, valy)
        print("val_score %s, train_score %s, accuracy %s" % (val_score, train_score, accuracy))
        if isinstance(model, LogisticRegression) or isinstance(model, SGDClassifier):
            return score, model.coef_
        else:
            return score, 0

    def compute_hyperparameter_ranges(self):
        """Compute the hyperparameter space using some heurtistics"""        
        exponent = np.floor(
            np.log10(np.abs(1 / self.trainX.shape[0]))).astype(int)
        self.gamma = np.logspace(exponent - 1, exponent + 4, self.param_space)
        self.c = np.logspace(exponent, 1, self.param_space)
        self.alpha = np.logspace(exponent, 1, self.param_space)
        self.l1_ratio = np.logspace(exponent, 0, self.param_space)

    def test(self):
        """
        If a test matrix exists, than one can perform a test on
        the best model selected.
        """        
        self.params = self.current_model.get_params()
        self.test_model = self.current_model.fit(self.trainX, self.trainy)
        self.test_score = self.test_model.score(self.testX, self.testy)
        if 'kernel' not in self.params:
            self.coeffs = self.test_model.coef_
        self.conf_score = self.test_model.decision_function(self.testX)
        print("This is the test score:%s" % self.test_score, "Model params=%s" %
              self.test_model.get_params())    


class clf_nn(clf):
    def __init__(self, model_type='binary', **kwargs):
        """
        Setup logistic regression from scikit-learn
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        
        Parameters:
        ======================================
        
        model_type -> String: indicates whethe to perform classification or regression
                      e.g. 'binary' 'reg'

        Attributes:
        ======================================

        layers -> [0-inf] Integer specifies the number of layers

        neurons -> List of positive integers: specifies the number of neurons per layer 

        keep -> [0-inf] Float The probability that a neuron will turn-off (applied to all layers)

        model_type -> String: indicates whethe to perform classification or regression
                      e.g. 'binary' 'reg'

        l1 -> [0-inf] Placeholder for l1 regularization

        l2 -> [0-inf] Placeholder for l1 regularization

        Notes:
        ======================================

        Neural networks are a type of algorithms that drew their original inspiration
        from neurophysiology, as an attempt to mimic how the brain functions. The
        simplest form of the neural net resembles the format of a logistic regression. 
        The complexity begins when multiple logistic functions are chained together
        in order to learn higher-level interactions from the data. Chaining
        multiple functions together, in this case, eliminates the possibility to solve for the
        weights analytically, i.e. there is no closed form solution, but neural networks have
        been proven to approximate any continuous function. Any function can be used as
        long as it has the property of being able to have its derivative taken. However,
        there are a few functions, known as rectified linear units (ReLUs) that have been
        shown to outperform the traditionally used logistic function.

        Here is implemented a Neural network using ReLUs and dropout for regularization.
        L1 and L2 will be implemented in the future.

        Examples:
        ======================================

        >>>    nn = clf_nn(model_type='binary', matrix_class=pme.custom,
                           feature_df=pme.custom.columns_df, grid_search_random=1,
                           original_cols=pme.custom.original_cols, bst=True, num_folds=500)                           

        >>>    parameters = {'layers': 3, 'neurons': [5,10,15],
                             'penalty': 'dropout', 'keep': 0.6, lin:reg':False}
        >>>    model = [parameters]
        >>>    nn.models = model
        >>>    nn.train()
        """        
        clf.__init__(self, **kwargs)
        self.layers = np.linspace(1, 10, self.param_space)  # default is usually 6
        self.neurons = [10 for x in self.layers]
        self.l1 = 0.5
        self.l2 = 0.5
        self.keep = 1
        self.model_type = model_type
        if self.model_type is 'multi':
            arry1 = np.tile(np.unique(self.trainy), self.trainX.shape[0])
            arry2 = np.repeat(np.arange(self.trainX.shape[0]),len(np.unique(self.trainy)))
            tuples = list(zip(arry2,arry1))
            index = pd.MultiIndex.from_tuples(tuples, names=['samples', 'y'])
            self.prediction_tracker = pd.DataFrame(index=index, columns=np.arange(self.num_folds))
    
    def get_model_params(self):
        print(self.models[self.max_index])

    # def log_model_search(self):


    # def linear_model_search(self):


    def score_model(self, trainX, trainy, valX, valy, model):
        """Function called by cross-validation in order to fit the model and then make predictions"""
        nn_model = nn.Graph(trainX=trainX.A, trainy=trainy, parameters=model)
        y_pred, logits = nn_model.fit_predict(valX.A, valy)
        score = self.score(y_pred, valy, logits)
        nn_model = 0
        return score

    def score(self, y_pred, datay, logits):
        """Function called by score_model that calculates all the predictions and stores them"""
        if self.model_type is 'reg':
            self.prediction_tracker.loc[self.vcoords, self.gc] = logits.reshape(-1)
            self.gc += 1
            print("R^2 %s" % y_pred)
            return y_pred, 0
        elif self.model_type is 'binary':
            logits = logits[:, 1]
            predictions = [1 if x >= 0.5 else 0 for x in y_pred[:, 1]]
            accuracy = accuracy_score(datay, predictions)
            val_score = brier_score_loss(datay, y_pred[:, 1])
            self.prediction_tracker.loc[self.vcoords, self.gc] = logits
            self.gc += 1
            print("brier_score %s, accuracy %s" % (val_score, accuracy))
            return val_score, 0
        elif self.model_type is 'multi':
            self.prediction_tracker = self.prediction_tracker.astype(object)
            y_pred = y_pred.argmax(axis=1)
            accuracy = accuracy_score(datay, y_pred)
            for i, v in enumerate(self.vcoords):
                self.prediction_tracker[self.gc][v] = logits[i]
            self.gc += 1
            print("accuracy %s" % (accuracy))
            return accuracy, 0

    # def compute_l1_l2_range(self):
        # auto = np.log10(1 / self.trainX.shape[0]).astype(int)
        # self.l1 = np.logspace(auto, 1, self.param_space)
        # self.l2 = np.logspace(auto, 1, self.param_space)

    # def test(self):