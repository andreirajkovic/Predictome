import numpy as np


class data_hangar:
    def __init__(self, sparse_matrix):
        """
        Create a train test split 80:20

        Parameters:
        ======================================

        sparse_matrix -> a sparse_matrix csr: Contains the explaintory and response variables

        Attributes:
        ======================================

        core_matrix -> a sparse_matrix csr: Contains the explaintory and response variables

        culled_matrix-> a sparse_matrix csr: Contains the explaintory and response variables after outliers have been removed

        testX -> a sparse_matrix csr: Contains the explainatory variables for the test split

        testy -> numpy array: Contains the response variables for the test split

        trainX -> a sparse_matrix csr: Contains the explainatory variables for the train split

        trainy -> numpy array: Contains the response variables for the train split

        snp_dist -> numpy array: each row is a sample and the columns represent the number of hets,homos,missing values ect

        rowindex -> numpy array: the original row indices saved after outlier removal has been performed

        test_index -> numpy array: Contains the original rows values that have been put in the test data

        train_index -> numpy array: Contains the original rows values that have been put in the train data

        Notes:
        ======================================

        Use this class only if enough data exists such that 80% of the effects can be explained by 20% of the data.
        If the dataset size is less than 20,000 this may not be the best method for dividing the data. Instead
        one should use bootstrapping. Also the data should ideally reflect accurate distributions of the classes.

        Examples:
        ======================================

        >>>  from splitData import data_hangar
        >>>  train_test = data_hangar('MetaSVM_score_<=_0.npz')
        >>>  train_test.load_data(True,row_data=pVCF.row_pos)
        """

        # input
        self.core_matrix = sparse_matrix
        self.culled_matrix = 0
        # test data
        self.testX = 0
        self.testy = 0
        # train data
        self.trainX = 0
        self.trainy = 0
        # important parameters
        self.snp_dist = 0
        self.rowindex = 0
        self.test_index = 0
        self.train_index = 0

    def check_multiclass(self):
        """Looks at the last col of the matrix to determine num of classes"""
        num_classes = len(np.unique(self.core_matrix[:, -1].toarray()))
        if num_classes > 2:
            return True
        else:
            return False

    def load_data(self, check_rows=False, row_data={}):
        """
        Transforms the design matrix into a suitable training 
        and test format.

        Parameters:
        ======================================

        check_rows -> Boolean: Whether to remove outliers based on row information

        row_data -> Dictionary: This should be in the same format as the variable row_pos
                                in the getAttributes class

        Notes:
        ======================================

        None 

        Examples:
        ======================================

        >>>  None
        """
        if check_rows is True:
            self.clean_data()
        elif check_rows is False:
            self.culled_matrix = self.core_matrix
        print('Data Selected')
        multiclass = self.check_multiclass()
        if multiclass is True:
            test_index, train_index = self.make_index_multiclass()
        elif multiclass is False:
            test_index, train_index = self.make_index_binary_classes()
        self.buildFolds(test_index, train_index)
        self.convert_test_index(row_data)
        print('Training and test split created')

    def clean_data(self):
        """Remove the rows that have outlier properties"""
        self.detectRowOutliers()  # compute genotype distribution
        remove_rowindex = np.unique(np.hstack(
            (self.removeRowOutliers(-1, skewed=True, bias=True),
             self.removeRowOutliers(0), self.removeRowOutliers(1))))
        self.rowindex = np.setdiff1d(np.linspace(
            0, self.core_matrix.shape[0] - 1,
            self.core_matrix.shape[0], dtype=int),
            remove_rowindex)
        print(self.rowindex.shape)
        self.culled_matrix = self.core_matrix.tocsr()[self.rowindex]

    def detectRowOutliers(self, verbose=True):
        """
        Compute the snp distributions

        Parameters:
        ======================================

        verbose -> Boolean: Shows the computation of the snp_distributions

        Notes:
        ======================================
        Each row has their number of unique values counted.
        These values are in the order of [NOCALL,HET,HOMO], depending
        on the type of design matrix. These values are then used to
        determine if an individual has too much or too few of
        a certain value.

        Examples:
        ======================================

        >>>  None
        """
        i = 0
        self.snp_dist = np.zeros(
            (self.core_matrix.shape[0], 4), dtype=np.int32)
        for row in range(self.core_matrix.shape[0]):
            snp_hits = np.unique(
                self.core_matrix[row, :-1].data, return_counts=True)
            if verbose:
                print(snp_hits, i)
            self.snp_dist[i, snp_hits[0]] = snp_hits[1]
            i += 1
        self.snp_dist = self.snp_dist[:, 1:]

    def removeRowOutliers(self, col, skewed=False, bias=False):
        """
        Compute the snp distributions

        Parameters:
        ======================================

        skewed -> Boolean: Whether to use a transformation

        bias-> Boolean: Which direction of the skew should be removed 

        Notes:
        ======================================
        Removing outliers is a bit tricky
        because our SNP distribution might be skewed
        for the nocall and ref. Therefore we
        will transform the data using a
        cube root / Anscombe transform.
        This will introduce a bias but may help
        remove some of the unwanted variance. A
        standard deviation of 1.75 is used to remove
        the outliers.

        Examples:
        ======================================

        >>>  None
        """        
        
        _std = np.std(self.snp_dist[:, col])
        _mean = np.mean(self.snp_dist[:, col])
        if skewed is True and bias is False:
            transformed_ = np.power(self.snp_dist[:, col] + 3 / 8, 1 / 4)
            # the sign is switched to indicate the rows we want to keep
            keep_left = np.where(transformed_ > np.mean(
                transformed_) + 1.75 * np.std(transformed_))[0]
            # the sign is switched to indicate the rows we want to keep
            keep_right = np.where(transformed_ < np.mean(
                transformed_) - 1.75 * np.std(transformed_))[0]
            print('Mean', np.mean(transformed_), 'STD', np.std(transformed_),
                  'Left', keep_left.shape, 'Right', keep_right.shape)
            return np.hstack((keep_left, keep_right))
        elif skewed is True and bias is True:
            transformed_ = np.power(self.snp_dist[:, col] + 3 / 8, 1 / 4)
            # the sign is switched to indicate the rows we want to keep
            keep_left = np.where(transformed_ > np.mean(
                transformed_) + 1.75 * np.std(transformed_))[0]
            print('Mean', np.mean(transformed_), 'STD',
                  np.std(transformed_), keep_left.shape)
            return keep_left
        elif skewed is False and bias is False:
            keep_left = np.where(
                self.snp_dist[:, col] > _mean + 1.75 * _std)[0]
            keep_right = np.where(
                self.snp_dist[:, col] < _mean - 1.75 * _std)[0]
            print('Mean', _mean, 'STD', _std, 'Left',
                  keep_left.shape, 'Right', keep_right.shape)
            return np.hstack((keep_left, keep_right))

    def make_index_multiclass(self):
        """
        Randomly assign row indices to a test and training split
        when the data is not binary. The classes are balanced in
        the test. This is assuming the data comes from a cohort
        control setup.
        """        
        # Count the number of classes
        labels = self.culled_matrix.tocsr()[:, -1].toarray().T[0]
        label_values, counts = np.unique(labels, return_counts=True)
        num_classes = len(label_values)
        # choose the indices
        num_samples = self.culled_matrix.shape[0]
        num_test_samples = num_samples * 0.20
        num_test_samples_per = int(num_test_samples // num_classes)
        self.test_index = []
        for label in label_values:
            # location of each label
            choices = np.where(labels == label)[0]
            choices = choices.astype(int)
            label_index = np.random.choice(choices, size=num_test_samples_per,
                                           replace=False)
            self.test_index += label_index.tolist()
        # make an 80 train :20 test split
        self.train_index = np.setdiff1d(np.arange(num_samples),
                                        self.test_index)
        return self.test_index, self.train_index

    def make_index_binary_classes(self):
        """
        Randomly assign row indices to a test and training split
        when the data is binary
        """
        # make an 80 train :20 test split
        case_i = np.where(self.culled_matrix.tocsr()[:, -1].toarray() == 1)[0]
        ctrl_i = np.where(self.culled_matrix.tocsr()[:, -1].toarray() == 0)[0]
        train_index_ctrl = np.random.choice(ctrl_i, size=round(
            (self.culled_matrix.shape[0] * .80) / 2), replace=False)
        train_index_case = np.random.choice(case_i, size=round(
            (self.culled_matrix.shape[0] * .80) / 2), replace=False)
        test_index_ctrl = np.setdiff1d(ctrl_i, train_index_ctrl)
        test_index_case = np.setdiff1d(case_i, train_index_case)
        # convert to correct indices
        self.test_index = np.hstack((test_index_case, test_index_ctrl))
        self.train_index = np.hstack((train_index_ctrl, train_index_case))
        return self.test_index, self.train_index

    def buildFolds(self, test_index, train_index, verbose=True):
        """Populate Training Data"""
        sparse_train_array = self.culled_matrix.tocsr()[train_index]
        self.trainX = sparse_train_array.tocsc()[:, :-1]
        self.trainy = sparse_train_array.tocsc()[:, -1].A.T[0]
        """Populate Test Data"""
        test_array = self.culled_matrix.tocsr()[test_index]
        self.testX = test_array.tocsc()[:, :-1]
        self.testy = test_array.tocsc()[:, -1].A.T[0]
        """Extract Important Data Features"""
        print('culled_matrix:', self.culled_matrix.shape)

    def convert_test_index(self, row_cnvrt):
        """since we manipulate the row positions we need
        to go back and determine which samples belong to the
        test rows. This is a very static method that requires
        to load the file from an output directory. I will change
        this to take the variable from the class obj parse_VCF"""
        self.rowid_sample = {}
        for key, value in row_cnvrt.items():
            if value in self.test_index:
                self.rowid_sample[value] = key
