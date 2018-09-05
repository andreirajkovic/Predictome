import numpy as np
import scipy
from scipy import sparse
from fisher import pvalue
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
import re



class divideTraining:
    def __init__(self, trainy, num_folds=10):
        """
        Randomly select a training and stratified validation set

        Parameters:
        ======================================

        trainy -> np.array labels from the design matrix

        num_folds -> integer k-folds

        Attributes:
        ======================================
        
        label_values -> np.array to determine the num_classes

        frozen_label_indices -> dictionary of the class and the row indicies belonging to that class

        fold_size-> number of samples that will be in the holdout dataset

        size-> np.array total indices for that training data size

        Returns:
        ======================================

        Nothing

        Notes:
        ======================================

        During training it is common to perform cross-validation in order to select certain hyperparameters.
        Here the divideTraining class splits the data and keeps memory of what indices were previously selected,
        such that the next time a split occurs the next values in the fold will be selected for the hold-out set.

        Examples:
        ======================================
        
        None

        """   

        self.label_values, counts = np.unique(trainy, return_counts=True)
        num_classes = len(self.label_values)
        self.frozen_label_indices = {}
        for label in self.label_values:
            self.frozen_label_indices[label] = np.where(trainy == label)[0]
        # divide by number of classes so data is stratified
        self.fold_size = trainy.shape[0] // (num_folds * num_classes)
        self.size = np.arange(trainy.shape[0])
        self.trainy = trainy

    def choseIndex(self):
        """
        Choose the incides for the valiation sets

        Parameters:
        ======================================

        None

        Attributes:
        ======================================
        
        None

        Returns:
        ======================================

        big_slice -> list: indicies of the training set
        small_slice -> list: indicies of the validation set

        Notes:
        ======================================

        None

        Examples:
        ======================================
        
        None

        """          
        self.check_indices_size()
        # populate the indicies
        small_slice = []
        for label in self.label_values:
            choice = self.frozen_label_indices[label]
            label_index = np.random.choice(choice, size=self.fold_size,
                                           replace=False)
            small_slice += label_index.tolist()
            self.frozen_label_indices[label] = np.setdiff1d(choice, label_index)
        big_slice = [[x] for x in np.setdiff1d(self.size, small_slice)]
        small_slice = [[x] for x in small_slice]
        return big_slice, small_slice

    def repopulateFolds(self):
        self.frozen_label_indices = {}
        for label in self.label_values:
            self.frozen_label_indices[label] = np.where(self.trainy == label)[0]

    def check_indices_size(self):
        for k, v in self.frozen_label_indices.items():
            if len(v) < self.fold_size:
                self.repopulateFolds()
                break


class bootstrapping:
   
    def __init__(self, y):
        """
        Select indices based on bootstrapping

        Parameters:
        ======================================

        y -> np.array of the labels from the design matrix

        Attributes:
        ======================================
        
        n_instances -> np.array Array of incides representing the number of sample rows

        Returns:
        ======================================

        None

        Notes:
        ======================================

        Bootstrapping is an alternative to cross-validation. Bootstrapping introduces more bias into the model, but less variance.
        This is due to duplicates that are added after each sampling event.  

        Examples:
        ======================================
        
        None

        """    
        self.n_instances = np.arange(len(y))
    
    def choseIndex(self):
        bst_train = np.random.choice(self.n_instances, size=len(self.n_instances))
        bst_test = np.setdiff1d(self.n_instances, bst_train)
        big_slice = [[x] for x in bst_train]
        small_slice = [[x] for x in bst_test]
        return big_slice, small_slice



def convert2csr(trainX):
    """
    Convert a array into a sparse row matrix

    Parameters:
    ======================================

    trainX -> sparse matrix

    Attributes:
    ======================================
    
    None

    Returns:
    ======================================

    csr_matrix

    Notes:
    ======================================

    Used to converted between the different formats of sparse matrices

    Examples:
    ======================================
    
    None

    """   

    return sparse.csr_matrix(trainX)


def convert2csc(trainX):
    """
    Convert a array into a sparse column matrix

    Parameters:
    ======================================

    trainX -> sparse matrix

    Attributes:
    ======================================
    
    None

    Returns:
    ======================================

    csc_matrix

    Notes:
    ======================================

    Used to converted between the different formats of sparse matrices

    Examples:
    ======================================
    
    None

    """ 
    return sparse.csc_matrix(trainX)


def generalFisherExact_sparse(col_prbs_n, col_prbs_d, size1, size2):
    """
    Use this function to compute the signifcance of each feature by fisher exact

    Parameters:
    ======================================
    
    col_prbs_n -> numpy array: a two dimensional array with each row a feature and each column the number of ref / alt for the control
    
    col_prbs_d -> numpy array: a two dimensional array with each row a feature and each column the number of ref / alt  for the case
    
    size1 -> number of rows for the control
    
    size2 -> number of rows for the case
    
    Returns:
    ======================================

    pvalue_array -> A vector array with n elements as the number of n features in the matrix
    or matrix.shape[1] with a p-value for each feature.
    
    odds_ratio -> A vector array with n elements as the number of n features in the matrix
    or matrix.shape[1] with the log_odds ratio for each feature.

    Notes:
    ======================================

    Fisher Exact Test evaluates the whether the case and control populations are independent.
    There are a few ways by which we could setup the table to select significant values. One is
    to assume that all loci produce alleles from the same distribution and that we are searching
    for loci that differ significantly from the case and control populations. We may also assume
    that the alleles contained in each loci are independent of one another eg heterozygote,
    and homozygote have independent distributions.  This method does not take into consideration
    that a heterozygous allele will have its population split between case and control, at least 
    in the case of allele and variant matrices.
    

    Examples:
    ======================================

    >>> pval2, lod outs=generalFisherExact_sparse(col_prbs_n, col_prbs_d, )

    """
    features = col_prbs_d.shape[0]
    min_val = cap(size1, size2)
    pvalue_array = np.ones(features, dtype=np.float)
    odds_ratio = np.zeros(features, dtype=np.float)
    CASE_REF = col_prbs_d[:, 0]
    CASE_ALT = col_prbs_d[:, 1]
    CTRL_REF = col_prbs_n[:, 0]
    CTRL_ALT = col_prbs_n[:, 1]
    NN = np.hstack((CASE_REF.reshape(-1, 1), CTRL_REF.reshape(-1, 1),
                    CASE_ALT.reshape(-1, 1), CTRL_ALT.reshape(-1, 1)))
    NN = NN.astype(int)
    proxy_column = NN[:, 2] - NN[:, 3]  # heuristics
    proxy_column = np.abs(proxy_column.reshape(-1, 1))
    grt_coords = np.where(proxy_column >= min_val)[0]
    any_coords = np.where(proxy_column.any(axis=1))[0]
    coords = np.hstack((any_coords, grt_coords))
    c = 0
    for feat in coords:
        pvalue_array[feat] = pvalue(*NN[feat]).two_tail
        if NN[feat, 2] > 0 and NN[feat, 1] > 0:
            odds_ratio[feat] = np.log(NN[feat, 0]) + np.log(NN[feat, 3]) - np.log(NN[feat, 2]) - np.log(NN[feat, 1])
        else:
            odds_ratio[feat] = np.inf
        print("Computing Fisher Exact: {:.2} % ".format(
            (c / len(coords)) * 100), end="\r")
        c += 1
    return pvalue_array, odds_ratio


def general_chi_squared_sparse(refs, c_table):
    """
    perform fisherexact test on a sparse variant matrix

    Parameters:
    ======================================
    matrix_ctrl -> A sparse CSC matrix that holds the control values
    matrix_case -> A sparse CSC matrix that holds the case values
    shapes -> a list of the dimensions for the case_matrix and then ctrl_matrix
    variant_locations -> A list of two dictionaries the first holding the heterozygous
    locations and the second holding the homozygous locations
    Returns:
    ======================================

    A vector array with n elements as the number of n features in the matrix
    or matrix.shape[1]

    Notes:
    ======================================

    Use this function to compute the signifcance of each feature. This method
    does not take into consideration that a heterozygous allele will have its
    population split between case and control. Therefore some accuracy will be
    lost in this method for now. A future verision will take a dictionary of
    the locations where the heterozygous features (0/x) are and then add a conditional
    to the algorithm such that it takes this information into account and then
    modifies the REF values

    Examples:
    ======================================

    >>> pval2=generalFisherExact_variant_sparse(col_prbs_n, col_prbs_d)
    """
    # het_locations = variant_locations[0]
    # homo_locations = variant_locations[1]
    features = c_table.shape[1]
    pvalue_array = np.ones(features, dtype=np.float)
    c = 0
    for feat in range(features):
        x2_table = np.vstack((c_table[:, feat], refs - c_table[:, feat]))
        chi2, p, dof, exp = chi2_contingency(x2_table)
        pvalue_array[feat] = p
        print("Computing chi^2 Exact: {:.2} % ".format(
            (c / features) * 100), end="\r")
        c += 1
    return pvalue_array


def cap(size1, size2):
    """
    This function finds the population value at which we
    begin to obtain a signiciant value. This may need to
    be modified depending on the training size.
    """
    lowest_sig = 0.05
    for n in range(1, 10):
        if size1 < size2:
            sig = pvalue(size2, size1 - n, 0, n).two_tail
            return n
        else:
            sig = pvalue(size1, size2 - n, 0, n).two_tail
            if sig <= 0.05:
                return n

def removeInsignificant(trainX):
    """
    Heuristics to remove features that will never yield a significant value
    Or remove regions where there are too many missing values in the column
    """
    if scipy.sparse.issparse(trainX) is False:
        sp_rw = convert2csr(trainX)
    else:
        sp_rw = trainX.tocsr()
    coords = sp_rw != 0
    row, col, _ = sparse.find(coords)
    sig_col = np.unique(col)[np.unique(col, return_counts=1)[1] >= cap(sp_rw.shape[0])]
    print("You have selected %s column(s)" % len(sig_col))
    return sig_col, sp_rw.tocsc()[:, sig_col]


def removeInvariant(trainX, threshold=0):
    """
    Remove columns with little or no variantion. Basically a column with all 0's or all 1's
    is not going to be informative when trying to separate normal and diseased individuals.
    Do this after imputation.
    """
    if scipy.sparse.issparse(trainX) is False:
        var_col = np.where(np.var(trainX, axis=0) > threshold)[0]
        print("You have selected %s column(s)" % len(var_col))
        return var_col, trainX[:, var_col]
    else:
        csc_X = trainX.tocsc()
        scaler = StandardScaler(with_mean=False)
        scaler.fit(csc_X)
        variance = scaler.var_
        var_col = np.where(variance > threshold)[0]
        print("You have selected %s column(s)" % len(var_col))
        trainX = csc_X[:, var_col].tocsr()
        return var_col, trainX

def removeMissing_classSize(trainX, trainy):
    """
    Small datasets will have features with lots of missing data, especially if
    that dataset comes from many different cohorts. In order to combat that we must
    either impute or remove all columns with missing data. This function
    determines the smallest class size in the dataset and removes columns with more
    missing values than the size of the smallest class.
    """
    thresh = np.min(np.unique(trainy, return_counts=1)[1])
    if scipy.sparse.issparse(trainX) is False:
        sp_rw = convert2csr(trainX)
    else:
        sp_rw = trainX.tocsr()
    coords = sp_rw == -1
    org_coords = np.linspace(0, trainX.shape[1] - 1, trainX.shape[1],
                             dtype=np.int)
    row, col, _ = sparse.find(coords)
    no_miss = np.setdiff1d(org_coords, col)
    sig_col = np.unique(col)[np.where(np.unique(col,
                                      return_counts=1)[1] < thresh)[0]]
    sig_col = np.hstack((sig_col, no_miss))
    sig_col.sort()
    print("You have selected %s column(s)" % len(sig_col))
    return sig_col, sp_rw.tocsc()[:, sig_col]


def removeMissing_threshold(trainX, threshold=0.01):
    """
    Small datasets will have features with lots of missing data, especially if
    that dataset comes from many different cohorts. In order to combat that we must
    either impute or remove all columns with missing data. This function
    takes a float value between 0-1 and selects only columns that contain the
    specified percentage of missing values or fewer.
    """    
    thres = int(trainX.shape[0] * threshold)
    if scipy.sparse.issparse(trainX) is False:
        sp_rw = convert2csr(trainX)
    else:
        sp_rw = trainX.tocsr()
    coords = sp_rw == -1
    org_coords = np.linspace(0, trainX.shape[1] - 1, trainX.shape[1],
                             dtype=np.int)
    row, col, _ = sparse.find(coords)
    no_miss = np.setdiff1d(org_coords, col)
    sig_col = np.unique(col)[np.where(np.unique(col,
                                      return_counts=1)[1] < thres)[0]]
    sig_col = np.hstack((sig_col, no_miss))
    sig_col.sort()
    print("You have selected %s column(s)" % len(sig_col))
    return sig_col, sp_rw.tocsc()[:, sig_col]


def convert_sparse_to_pseudo_VCF(feature_def, sample_def, sparse_matrix):
    """
    Assumption is that the matrix is formatted in such a way that the
    order of the feature defs and sample defitions are aligned with the
    matrix
    convert_sparse_to_pseudo_VCF(loci_df, wd, matrix)
    """
    parse_gt = {1: '0/1', 2: '1/1', -1: './.', 0: '0/0'}
    num_classes = np.unique(sparse_matrix[:, -1].A.T[0])
    for label in num_classes:
        row_index = np.where(sparse_matrix[:, -1].A.T[0] == label)[0]
        list_of_samples = sample_def[row_index].values.tolist()
        first_list = "\t".join([str(x) for x in list_of_samples])
        with open('/Volumes/Backup/%s.vcf' % label, 'w') as vcfout:
            vcfout.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t%s\n" % first_list)
            sparse_matrix = sparse_matrix.tocsc()
            for i in range(sparse_matrix.shape[1] - 1):  # travel across the features and make vcf line
                raw_values = sparse_matrix[row_index, i].A.T[0]
                variant_values = "\t".join([parse_gt[x] for x in raw_values])
                chrm = feature_def['CHR'][i]
                pos = feature_def['POS'][i]
                ref = feature_def['REF'][i]
                alt = feature_def['ALT'][i]
                """emulate X   154227805   .   C   A  0/0 ./. 1/1"""
                vcfout.write("%s\t%s\t.\t%s\t%s\t%s\n" % (chrm, pos, ref, alt, variant_values))


def chi2_hadoop(sparse_matrix, n_copies=50):
    """
    The chi2 calculations can take a long time to compute on a personal computer.
    It is therefore suggested to use a cluster approach and compute the chi-square
    on a number of pre-computed contigency tables
    """ 
    total_rows = len(np.unique(sparse_matrix[:, -1].A.T[0]))
    total_col = (sparse_matrix.shape[1] - 1) * n_copies
    matrix_to_print = np.zeros((total_rows  + 2, total_col), dtype=np.int).T
    with open("reference_count.txt", "w+") as fout:
        for n in range(n_copies):
            column_i = np.arange(sparse_matrix[1] - 1)
            bsidx = np.random.choice(np.arange(sparse_matrix.shape[0]), size=sparse_matrix.shape[0])
            bootstrap_matrix = sparse_matrix[bsidx.tolist()]  # we dont actually care about row info
            y_lab = bootstrap_matrix[:, -1].A.T[0]
            _ , refs = np.unique(y_lab, return_counts=1)
            rf = np.array2string(refs, separator=',')
            rf = rf.replace(" ","")
            rf = rf.replace("[","")
            rf = rf.replace("]","")
            print(rf)
            fout.write(rf+"\n")
            c_table = np.zeros([len(np.unique(y_lab)), bootstrap_matrix.shape[1] - 1], dtype=int)
            for u in np.unique(y_lab):
                c_table[u] = bootstrap_matrix[np.where(y_lab == u)[0].tolist(),:-1].sum(axis=0)
            s = n * len(column_i)
            e = (n + 1) * len(column_i)    
            matrix_to_print[s: e, 2:] = c_table.T
            matrix_to_print[s: e, 0] = column_i
            matrix_to_print[s: e, 1] = n

    fout.close()
    np.savetxt('c_table.txt', matrix_to_print, delimiter=',', fmt='%i')
    print("COMPLETE")
