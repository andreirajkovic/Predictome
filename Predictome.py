import numpy as np
import pandas as pd
from Matrices.src.buildMatrix import loci_matrix, allele_matrix, variant_matrix, custom_matrix, load_sparse_csr
from Classifiers.src.classifiers import clf_log, clf_nn, clf_gb
from Matrices.src.splitData import data_hangar
from Databases.src.buildDatabase import generate_featuretables


class Predictome:
    def __init__(self):
        """
        Build the design matrix from the VCF files
        
        Parameters:
        ======================================
        
        None

        Attributes:
        ======================================

        custom -> buildMatrix class Custom placeholder

        loci -> buildMatrix class Loci placeholder

        allele -> buildMatrix class Allele placeholder

        variant -> buildMatrix class variant placeholder
        
        table_builer -> table containing feature annotations
        
        filtered_matrix ->  matrix features selected on domain knowledge
        
        exome_data ->  holds the training and test data
    
        tables -> list of dataframes with annotation information
        
        Notes:
        ======================================

        This class serves as a conduit between the buildMatrix, Classifier, and Annotation Tables

        Examples:
        ======================================

        >>> import Predictome as pme
        >>> import pme.design_matrix.build_from_VCF
        """
        self.custom = 0
        self.loci = 0
        self.variant = 0
        self.allele = 0
        self.table_builder = 0
        self.filtered_matrix = 0
        self.exome_data = 0
        self.tables = []

    def build_from_VCF(self, matrix_type='variant',
                       vcf_folders_by_labels=['CASE', 'CTRL'],
                       matrix=None, df_for_filter=None, row_data=None):
        """
        Setup data_matrix from vcf file

        Parameters:
        ======================================

        matrix_type -> ['variant', 'loci', 'allele', 'custom'] select a matrix type using one
        of the strings shown.

        vcf_folders_by_labels -> The vcf file names also a String type.

        matrix -> .npz file holding a premade matrix

        df_for_filter -> premade csv file holding all the relevant snp information eg column
                         positions
        row_data -> saved csv or pickled file that contains the position of the rows in a
                    dictionary format.

        Notes:
        ======================================

        Variant has the greatest flexibility in representations producing the most
        features; Allele is an intermediate form with some linear hiearchy in-place;
        Loci has the least flexiability with a strong linear representation of the
        features. Use the custom option if loading a matrix directly into the class.
        Custom matrix should be loaded only if you have the feature_pos dataframe, sample_pos
        dictionary, and sparse matrix.

        The vcf_folders_by_labels should be divided prior into folders by
        different class labels.

        Examples:
        ======================================

        >>> pme.build_from_VCF(matrix_type='variant',
                              vcf_folders_by_labels=['testing_ctrl','testing_case'])
        >>> pme.variant.make_matrix()

        """
        if matrix_type == 'variant':
            self.variant = variant_matrix(class_folders=vcf_folders_by_labels)
        elif matrix_type == 'loci':
            self.loci = loci_matrix(class_folders=vcf_folders_by_labels)
        elif matrix_type == 'allele':
            self.allele = allele_matrix(class_folders=vcf_folders_by_labels)
        elif matrix_type == 'custom':
            self.custom = custom_matrix(matrix=matrix, df_for_filter=df_for_filter,
                                        row_data=row_data)

    def make_snptable(self,
                      tables_features=[[10, 18, 19, 152, 188, 224, 232, 236],
                                       [2, 3, 59, 188, 189]]):
        """
        Setup data_matrix from vcf file

        Parameters:
        ======================================

        tables_features -> [0, 1, 2, 5], [10, 22, 30] list

        Notes:
        ======================================

        Function takes a list(s) of features that corresponds to a file with columns
        that contain information about the SNPs. In this case the dbNSFP3.4a is used
        to find out information of the SNPs and that information is then used to eventually
        filter SNPs and collect relevant information on them as well.

        Examples:
        ======================================

        >>> pme.make_snptable(tables_features=[[10, 18, 19, 152, 188, 224, 232, 236],
                                              [2, 3, 59, 188, 189]])
        >>> pme.variant.filter_matrix(by='keep_annotated', tables=pme.tables[0])
        >>> Select features ...
        >>> Provide comparator: eg >, <, >= > >
        >>> Provide threshold: eg Float or String 10
        >>> pme.variant.filter_matrix(by='missing')
        >>> pme,reliability()
        >>> pme.variant.impute()
        >>> pme.variant.filter_matrix(by='invariant')

        """
        self.table_builder = generate_featuretables()  # initialize class
        for i, feat in enumerate(tables_features):
            ff = self.table_builder.filtered_files[i]
            cdf = self.table_builder.column_defintion_files[i]
            self.tables.append(self.table_builder.general_table(feat, cdf, ff))

    def split_train_test(self, sparse_matrix, row_data, check_rows=False):
        """
        Divide the matrix into test and training data

        Parameters:
        ======================================

        sparse_matrix ->  sparse csr matrix with the -1 column containing the label info

        row_data -> dictionary with sample name as they key and row position as value

        check_rows -> If your sample size is large >= 10,000 you should perform a row check
        to remove samples that are strongly skewed to have a certain variant call.

        Notes:
        ======================================

        Function takes a sparse matrix and sends it to the splitData class where
        the data is shuffled and split 80:20 training, testing respectively. Can also
        perform a bit of quality control and clean up rows that deviate statistically.

        Examples:
        ======================================

        pme.split_train_test(pme.variant.gt_matrix, pme.variant.atr.row_pos, check_rows=False)

        """
        self.exome_data = data_hangar(sparse_matrix)
        self.exome_data.load_data(check_rows=check_rows,
                                  row_data=row_data)

    def shuffle_train_test(self, multiclass=False):
        """Function to randomize the training and test datasets again"""
        if multiclass is False:
            test_index, train_index = self.exome_data.make_index_binary_classes()
        else:
            test_index, train_index = self.exome_data.make_index_multiclass()
        self.exome_data.buildFolds(test_index, train_index)
