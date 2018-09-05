from fisher import pvalue
import itertools
import numpy as np
import operator
import pickle
import pandas as pd


class filter_palette:
    def __init__(self, loci_variant_dataframe, save_to_output=False):
        """
        Filter Features using domain knowledge

        Parameters:
        ======================================

        loci_variant_position-> panda df: contains all the relevant information to select matrix columns that are going to be kept or removed.

        save_to_output -> Boolean: Save the matrix

        Attributes:
        ======================================

        gnomad_file -> String: Path to the gnomad dataset

        col_translator-> panda df: contains all the relevant information to select matrix columns that are going to be kept or removed.

        standard_filters -> list strings: two forms of generic filtering methods that can be built from the gnomad dataset

        filtered_columns -> list: contains the columns to be kept or removed

        filename -> String: Name of the filtered matrix, named by type of filter used

        save -> Boolean: Whether to save the matrix or not default is not


        Notes:
        ======================================

        We have 2 standard filter methods based on the gNOMAD dataset
        whereby we remove snps if they are uncommon for a certain ethnicitiy
        or the alleles are too common to the population. Then there is the
        general method to filter using custom databases/annotations of the
        snps. This allows for more flexibility when deciding which features
        will be the most important. Though it must be understood that this
        particular setup only removes features. I will add a keep feature
        mechanism, but this may become a very harsh filter since the marjority
        of the snps will not actually be covered by the database.

        Examples:
        ======================================

        >>>  None
        """
        self.gnomad_file = 'Matrices/input/gNOMAD_filter'
        self.col_translator = loci_variant_dataframe
        self.standard_filters = ['ethinic_filter_gNOMAD', 'rare_allels_gNOMAD']
        self.filtered_columns = []
        self.filename = ''
        self.save = save_to_output

    def choose_data_from_df(self, dataframe):
        chrm_pos = dataframe['chrm_pos']
        custom_filters = list(dataframe.columns)
        filters = self.standard_filters + custom_filters
        # prompt user to select features need comma to separate
        selection = input('Select features %s ' % filters)
        selection = [x for x in selection.split(',')]
        for selected in selection:
            if selected not in self.standard_filters:
                df = dataframe[selected].dropna()
                if isinstance(df.iloc[0], str) is True:
                    df = df[df != '.']
                oper = input('Provide comparator: eg >, <, >= ')
                user_thresh = input('Provide threshold: eg Float or String ')
                self.filename += '_'.join([str(selected), oper, user_thresh])
                try:
                    thresh_float = float(user_thresh)
                    if 'ref' and 'alt' in dataframe.columns:
                        chrm_pos_ref_alt = dataframe['chrm_pos'].map(str) + "_" + dataframe['ref'].map(str) + "_" + dataframe['alt'].map(str)
                        self.general_filter(df=df, chrm_pos=chrm_pos_ref_alt,
                                            oper=oper, thresh=thresh_float)
                    else:
                        self.general_filter(df=df, chrm_pos=chrm_pos,
                                            oper=oper, thresh=thresh_float)
                except ValueError:
                    thresh_string = user_thresh
                    if 'ref' and 'alt' in dataframe.columns:
                        chrm_pos_ref_alt = dataframe['chrm_pos'].map(str) + "_" + dataframe['ref'].map(str) + "_" + dataframe['alt'].map(str)
                        self.general_filter(df=df, chrm_pos=chrm_pos_ref_alt,
                                            oper=oper, thresh=thresh_string)
                    else:
                        self.general_filter(df=df, chrm_pos=chrm_pos,
                                            oper=oper, thresh=thresh_string)
            elif selected == 'ethinic_filter_gNOMAD':
                self.filename += '_'.join([str(selected)])
                self.ethinic_filter_gNOMAD()
            elif selected == 'rare_allels_gNOMAD':
                user_thresh = input('Provide threshold: Float ')
                self.filename += '_'.join([str(selected), user_thresh])
                self.rare_allels_gNOMAD(user_thresh)
        print("%s column(s) are %s %s when you applied the %s filter(s)" % (len(self.filtered_columns), user_thresh, oper, ' '.join(selection)))

    def general_filter(self, df, chrm_pos, oper='>', thresh=10):
        """
        pass a dataframe with annotation data that can carry either a 
        loci format or an allele format. Allele format takes variant 
        information into account were as loci information is purely 
        based on the chromosome position data not the variant call
        data. In the future this method will be restricted to cateogorical
        calls such as damaging or tolerate. I think using threshold values 
        will need to be something that ends up in the cross validation section.
        """

        comparator = {"<=": operator.le, "<": operator.lt,
                      "==": operator.eq, "!=": operator.ne,
                      ">=": operator.ge, ">": operator.gt}
        ops = comparator[oper]
        filtered_index = df[ops(df, thresh)].index
        chrm_pos_index = chrm_pos[filtered_index]
        # for when your table has variant specific information
        if len(chrm_pos.loc[0].split("_")) > 2:
            self.filtered_columns = self.col_translator[self.col_translator['chrm_pos_ref_alt'].isin(chrm_pos_index)].index
        # for when your table has loci specific information
        else:
            self.filtered_columns = self.col_translator[self.col_translator['chrm_pos'].isin(chrm_pos_index)].index

    def build_remove_filtered_matrix(self, sparse_matrix):
        """
        typical to use when getting rid of tolerant mutations.
        Pass through a sparse_matrix without the labels
        """
        cols = sparse_matrix.shape[1]
        rg_cols = np.arange(cols)
        new_cols = np.setdiff1d(rg_cols, self.filtered_columns.values)
        filtered_matrix = sparse_matrix[:, new_cols]
        if self.save is True:
            save_sparse_csr('FilteredMatrix/%s' % self.filename,
                            filtered_matrix)
        return new_cols, filtered_matrix

    def build_keep_filtered_matrix(self, sparse_matrix):
        """
        typical to use when selecting a set of genes based on expression patterns
        or pathway associations.
        Pass through a sparse_matrix without the labels
        """
        filtered_matrix = sparse_matrix[:, self.filtered_columns]
        if self.save is True:
            save_sparse_csr('FilteredMatrix/%s' % self.filename,
                            filtered_matrix)
        return self.filtered_columns, filtered_matrix

    def ethinic_filter_gNOMAD(self):
        """
        Pass a file with the gnomad dataset filter for the genes of interest.
        For example a dataset might intersects with 800,000 snps (determined by
        wc -l of the file. We will test to see if any of those are candidates
        for removal. The pvalue threshold will be 0.05 / 28 (28 being the number
        of ethnic combinations). If 1 of 28 combinations are indeed a hit for
        signif. then we toss that SNP from the dataset as it could be missused
        for separating disease vs non-dieases by ethnicity instead of by diease
        bearing snps.
        """
        cutoff = 0.05 / 28
        snps_to_remove = []
        with open(self.gnomad_file, 'r') as fin:
            line_num = 0
            for line in fin:
                div = line.split("\t")
                chrm_info = div[0]
                genotype_list = []
                for ethnicity in range(8):  # 8 for number of ethn. in gnomad
                    genotype_list.append(div[1].split(",")[ethnicity])
                for i in itertools.combinations(genotype_list, 2):
                    n11 = int(i[0].split(" ")[0])
                    n12 = int(i[0].split(" ")[1])
                    n21 = int(i[1].split(" ")[0])
                    n22 = int(i[1].split(" ")[1])
                    pval = pvalue(n11, n21, n12, n22).two_tail
                    if pval <= cutoff:
                        snps_to_remove.append(chrm_info)
                        break
                line_num += 1
                print(line_num / 15008010, end='\r')  # percent complete
        self.filtered_columns = self.col_translator[self.col_translator['chrm_pos_ref_alt'].isin(snps_to_remove)].index

    def rare_allels_gNOMAD(self, thershold=0.01):
        """
        If an allele is commonly found in the gNOMAD dataset then we remove
        that snp using a specified threshold.
        """        
        t = thershold
        snps_to_remove = []
        with open(self.gnomad_file, 'r') as fin:
            line_num = 0
            for line in fin:
                div = line.split("\t")
                chrm_info = div[0]
                genotype_list = []
                for ethn in range(8):  # 8 for number of ethn. in gnomad
                    genotype_list.append(div[1].split(",")[ethn])
                for i in genotype_list:
                    n11 = int(i.split(" ")[0])
                    n12 = int(i.split(" ")[1])
                    if self.safe_division(n12, n11) >= t:
                        snps_to_remove.append(chrm_info)
                        break
                line_num += 1
                print(line_num / 15008010, end='\r')  # percent complete
        self.filtered_columns = self.col_translator[self.col_translator['chrm_pos_ref_alt'].isin(snps_to_remove)].index

    def safe_division(self, n, d):
        return n / d if d else 0
