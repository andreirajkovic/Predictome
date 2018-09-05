from functools import reduce
import pandas as pd
import numpy as np
import pickle
import os




class RangeDict(dict):
    def __getitem__(self, item):
        if type(item) != range:
            for key in self:
                if item in key:
                    return self[key]
        else:
            return super().__getitem__(item)

class snp2gene:
    def __init__(self, chrm_pos_dic):
        """
        Convert snp to a ucsc gene id 

        Parameters:
        ======================================

        chrm_pos_dic -> Dictionary: This is the columns_pos variable from getAttributes after a certain filter has been applied i.e. columns_pos['loci']

        Attributes:
        ======================================

        knownGene_table -> String: Path to the text file containing the known genes at a particular region of the human genome

        snps_matrix_pos-> Dictionary: equivalent to the getAttributes variable columns_pos


        Notes:
        ======================================

        This method is to convert all the identified snps from the vcf files into
        gene ids that can then be used by other databases to determine a variety
        of features e.g. clinical significance of a certain gene.

        Examples:
        ======================================

        >>>  None
        """
        self.knownGene_table = 'Databases/input/knownGene.txt'  # file from UCSC genome browser
        self.snps_matrix_pos = chrm_pos_dic

    def read_knownGene(self, verbose=False):
        ucsc_id_region = build_chrm_dic()
        with open(self.knownGene_table, 'r') as fin:
            for line in fin:
                split_line = line.split("\t")
                geneid = split_line[0]
                chrm = split_line[1][3:]
                if chrm in ucsc_id_region:
                    region = range(int(split_line[3]), int(split_line[4]) + 1)
                    ucsc_id_region[chrm][region] = geneid

        with open('Databases/output/key_ucsc_id', 'w') as fout:
            total_lines = len(self.snps_matrix_pos)
            count = 0
            for k, v in self.snps_matrix_pos.items():
                chrm = str(k.split("_")[0])
                pos = int(k.split("_")[1])
                if chrm in ucsc_id_region:
                    geneid = ucsc_id_region[chrm][pos]
                    if geneid is not None:
                        fout.write(k + ' ' + geneid + '\n')
                count += 1
                print(count / total_lines, end='\r')

    def build_chrm_dic():
        d = {}
        for x in range(1, 23):
            d[str(x)] = RangeDict({})
        d['X'] = RangeDict({})
        d['Y'] = RangeDict({})
        return d

    def check(self, split_line, keys):
        """Print some of the parameters when building the dictionary"""
        print('keys', keys)
        print('value', split_line[0])
        print('chrm', split_line[1][3:])
        print('start', int(split_line[3]))
        print('end', int(split_line[4]))


class generate_featuretables:
    def __init__(self):
        """
        Build a table that gives new information to the column attributes

        Parameters:
        ======================================

        None

        Attributes:
        ======================================

        filtered_files -> Tab separated file that contains information on each feature

        column_defintion_files -> Text file that describes the column type e.g. numeric 
                                  or string

        df_final -> List: List of dataframes after performing a join operation


        Notes:
        ======================================

        Contained in the dbNSFP3.4a readme file are the list of columns to
        chose from. The class is intended to house a single database type.
        For example dbNSFP contains variant information as well gene information.
        Therefore two separate tables should be formed.
        Build a table containing information pertainant to the snp/gene
        Pandas will generate two separate tables then join on the chrm_pos key
        FILES: expression_filter # 10_100003785 R3HCC1L,4.7054693698883066,...
        variant_filter    # 10_100008728 C,A,-0.9291,.,.

        Table 1: COLUMNS chrm_pos, genename, FallopianTube , Ovary, Testis , Uterus, Vagina
        Table 2: COLUMNS chrm_pos, ref, alt, MetaSVMScore, clinvar_clnsig, clinvar_trait

        Examples:
        ======================================

        >>>  None
        """        
        self.filtered_files, self.column_defintion_files = self.import_files()
        self.df_final = []

    def general_table(self, selected=[], cdf='../fp', filter_file='../fp'):
        """
        Given a set of columns to select from the filter file create a
        dataframe populated with the values from the selected columns with
        the correct type (eg numeric vs str this is defined in the column definition file)
        """
        rows = line_count(filter_file)
        cols = len(selected) + 1  # add one for the key column
        data = np.empty([rows, cols]).astype(object)
        column_defitions = pickle.load(open(cdf, 'rb'))
        columns = ['chrm_pos'] + [column_defitions[x][0] for x in selected]
        numeric_columns = [column_defitions[x][0] for x in selected if column_defitions[x][1] == 'numeric']
        with open(filter_file, 'r') as fin:
            for i, line in enumerate(fin):
                line = line.strip()
                data[i] = line.split('\t')
        df = pd.DataFrame(data=data, columns=columns)
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric,
                                                        errors='coerce')
        return df

    def import_files(self):
        """
        This function takes a path as an argument in order to
        search for all the tables files that belong to a specific
        table type such as gene or variant. Not counting redundant
        files so make sure all files are labeled uniquely.
        """
        table_path = []
        cdf_path = []
        redundant_files = []
        for dirName, subdirList, fileList in os.walk('Databases/input/dbNSFP3.4a/active_tables'):
            for fname in fileList:
                if fname.endswith('.tab') and fname not in redundant_files:
                    redundant_files.append(fname)
                    table_path.append(os.path.join(dirName, fname))
                if fname.endswith('.p') and fname not in redundant_files:
                    redundant_files.append(fname)
                    cdf_path.append(os.path.join(dirName, fname))
        return table_path, cdf_path

    def join_dataframes(self, dfs):
        """Use if joining 2 or more tables"""
        if len(dfs) > 1:
            for df in dfs[1:]:
                dfs[0] = dfs[0].join(df.set_index('chrm_pos'), on='chrm_pos')
            self.df_final = dfs[0]
        else:
            self.df_final = dfs

    def save_df(self, table):
        filename = '_'.join([x[:2] for x in table.columns])
        table.to_csv('../output/tables/' + filename + '.csv')


class generate_sampletables:
    """
    Tables that give additional information to the rows are important
    to determine how the accuracy, or some numerical output from the classifier
    behaves with respect to that addtional categorical label that is not used
    in the classifier. For instance Age, Braak Score, Cohort ect. If no other
    attributes are known to the rows other than the ylabel, we can use unsupervised
    learning on the whole dataset to determine whether the structure of the data
    naturally clusters into groups. These clusters can then become the labels that will
    be attached to the rows and can be used to reveal the complexity of the data, but
    this has to be done after feature selection otherwise clustering in high-dimensionality
    is meaningless since the distances between points become so far that similarity
    measurements are useless such as euclidean distance. So the algorithm might go
    pass the filtered preprocessed sparse_matrix (training and test) then apply some
    sort of dimension reduction
    """
    def __init__(self, sparse_matrix, phen_table=0):
        self.phen_table = phen_table
        self.sparse_matrix = sparse_matrix

"""UTILITY FUNCTIONS"""


def line_count(f):
    with open(f, 'rb') as fin:
        for i, l in enumerate(fin):
            pass
        return i + 1
