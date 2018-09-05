import vcf
import os
import numpy as np
import re
import pickle


class parse_VCF:
    def __init__(self, filepaths=['.']):
        """
        parse VCF for relevant attributes 

        Parameters:
        ======================================

        filepaths -> list of strings that represent the location of the VCF files

        Attributes:
        ======================================

        blacklisted_cols -> Dictionary: features that have mismatching number of samples to the number of genotyped values

        genotype_converter-> Dictionary: numerical conversion from VCF genotype format

        types_of_gt -> list strings: the unique genotypes from the VCF files

        columns_pos -> Nested Dictionary:  The column positions for each type of design matrix format

        row_pos -> Dictionary: Sample names key and row index value

        filepaths -> list of strings that represent the location of the VCF files

        loci_count -> Int: number of loci features

        allele_count -> Int: number of allele features

        variant_count -> Int: number of variant features

        row_count -> Int: number of samples

        Notes:
        ======================================

        The buildMatrix class uses this class to determine the number of columns in the design matrix,
        the number of rows, what features and samples go where.

        Examples:
        ======================================

        >>>  None
        """

        self.blacklisted_cols = {}
        self.genotype_converter = {}
        self.types_of_gt = []
        self.columns_pos = {'loci': {}, 'variant': {}, 'allele': {}}
        self.row_pos = {}
        self.filepaths = filepaths
        self.loci_count = 0
        self.allele_count = 0
        self.variant_count = 0
        self.row_count = 0
        self.column_crawl_ref_alt()  # initiates the parsing

    def row_hashing(self, key):
        """maps each sample to a row index"""
        if key in self.row_pos:
            return
        else:
            self.row_pos[key] = self.row_count
            self.row_count += 1

    def row_crawl(self, root='case'):
        """
        Crawl through a folder and populate the row dictionary with the relevant sample names

        Parameters:
        ======================================

        root -> the name of the folder, specifing a label, that contains the VCF file(s) 

        Notes:
        ======================================

        None

        Examples:
        ======================================

        >>>  None
        """
        paths = self.import_files(root)  # list of files
        for path in paths:
            samples = self.sample_names(path)
            [self.row_hashing(x) for x in samples]

    def loci_hashing(self, key):
        """
        Builds the loci dictionary        

        Parameters:
        ======================================

        key -> String: Chromosome Position

        Notes:
        ======================================

        Hashes the chromosome + position of each snp with an index
        value that will correspond to its position in the matrix.
        This is a loci format therefore the values will be ./., 
        0/0, 1/1, 0/1

        Modifies the attributes loci_count and columns_pos

        Examples:
        ======================================

        >>>  None
        """
        if key in self.columns_pos['loci']:
            return
        else:
            self.columns_pos['loci'][key] = self.loci_count
            self.loci_count += 1

    def allele_hashing(self, key):
        """
        Builds the allele dictionary        

        Parameters:
        ======================================

        key -> String: Chromosome Position

        Notes:
        ======================================

        The allele format of the matrix represents the columns as
        the ALTs and REFs that are observed at that loci. eg the
        chrm_pos_ref_alt is 12_1789942_A_T equates to 
        A:10, T:11 (the integers being the different column
        locations. However, the key will be represented numerically rather than by letters to facilitate the
        key2value translation. For example the matrix encounters a chrm_pos_ref_alt with more than 1 ALT 12_1789942_A_T,C.
        How do we efficiently place this in our matrix? Ideally we
        would have all representations ie AT TT TC CA CC, however
        this would just be the same as variant hashing. Instead we
        assume A T C with the max variant call value going to that
        key eg 0/0 goes to A 0/1 goes to T 1/1 goes to T 1/2 goes
        to C 0/2 goes to C ect.

        Examples:
        ======================================

        >>>  None
        """

        s_dic = {}
        if key in self.columns_pos['allele']:
            return
        else:
            ref_alts = ",".join(key.split("_")[2:])
            sub_keys = [i for i, x in enumerate(ref_alts.split(","))]
            for sub_key in sub_keys:
                s_dic[sub_key] = self.allele_count
                self.allele_count += 1
            self.columns_pos['allele'][key] = s_dic

    def variant_hashing(self, key, sub_keys=['0/1', '1/1']):
        """
        Builds the variant dictionary        

        Parameters:
        ======================================

        key -> String: Chromosome Position

        Notes:
        ======================================

        This should be used when attempted to make a completely cateogorical
        matrix where homozygous and heterozygous are considered unique.
        INPUT 2 43793963 TA T,TAAA
        {'2_43793963': {'T': {'HET_T': 0,'HET_T_TAAA': 1,'HOMO_T': 2}, 
        'G': {'HET_TAAA': 3,'HOMO_TAAA': 4}}}
        {'2_43793963_TA_T,TAAA': {'./.':[0,1,2,3,4],'0/1':0,'0/2':1, '1/1':2, '1/2':3, '2/2':4}}
        
        vcf files might have different genotype distributions that are not shared.
        In order to not overwrite our subkey dictionary we have to do a few checks
        and then add a new key and a new value.
        
        Examples:
        ======================================

        >>>  None
        """

        missing_counts = []
        s_dic = {}
        if key in self.columns_pos['variant']:
            old_sub_keys = self.columns_pos['variant'][key]
            missing_keys = [x for x in sub_keys if x not in old_sub_keys]
            if len(missing_keys) > 0:
                # we do not watch to change the old values
                preserve = [(k, v) for k, v in old_sub_keys.items()]
                for miss_key in missing_keys:
                    s_dic[miss_key] = self.variant_count
                    missing_counts.append(self.variant_count)
                    self.variant_count += 1
                for old_key_value in preserve:
                    s_dic[old_key_value[0]] = old_key_value[1]
                    if old_key_value[0] != './.':  # we don't want to append the old list
                        missing_counts.append(old_key_value[1])
                s_dic['./.'] = missing_counts
                self.columns_pos['variant'][key] = s_dic
            else:
                return
        else:
            for s_key in sub_keys:
                s_dic[s_key] = self.variant_count
                missing_counts.append(self.variant_count)
                self.variant_count += 1
            s_dic['./.'] = missing_counts
            self.columns_pos['variant'][key] = s_dic

    def column_crawl_ref_alt(self):
        """
        Main function used to populate all the classes variables        

        Parameters:
        ======================================

        None

        Notes:
        ======================================

        This function takes the folder path and parses each file in that folder.
        Parsing is accomplished by use of regular expressions to extract the
        genotype information from the vcf file and the pyVCF library to extract
        sample names out. Each dictionary described above is populated in case
        different verisions of the matrix are to be generated. 

        Sometimes vcf files are not properly created and can have mismatches
        between the number of samples and the number of genotypes present.
        Usually there are fewer genotypes for some SNP. If this happens
        we ignore this SNP altogther including the matrix population phase.    
        
        Examples:
        ======================================

        >>>  None
        """

        paths = self.import_files(self.filepaths)  # choose files
        for path in paths:
            samples = self.sample_names(path)
            with open(path, 'r') as fin:
                # vcf_chunk = fin.readlines()
                for line in fin:
                    if not line.startswith("#"):
                        raw = line.split("\t")
                        chrm_pos_ref_alt = raw[0] + "_" + raw[1] + "_"  \
                                                  + raw[3] + "_" + raw[4]
                        genotype = re.findall('((?<=\\t).\/.)', line)
                        self.genotype_distribtion(genotype)
                        unique_genotypes = np.unique(genotype)
                        if len(genotype) != len(samples):
                            self.is_gt_inconsistent(chrm_pos_ref_alt)
                            continue
                        elif len(chrm_pos_ref_alt) > 0 and chrm_pos_ref_alt not in self.blacklisted_cols:
                            self.allele_hashing(chrm_pos_ref_alt)
                            self.loci_hashing(chrm_pos_ref_alt)
                            if ',' not in raw[4]:
                                self.variant_hashing(chrm_pos_ref_alt)
                            elif ',' in raw[4]:
                                sub_keys = self.create_subkeys(unique_genotypes)
                                self.variant_hashing(chrm_pos_ref_alt, sub_keys)
            fin.close()
            print(len(self.columns_pos), path, len(self.blacklisted_cols))
        self.genotype_dictionary()

    def is_gt_inconsistent(self, chrm_pos):
        """
        takes the parsed chromosome position and checks
        if it in the blacklisted folder
        """
        if chrm_pos in self.blacklisted_cols:
            return
        else:
            self.blacklisted_cols[chrm_pos] = 1

    def create_subkeys(self, unique_genotypes):
        """
        Creates subkeys for when a vcf has multiple allele 
        assignments for that chromosome position
        """
        sub_keys = [x for x in unique_genotypes if x != './.' and x != '0/0']
        return sub_keys

    def import_files(self, root=['.']):
        """
        This function takes a path as an argument in order to
        search for all the vcf files that belong to a specific
        condition such as control or case not counting redundant
        files so make sure all files are labeled uniquely.
        """
        path = []
        redundant_files = []
        for r in root:
            if len(list(os.walk('Matrices/input/VCF/' + r))) > 0:
                input_path = 'Matrices/input/VCF/' + r
            else:
                input_path = r
            for dirName, subdirList, fileList in os.walk(input_path):
                for fname in fileList:
                    if fname.endswith('.vcf') and fname not in redundant_files:
                        redundant_files.append(fname)
                        path.append(os.path.join(dirName, fname))
        return path

    def sample_names(self, vcf_file):
        """
        a wrapper function to get the sample names 
        using the pyVCF library
        """
        vcf_reader = vcf.Reader(open(vcf_file, 'r'))
        return vcf_reader.samples

    def genotypeparser(self, genotype):
        """
        Genotype Break Down
        ['./.' '0/0' '0/1' '1/1' '1/2' '2/1' '2/2']
        In reality these should all be assigned
        their own separate cateogry, but that might
        generate a matrix that is too big to fit into
        memory, so we cheat a bit and assume that 2/1
        is het and 2/2 is homo.
        """
        genotype_cvrt = [[self.genotype_converter[x]] for x in genotype]
        return genotype_cvrt

    def genotype_distribtion(self, genotype):
        """
        A function to determines the different types of genotypes
        captured from all the vcf files
        """
        self.types_of_gt += genotype
        self.types_of_gt = np.unique(self.types_of_gt).tolist()

    def genotype_dictionary(self):
        """
        A function to build the genotype dictionary that 
        converts the raw vcf version of a genotype into a
        numeric representation
        """
        for gt in self.types_of_gt:
            split_gt = gt.strip().split("/")  # 0/0
            if split_gt[0] == '0' and split_gt[1] == '0':
                self.genotype_converter[gt] = 0
            elif split_gt[0] == '.' and split_gt[1] == '.':
                self.genotype_converter[gt] = -1
            elif split_gt[0] != split_gt[1]:
                self.genotype_converter[gt] = 1
            elif split_gt[0] == split_gt[1]:
                self.genotype_converter[gt] = 2

    def save(self):
        """
        We need to save the positions of the snps
        in order to figure out which features are
        the most important to the classifier and
        in order to select ucscids from the knownGene
        UCSC tables.
        """
        pickle.dump(self.columns_pos,
                    open('Matrices/output/snps_matrix_pos.p', 'wb'))
        """We also need to save the genotype converter incase the
         case we are using old files"""
        pickle.dump(self.genotype_converter,
                    open('Matrices/output/genotype_converter.p', 'wb'))
        """We also need to save the row pos in order to know what
         patient is at which row position"""
        pickle.dump(self.row_pos,
                    open('Matrices/output/sample_matrix_pos.p', 'wb'))
