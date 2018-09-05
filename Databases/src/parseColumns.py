import pickle


def column_parsing(files=['../input/dbNSFP3.4a/dbNSFP_gene-columns.txt',
                          '../input/dbNSFP3.4a/dbNSFP_variant-columns.txt']):
    '''
    Parse the dbNSFP3.4a columns so that we can use a numeric
    input to identify the column of choice. This will help generalize the buildDatabases function

    This is a manual method for parsing the columns

    '''


    type_dictionary = {'1': 'numeric', '2': 'string'}
    for f in files:
        parsed_columns = []
        output_name = f.split("/")[-1].split(".")[0]
        with open(f, 'r') as fin:
            for line in fin:
                line = line.strip()
                print(line)
                # type numeric: 1, string: 2
                column = line.split(":")[0]
                column_type = line.split("\t")[-1]
                parsed_columns.append((column, type_dictionary[column_type]))
        pickle.dump(parsed_columns,
                    open('../output/parsed_%s.p' % output_name, 'wb'))
