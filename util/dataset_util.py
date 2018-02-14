import gensim
import pandas as pd

def check_and_load_dataset():
    print('check')

def load_wnid_list():
    wnid_list = []
    description_list = []
    with open('wnid_list.txt') as f:
        input_lines = f.read().split('\n')
        for input_line in input_lines:
            fields = input_line.split('\t')
            wnid = fields[0]
            description = fields[1] if len(fields) > 1 else 'unknown'
            wnid_list.append(wnid)
            description_list.append(description)
    combined = {'wnid': wnid_list, 'description': description_list}
    output = pd.DataFrame(combined)
    return output
