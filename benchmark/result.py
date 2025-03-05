#import pickle
import shutil
import os
import pandas as pd

class BenchResult:
    def __init__(self, benchmark_filename, duplicates_columns = ['name','pair','dataset_name']):
        
        self.benchmark_filename = benchmark_filename
        self.duplicates_columns = duplicates_columns
    # def save_result(self, filename_save, pair_name, all_preds, all_labels, all_sources):
    #     # save predict result
    #     if os.path.exists(filename_save):
    #         shutil.copyfile(filename_save, filename_save + '.bak')
    #         dict_all_result = pickle.load(open(filename_save, 'rb'))
    #     else:
    #         dict_all_result = {}
    #     key_name = self.model_name + '_' + pair_name
    #     dict_all_result[key_name] = {'preds': all_preds, 'labels': all_labels, 'sources': all_sources}
    #     pickle.dump(dict_all_result, open(filename_save, 'wb'))

    def save_result(self, result):
        if os.path.exists(self.benchmark_filename):
            df = pd.read_csv(self.benchmark_filename).fillna('')
        else:
            df = None
        
        df_new = pd.DataFrame(result,index=[1])
        if df is not None:
            df = pd.concat([df,df_new])
        else:
            df = df_new

        if len(self.duplicates_columns) > 0:
            df.drop_duplicates(subset = self.duplicates_columns, keep='last', inplace=True)

        if os.path.exists(self.benchmark_filename):
            shutil.copyfile(self.benchmark_filename, self.benchmark_filename + '.bak')

        df.to_csv(self.benchmark_filename,index=False)
