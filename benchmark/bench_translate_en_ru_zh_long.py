import os
import time
import json

import torch
import pandas as pd
from tqdm import tqdm
import datetime

import datasets


from benchmark.dataset import BenchDataset
from benchmark.utils import append_to_file
from benchmark.bench_translate_constants import result_dirname, result_filename

torch.set_float32_matmul_precision = 'high'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def _create_timestamp() -> str:
    dt = datetime.datetime.now()
    return dt.strftime('%Y%m%d_%H%M%S') 


list_pairs = [
              ('en','ru'),
              ('ru','en'),
              ('en','zh'),
              ('zh','en'),
              ('ru','zh'),
              ('zh','ru'),
              ]


def run_benchmark(model_name:str, model_name_suffix:str, bench_model_lambda, batch_size:int):

    model_name_lower = model_name.lower().replace('/','_')

    filename_models_name = os.path.join(result_dirname, 'models_name.jsonl')
    dict_modelname = {'model': model_name, 'model_name_lower': model_name_lower}

    filename_sessions = os.path.join(result_dirname, result_filename)

    if os.path.exists(filename_models_name):
        df_models = pd.read_json(filename_models_name, lines=True, orient='records')
        df_tmp = pd.DataFrame([dict_modelname])
        df_models = pd.concat([df_models, df_tmp])
        df_models.drop_duplicates(subset='model', keep='last', inplace=True)
    else:
        df_models = pd.DataFrame([dict_modelname])
    df_models.to_json(filename_models_name, lines=True, orient='records', force_ascii=False)

    ds_src = datasets.load_dataset('datasets/translate_en_ru_zh_alpaca')
    list_ds = ds_src['test'].to_list()
    #list_ds = list_ds[:22]

    for lang_src, lang_tgt in list_pairs:

        save_model_filename = lang_src + '_' + lang_tgt + '_' + model_name_lower + model_name_suffix
        save_model_filename = os.path.join(result_dirname, save_model_filename)

        # bench_model = TranslateModelQwen25_Instruct(model_name, 
        #                                             lang_src = lang_src, lang_dst = lang_tgt, 
        #                                             device='cuda', 
        #                                             generation_config=generation_config)
        bench_model = bench_model_lambda(lang_src = lang_src, lang_tgt = lang_tgt)

        log_prompt = bench_model.get_prompt_string('hello world!')
        print("model:", model_name, ', prompt:', log_prompt)

        current_timestamp = _create_timestamp()

        target_ds = []
        for idx in range(len(list_ds)):
            item_ds = list_ds[idx].copy()
            item_ds['input'] = item_ds['output'][lang_src]
            item_ds['target'] = item_ds['output'][lang_tgt]

            add_inst_ds = {'format': 'text', 'category': item_ds['category'], 
                        'input': item_ds['instruction'][lang_src],
                        'target': item_ds['instruction'][lang_tgt]}
            
            item_ds['lang_src'] = lang_src
            item_ds['lang_tgt'] = lang_tgt
            item_ds['timestamp'] = current_timestamp

            add_inst_ds['lang_src'] = lang_src
            add_inst_ds['lang_tgt'] = lang_tgt
            add_inst_ds['timestamp'] = current_timestamp

            item_ds.pop('instruction')
            item_ds.pop('output')
            item_ds.pop('index')

            target_ds.append(item_ds)
            target_ds.append(add_inst_ds)

        bench_ds = BenchDataset(target_ds, batch_size = batch_size)

        start_time = time.time()

        list_result = []
        for list_dict_batch in tqdm(bench_ds):
            batch = []
            batch_labels = []
            for item in list_dict_batch:
                batch.append(item['input'])
                batch_labels.append(item['target'])
            result = bench_model.translate_batch(batch, batch_labels)
            list_result.extend(result)
            #break

        end_time = time.time()

        for idx in range(len(list_result)):
            target_ds[idx]['result'] = list_result[idx]

        df_target = pd.DataFrame(target_ds)
        df_target.to_json(save_model_filename + '.jsonl', lines=True, orient='records', force_ascii=False)
        df_target.to_csv(save_model_filename + '.csv', index=False)

        execution_time = end_time - start_time

        dict_session = {'model': model_name, 
                        'model_name_lower': model_name_lower,
                        'model_name_suffix': model_name_suffix,
                        'lang_src': lang_src,
                        'lang_tgt': lang_tgt,
                        'execution_time': execution_time,
                        'current_timestamp': current_timestamp,
                        'filename': save_model_filename + '.jsonl',
                        'metrics': False
                        }
        dict_session_str = json.dumps(dict_session)

        append_to_file(filename_sessions, dict_session_str)

