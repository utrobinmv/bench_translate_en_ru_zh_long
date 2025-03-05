import os
import pandas as pd

from benchmark.bench_translate_constants import result_dirname, result_filename
from benchmark.utils import save_json_to_file

from benchmark.metrics import BenchmarkMetrics, BenchmarkMetricsComet, BenchmarkMetricsBertScore, BenchmarkMetricsTer

from benchmark.result import BenchResult

result_dirname = os.getenv('BENCH_RESULT_TRANSLATE_DIR', 'result_translate')
complete_dirname = os.getenv('BENCH_COMPLETE_TRANSLATE_DIR', 'complete_translate')

save_result_all_filename = 'result/bench_en_ru_zh_translate_long.csv'
save_result_category_filename = 'result/bench_translate_long.csv'


def load_result(filename:str):
    df_target = pd.read_json(filename, lines=True, orient='records')
    targets = df_target.to_dict('records')
    del df_target

    labels_all = []
    preds_all = []
    sources_all = []
    labels = {}
    preds = {}
    sources = {}
    for item in targets:
        labels_all.append(item['target'])
        preds_all.append(item['input'])
        sources_all.append(item['result'])

        format = item['format']
        if format not in labels.keys():
            labels[format] = []
            preds[format] = []
            sources[format] = []
        
        labels[format].append(item['target'])
        sources[format].append(item['input'])
        preds[format].append(item['result'])

        category = item['category']
        if category == 'x':
            continue

        if category not in labels.keys():
            labels[category] = []
            preds[category] = []
            sources[category] = []

        labels[category].append(item['target'])
        sources[category].append(item['input'])
        preds[category].append(item['result'])   

    return labels, sources, preds, labels_all, sources_all, preds_all

def save_result_all(dict_result_all):
    # save benchmark result
    bench_all_result = BenchResult(save_result_all_filename, ['model_name','model_name_suffix','pair'])
    for filename in dict_result_all.keys():
        metric_result = dict_result_all[filename]
        bench_all_result.save_result(metric_result)

def save_result_by_category(dict_category_result):
    # save benchmark result
    bench_result = BenchResult(save_result_category_filename, ['model_name','model_name_suffix','pair','category'])
    for filename in dict_category_result.keys():
        for category in dict_category_result[filename].keys():
            metric_result = dict_category_result[filename][category]
            bench_result.save_result(metric_result)

def calc_metrics(calculate_all = True, calculate_by_category = True):

    filename_sessions = os.path.join(result_dirname, result_filename)
    if os.path.exists(filename_sessions):
        dict_result_all = {}
        
        dict_category_result = {}

        df = pd.read_json(filename_sessions, lines=True, orient="records")
        df_metrics = df[df['metrics'] == False]
        df_metrics.drop_duplicates(subset='filename', keep='last', inplace=True)
    
        metrics = {}
        for _, rows in df_metrics.iterrows():
            filename = rows['filename']

            labels, sources, preds, labels_all, sources_all, preds_all = load_result(filename)

            lang_src = rows['lang_src']
            lang_tgt = rows['lang_tgt']
            model_name = rows['model']
            model_name_suffix = rows['model_name_suffix']

            pair = [lang_src, lang_tgt]
            pair = '-'.join(pair)

            if lang_tgt not in metrics.keys():
                metrics[lang_tgt] = BenchmarkMetrics(lang_tgt, 'cpu')

            if calculate_all:
                metric_result = metrics[lang_tgt].compute_metrics(preds_all, labels_all, sources_all)
                metric_result['model_name'] = model_name
                metric_result['model_name_suffix'] = model_name_suffix
                metric_result['pair'] = pair
                metric_result['gpu_time'] = rows['execution_time']

                dict_result_all[filename] = metric_result

            if calculate_by_category:
                dict_category_result[filename] = {}

                for category in labels.keys():
                    metric_result = metrics[lang_tgt].compute_metrics(preds[category], labels[category], sources[category])
                    metric_result['model_name'] = model_name
                    metric_result['model_name_suffix'] = model_name_suffix
                    metric_result['pair'] = pair
                    metric_result['category'] = category

                    print(' = ', category, metric_result)

                    dict_category_result[filename][category] = metric_result

        if len(df_metrics) > 0:

            if calculate_all:
                save_result_all(dict_result_all)
            if calculate_by_category:
                save_result_by_category(dict_result_all, dict_category_result)

            metrics_comet = BenchmarkMetricsComet('cuda')
            for filename in dict_result_all.keys():
                labels, sources, preds, labels_all, sources_all, preds_all = load_result(filename)
                if calculate_all:
                    metric_result = metrics_comet.compute_metrics(preds_all, labels_all, sources_all)
                    dict_result_all[filename].update(metric_result)

                if calculate_by_category:
                    for category in labels.keys():
                        metric_result = metrics_comet.compute_metrics(preds[category], labels[category], sources[category])

                        dict_category_result[filename][category].update(metric_result)

            if calculate_all:
                save_result_all(dict_result_all)
            if calculate_by_category:
                save_result_by_category(dict_result_all, dict_category_result)

            metrics_bert = {}
            for _, rows in df_metrics.iterrows():
                filename = rows['filename']

                labels, sources, preds, labels_all, sources_all, preds_all = load_result(filename)

                #lang_src = rows['lang_src']
                lang_tgt = rows['lang_tgt']
                #model_name = rows['model']

                if lang_tgt not in metrics_bert.keys():
                    metrics_bert[lang_tgt] = BenchmarkMetricsBertScore(lang_tgt, 'cuda')

                if calculate_all:
                    metric_result = metrics_bert[lang_tgt].compute_metrics(preds_all, labels_all, sources_all)
                    dict_result_all[filename].update(metric_result)

                if calculate_by_category:
                    for category in labels.keys():
                        metric_result = metrics_bert[lang_tgt].compute_metrics(preds[category], labels[category], sources[category])

                        dict_category_result[filename][category].update(metric_result)

            if calculate_all:
                save_result_all(dict_result_all)
            if calculate_by_category:
                save_result_by_category(dict_result_all, dict_category_result)

            metrics_ter = BenchmarkMetricsTer('cpu')
            if calculate_all:
                for filename in dict_result_all.keys():
                    labels, sources, preds, labels_all, sources_all, preds_all = load_result(filename)
                    metric_result = metrics_ter.compute_metrics(preds_all, labels_all, sources_all)
                    dict_result_all[filename].update(metric_result)

            if calculate_by_category:
                for filename in dict_category_result.keys():
                    for category in labels.keys():
                        metric_result = metrics_ter.compute_metrics(preds[category], labels[category], sources[category])

                        dict_category_result[filename][category].update(metric_result)

            if calculate_all:
                save_result_all(dict_result_all)
            if calculate_by_category:
                save_result_by_category(dict_result_all, dict_category_result)

            # remove benchmark todo metrics
            if calculate_all and calculate_by_category:
                list_filenames = list(set(df_metrics['filename']))
                save_json_to_file(filename_sessions, list_filenames)




