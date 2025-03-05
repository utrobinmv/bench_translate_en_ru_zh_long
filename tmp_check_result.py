import pickle

from benchmark.result import BenchResult
from benchmark.metrics import BenchmarkMetrics

(preds, labels, sources) = pickle.load(open("1.pkl", 'rb'))

lang_tgt = 'ru'

save_result_all_filename = 'result/bench_en_ru_zh_translate_long.csv'


metrics = BenchmarkMetrics(lang_tgt, 'cuda')
bench_all_result = BenchResult(save_result_all_filename, ['model_name','pair'])

for key in labels.keys():
    preds_all = preds[key]
    labels_all = labels[key]
    sources_all = sources[key]
    print(' - key:', key, len(preds_all), len(labels_all), len(sources_all))

    metric_result = metrics.compute_metrics(preds_all, labels_all, sources_all)
