import razdel
import evaluate
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

import time
import functools

def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class BenchmarkMetrics:
    def __init__(self, lang: str, device: str=None):
        self.device = device
        self.lang = lang

        self.metric_sacrebleu = evaluate.load("sacrebleu")
        self.metric_bleu = evaluate.load("bleu")
        self.metric_chrf = evaluate.load("chrf")
        self.metric_meteor = evaluate.load('meteor')
        self.metric_rouge = evaluate.load('rouge')
        #self.metric_ter = evaluate.load('ter')
        #self.metric_bertscore = evaluate.load("bertscore", device=self.device) #, batch_size=2

        #self.metric_comet = evaluate.load('evaluate-metric/comet')

    @log_execution_time
    def compute_metrics(self, preds: list[str], labels: list[str], sources: list[str]):
        
        assert len(preds) == len(labels)
        assert len(sources) == len(labels)

        print('bleu start')

        if self.lang == 'zh':
            result_bleu = self.metric_bleu.compute(predictions=preds, references=labels, tokenizer=TokenizerZh())
        else:
            result_bleu = self.metric_bleu.compute(predictions=preds, references=labels)

        print('bleu end')


        #print(result_bleu)
        #print('len: ', len(preds))


        result = {"bleu": result_bleu["bleu"]*100}

        print('sacrebleu start')

        if self.lang == 'zh':
            result_sacrebleu = self.metric_sacrebleu.compute(predictions=preds, references=labels, tokenize='zh', lowercase=True)    
        else:
            result_sacrebleu = self.metric_sacrebleu.compute(predictions=preds, references=labels, lowercase=True)
        result["sacrebleu"] = result_sacrebleu["score"]

        print('sacrebleu end')

        print('chrf start')

        result_chrf = self.metric_chrf.compute(predictions=preds, references=labels)
        result["chrf"] = result_chrf["score"]

        print('chrf end')

        # print('ter start')

        # result_ter = self.metric_ter.compute(predictions=preds, references=labels, case_sensitive=False)
        # result["ter"] = result_ter["score"]

        # print('ter end')

        # print('comet start')

        # gpus = 0
        # if gpus != 'cpu':
        #     gpus = 1

        # result_comet = self.metric_comet.compute(predictions=preds, references=labels, sources=sources, gpus=gpus)
        # result["comet"] = result_comet["mean_score"]

        # print('comet end')

        def tokenize(text):
            return [token.text.lower() for token in razdel.tokenize(text)]

        print('rouge start')

        if self.lang == 'zh':
            result_rouge = self.metric_rouge.compute(predictions=preds, references=labels, tokenizer=TokenizerZh())
        else:
            result_rouge = self.metric_rouge.compute(predictions=preds, references=labels, tokenizer=tokenize)

        for key in result_rouge.keys():
            result[key] = result_rouge[key]

        print('rouge end')

        print('meteor start')

        def tokenize_meteor(text):
            return ' '.join([token.text for token in razdel.tokenize(text)])

        if self.lang == 'zh':
            tokenize_meteor = TokenizerZh()

        preds_tokens = [tokenize_meteor(x) for x in preds]
        labels_tokens = [tokenize_meteor(x) for x in labels]

        result_meteor = self.metric_meteor.compute(predictions=preds_tokens, references=labels_tokens)
        result["meteor"] = result_meteor["meteor"]
        print('meteor end')

        # print('bert start')
        # result_bertscore = self.metric_bertscore.compute(predictions=preds, references=labels, lang=self.lang)
        # result_bertscore.pop('hashcode')
        # for key in result_bertscore.keys():
        #    result['bertscore_'+key] = result_bertscore[key][0]
        # print('bert end')

        # preds = preds.cpu().detach().numpy()

        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        # result = {k: round(v, 4) for k, v in result.items()}
        
        #print(result)
        
        return result

class BenchmarkMetricsComet(BenchmarkMetrics):
    def __init__(self, device: str=None):
        self.device = device
        # self.lang = lang

        #self.metric_ter = evaluate.load('ter')
        #self.metric_bertscore = evaluate.load("bertscore", device=self.device) #, batch_size=2

        self.metric_comet = evaluate.load('evaluate-metric/comet')

    def compute_metrics(self, preds: list[str], labels: list[str], sources: list[str]):
        
        assert len(preds) == len(labels)
        assert len(sources) == len(labels)

        result = {}

        print('comet start')

        gpus = 0
        if gpus != 'cpu':
            gpus = 1

        result_comet = self.metric_comet.compute(predictions=preds, references=labels, sources=sources, gpus=gpus)
        result["comet"] = result_comet["mean_score"]

        print('comet end')
        
        return result


class BenchmarkMetricsBertScore(BenchmarkMetrics):
    def __init__(self, lang: str, device: str=None):
        self.device = device
        self.lang = lang

        self.metric_bertscore = evaluate.load("bertscore", device=self.device) #, batch_size=2

    def compute_metrics(self, preds: list[str], labels: list[str], sources: list[str]):
        
        assert len(preds) == len(labels)
        assert len(sources) == len(labels)

        result = {}

        print('bert start')
        result_bertscore = self.metric_bertscore.compute(predictions=preds, references=labels, lang=self.lang)
        result_bertscore.pop('hashcode')
        for key in result_bertscore.keys():
           result['bertscore_'+key] = result_bertscore[key][0]
        print('bert end')
        
        return result


class BenchmarkMetricsTer(BenchmarkMetrics):
    def __init__(self, lang: str, device: str=None):
        self.device = device
        self.lang = lang

        self.metric_ter = evaluate.load('ter')

    def compute_metrics(self, preds: list[str], labels: list[str], sources: list[str]):
        
        assert len(preds) == len(labels)
        assert len(sources) == len(labels)

        result = {}

        print('ter start')

        gpus = 0
        if gpus != 'cpu':
            gpus = 1

        result_ter = self.metric_ter.compute(predictions=preds, references=labels, case_sensitive=True)
        result["ter"] = result_ter["score"]

        print('ter end')
        
        return result
