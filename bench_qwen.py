from benchmark.models.model_qwen25 import TranslateModelQwen25_Instruct
from benchmark.bench_translate_en_ru_zh_long import run_benchmark

batch_size = 8

#model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
model_name = '/data/models/qwen25_0_5b_translate_en_ru_zh_step_02_2024_12_26_20-39-35'

bench_model = TranslateModelQwen25_Instruct(model_name, lang_src = 'ru', lang_tgt = 'en', device='cuda')

model_name_suffix = ''

generation_config = bench_model.model.generation_config
generation_config.max_new_tokens = 4096
generation_config.repetition_penalty = 1
generation_config.temperature = 1
generation_config.do_sample = False
generation_config.top_k = None
generation_config.top_p = 1

model_name_suffix = 'gen1'

generation_config = bench_model.model.generation_config
generation_config.max_new_tokens = 4096
generation_config.repetition_penalty = 1.1
generation_config.temperature = 0.7
generation_config.do_sample = True
generation_config.top_k = 20
generation_config.top_p = 0.8

bench_model_lambda = lambda lang_src, lang_tgt: TranslateModelQwen25_Instruct(model_name, 
                                                                       lang_src = lang_src, lang_tgt = lang_tgt, 
                                                                       device='cuda', 
                                                                       generation_config=generation_config)

#bench_model = bench_model_lambda(lang_src = 'ru', lang_tgt = 'en')

run_benchmark(model_name, model_name_suffix, bench_model_lambda, batch_size = batch_size)

