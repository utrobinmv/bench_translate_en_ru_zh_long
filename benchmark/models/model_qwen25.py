import copy
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BatchEncoding

class TranslateModelQwen25_Instruct:
    def __init__(self, name_model, lang_src, lang_tgt, device = None, generation_config = None, prefix='', amp = ''):
        self.tokenizer = AutoTokenizer.from_pretrained(name_model)
        if amp == 'bfloat16':
            self.model = AutoModelForCausalLM.from_pretrained(name_model,torch_dtype=torch.bfloat16)
        elif amp == 'int8':
            self.model = AutoModelForCausalLM.from_pretrained(name_model,load_in_8bit=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(name_model)
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt

        self.device = device

        if self.device is not None:
            self.model.to(self.device)

        self.prefix = prefix

        if generation_config is None:
            self.generation_config = self.model.generation_config
        else:
            self.generation_config = generation_config

        self.model.eval()

    def test_prompt(self, text):
        pass

    def get_prompt_string(self, text):
        target_language = ''
        if self.lang_tgt == 'en':
            target_language = 'English'
        if self.lang_tgt == 'ru':
            target_language = 'Russian'
        if self.lang_tgt == 'zh':
            target_language = 'Chinese'

        messages = [
            {"role": "user", "content": f"Translate the following sentence to {target_language}: {text}"},
        ]
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return encoded
        
    def translate_batch(self, batch: list[str], batch_labels: list[str] | None = None):
        # if self.prefix != '':

        new_batch = []
        for text in batch:
            encoded = self.get_prompt_string(text)
            #print(encoded)
            new_batch.append(encoded)

        src_batch = self.tokenizer(new_batch, return_tensors="pt", truncation=True, padding=True, padding_side='left') #, padding='longest', max_length=2048)

        # print('src_batch:',src_batch)

        cut_len = src_batch['input_ids'].shape[1]

        #a = self.tokenizer.batch_decode(src_batch['input_ids'])
        #print(a)

        generation_config = copy.deepcopy(self.generation_config)

        if batch_labels is not None:
            max_length = 0
            tmp_labels = self.tokenizer(batch_labels, return_attention_mask=False, return_length=True, return_tensors="pt", truncation=True, padding=True, padding_side='left')
            max_length = tmp_labels['length'].max().item()
            
            max_length = int(min(max_length * 1.5, self.generation_config.max_new_tokens))
            generation_config.max_new_tokens = max_length

        if self.device is not None:
            for key in src_batch.keys():
                src_batch[key] = src_batch[key].to(self.device)

        generated_tokens = self.model.generate(
                **src_batch,
                generation_config=generation_config
            )

        generated_tokens = generated_tokens[:,cut_len:]

        out = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return out

