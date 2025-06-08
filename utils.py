import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from MAGE.deployment import preprocess


class MAGEDetector():
    @torch.no_grad()
    def __init__(self, device='auto'):
        model_dir = "yaful/MAGE" # model in the online demo
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, device_map=device)
    
    @torch.no_grad()    
    def get_scores(self, texts, deploy=False):
        # larger the score, more likely to be AI text
        texts_ = [preprocess(text) for text in texts]
        toks = self.tokenizer(texts_, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        outputs = self.model(**toks)
        scores = torch.nn.Softmax()(outputs.logits)
        scores = [score[0].item() for score in scores]
        # scores = [np.log(score[0].item()+1e-16) for score in scores]
        return np.stack(scores)

class OpenAIRoberta():
    # Trained to detect GPT-2 output
    @torch.no_grad()
    def __init__(self, model_name='openai_roberta_base', device='auto' if torch.cuda.is_available() else 'cpu'):
        size = model_name.split('_')[-1]
        model_dir = f"openai-community/roberta-{size}-openai-detector" 
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, device_map=device)
    
    @torch.no_grad()
    def get_scores(self, texts=["Hello world! Is this content AI-generated?"], deploy=False):
        tokenized_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        logits = self.model(**tokenized_input).logits
        scores = torch.nn.Softmax(dim=1)(logits) # [[fake_score, real_score], ..., [fake_score, real_score]] 
        fake_scores, real_scores = scores[:,0], scores[:,1]
        return fake_scores.cpu().numpy() # lower fake score => classify as human text (so minimize fake scores)

class RADAR():
    @torch.no_grad()
    def __init__(self, device='auto'):
        self.device = device
        model_dir = "TrustSafeAI/RADAR-Vicuna-7B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, model_max_length=512)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir,device_map=device)
    
    @torch.no_grad()    
    def get_scores(self, texts, deploy=False):
        tokenized_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        logits = self.model(**tokenized_input).logits
        scores = torch.nn.Softmax(dim=1)(logits) # [[fake_score, real_score], ..., [fake_score, real_score]] 
        fake_scores, real_scores = scores[:,0], scores[:,1]
        return fake_scores.cpu().numpy() # lower fake score => classify as human text (so minimize fake scores)
    

class Paraphraser():
    @torch.no_grad()
    def __init__(self, name="meta-llama/Meta-Llama-3-8B-Instruct", classifier=None):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, device_map="auto", torch_dtype=torch.float16, use_cache=False, low_cpu_mem_usage=True)
        if self.tokenizer.pad_token is None:   
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        self.model.generation_config.top_p = 0.99
        self.model.generation_config.top_k = 50
        self.model.generation_config.temperature = 0.6
        self.classifier = classifier
    
    @torch.no_grad()
    def paraphrase(self, contents, batch_size=10, max_new_tokens=2048, top_p=0.9, adversarial=1., option=None, deterministic=True):
        """
        Paraphrases the given contents using the model
        
        Params: 
            contents: passages in list
            batch_size: batch size for inference
            max_new_tokens: maximum tokens to generate
            option: various options for adversarial score composition
                None: uses top_p for sampling
                1: uses only next_tokens with `probs > adversarial`
                2: selects best option from `probs - float(adversarial)*adv_scores`
            adversarial: adversarial parameters
            deterministic: if True, uses deterministic sampling
        Returns:
            paraphrased contents
        """
        system_prompt = """You are a rephraser. Given any input text, you are supposed to rephrase the text without changing its meaning and content, while maintaining the text quality. Also, it is important for you to output a rephrased text that has a different style from the input text. You can not just make a few changes to the input text. The input text is given below. Print your rephrased output text between tags <TAG> and </TAG>."""
        self.model.generation_config.top_p = top_p
        
        responses = []
        for b in range(0, len(contents), batch_size):
            inputs = [self.tokenizer.apply_chat_template([ {"role": "system", "content": system_prompt}, {"role": "user", "content": content}], tokenize=False, add_generation_prompt=True) for content in contents[b: b+batch_size]]
            inputs = [inp + "<TAG> " for inp in inputs]
            tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.model.device)
            input_ids = tokenized_inputs["input_ids"]
            attention_mask = tokenized_inputs["attention_mask"]
            past_key_values = None
            finished = torch.zeros(len(input_ids), dtype=torch.bool, device=self.model.device)
            generated_tokens = [[] for _ in range(len(input_ids))]

            for t in range(max_new_tokens):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=True)
                logits = outputs.logits[:, -1, :]  
                past_key_values = outputs.past_key_values
                probs = torch.nn.Softmax()(logits.float() / self.model.generation_config.temperature) # convert to float32 to avoid inf in softmax

                # Apply top_p, top_k masking                
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                mask = probs_sum - probs_sort > self.model.generation_config.top_p
                probs_sort[mask] = 0.0
                if option == 1:
                    assert 0 < adversarial <= 1
                    mask = probs_sort < adversarial
                    # assert that not all values are masked
                    max_indices = probs_sort.argmax(dim=1, keepdim=True)
                    mask.scatter_(1, max_indices, False)
                    probs_sort[mask] = 0.0
                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  
                next_tokens, prob_scores = [], []
                for i in range(len(probs_idx)):
                    next_tokens.append(probs_idx[i][probs_sort[i] > 0.0].cpu().detach().numpy().tolist()[: self.model.generation_config.top_k])
                    prob_scores.append(probs_sort[i][probs_sort[i] > 0.0].cpu().detach().numpy().tolist()[: self.model.generation_config.top_k])
                    
                # Get classifier score
                sampled_tokens = []
                if bool(adversarial):
                    for i in range(len(next_tokens)): # for each sample in batch
                        if len(next_tokens[i]) == 1:
                            sampled_tokens.append(next_tokens[i][0])
                            continue
                        toks = torch.tensor(next_tokens[i], device=self.model.device).unsqueeze(1)
                        inps = torch.tensor(generated_tokens[i], dtype=input_ids.dtype).unsqueeze(0).expand(toks.shape[0], -1).to(self.model.device)
                        next_toks = torch.cat([inps, toks], dim=-1)
                        next_words = self.tokenizer.batch_decode(next_toks, skip_special_tokens=True)
                        adv_scores = []
                        for j in range(0, len(next_words), batch_size):
                            adv_scores.extend(self.classifier.get_scores(next_words[j: j+batch_size])) # only the partial paraphrased output is fed into classifier
                        if option == 2:
                            adv_scores = -np.array(prob_scores[i]) + float(adversarial)*np.array(adv_scores)
                        if deterministic:
                            idx = np.argmin(adv_scores) # minimize the score
                            sampled_tokens.append(next_tokens[i][idx])
                        else:
                            adv_scores = -np.array(adv_scores)
                            adv_scores += -adv_scores.min() + 1e-9
                            adv_scores[adv_scores<0] = 0.
                            adv_scores /= adv_scores.sum()
                            idx = np.random.choice(len(next_tokens[i]), p=adv_scores)
                            sampled_tokens.append(next_tokens[i][idx])
                        
                else:
                    # always non-deteministic sampling for non-adversarial case
                    for i in range(len(next_tokens)):
                        tok = np.random.choice(next_tokens[i], p=prob_scores[i]/np.sum(prob_scores[i]))
                        sampled_tokens.append(tok)

                # Update generated tokens
                for i in range(len(input_ids)):
                    if not finished[i]:
                        generated_tokens[i].append(sampled_tokens[i])
                        if sampled_tokens[i] == self.tokenizer.eos_token_id:
                            finished[i] = True
                if finished.all():
                    break
                
                input_ids = torch.tensor(sampled_tokens, dtype=input_ids.dtype).unsqueeze(1).to(self.model.device)
                attention_mask = torch.cat([attention_mask, torch.ones((len(input_ids), 1), dtype=torch.long, device=self.model.device)], dim=1)

            for i in range(len(input_ids)):
                response_text = self.tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
                response_text = response_text.replace("<TAG>", "").replace("</TAG>", "").strip()
                weirds = ["Note: I rephrased", "Note: I've rephrased", "Note: I have rephrased", "(Note:"]
                for weird in weirds:
                    if weird in response_text:
                        response_text = response_text.split(weird)[0].strip()
                        break
                responses.append(response_text)
                
        # responses = [resp.replace("<TAG>", "").replace("</TAG>", "").strip() for resp in responses]
        return responses
    
   
