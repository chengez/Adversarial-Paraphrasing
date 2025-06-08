from extended_watermark_processor import WatermarkDetector
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_from_disk
import argparse
from tqdm import tqdm



model_path = "/fs/cml-scratch/yzcheng/cache2/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")  # Adjust `device_map` as needed
if tokenizer.pad_token is None:   
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

ds = load_from_disk('uni_wm/wm_mage')
texts = ds['text']

watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.25, # should match original setting
                                        seeding_scheme="unigram", # should match original setting
                                        device='cuda', # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)

np.random.seed(0)
detect_results = []
count_true = 0
for t in tqdm(texts):
    score_dict = watermark_detector.detect(t) # or any other text of interest to analyze
    detect_results.append(score_dict)
    count_true += int(score_dict['prediction'] == True)

print(f"Among {len(texts)} texts, {count_true} were identified as AI generated")

