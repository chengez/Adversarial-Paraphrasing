from extended_watermark_processor import WatermarkLogitsProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import numpy as np
from datasets import load_dataset
import argparse
import time
from tqdm import tqdm
from datasets import Dataset
import pandas as pd


def main(args):
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")  # Adjust `device_map` as needed
    if tokenizer.pad_token is None:   
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"


    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                gamma=0.25,
                                                delta=2.0,
                                                seeding_scheme="unigram")
    # Note:
    # You can turn off self-hashing by setting the seeding scheme to `minhash`.

    np.random.seed(0)
    dataset = load_dataset("yaful/MAGE")['test']
    labels = np.array([x['label'] for x in dataset]) # label == 0 is AI text
    idx = np.arange(len(labels))[labels == 0]
    idx = np.random.choice(idx, args.num_samples*10, replace=False)
    def condition(text):
        return args.n_words_sample <= len(text.split(' ')) <= 2*args.n_words_sample
    texts = [dataset[int(i)]['text'] for i in idx if condition(dataset[int(i)]['text'])]
    if len(texts) > args.num_samples: # default first 2000
        texts = texts[: args.num_samples]

    print("Num of Texts:", len(texts))

    possible_tok_lens = []
    all_texts = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch_texts = texts[i: i+args.batch_size]
        inputs = [tokenizer.apply_chat_template(
            [{"role": "user", "content": ' '.join(text.split(' ')[:20])}],
            tokenize=False,
            add_generation_prompt=True
        ) for text in batch_texts]

        tokenized_input = tokenizer(inputs, return_tensors='pt', padding=True).to(model.device)
        
        if args.debug:
            possible_tok_lens.append(tokenized_input['input_ids'].shape[-1]) # min #token=199, max=614, mean~=264 
        
        output_tokens = model.generate(**tokenized_input, min_new_tokens=200, max_new_tokens=600, repetition_penalty=1.25, logits_processor=LogitsProcessorList([watermark_processor]))

        # if decoder only model, then we need to isolate the newly generated tokens as only those are watermarked, the input/prompt is not
        output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

        output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        all_texts += output_texts

    
    df = pd.DataFrame({'text': all_texts})
    dataset = Dataset.from_pandas(df)

    # Save the dataset to disk
    dataset.save_to_disk(args.save_path)


# REMEMBER TO CHANGE THE "SAVE DATA NAME" WHEN GENERATING THE REMAINING 1000 SAMPLES
if __name__ == "__main__":

    parser = argparse.ArgumentParser()  
    parser.add_argument("--model_path", type=str, default="/fs/cml-scratch/yzcheng/cache2/Llama-3.1-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--n_words_sample", type=int, default=100)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="uni_wm/wm_mage")
    args = parser.parse_args()
    print("*"*20, "\n", args, "\n", "*"*20, "\n")
    main(args)
