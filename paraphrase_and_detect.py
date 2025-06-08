import utils 
from datasets import load_dataset, load_from_disk
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def main(args):
    np.random.seed(0)
    if args.dataset == "mage":
        np.random.seed(0)
        dataset = load_dataset("yaful/MAGE")['test']
        labels = np.array([x['label'] for x in dataset]) # label == 0 is AI text
        idx = np.arange(len(labels))[labels == 0]
        idx = np.random.choice(idx, args.num_samples*10, replace=False)
        def condition(text):
            return args.n_words_sample <= len(text.split(' ')) <= 2*args.n_words_sample
        texts = [dataset[int(i)]['text'] for i in idx if condition(dataset[int(i)]['text'])]
        if len(texts) > args.num_samples:
            texts = texts[: args.num_samples]
        print("Num of Texts:", len(texts))
    elif args.dataset == "kgwwm_mage":
        dataset = load_from_disk("kgw_wm/wm_mage")
        texts = dataset['text']
    elif args.dataset == "uniwm_mage":
        dataset = load_from_disk("uni_wm/wm_mage")
        texts = dataset['text']            
    
    print("Loaading guidance classifier...")
    if args.guidance_classifier == "mage":
        guidance_classifier = utils.MAGEDetector()
    elif args.guidance_classifier == "openai_roberta_base" or args.guidance_classifier == "openai_roberta_large":
        guidance_classifier = utils.OpenAIRoberta(model_name=args.guidance_classifier)
    elif args.guidance_classifier == "radar":
        guidance_classifier = utils.RADAR()

    print("Loaading deploy classifier...")
    if args.deploy_classifier == "mage":
        deploy_classifier = utils.MAGEDetector()
    elif args.deploy_classifier == "openai_roberta_base" or args.deploy_classifier == "openai_roberta_large":
        deploy_classifier = utils.OpenAIRoberta(model_name=args.deploy_classifier)
    elif args.deploy_classifier == "radar":
        deploy_classifier = utils.RADAR()
    elif args.deploy_classifier == "kgw_wm":
        from kgw_wm.extended_watermark_processor import WatermarkDetector
        tokenizer = AutoTokenizer.from_pretrained(f"{args.hf_cache_dir}/Llama-3.1-8B-Instruct")
        watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.25, # should match original setting
                                        seeding_scheme="selfhash", # should match original setting
                                        device='cuda', # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)
    elif args.deploy_classifier == 'uni_wm':
        from uni_wm.extended_watermark_processor import WatermarkDetector
        tokenizer = AutoTokenizer.from_pretrained(f"{args.hf_cache_dir}/Llama-3.1-8B-Instruct")
        watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.25, # should match original setting
                                        seeding_scheme="unigram", # should match original setting
                                        device='cuda', # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)
    elif args.deploy_classifier == "fastdetectgpt" or args.deploy_classifier == "gltr":
        from zs_detectors.detector import get_detector
        deploy_classifier = get_detector(args.deploy_classifier)


    paraphraser = utils.Paraphraser(
        name=args.model,
        classifier=guidance_classifier,
        )
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch_texts = texts[i: i+args.batch_size]
        outputs = paraphraser.paraphrase(
            batch_texts,
            batch_size=len(batch_texts),
            max_new_tokens=1024,
            top_p=args.top_p,
            adversarial=args.adversarial,
            option=args.option,
            deterministic=args.deterministic,
        )
        print("paraphrase complete for this batch")
        if args.deploy_classifier == "kgw_wm" or args.deploy_classifier == "uni_wm": # Watermark-based detectors
            inp_scores, out_scores = [],[]
            for i_t, o_t in zip(batch_texts, outputs):
                in_score_dict = watermark_detector.detect(i_t)
                try:
                    out_score_dict = watermark_detector.detect(o_t)
                except:
                    out_score_dict = {"z_score": -9999}
                    # warnings.warn(f"Input '{i_t}' resulted in output '{o_t}'")
                inp_scores.append(in_score_dict["z_score"])
                out_scores.append(out_score_dict["z_score"])
        elif args.deploy_classifier == "fastdetectgpt" or args.deploy_classifier == "gltr": # ZS detectors
            inp_scores = deploy_classifier.inference(batch_texts)
            out_scores = deploy_classifier.inference(outputs)
        else: # NN-based classifiers
            inp_scores = deploy_classifier.get_scores(batch_texts)
            out_scores = deploy_classifier.get_scores(outputs)
        
        for input_text, output_text, i_s, o_s in zip(batch_texts, outputs, inp_scores, out_scores):
            print(f"\n\n\n{'*'*20}\n<input>{input_text}</input>\n<inp_score>{i_s}</inp_score>\n\n<output>{output_text}</output>\n<out_score>{o_s}</out_score>", flush=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mage", "kgwwm_mage", "uniwm_mage"],
        default="mage",
    )    
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--guidance_classifier", type=str, choices=["mage", "openai_roberta_base", "openai_roberta_large","radar"], default="mage")
    parser.add_argument("--deploy_classifier", type=str, choices=["mage", "openai_roberta_base", "openai_roberta_large","radar", "kgw_wm", "uni_wm", "fastdetectgpt", "gltr"], default="mage")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--adversarial", type=float, default=1.0)
    parser.add_argument("--option", type=int, default=None)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--n_words_sample", type=int, default=100)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--hf_cache_dir", type=str, default="/fs/cml-scratch/yzcheng/cache2")
    args = parser.parse_args()
    print("*"*20, "\n", args, "\n", "*"*20, "\n")
    main(args)