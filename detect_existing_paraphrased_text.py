import utils 
from text_loader import *
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def main(args):
    # load original texts
    if args.deploy_classifier == "kgw_wm":
        original_texts = load_wm_initial_text("kgw")
    elif args.deploy_classifier == "uni_wm":
        original_texts = load_wm_initial_text("uni")
    else:
        original_texts = load_initial_ai_text(num_samples=2000)
    # load paraphrased texts
    paraphrased_texts = load_from_disk(args.paraphrased_texts_path)['text']

    print("Loading deploy classifier...")
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

    detect_results = []
    for i in tqdm(range(0, len(paraphrased_texts), args.batch_size)):
        batch_texts = original_texts[i: i+args.batch_size]
        outputs = paraphrased_texts[i: i+args.batch_size]
        if args.deploy_classifier == "kgw_wm" or args.deploy_classifier == "uni_wm": # Watermark-based detectors
            inp_scores, out_scores = [],[]
            for i_t, o_t in zip(batch_texts, outputs):
                in_score_dict = watermark_detector.detect(i_t)
                try:
                    out_score_dict = watermark_detector.detect(o_t)
                except:
                    out_score_dict = {"z_score": -9999}
                    # warnings.warn(f"Input '{i_t}' resulted in output '{o_t}'")
                detect_results.append((in_score_dict, out_score_dict))
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
    parser.add_argument("--hf_cache_dir", type=str, default="/fs/cml-scratch/yzcheng/cache2")
    parser.add_argument("--paraphrased_texts_path", type=str, default="outputs/guided_generations_mage/adv/radar")
    parser.add_argument("--deploy_classifier", type=str, choices=["mage", "openai_roberta_base", "openai_roberta_large","radar", "kgw_wm", "uni_wm"], default="mage")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()
    print("*"*20, "\n", args, "\n", "*"*20, "\n")
    main(args)