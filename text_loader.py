from datasets import load_dataset, load_from_disk
import numpy as np


def word_count_condition(text, n_words_sample=100):
    return n_words_sample <= len(text.split(' ')) <= 2*n_words_sample

# Used to load initial ai texts from MAGE dataset
def load_initial_ai_text(num_samples=2000):
    np.random.seed(0)
    dataset = load_dataset("yaful/MAGE")['test']
    labels = np.array([x['label'] for x in dataset]) # label == 0 is AI text
    idx = np.arange(len(labels))[labels == 0]
    idx = np.random.choice(idx, num_samples*10, replace=False)
    texts = [dataset[int(i)]['text'] for i in idx if word_count_condition(dataset[int(i)]['text'])]
    if len(texts) > num_samples:
        texts = texts[: num_samples]

    return texts


# Used to load initial human texts from MAGE dataset
def load_initial_human_text(num_samples=2000):
    np.random.seed(0)
    dataset = load_dataset("yaful/MAGE")['test']
    labels = np.array([x['label'] for x in dataset]) # label == 1 is human text
    idx = np.arange(len(labels))[labels == 1]
    idx = np.random.choice(idx, num_samples*10, replace=False)
    texts = [dataset[int(i)]['text'] for i in idx if word_count_condition(dataset[int(i)]['text'])]
    if len(texts) > num_samples:
        texts = texts[: num_samples]

    return texts


def load_wm_initial_text(wm_name):
    if wm_name == "kgw":
        dataset = load_from_disk(f"kgw_wm/wm_mage")
    elif wm_name == "uni":
        dataset = load_from_disk(f"uni_wm/wm_mage")
    else:
        raise ValueError("Invalid wm_name. Choose 'kgw' or 'uni'.")
    texts = dataset['text']

    return texts



