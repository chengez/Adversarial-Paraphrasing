import re
from datasets import Dataset

# Path to your log file
log_file_path = "logging_transfer_data-uniwmmage/logging.radar-uniwm"

# Read file content
with open(log_file_path, "r", encoding="utf-8") as f:
    log_content = f.read()

# Find all <output>...</output> blocks (non-greedy, multiline)
sentences = re.findall(r"<output>(.*?)</output>", log_content, re.DOTALL)

# Strip whitespace from each sentence
sentences = [s.strip() for s in sentences]

# Verify that there are exactly 2000
num_sentences = len(sentences)
if num_sentences != 2000:
    raise ValueError(f"❌ Expected 2000 sentences, but found {num_sentences}.")

# Create Hugging Face dataset
dataset = Dataset.from_dict({"text": sentences})

# Save to disk
dataset.save_to_disk("outputs/guided_generations_uniwmmage/adv/radar")

print("✅ Exactly 2000 sentences found. Dataset saved successfully.")
