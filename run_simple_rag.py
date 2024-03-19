from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

import torch
import os

embedd_model = "HooshvareLab/bert-fa-base-uncased"
llm_model = "asedmammad/PersianMind-v1.0-GGUF"

MODEL_ID = "asedmammad/PersianMind-v1.0-GGUF"
MODEL_BASENAME = "PersianMind-v1.0.q4_K_M.gguf"

# Load the embedding model
embedding_model = BertModel.from_pretrained(embedd_model)

tokenizer = AutoTokenizer.from_pretrained("aidal/Persian-Mistral-7B", cache_dir="./models/")
model = AutoModelForCausalLM.from_pretrained("aidal/Persian-Mistral-7B", cache_dir="./models/")

input_text = "پایتخت ایران کجاست؟"
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))

# Load the inference LLM

llm_filename = "persian_llama_7b.Q4_K_M.gguf"
llm_directory = r'./models\models--mostafaamiri--persian-llama-7b-GGUF-Q4\snapshots\982b03b059d05d08e94c1fff252e32b8e49f23a4'
llm_path = os.path.join(llm_directory, llm_filename)

inference_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="./models/")
inference_model = AutoModelForCausalLM.from_pretrained(llm_path, cache_dir="./models/") # ,local_files_only=True)

# Example Persian text
text = "ایران کشوری در آسیای جنوب غربی است. پایتخت آن تهران است. جمعیت ایران حدود ۸۳ میلیون نفر است. زبان رسمی ایران فارسی است."

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained(embedd_model)
input_ids = tokenizer.encode(text, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    outputs = embedding_model(input_ids)
    embeddings = outputs.last_hidden_state

# Example question
question = "زبان رسمی ایران چیست؟"

# Tokenize the question
question_input_ids = inference_tokenizer.encode(question, return_tensors="pt")

# Compute similarity between question and text embeddings
similarity_scores = torch.mv(embeddings[0], question_input_ids.squeeze().float())

# Find the most relevant part of the text
relevant_part_start = torch.argmax(similarity_scores).item()
relevant_part_end = relevant_part_start + 1  # Assuming single word answer
relevant_part = tokenizer.decode(input_ids[0][relevant_part_start:relevant_part_end])

# Prompt the inference LLM with the relevant part
prompt = f"از متن داده شده، پاسخ به سوال '{question}' '{relevant_part}' است. توضیح دهید که چرا این پاسخ صحیح است."

# Generate the answer
output_ids = inference_model.generate(inference_tokenizer.encode(prompt, return_tensors="pt"), max_length=200)
answer = inference_tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(answer)
