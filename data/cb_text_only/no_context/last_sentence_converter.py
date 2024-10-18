import os
import json
import re

input_directory_path = '/home/jmurzaku/sad/sad-training/data/cb_text_only'
output_directory_path = '/home/jmurzaku/sad/sad-training/data/cb_text_only/no_context'

os.makedirs(output_directory_path, exist_ok=True)

files = ['train.jsonl', 'test.jsonl', 'dev.jsonl']

def get_last_sentence(text):
    if ':' in text:
        parts = text.split(':')
        return parts[-1].strip() if len(parts) > 1 else text.strip()
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    return sentences[-1].strip() if sentences else text.strip()

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?;:(){}[\]%"\'’%])', r'\1', text)
    
    contraction_suffixes = [
        r"n['’]t",   # n't
        r"['’]m",    # 'm
        r"['’]s",    # 's
        r"['’]re",   # 're
        r"['’]ll",   # 'll
        r"['’]ve",   # 've
        r"['’]d",    # 'd
        r"['’]em",   # 'em (optional, e.g., 'em for 'them')
    ]
    pattern = r'\s+(' + '|'.join(contraction_suffixes) + r')\b'
    text = re.sub(pattern, r'\1', text)
    
    text = re.sub(r'\$\s+', r'$', text)
    
    return text

def process_jsonl_file(file_path, output_directory):
    updated_data = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if 'text' in entry:
                last_sentence = get_last_sentence(entry['text'])
                entry['text'] = normalize_text(last_sentence)
            updated_data.append(entry)
    
    file_name = os.path.basename(file_path)
    new_file_path = os.path.join(output_directory, file_name)
    
    with open(new_file_path, 'w') as out_f:
        for entry in updated_data:
            out_f.write(json.dumps(entry) + '\n')
    print(f"Processed file saved at: {new_file_path}")

for file_name in files:
    file_path = os.path.join(input_directory_path, file_name)
    process_jsonl_file(file_path, output_directory_path)
