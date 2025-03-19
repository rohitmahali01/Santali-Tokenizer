!pip install datasets

from google.colab import files
uploaded = files.upload()

file_name = list(uploaded.keys())[0]

# Read the file
with open("olchiki.txt", 'r', encoding='utf-8') as f:  # Replace olchiki.txt using the name of your file
    text = f.read()

print(text[:500])  # Print the first 500 characters

from tqdm.auto import tqdm  # For progress bar

# Get the uploaded file name dynamically
file_name = list(uploaded.keys())[0]

# Read the file contents
with open(file_name, 'r', encoding='utf-8') as file:
    lines = file.readlines()  # Read file line by line

# Initialize variables
text_data = []
file_count = 0
batch_size = 5000  # Save every 5000 lines

# Process each line from the file
for line in tqdm(lines):
    text = line.strip().replace('\n', '')  # Remove extra whitespace and newlines
    text_data.append(text)

    # Save every 5,000 samples
    if len(text_data) >= batch_size:
        file_path = f'text_{file_count}.txt'  # Saves files in Colab working directory
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(text_data))

        text_data = []  # Clear the list for the next batch
        file_count += 1  # Update file count

# Save remaining text (if any)
if text_data:
    file_path = f'text_{file_count}.txt'
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(text_data))

print(f"Splitting complete! {file_count + 1} files saved.")

!pip install transformers

import glob
paths = glob.glob("text_*.txt")

import os
from tokenizers import ByteLevelBPETokenizer
import glob  # To get all text files dynamically

# Get all text files from the working directory
paths = glob.glob("text_*.txt")  # Finds all saved text files (e.g., text_0.txt, text_1.txt)

# Initialize the tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on the uploaded text files
tokenizer.train(files=paths, vocab_size=30000, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

# Create the directory if it doesn't exist
directory_name = "santali_tokenizer"
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

# Save the trained tokenizer to the specified directory
tokenizer.save_model(directory_name)
