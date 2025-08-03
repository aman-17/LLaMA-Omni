import json
import random

def create_subset(input_file, output_file, percentage=0.2):
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    subset_size = int(total_samples * percentage / 100)
    
    print(f"Original dataset size: {total_samples}")
    print(f"Creating subset of {subset_size} samples ({percentage}%)")
    
    # Randomly sample
    random.seed(42)  # For reproducibility
    subset = random.sample(data, subset_size)
    
    print(f"Writing subset to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(subset, f, indent=2)
    
    print(f"Done! Created {output_file} with {len(subset)} samples")

if __name__ == "__main__":
    create_subset(
        "InstructS2S-200K/instruct_en_val.json",
        "InstructS2S-200K/instruct_en_val_small.json",
        percentage=0.1
    )