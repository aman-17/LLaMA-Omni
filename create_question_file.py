#!/usr/bin/env python3
"""
Convert validation data to official infer.py question format
"""
import json
import argparse

def create_question_file(val_file, output_file, num_samples=5):
    """Convert validation data to question.json format"""
    
    with open(val_file, 'r') as f:
        val_data = json.load(f)
    
    questions = []
    for i, item in enumerate(val_data[:num_samples]):
        question = {
            "id": item.get("id", f"sample_{i}"),
            "speech": item["audio"],
            "conversations": [
                {
                    "from": "human", 
                    "value": "<speech>\nTranscribe the speech:"
                }
            ]
        }
        questions.append(question)
    
    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=2)
    
    print(f"Created {output_file} with {len(questions)} questions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", default="./InstructS2S-200K/instruct_en_val_small.json")
    parser.add_argument("--output", default="./validation_questions.json") 
    parser.add_argument("--num_samples", type=int, default=5)
    
    args = parser.parse_args()
    create_question_file(args.val_file, args.output, args.num_samples)