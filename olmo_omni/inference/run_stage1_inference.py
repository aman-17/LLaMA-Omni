import argparse
import json
import os

import librosa
import numpy as np
import torch
import whisper
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from olmo_omni.constants import DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX
from olmo_omni.model.builder import load_pretrained_model


def load_audio(audio_path, sr=16000):
    """Load and preprocess audio file"""
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio


def preprocess_speech(audio, speech_encoder):
    """Convert audio to speech features using Whisper encoder"""
    # Convert to mel spectrogram (similar to Whisper preprocessing)
    mel = whisper.audio.log_mel_spectrogram(audio)
    mel = mel.unsqueeze(0)  # Add batch dimension

    # Extract features using speech encoder
    with torch.no_grad():
        speech_features = speech_encoder(mel.cuda())

    return speech_features


def run_stage1_inference(args):
    """Run Stage 1 inference (speech-to-text only)"""
    model_path = args.model_path

    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    tokenizer, model, context_len = load_pretrained_model(
        model_path=model_path, model_base=None, s2s=False
    )

    model.eval()
    model = model.cuda()

    # Load questions
    with open(args.question_file, "r") as f:
        questions = json.load(f)

    results = []

    for idx, question in enumerate(tqdm(questions, desc="Processing")):
        try:
            # Load audio
            audio_path = question["audio"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(
                    os.path.dirname(args.question_file), audio_path
                )

            audio = load_audio(audio_path)

            # Preprocess speech
            speech_features = preprocess_speech(audio, model.get_model().speech_encoder)
            speech_lengths = torch.tensor(
                [speech_features.shape[1]], dtype=torch.long
            ).cuda()

            # Prepare text prompt
            prompt = question.get("text", "")
            if prompt:
                # Add speech token to prompt
                prompt_with_speech = DEFAULT_SPEECH_TOKEN + prompt
            else:
                prompt_with_speech = DEFAULT_SPEECH_TOKEN + "Transcribe the speech."

            # Tokenize
            input_ids = tokenizer(
                prompt_with_speech,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=context_len,
            ).input_ids.cuda()

            attention_mask = torch.ones_like(input_ids)

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    speech=speech_features,
                    speech_lengths=speech_lengths,
                    max_new_tokens=512,
                    temperature=args.temperature if args.temperature > 0 else None,
                    do_sample=args.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the input prompt from response
            if response.startswith(prompt_with_speech):
                response = response[len(prompt_with_speech) :].strip()

            result = {
                "question_id": question.get("id", idx),
                "audio_path": audio_path,
                "prompt": prompt,
                "response": response,
            }
            results.append(result)

            print(f"Question {idx + 1}: {prompt}")
            print(f"Response: {response}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing question {idx}: {e}")
            result = {
                "question_id": question.get("id", idx),
                "audio_path": (
                    audio_path
                    if "audio_path" in locals()
                    else question.get("audio", "")
                ),
                "prompt": question.get("text", ""),
                "response": f"Error: {str(e)}",
            }
            results.append(result)

    # Save results
    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
    with open(args.answer_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Results saved to {args.answer_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Speech-to-Text Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--question_file", type=str, required=True, help="Path to questions JSON file"
    )
    parser.add_argument(
        "--answer_file", type=str, required=True, help="Path to save answers"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )

    args = parser.parse_args()

    run_stage1_inference(args)


if __name__ == "__main__":
    main()
