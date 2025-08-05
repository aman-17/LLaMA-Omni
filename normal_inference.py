import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model classes to ensure registration happens
from omni_speech.model.language_model.omni_speech_olmo import OmniSpeechOlmoForCausalLM, OmniSpeechOlmoConfig
from omni_speech.model.builder import load_pretrained_model
from omni_speech.utils import disable_torch_init

# Load your trained OmniSpeech model properly
print("🔄 Starting model loading...")
disable_torch_init()
model_path = "./outputs/olmo7b_tiny_stage1/best_model"
print(f"📁 Model path: {model_path}")
print("🚀 Loading model...")

tokenizer, olmo, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,  # Load from local checkpoint
    is_lora=False,
    s2s=False
)
print("✅ Model loaded successfully!")
print("🔤 Preparing input...")
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
print("🎯 Moving to CUDA...")
inputs = {k: v.to('cuda') for k,v in inputs.items()}
olmo = olmo.to('cuda')
print("⚡ Generating response...")
response = olmo.generate(
    inputs['input_ids'], 
    attention_mask=inputs.get('attention_mask'),
    max_new_tokens=100, 
    do_sample=True, 
    top_k=50, 
    top_p=0.95
)
print("📝 Output:")
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])