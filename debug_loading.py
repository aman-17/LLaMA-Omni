#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model classes to ensure registration happens
print("🔄 Importing model classes...")
from omni_speech.model.language_model.omni_speech_olmo import OmniSpeechOlmoForCausalLM, OmniSpeechOlmoConfig

print("🔄 Importing builder...")
from omni_speech.model.builder import is_olmo_model

model_path = "./outputs/olmo7b_tiny_stage1/best_model"

print(f"📁 Model path: {model_path}")
print(f"🔍 is_olmo_model check: {is_olmo_model(model_path)}")

print("🔄 Loading config...")
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained(model_path)
    print(f"✅ Config loaded: {type(config)}")
    print(f"🏗️  Model type: {getattr(config, 'model_type', 'NOT_FOUND')}")
    print(f"🏗️  Architectures: {getattr(config, 'architectures', 'NOT_FOUND')}")
except Exception as e:
    print(f"❌ Config loading failed: {e}")

print("🔄 Testing model class loading...")
try:
    model = OmniSpeechOlmoForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")