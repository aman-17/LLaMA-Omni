sudo apt install -y ffmpeg
#cd /myfiles/amanr/LLaMA-Omni/
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -e .
pip install ffmpeg
bash training/scripts/run_stage1.sh
# python olmo_omni/inference/run_llama_omni2.py --model_path outputs/olmo1b_large_stage1_tiny_data/checkpoint-epoch-3/ --question_file olmo_omni/examples/questions.json --answer_file olmo_omni/examples/answers.jsonl --temperature 0.5
