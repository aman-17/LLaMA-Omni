import argparse
import os
import sys
import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omni_speech.arguments import ModelArguments, DataArguments, TrainingArguments
from stage1_trainer import Stage1Trainer
from stage2_trainer import Stage2Trainer


@dataclass
class TrainingConfig:
    stage: int = field(
        default=1,
        metadata={"help": "Training stage: 1 for speech-to-text, 2 for speech generation"}
    )
    stage1_model_path: str = field(
        default=None,
        metadata={"help": "Path to Stage 1 trained model (required for Stage 2)"}
    )
    kmeans_model_path: str = field(
        default=None,
        metadata={"help": "Path to K-means model for speech unit extraction (required for Stage 2)"}
    )
    validation_data_path: str = field(
        default=None,
        metadata={"help": "Path to validation dataset"}
    )


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_arguments(training_config: TrainingConfig, model_args: ModelArguments, 
                      data_args: DataArguments, training_args: TrainingArguments):
    if training_config.stage not in [1, 2]:
        raise ValueError("Stage must be 1 or 2")
    if training_config.stage == 2:
        if not training_config.stage1_model_path:
            raise ValueError("Stage 1 model path is required for Stage 2 training")
        if not training_config.kmeans_model_path:
            raise ValueError("K-means model path is required for Stage 2 training")
        if not os.path.exists(training_config.stage1_model_path):
            raise ValueError(f"Stage 1 model path does not exist: {training_config.stage1_model_path}")
        if not os.path.exists(training_config.kmeans_model_path):
            raise ValueError(f"K-means model path does not exist: {training_config.kmeans_model_path}")
    
    if not os.path.exists(data_args.data_path):
        raise ValueError(f"Data path does not exist: {data_args.data_path}")
    
    if training_config.validation_data_path and not os.path.exists(training_config.validation_data_path):
        raise ValueError(f"Validation data path does not exist: {training_config.validation_data_path}")


def main():
    parser = HfArgumentParser((TrainingConfig, ModelArguments, DataArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        training_config, model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        training_config, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    data_args.validation_data_path = training_config.validation_data_path
    try:
        validate_arguments(training_config, model_args, data_args, training_args)
    except ValueError as e:
        logger.error(f"Argument validation failed: {e}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("LLaMA-Omni Training")
    logger.info("=" * 50)
    logger.info(f"Stage: {training_config.stage}")
    logger.info(f"Output directory: {training_args.output_dir}")
    logger.info(f"Data path: {data_args.data_path}")
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Epochs: {training_args.num_train_epochs}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info("=" * 50)
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Using {device_count} GPU(s)")
        if device_count > 1:
            logger.warning("Multi-GPU training not fully implemented. Using single GPU.")
    else:
        logger.info("Using CPU")

    os.makedirs(training_args.output_dir, exist_ok=True)
    config_path = os.path.join(training_args.output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        import json
        from dataclasses import asdict
        
        def make_serializable(obj):
            if hasattr(obj, '__dict__'):
                return {k: make_serializable(v) for k, v in obj.__dict__.items() 
                       if not k.startswith('_') and not callable(v)}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                try:
                    json.dumps(obj)  # Test if serializable
                    return obj
                except TypeError:
                    return str(obj)  # Convert to string if not serializable
        
        config_dict = {
            'training_config': make_serializable(training_config),
            'model_args': make_serializable(model_args),
            'data_args': make_serializable(data_args),
            'training_args': make_serializable(training_args)
        }
        json.dump(config_dict, f, indent=2)
    
    try:
        if training_config.stage == 1:
            logger.info("Starting Stage 1 training (Speech-to-Text)")
            
            trainer = Stage1Trainer(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                model_path=training_args.resume_from_checkpoint
            )
            
            trained_model = trainer.train()
            logger.info("Stage 1 training completed successfully!")
            
        elif training_config.stage == 2:
            logger.info("Starting Stage 2 training (Speech Generation)")
            
            trainer = Stage2Trainer(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                stage1_model_path=training_config.stage1_model_path,
                kmeans_model_path=training_config.kmeans_model_path
            )
        
            trained_model = trainer.train()
            logger.info("Stage 2 training completed successfully!")
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()