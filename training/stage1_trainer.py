"""
Stage 1 Training: Speech-to-Text Instruction Following
Trains the speech adaptor and LLM while keeping speech encoder frozen
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
import logging
from tqdm import tqdm
from typing import Dict, Optional
import wandb
from omni_speech.model.language_model.omni_speech_llama import OmniSpeechLlamaForCausalLM
from omni_speech.model.builder import load_pretrained_model
from omni_speech.arguments import ModelArguments, DataArguments, TrainingArguments
from data_utils import create_data_loader
import json


class Stage1Trainer:
    """Trainer for Stage 1: Speech-to-Text Instruction Following"""
    
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        model_path: Optional[str] = None
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        self.setup_model(model_path)
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()
        
        # Setup wandb if enabled
        if self.training_args.report_to == "wandb":
            wandb.init(
                project="llama-omni-stage1",
                config={
                    **vars(model_args),
                    **vars(data_args),
                    **vars(training_args)
                }
            )
    
    def setup_model(self, model_path: Optional[str] = None):
        """Initialize the model and tokenizer"""
        if model_path:
            # Load from checkpoint
            self.tokenizer, self.model, _ = load_pretrained_model(
                model_path=model_path,
                model_base=self.model_args.model_name_or_path,
                s2s=False
            )
        else:
            # Initialize from base model
            self.tokenizer, self.model, _ = load_pretrained_model(
                model_path=self.model_args.model_name_or_path,
                model_base=None,
                s2s=False
            )
        
        # Freeze speech encoder
        for param in self.model.get_model().speech_encoder.parameters():
            param.requires_grad = False
        
        # Freeze LLM if specified
        if self.model_args.freeze_backbone:
            for param in self.model.get_model().model.parameters():
                param.requires_grad = False
        
        # Only train speech projector if specified
        if self.model_args.tune_speech_projector:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_model().speech_projector.parameters():
                param.requires_grad = True
        
        self.logger.info(f"Model initialized. Trainable parameters: {self.count_trainable_params()}")
    
    def count_trainable_params(self) -> int:
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def setup_data_loaders(self):
        """Setup training and validation data loaders"""
        self.train_loader = create_data_loader(
            data_path=self.data_args.data_path,
            tokenizer=self.tokenizer,
            data_args=self.data_args,
            batch_size=self.training_args.per_device_train_batch_size,
            stage=1,
            num_workers=4,
            shuffle=True
        )
        
        # Setup validation loader if validation data is provided
        if hasattr(self.data_args, 'validation_data_path') and self.data_args.validation_data_path:
            self.val_loader = create_data_loader(
                data_path=self.data_args.validation_data_path,
                tokenizer=self.tokenizer,
                data_args=self.data_args,
                batch_size=self.training_args.per_device_eval_batch_size,
                stage=1,
                num_workers=4,
                shuffle=False
            )
        else:
            self.val_loader = None
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Separate learning rates for different components
        param_groups = []
        
        # Speech projector parameters
        speech_projector_params = list(self.model.get_model().speech_projector.parameters())
        if speech_projector_params:
            lr = self.training_args.speech_projector_lr or self.training_args.learning_rate
            param_groups.append({
                'params': speech_projector_params,
                'lr': lr,
                'name': 'speech_projector'
            })
        
        # LLM parameters (if not frozen)
        if not self.model_args.freeze_backbone:
            llm_params = list(self.model.get_model().model.parameters())
            param_groups.append({
                'params': llm_params,
                'lr': self.training_args.learning_rate,
                'name': 'llm'
            })
        
        # LM head parameters
        lm_head_params = list(self.model.lm_head.parameters())
        param_groups.append({
            'params': lm_head_params,
            'lr': self.training_args.learning_rate,
            'name': 'lm_head'
        })
        
        self.optimizer = AdamW(
            param_groups,
            lr=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay
        )
        
        # Calculate total training steps
        total_steps = len(self.train_loader) * self.training_args.num_train_epochs
        warmup_steps = int(total_steps * 0.03)  # 3% warmup as mentioned in paper
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def forward_step(self, batch: Dict) -> Dict:
        """Forward pass through the model"""
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.model.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            speech=batch['speech_features'],
            speech_lengths=batch['speech_lengths']
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states
        }
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Skip empty batches
            if not batch:
                continue
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward_step(batch)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.training_args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.training_args.max_grad_norm
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.training_args.report_to == "wandb" and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'step': epoch * len(self.train_loader) + batch_idx
                })
        
        return {'train_loss': total_loss / num_batches if num_batches > 0 else 0}
    
    def validate(self) -> Dict:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if not batch:
                    continue
                
                outputs = self.forward_step(batch)
                total_loss += outputs['loss'].item()
                num_batches += 1
        
        return {'val_loss': total_loss / num_batches if num_batches > 0 else 0}
    
    def save_checkpoint(self, epoch: int, output_dir: str):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state
        checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save speech projector separately
        speech_projector_path = os.path.join(checkpoint_path, "speech_projector.bin")
        speech_projector_state = self.model.get_model().speech_projector.state_dict()
        torch.save(speech_projector_state, speech_projector_path)
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'model_args': self.model_args,
            'data_args': self.data_args,
            'training_args': self.training_args
        }
        torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"))
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting Stage 1 training...")
        self.logger.info(f"Total epochs: {self.training_args.num_train_epochs}")
        self.logger.info(f"Batch size: {self.training_args.per_device_train_batch_size}")
        self.logger.info(f"Learning rate: {self.training_args.learning_rate}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.training_args.num_train_epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch metrics
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['train_loss']:.4f}")
            if val_metrics:
                self.logger.info(f"Epoch {epoch}: Val Loss = {val_metrics['val_loss']:.4f}")
            
            if self.training_args.report_to == "wandb":
                wandb.log({
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch
                })
            
            # Save checkpoint
            if epoch % self.training_args.save_steps == 0 or epoch == self.training_args.num_train_epochs:
                self.save_checkpoint(epoch, self.training_args.output_dir)
            
            # Save best model
            if val_metrics and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_model_path = os.path.join(self.training_args.output_dir, "best_model")
                self.save_checkpoint(epoch, best_model_path)
                self.logger.info(f"New best model saved with val_loss: {best_val_loss:.4f}")
        
        self.logger.info("Stage 1 training completed!")
        
        # Save final model
        final_model_path = os.path.join(self.training_args.output_dir, "final_model")
        self.save_checkpoint(self.training_args.num_train_epochs, final_model_path)
        
        return self.model