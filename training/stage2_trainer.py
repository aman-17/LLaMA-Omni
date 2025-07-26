"""
Stage 2 Training: Speech Generation with CTC
Trains the speech decoder while keeping all other components frozen
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import logging
from tqdm import tqdm
from typing import Dict, Optional
import wandb
from omni_speech.model.language_model.omni_speech2s_llama import OmniSpeech2SLlamaForCausalLM
from omni_speech.model.builder import load_pretrained_model
from omni_speech.arguments import ModelArguments, DataArguments, TrainingArguments
from data_utils import create_data_loader


class CTCLoss(nn.Module):
    """CTC Loss for speech unit prediction"""
    
    def __init__(self, blank_idx: int = 0, reduction: str = 'mean'):
        super().__init__()
        self.blank_idx = blank_idx
        self.reduction = reduction
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs: (batch_size, max_time, vocab_size) - log probabilities
            targets: (batch_size, max_target_len) - target unit sequences
            input_lengths: (batch_size,) - lengths of input sequences
            target_lengths: (batch_size,) - lengths of target sequences
        """
        # CTC expects (max_time, batch_size, vocab_size)
        log_probs = log_probs.transpose(0, 1)
        
        # Flatten targets and remove padding (-1)
        targets_flat = []
        target_lengths_clean = []
        
        for i, target_len in enumerate(target_lengths):
            target = targets[i][:target_len]
            # Remove padding tokens (-1)
            target = target[target != -1]
            targets_flat.extend(target.tolist())
            target_lengths_clean.append(len(target))
        
        targets_flat = torch.tensor(targets_flat, dtype=torch.long, device=log_probs.device)
        target_lengths_clean = torch.tensor(target_lengths_clean, dtype=torch.long, device=log_probs.device)
        
        return self.ctc_loss(log_probs, targets_flat, input_lengths, target_lengths_clean)


class Stage2Trainer:
    """Trainer for Stage 2: Speech Generation with CTC"""
    
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        stage1_model_path: str,
        kmeans_model_path: str
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.kmeans_model_path = kmeans_model_path
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        self.setup_model(stage1_model_path)
        
        # Setup CTC loss
        self.ctc_loss = CTCLoss(blank_idx=model_args.unit_vocab_size)  # Use vocab_size as blank
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()
        
        # Setup wandb if enabled
        if self.training_args.report_to == "wandb":
            wandb.init(
                project="llama-omni-stage2",
                config={
                    **vars(model_args),
                    **vars(data_args),
                    **vars(training_args)
                }
            )
    
    def setup_model(self, stage1_model_path: str):
        """Initialize the model with Stage 1 weights"""
        # Load the speech-to-speech model
        self.tokenizer, self.model, _ = load_pretrained_model(
            model_path=stage1_model_path,
            model_base=None,
            s2s=True  # Load speech-to-speech version
        )
        
        # Freeze all components except speech decoder
        for name, param in self.model.named_parameters():
            if 'speech_decoder' not in name and 'speech_generator' not in name:
                param.requires_grad = False
        
        # Only train speech decoder
        for param in self.model.get_model().speech_generator.parameters():
            param.requires_grad = True
        
        self.logger.info(f"Model initialized. Trainable parameters: {self.count_trainable_params()}")
    
    def count_trainable_params(self) -> int:
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def setup_data_loaders(self):
        """Setup training and validation data loaders for Stage 2"""
        self.train_loader = create_data_loader(
            data_path=self.data_args.data_path,
            tokenizer=self.tokenizer,
            data_args=self.data_args,
            batch_size=self.training_args.per_device_train_batch_size,
            stage=2,
            kmeans_model_path=self.kmeans_model_path,
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
                stage=2,
                kmeans_model_path=self.kmeans_model_path,
                num_workers=4,
                shuffle=False
            )
        else:
            self.val_loader = None
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Only optimize speech decoder parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=2e-4,  # Higher learning rate for stage 2 as mentioned in paper
            weight_decay=self.training_args.weight_decay
        )
        
        # Calculate total training steps
        total_steps = len(self.train_loader) * self.training_args.num_train_epochs
        warmup_steps = int(total_steps * 0.03)  # 3% warmup
        
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
        
        # Forward pass through LLM (frozen) to get hidden states
        with torch.no_grad():
            llm_outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                speech=batch['speech_features'],
                speech_lengths=batch['speech_lengths'],
                output_hidden_states=True
            )
        
        # Get hidden states for speech generation
        hidden_states = llm_outputs.hidden_states[-1]  # Last layer hidden states
        
        # Generate speech units using speech decoder
        speech_outputs = self.model.get_model().speech_generator(
            hidden_states=hidden_states,
            attention_mask=batch['attention_mask']
        )
        
        # Calculate CTC loss
        log_probs = F.log_softmax(speech_outputs.logits, dim=-1)
        
        # Get sequence lengths for CTC
        input_lengths = batch['attention_mask'].sum(dim=1)
        target_lengths = batch['unit_lengths']
        
        ctc_loss = self.ctc_loss(
            log_probs=log_probs,
            targets=batch['speech_units'],
            input_lengths=input_lengths,
            target_lengths=target_lengths
        )
        
        return {
            'loss': ctc_loss * self.model_args.ctc_loss_weight,
            'logits': speech_outputs.logits,
            'ctc_loss': ctc_loss
        }
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch"""
        self.model.train()
        
        # Keep frozen components in eval mode
        self.model.get_model().speech_encoder.eval()
        self.model.get_model().speech_projector.eval()
        self.model.get_model().model.eval()
        
        total_loss = 0
        total_ctc_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Skip empty batches or batches without speech units
            if not batch or 'speech_units' not in batch:
                continue
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward_step(batch)
            loss = outputs['loss']
            ctc_loss = outputs['ctc_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.training_args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    self.training_args.max_grad_norm
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_ctc_loss += ctc_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ctc_loss': f"{ctc_loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.training_args.report_to == "wandb" and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_ctc_loss': ctc_loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'step': epoch * len(self.train_loader) + batch_idx
                })
        
        return {
            'train_loss': total_loss / num_batches if num_batches > 0 else 0,
            'train_ctc_loss': total_ctc_loss / num_batches if num_batches > 0 else 0
        }
    
    def validate(self) -> Dict:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        total_ctc_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if not batch or 'speech_units' not in batch:
                    continue
                
                outputs = self.forward_step(batch)
                total_loss += outputs['loss'].item()
                total_ctc_loss += outputs['ctc_loss'].item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches if num_batches > 0 else 0,
            'val_ctc_loss': total_ctc_loss / num_batches if num_batches > 0 else 0
        }
    
    def save_checkpoint(self, epoch: int, output_dir: str):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state
        checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save full model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save speech decoder separately
        speech_decoder_path = os.path.join(checkpoint_path, "speech_decoder.bin")
        speech_decoder_state = self.model.get_model().speech_generator.state_dict()
        torch.save(speech_decoder_state, speech_decoder_path)
        
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
        self.logger.info("Starting Stage 2 training...")
        self.logger.info(f"Total epochs: {self.training_args.num_train_epochs}")
        self.logger.info(f"Batch size: {self.training_args.per_device_train_batch_size}")
        self.logger.info(f"Learning rate: 2e-4")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.training_args.num_train_epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch metrics
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['train_loss']:.4f}, "
                           f"Train CTC Loss = {train_metrics['train_ctc_loss']:.4f}")
            if val_metrics:
                self.logger.info(f"Epoch {epoch}: Val Loss = {val_metrics['val_loss']:.4f}, "
                               f"Val CTC Loss = {val_metrics['val_ctc_loss']:.4f}")
            
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
        
        self.logger.info("Stage 2 training completed!")
        
        # Save final model
        final_model_path = os.path.join(self.training_args.output_dir, "final_model")
        self.save_checkpoint(self.training_args.num_train_epochs, final_model_path)
        
        return self.model