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
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.setup_model(model_path)
        self.setup_data_loaders()
        self.setup_optimizer_and_scheduler()
        
        # Handle both string and list formats for report_to
        report_to = self.training_args.report_to
        if isinstance(report_to, list):
            wandb_enabled = "wandb" in report_to
        else:
            wandb_enabled = report_to == "wandb" or "wandb" in str(report_to)
        
        if wandb_enabled:
            try:
                wandb.init(
                    project="aolmo",
                    name=self.training_args.run_name,
                    config={
                        **vars(model_args),
                        **vars(data_args),
                        **vars(training_args)
                    },
                    reinit=True
                )
                self.logger.info(f"Initialized wandb run: {wandb.run.url}")
            except Exception as e:
                self.logger.error(f"Failed to initialize wandb: {e}")
    
    def setup_model(self, model_path: Optional[str] = None):
        if model_path:
            self.tokenizer, self.model, _ = load_pretrained_model(
                model_path=model_path,
                model_base=self.model_args.model_name_or_path,
                s2s=False,
                model_args=self.model_args
            )
        else:
            self.tokenizer, self.model, _ = load_pretrained_model(
                model_path=self.model_args.model_name_or_path,
                model_base=None,
                s2s=False,
                model_args=self.model_args
            )
        if not hasattr(self.model.get_model(), 'speech_projector') or self.model.get_model().speech_projector is None:
            self.model.get_model().initialize_speech_modules(self.model_args)
        for param in self.model.get_model().speech_encoder.parameters():
            param.requires_grad = False
        if self.model_args.freeze_backbone:
            model_core = self.model.get_model()
            if hasattr(model_core, 'model'):
                for param in model_core.model.parameters():
                    param.requires_grad = False
            else:
                for name, param in model_core.named_parameters():
                    if not name.startswith('speech_'):
                        param.requires_grad = False
        
        if self.model_args.tune_speech_projector:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_model().speech_projector.parameters():
                param.requires_grad = True
        
        self.logger.info(f"Model initialized. Trainable parameters: {self.count_trainable_params()}")
    
    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def setup_data_loaders(self):
        self.train_loader = create_data_loader(
            data_path=self.data_args.data_path,
            tokenizer=self.tokenizer,
            data_args=self.data_args,
            batch_size=self.training_args.per_device_train_batch_size,
            stage=1,
            num_workers=4,
            shuffle=True
        )
        
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
        param_groups = []
        speech_projector_params = list(self.model.get_model().speech_projector.parameters())
        if speech_projector_params:
            lr = self.training_args.speech_projector_lr or self.training_args.learning_rate
            param_groups.append({
                'params': speech_projector_params,
                'lr': lr,
                'name': 'speech_projector'
            })
        
        if not self.model_args.freeze_backbone:
            model_core = self.model.get_model()
            if hasattr(model_core, 'model'):
                llm_params = list(model_core.model.parameters())
            else:
                llm_params = []
                for name, param in model_core.named_parameters():
                    if not name.startswith('speech_'):
                        llm_params.append(param)
            
            param_groups.append({
                'params': llm_params,
                'lr': self.training_args.learning_rate,
                'name': 'llm'
            })
        
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
        
        total_steps = len(self.train_loader) * self.training_args.num_train_epochs
        warmup_steps = int(total_steps * 0.03)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def forward_step(self, batch: Dict) -> Dict:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.model.device)
                
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
        self.model.train()
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            if not batch:
                continue
        
            self.optimizer.zero_grad()
            outputs = self.forward_step(batch)
            loss = outputs['loss']
            loss.backward()
            if self.training_args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.training_args.max_grad_norm
                )
            
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            # Check wandb logging condition
            report_to = self.training_args.report_to
            if isinstance(report_to, list):
                wandb_enabled = "wandb" in report_to
            else:
                wandb_enabled = report_to == "wandb" or "wandb" in str(report_to)
                
            if wandb_enabled and batch_idx % 10 == 0:
                total_grad_norm = 0
                param_count = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                        param_count += 1
                total_grad_norm = total_grad_norm ** 0.5
                
                speech = batch['speech_features']
                speech_lengths = batch['speech_lengths']
                
                try:
                    log_data = {
                        'train/loss': loss.item(),
                        'train/avg_loss': total_loss / num_batches,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/gradient_norm': total_grad_norm,
                        'train/epoch': epoch,
                        'train/step': epoch * len(self.train_loader) + batch_idx,
                        'train/batch_idx': batch_idx,
                        'system/epoch_progress': batch_idx / len(self.train_loader),
                        'speech/min_value': speech.min().item(),
                        'speech/max_value': speech.max().item(),
                        'speech/mean_value': speech.mean().item(),
                        'speech/std_value': speech.std().item(),
                        'speech/avg_length': speech_lengths.float().mean().item(),
                        'speech/max_length': speech_lengths.max().item(),
                        'speech/min_length': speech_lengths.min().item(),
                        'speech/batch_size': speech.shape[0]
                    }
                    wandb.log(log_data)
                except Exception as e:
                    self.logger.warning(f"Failed to log to wandb: {e}")
            
            # Step-based validation every 2000 steps
            current_step = epoch * len(self.train_loader) + batch_idx
            if hasattr(self.training_args, 'eval_steps') and self.training_args.eval_steps > 0:
                if current_step > 0 and current_step % self.training_args.eval_steps == 0:
                    self.logger.info(f"Running validation at step {current_step}")
                    val_metrics = self.validate()
                    if val_metrics:
                        self.logger.info(f"Step {current_step}: Val Loss = {val_metrics['val_loss']:.4f}")
                        
                        # Log validation metrics to wandb
                        if wandb_enabled:
                            try:
                                wandb.log({
                                    'val/loss': val_metrics['val_loss'],
                                    'val/step': current_step,
                                    'val/epoch': epoch + (batch_idx / len(self.train_loader))
                                })
                            except Exception as e:
                                self.logger.warning(f"Failed to log validation to wandb: {e}")
        
        return {'train_loss': total_loss / num_batches if num_batches > 0 else 0}
    
    def validate(self) -> Dict:
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
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        speech_projector_path = os.path.join(checkpoint_path, "speech_projector.bin")
        speech_projector_state = self.model.get_model().speech_projector.state_dict()
        torch.save(speech_projector_state, speech_projector_path)
        
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
        self.logger.info("Starting Stage 1 training...")
        self.logger.info(f"Total epochs: {self.training_args.num_train_epochs}")
        self.logger.info(f"Batch size: {self.training_args.per_device_train_batch_size}")
        self.logger.info(f"Learning rate: {self.training_args.learning_rate}")
        best_val_loss = float('inf')
        for epoch in range(1, self.training_args.num_train_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['train_loss']:.4f}")
            if val_metrics:
                self.logger.info(f"Epoch {epoch}: Val Loss = {val_metrics['val_loss']:.4f}")
            
            # Check wandb logging for epoch metrics
            report_to = self.training_args.report_to
            if isinstance(report_to, list):
                wandb_enabled = "wandb" in report_to
            else:
                wandb_enabled = report_to == "wandb" or "wandb" in str(report_to)
                
            if wandb_enabled:
                log_dict = {
                    'epoch/train_loss': train_metrics['train_loss'],
                    'epoch/epoch_num': epoch,
                    'system/total_epochs': self.training_args.num_train_epochs,
                    'system/progress': epoch / self.training_args.num_train_epochs
                }
                
                if val_metrics:
                    log_dict['epoch/val_loss'] = val_metrics['val_loss']
                    log_dict['epoch/val_train_diff'] = val_metrics['val_loss'] - train_metrics['train_loss']
                
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                log_dict.update({
                    'model/total_params': total_params,
                    'model/trainable_params': trainable_params,
                    'model/frozen_params': total_params - trainable_params
                })
                
                wandb.log(log_dict)
            
            if epoch % self.training_args.save_steps == 0 or epoch == self.training_args.num_train_epochs:
                self.save_checkpoint(epoch, self.training_args.output_dir)
            if val_metrics and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_model_path = os.path.join(self.training_args.output_dir, "best_model")
                self.save_checkpoint(epoch, best_model_path)
                self.logger.info(f"New best model saved with val_loss: {best_val_loss:.4f}")
        
        self.logger.info("Stage 1 training completed!")
        final_model_path = os.path.join(self.training_args.output_dir, "final_model")
        self.save_checkpoint(self.training_args.num_train_epochs, final_model_path)
        
        return self.model