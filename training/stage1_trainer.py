import logging
import os
from typing import Dict, Optional

import torch
import torch.distributed as dist
from data_utils import create_data_loader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import wandb
from olmo_omni.arguments import DataArguments, ModelArguments, TrainingArguments
from olmo_omni.model.builder import load_pretrained_model


class Stage1Trainer:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        model_path: Optional[str] = None,
        is_distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        if rank == 0:
            logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        self.setup_model(model_path)
        self.setup_data_loaders()
        self.setup_optimizer_and_scheduler()

        if self.rank == 0:
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
                            **vars(training_args),
                        },
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
                model_args=self.model_args,
            )
        else:
            self.tokenizer, self.model, _ = load_pretrained_model(
                model_path=self.model_args.model_name_or_path,
                model_base=None,
                s2s=False,
                model_args=self.model_args,
            )

        underlying_model = self.model
        if (
            not hasattr(underlying_model.get_model(), "speech_projector")
            or underlying_model.get_model().speech_projector is None
        ):
            underlying_model.get_model().initialize_speech_modules(self.model_args)
        for param in underlying_model.get_model().speech_encoder.parameters():
            param.requires_grad = False
        if self.model_args.freeze_backbone:
            model_core = underlying_model.get_model()
            if hasattr(model_core, "model"):
                for param in model_core.model.parameters():
                    param.requires_grad = False
            else:
                for name, param in model_core.named_parameters():
                    if not name.startswith("speech_"):
                        param.requires_grad = False

        if self.model_args.tune_speech_projector:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in underlying_model.get_model().speech_projector.parameters():
                param.requires_grad = True

        if self.is_distributed:
            device = torch.device(f"cuda:{self.local_rank}")
            self.model = self.model.to(device)
            self.model = DDP(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank
            )

        if self.rank == 0:
            self.logger.info(
                f"Model initialized. Trainable parameters: {self.count_trainable_params()}"
            )

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_underlying_model(self):
        if self.is_distributed:
            return self.model.module
        return self.model

    def setup_data_loaders(self):
        num_workers = 0 if self.is_distributed else 4
        self.train_loader = create_data_loader(
            data_path=self.data_args.data_path,
            tokenizer=self.tokenizer,
            data_args=self.data_args,
            batch_size=self.training_args.per_device_train_batch_size,
            stage=1,
            num_workers=num_workers,
            shuffle=True,
            is_distributed=self.is_distributed,
            rank=self.rank,
            world_size=self.world_size,
        )

        if (
            hasattr(self.data_args, "validation_data_path")
            and self.data_args.validation_data_path
        ):
            self.val_loader = create_data_loader(
                data_path=self.data_args.validation_data_path,
                tokenizer=self.tokenizer,
                data_args=self.data_args,
                batch_size=self.training_args.per_device_eval_batch_size,
                stage=1,
                num_workers=num_workers,
                shuffle=False,
                is_distributed=self.is_distributed,
                rank=self.rank,
                world_size=self.world_size,
            )
        else:
            self.val_loader = None

    def setup_optimizer_and_scheduler(self):
        param_groups = []
        underlying_model = self.get_underlying_model()
        speech_projector_params = list(
            underlying_model.get_model().speech_projector.parameters()
        )
        if speech_projector_params:
            lr = (
                self.training_args.speech_projector_lr
                or self.training_args.learning_rate
            )
            param_groups.append(
                {
                    "params": speech_projector_params,
                    "lr": lr,
                    "name": "speech_projector",
                }
            )

        if not self.model_args.freeze_backbone:
            model_core = underlying_model.get_model()
            if hasattr(model_core, "model"):
                llm_params = list(model_core.model.parameters())
            else:
                llm_params = []
                for name, param in model_core.named_parameters():
                    if not name.startswith("speech_"):
                        llm_params.append(param)

            param_groups.append(
                {
                    "params": llm_params,
                    "lr": self.training_args.learning_rate,
                    "name": "llm",
                }
            )

        lm_head_params = list(underlying_model.lm_head.parameters())
        param_groups.append(
            {
                "params": lm_head_params,
                "lr": self.training_args.learning_rate,
                "name": "lm_head",
            }
        )

        self.optimizer = AdamW(
            param_groups,
            lr=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay,
        )

        total_steps = len(self.train_loader) * self.training_args.num_train_epochs
        warmup_steps = int(total_steps * 0.02)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def forward_step(self, batch: Dict) -> Dict:
        underlying_model = self.get_underlying_model()
        model_dtype = next(underlying_model.parameters()).dtype

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(underlying_model.device)
                if batch[key].dtype.is_floating_point:
                    batch[key] = batch[key].to(dtype=model_dtype)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            speech=batch["speech_features"],
            speech_lengths=batch["speech_lengths"],
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
        }

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()

        if self.is_distributed and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0
        num_batches = 0

        micro_batch_size = getattr(self.training_args, "micro_batch_size", None)

        if self.rank == 0:
            print(
                f"Starting epoch {epoch}, data loader length: {len(self.train_loader)}"
            )
            if micro_batch_size:
                print(f"Using micro-batching with micro_batch_size: {micro_batch_size}")
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        else:
            progress_bar = self.train_loader

        for batch_idx, batch in enumerate(progress_bar):
            if not batch:
                continue

            self.optimizer.zero_grad()

            if micro_batch_size and micro_batch_size < batch["input_ids"].size(0):
                batch_size = batch["input_ids"].size(0)
                total_loss_batch = 0.0
                num_microbatches = 0

                for start_idx in range(0, batch_size, micro_batch_size):
                    end_idx = min(start_idx + micro_batch_size, batch_size)
                    microbatch = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            microbatch[key] = value[start_idx:end_idx]
                        else:
                            microbatch[key] = value[start_idx:end_idx]

                    outputs = self.forward_step(microbatch)
                    loss = outputs["loss"]

                    scaled_loss = loss / (
                        (batch_size + micro_batch_size - 1) // micro_batch_size
                    )
                    scaled_loss.backward()

                    total_loss_batch += loss.detach().item()
                    num_microbatches += 1

                avg_loss = total_loss_batch / num_microbatches
            else:
                outputs = self.forward_step(batch)
                loss = outputs["loss"]
                loss.backward()
                avg_loss = loss.item()

            if self.training_args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.training_args.max_grad_norm
                )

            self.optimizer.step()
            self.scheduler.step()
            total_loss += avg_loss
            num_batches += 1

            if self.rank == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "avg_loss": f"{total_loss / num_batches:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            report_to = self.training_args.report_to
            if isinstance(report_to, list):
                wandb_enabled = "wandb" in report_to
            else:
                wandb_enabled = report_to == "wandb" or "wandb" in str(report_to)

            if self.rank == 0:

                if wandb_enabled and batch_idx % 10 == 0:
                    total_grad_norm = 0
                    param_count = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.data.norm(2).item() ** 2
                            param_count += 1
                    total_grad_norm = total_grad_norm**0.5

                    speech = batch["speech_features"]
                    speech_lengths = batch["speech_lengths"]

                    try:
                        log_data = {
                            "train/loss": avg_loss,
                            "train/avg_loss": total_loss / num_batches,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/gradient_norm": total_grad_norm,
                            "train/epoch": epoch,
                            "train/step": epoch * len(self.train_loader) + batch_idx,
                            "train/batch_idx": batch_idx,
                            "system/epoch_progress": batch_idx / len(self.train_loader),
                            "train/batch_size": speech.shape[0],
                        }
                        if micro_batch_size:
                            log_data["train/micro_batch_size"] = micro_batch_size
                        wandb.log(log_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to log to wandb: {e}")

            current_step = epoch * len(self.train_loader) + batch_idx
            if (
                hasattr(self.training_args, "eval_steps")
                and self.training_args.eval_steps > 0
            ):
                if (
                    current_step > 0
                    and current_step % self.training_args.eval_steps == 0
                ):
                    self.logger.info(f"Running validation at step {current_step}")
                    val_metrics = self.validate()
                    if val_metrics:
                        self.logger.info(
                            f"Step {current_step}: Val Loss = {val_metrics['val_loss']:.4f}"
                        )

                        if self.rank == 0 and wandb_enabled:
                            try:
                                wandb.log(
                                    {
                                        "val/loss": val_metrics["val_loss"],
                                        "val/step": current_step,
                                        "val/epoch": epoch
                                        + (batch_idx / len(self.train_loader)),
                                    }
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to log validation to wandb: {e}"
                                )

        return {"train_loss": total_loss / num_batches if num_batches > 0 else 0}

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
                total_loss += outputs["loss"].item()
                num_batches += 1

        return {"val_loss": total_loss / num_batches if num_batches > 0 else 0}

    def save_checkpoint(self, epoch: int, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model_to_save = self.model.module if self.is_distributed else self.model

        try:
            model_to_save.save_pretrained(checkpoint_path, safe_serialization=True)
            if self.rank == 0:
                self.logger.info(
                    f"Model saved in safetensors format to {checkpoint_path}"
                )
        except (AttributeError, RuntimeError) as e:
            try:
                model_to_save.save_pretrained(checkpoint_path, safe_serialization=False)
                if self.rank == 0:
                    self.logger.info(
                        f"Model saved in pytorch format to {checkpoint_path}"
                    )
            except (AttributeError, RuntimeError) as e2:
                if self.rank == 0:
                    self.logger.warning(
                        f"save_pretrained failed ({e2}), using fallback method"
                    )
                model_state_path = os.path.join(checkpoint_path, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), model_state_path)

        try:
            self.tokenizer.save_pretrained(checkpoint_path)
        except AttributeError:
            if hasattr(self.tokenizer, "state_dict"):
                tokenizer_path = os.path.join(checkpoint_path, "tokenizer.bin")
                torch.save(self.tokenizer.state_dict(), tokenizer_path)

        if hasattr(model_to_save, "get_model") and hasattr(
            model_to_save.get_model(), "speech_projector"
        ):
            speech_projector_path = os.path.join(
                checkpoint_path, "speech_projector.bin"
            )
            speech_projector_state = (
                model_to_save.get_model().speech_projector.state_dict()
            )
            torch.save(speech_projector_state, speech_projector_path)

        training_state = {
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "model_args": self.model_args,
            "data_args": self.data_args,
            "training_args": self.training_args,
        }
        torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"))

        if hasattr(model_to_save, "config"):
            config_path = os.path.join(checkpoint_path, "config.json")
            with open(config_path, "w") as f:
                import json

                if hasattr(model_to_save.config, "__dict__"):
                    config_dict = {
                        k: v
                        for k, v in model_to_save.config.__dict__.items()
                        if not callable(v) and not k.startswith("_")
                    }
                    json.dump(config_dict, f, indent=2, default=str)

        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def train(self):
        if self.rank == 0:
            self.logger.info("Starting Stage 1 training...")
            self.logger.info(f"Total epochs: {self.training_args.num_train_epochs}")
            self.logger.info(
                f"Batch size: {self.training_args.per_device_train_batch_size}"
            )
            self.logger.info(f"Learning rate: {self.training_args.learning_rate}")
            print(
                f"About to start training loop with {self.training_args.num_train_epochs} epochs"
            )

        best_val_loss = float("inf")
        for epoch in range(1, self.training_args.num_train_epochs + 1):
            if self.rank == 0:
                print(f"Starting epoch {epoch}/{self.training_args.num_train_epochs}")
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            if self.rank == 0:
                self.logger.info(
                    f"Epoch {epoch}: Train Loss = {train_metrics['train_loss']:.4f}"
                )
                if val_metrics:
                    self.logger.info(
                        f"Epoch {epoch}: Val Loss = {val_metrics['val_loss']:.4f}"
                    )

                report_to = self.training_args.report_to
                if isinstance(report_to, list):
                    wandb_enabled = "wandb" in report_to
                else:
                    wandb_enabled = report_to == "wandb" or "wandb" in str(report_to)

                if wandb_enabled:
                    log_dict = {
                        "epoch/train_loss": train_metrics["train_loss"],
                        "epoch/epoch_num": epoch,
                        "system/total_epochs": self.training_args.num_train_epochs,
                        "system/progress": epoch / self.training_args.num_train_epochs,
                    }
                    if val_metrics:
                        log_dict["epoch/val_loss"] = val_metrics["val_loss"]
                        log_dict["epoch/val_train_diff"] = (
                            val_metrics["val_loss"] - train_metrics["train_loss"]
                        )

                    wandb.log(log_dict)

            if self.rank == 0:
                if (
                    epoch % self.training_args.save_steps == 0
                    or epoch == self.training_args.num_train_epochs
                ):
                    self.save_checkpoint(epoch, self.training_args.output_dir)
                if val_metrics and val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint(epoch, self.training_args.output_dir)
                    self.logger.info(
                        f"New best model saved with val_loss: {best_val_loss:.4f}"
                    )

            if self.is_distributed:
                dist.barrier()

        if self.rank == 0:
            self.logger.info("Stage 1 training completed!")
            final_model_path = os.path.join(
                self.training_args.output_dir, "final_model"
            )
            self.save_checkpoint(self.training_args.num_train_epochs, final_model_path)

        return self.model
