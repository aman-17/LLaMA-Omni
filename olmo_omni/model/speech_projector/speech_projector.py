# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/projector.py


import torch
import torch.nn as nn


class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = config.speech_encoder_hidden_size
        self.llm_dim = config.hidden_size
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.hidden_size)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class ConvAttentionProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.speech_encoder_hidden_size
        self.llm_dim = config.hidden_size
        self.compression_ratio = config.speech_encoder_ds_rate
        self.conv1d = nn.Conv1d(
            self.encoder_dim,
            self.encoder_dim,
            kernel_size=self.compression_ratio,
            stride=self.compression_ratio,
            padding=0,
        )
        self.attention = nn.MultiheadAttention(
            self.encoder_dim, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(self.encoder_dim)
        self.projector = nn.Linear(self.encoder_dim, self.llm_dim)

    def forward(self, x):
        # x: [batch, seq_len, dim]
        x = x.transpose(1, 2)  # [batch, dim, seq_len]
        x = self.conv1d(x)  # Temporal compression
        x = x.transpose(1, 2)  # [batch, compressed_seq, dim]
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        return self.projector(x)


class HierarchicalPoolingProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.speech_encoder_hidden_size
        self.llm_dim = config.hidden_size
        self.k = config.speech_encoder_ds_rate
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.attention_pool = nn.Sequential(
            nn.Linear(self.encoder_dim, 1), nn.Softmax(dim=1)
        )
        self.combine = nn.Linear(self.encoder_dim * 3, self.encoder_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.encoder_dim * 2, self.llm_dim),
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_chunks = seq_len // self.k
        if seq_len % self.k != 0:
            x = x[:, : num_chunks * self.k, :]
        x_chunks = x.view(batch_size, num_chunks, self.k, dim)
        avg_pooled = x_chunks.mean(dim=2)  # [batch, num_chunks, dim]
        max_pooled = x_chunks.max(dim=2)[0]
        attention_weights = self.attention_pool(
            x_chunks.view(batch_size * num_chunks, self.k, dim)
        )
        attention_pooled = (
            x_chunks.view(batch_size * num_chunks, self.k, dim) * attention_weights
        ).sum(dim=1)
        attention_pooled = attention_pooled.view(batch_size, num_chunks, dim)
        combined = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=-1)
        compressed = self.combine(combined)
        return self.projector(compressed)


class LearnableTokenCompressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.speech_encoder_hidden_size
        self.llm_dim = config.hidden_size
        self.compression_ratio = config.speech_encoder_ds_rate
        self.num_compress_tokens = config.max_speech_tokens // self.compression_ratio
        self.compress_tokens = nn.Parameter(
            torch.randn(1, self.num_compress_tokens, self.encoder_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            self.encoder_dim, num_heads=8, batch_first=True
        )
        self.norm1 = nn.LayerNorm(self.encoder_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim * 4),
            nn.GELU(),
            nn.Linear(self.encoder_dim * 4, self.encoder_dim),
        )
        self.norm2 = nn.LayerNorm(self.encoder_dim)
        self.projector = nn.Linear(self.encoder_dim, self.llm_dim)

    def forward(self, x):
        batch_size = x.size(0)
        compress_tokens = self.compress_tokens.expand(batch_size, -1, -1)
        compressed, _ = self.cross_attention(compress_tokens, x, x)
        compressed = self.norm1(compressed + compress_tokens)
        ffn_out = self.ffn(compressed)
        compressed = self.norm2(compressed + ffn_out)
        return self.projector(compressed)


class HiggsStyleAudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.speech_encoder_hidden_size
        self.llm_dim = config.hidden_size
        self.k = config.speech_encoder_ds_rate
        self.linear = nn.Linear(self.encoder_dim, self.llm_dim, bias=True)

        self.use_compression = getattr(config, "compress_speech_tokens", False)
        if self.use_compression:
            self.compression_type = getattr(
                config, "compression_type", "attention_pool"
            )

            if self.compression_type == "attention_pool":
                self.attention_pool = nn.MultiheadAttention(
                    self.llm_dim, num_heads=8, batch_first=True
                )
                self.query_tokens = nn.Parameter(
                    torch.randn(1, config.max_speech_tokens // self.k, self.llm_dim)
                )
            elif self.compression_type == "conv_pool":
                self.conv_compress = nn.Conv1d(
                    self.llm_dim,
                    self.llm_dim,
                    kernel_size=self.k,
                    stride=self.k,
                    padding=0,
                )
            elif self.compression_type == "adaptive_pool":
                self.adaptive_pool = nn.AdaptiveAvgPool1d(
                    config.max_speech_tokens // self.k
                )

    def forward(self, x):
        projected = self.linear(x)  # [batch, seq_len, llm_dim]
        if not self.use_compression:
            return projected
        if self.compression_type == "attention_pool":
            batch_size = x.size(0)
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            compressed, _ = self.attention_pool(query_tokens, projected, projected)
            return compressed

        elif self.compression_type == "conv_pool":
            x_conv = projected.transpose(1, 2)  # [batch, llm_dim, seq_len]
            compressed = self.conv_compress(x_conv).transpose(1, 2)
            return compressed

        elif self.compression_type == "adaptive_pool":
            x_pool = projected.transpose(1, 2)  # [batch, llm_dim, seq_len]
            compressed = self.adaptive_pool(x_pool).transpose(1, 2)
            return compressed

        return projected


class MultiScaleAudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.speech_encoder_hidden_size
        self.llm_dim = config.hidden_size
        self.k = config.speech_encoder_ds_rate
        self.scales = [1, 2, 4]  # Different temporal scales
        self.scale_projectors = nn.ModuleList(
            [nn.Linear(self.encoder_dim * scale, self.llm_dim) for scale in self.scales]
        )
        self.scale_attention = nn.MultiheadAttention(
            self.llm_dim, num_heads=8, batch_first=True
        )
        self.final_compress = nn.Conv1d(
            self.llm_dim, self.llm_dim, kernel_size=self.k, stride=self.k, padding=0
        )
        self.norm = nn.LayerNorm(self.llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        scale_outputs = []
        for i, scale in enumerate(self.scales):
            if seq_len % scale == 0:
                x_scale = x.view(batch_size, seq_len // scale, dim * scale)
                projected = self.scale_projectors[i](x_scale)
                scale_outputs.append(projected)
            else:
                pad_len = scale - (seq_len % scale)
                x_padded = torch.cat([x, x[:, -pad_len:, :]], dim=1)
                x_scale = x_padded.view(
                    batch_size, (seq_len + pad_len) // scale, dim * scale
                )
                projected = self.scale_projectors[i](x_scale)
                scale_outputs.append(projected)

        if len(scale_outputs) > 1:
            combined, _ = self.scale_attention(
                scale_outputs[0],
                torch.cat(scale_outputs, dim=1),
                torch.cat(scale_outputs, dim=1),
            )
        else:
            combined = scale_outputs[0]
        combined = combined.transpose(1, 2)
        compressed = self.final_compress(combined).transpose(1, 2)
        return self.norm(compressed)
