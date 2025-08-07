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

class WeightedEncoderProjectorConcat(nn.Module):
      def __init__(self, config):
          super().__init__()
          self.k = config.speech_encoder_ds_rate
          self.encoder_dim = config.speech_encoder_hidden_size
          self.temporal_weights = nn.Parameter(torch.ones(self.k) / self.k)
          self.softmax = nn.Softmax(dim=-1)
          self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
          self.linear2 = nn.Linear(2048, config.hidden_size)
          self.residual = nn.Linear(self.encoder_dim, config.hidden_size)

      def forward(self, x):
          batch_size, seq_len, dim = x.size()
          # Truncate to multiple of k
          x = x[:, :seq_len - (seq_len % self.k), :]
          seq_len = x.size(1)
          x_reshaped = x.view(batch_size, seq_len // self.k, self.k, dim)
          weights = self.softmax(self.temporal_weights).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
          x_weighted = (x_reshaped * weights).view(batch_size, seq_len // self.k, dim * self.k)
          main_path = self.linear2(F.relu(self.linear1(x_weighted)))
          center_frame = x_reshaped[:, :, self.k//2, :]
          residual_path = self.residual(center_frame)
          return main_path + 0.1 * residual_path


class SpectralEncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.speech_encoder_ds_rate
        self.encoder_dim = config.speech_encoder_hidden_size
        self.register_buffer('dct_basis', self._get_dct_basis(self.k))
        self.freq_selector = nn.Linear(self.encoder_dim, self.k)
        self.sigmoid = nn.Sigmoid()
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_dim * self.k, 2048),
            nn.ReLU(),
            nn.Linear(2048, config.hidden_size)
        )
    def _get_dct_basis(self, k):
        basis = torch.zeros(k, k)
        for i in range(k):
            for j in range(k):
                if i == 0:
                    basis[i, j] = 1.0 / math.sqrt(k)
                else:
                    basis[i, j] = math.sqrt(2.0/k) * math.cos(math.pi * i * (2*j + 1) / (2*k))
        return basis

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        x = x[:, :seq_len - (seq_len % self.k), :]
        seq_len = x.size(1)
        x_windows = x.view(batch_size, seq_len // self.k, self.k, dim)
        x_freq = torch.matmul(x_windows, self.dct_basis.T)  # [B, W, k, D]
        freq_weights = self.sigmoid(self.freq_selector(x_freq.mean(dim=2)))  # [B, W, k]
        x_freq_weighted = x_freq * freq_weights.unsqueeze(-1)
        x_time = torch.matmul(x_freq_weighted, self.dct_basis)
        x_concat = x_time.view(batch_size, seq_len // self.k, dim * self.k)

        return self.projector(x_concat)


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
