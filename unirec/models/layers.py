import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _pair
from typing import Literal


class ScaledDotProductAttentionLogits(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttentionLogits, self).__init__()

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        mask: Tensor | None = None,
        clip: float | None = None,
        temp: float = 1.0,
    ) -> Tensor:
        matmul_qk = torch.matmul(
            q, torch.transpose(k, -2, -1)
        )  # matmul query with key (batch_size, seq_len_q, seq_len_k)
        dk = float(k.shape[-1])
        attention_logits = matmul_qk / dk**0.5  # scaling by key dimension
        attention_logits = attention_logits / temp  # softmax annealing applied
        if clip is not None:
            attention_logits = clip * torch.tanh(
                attention_logits
            )  # logit clipping applied
        if mask is not None:
            attention_logits += mask * -1e9  # masking applied

        return attention_logits


class AdditiveAttentionLogits(nn.Module):
    def __init__(self):
        super(AdditiveAttentionLogits, self).__init__()

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        clip: float | None = None,
        temp: float = 1.0,
    ) -> Tensor:
        attention_logits = torch.sum(
            v * torch.tanh(q + k), -1
        )  # additive attention - NCO(8), (batch_size, key_len)
        attention_logits = (
            attention_logits / temp
        )  # softmax annealing applied - NCO(15)
        if clip is not None:
            attention_logits = clip * torch.tanh(
                attention_logits
            )  # logit clipping applied - NCO(16)
        if mask is not None:
            attention_logits += mask * -1e9  # masking applied - NCO(8)

        return attention_logits


class MultiHeadAttention(nn.Module):
    def __init__(self, d_q: int, d_k: int, d_v: int, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_q, d_model)
        self.wk = nn.Linear(d_k, d_model)
        self.wv = nn.Linear(d_v, d_model)
        self.scaled_dot_product_attention_logits = ScaledDotProductAttentionLogits()
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(d_model, d_model)
        # self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=1, strides=1, padding='valid')

    def split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x.permute(0, 2, 1, 3)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        batch_size = q.shape[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        attention_logits = self.scaled_dot_product_attention_logits(q, k, mask=mask)
        attention_weights = self.softmax(attention_logits)
        attention = torch.matmul(attention_weights, v)

        # (batch_size, seq_len_q, num_heads, depth)
        attention = attention.permute(0, 2, 1, 3)

        # (batch_size, seq_len_q, d_model)
        concat_attention = torch.reshape(attention, (batch_size, -1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        out = self.linear(concat_attention)

        return out, attention_weights


class PointWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_x: int, d_ff: int, d_model: int):
        super(PointWiseFeedForwardNetwork, self).__init__()

        self.linear_ff = nn.Linear(d_x, d_ff)
        self.relu = nn.ReLU()
        self.linear_model = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        net = self.linear_ff(x)  # (batch_size, seq_len, dff)
        net = self.relu(net)
        out = self.linear_model(net)  # (batch_size, seq_len, d_model)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_x: int, d_model: int, num_heads: int, d_ff: int, rate: float = 0.1
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_x, d_x, d_x, d_model, num_heads)
        self.dropout1 = nn.Dropout(p=rate)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = PointWiseFeedForwardNetwork(d_model, d_ff, d_model)
        self.dropout2 = nn.Dropout(p=rate)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_x: int, d_model: int, num_heads: int, d_ff: int, rate: float = 0.1
    ):
        super(TransformerDecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_x, d_x, d_x, d_model, num_heads)
        self.dropout1 = nn.Dropout(p=rate)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.mha2 = MultiHeadAttention(d_model, d_model, d_model, d_model, num_heads)
        self.dropout2 = nn.Dropout(p=rate)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = PointWiseFeedForwardNetwork(d_model, d_ff, d_model)
        self.dropout3 = nn.Dropout(p=rate)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self, x: Tensor, enc_output: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, _ = self.mha1(x, x, x, mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, _ = self.mha2(
            out1, enc_output, enc_output, mask
        )  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_x: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        rate: float = 0.1,
    ):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.linear = nn.Linear(d_x, d_model)
        self.dropout = nn.Dropout(p=rate)

        for i in range(num_layers):
            setattr(
                self,
                f"enc_layers_{i}",
                TransformerEncoderLayer(d_model, d_model, num_heads, d_ff, rate),
            )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        seq_len = x.shape[1]
        net = self.linear(x)
        net *= self.d_model**0.5

        net = self.dropout(net)

        for i in range(self.num_layers):
            net = getattr(self, f"enc_layers_{i}")(net, mask)
        out = net

        return out  # (batch_size, input_seq_len, d_model)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_x: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        rate: float = 0.1,
    ):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=rate)

        for i in range(self.num_layers):
            setattr(
                self,
                f"dec_layers_{i}",
                TransformerDecoderLayer(d_x, d_model, num_heads, d_ff, rate),
            )

    def forward(
        self, x: Tensor, enc_output: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        seq_len = x.shape[1]
        net = x
        net *= self.d_model**0.5
        net = self.dropout(net)

        for i in range(self.num_layers):
            net = getattr(self, f"dec_layers_{i}")(net, enc_output, mask)
        out = net
        return out  # (batch_size, target_seq_len, d_model)


AttentionMethod = Literal["additive", "scaled_dot_product"]


class PointerNetwork(nn.Module):
    def __init__(
        self,
        d_q: int,
        d_k: int,
        d_model: int,
        attention_method: AttentionMethod,
    ):
        super(PointerNetwork, self).__init__()

        self.d_model = d_model
        self.attention_method = attention_method
        self.wq = nn.Linear(d_q, d_model)
        self.wk = nn.Linear(d_k, d_model)

        self.attention_logits: ScaledDotProductAttentionLogits | AdditiveAttentionLogits
        if attention_method == "scaled_dot_product":
            self.attention_logits = ScaledDotProductAttentionLogits()
        elif attention_method == "additive":
            self.v = nn.Parameter(torch.randn((1, d_model)), requires_grad=True)
            self.attention_logits = AdditiveAttentionLogits()

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        mask: Tensor | None = None,
        clip: float = 10.0,
        temp: float = 1.0,
    ) -> Tensor:
        q = self.wq(q)
        q = torch.unsqueeze(q, 1)
        k = self.wk(k)  # (batch_size, seq_len, d_model)

        if self.attention_method == "scaled_dot_product":
            pointer_logits = self.attention_logits(
                q, k, mask=mask, clip=clip, temp=temp
            )
        elif self.attention_method == "additive":
            pointer_logits = self.attention_logits(
                q, k, self.v, mask=mask, clip=clip, temp=temp
            )

        pointer_logits = torch.squeeze(pointer_logits, 1)

        return pointer_logits


class LstmPointerNetwork(nn.Module):
    def __init__(
        self,
        d_q: int,
        d_k: int,
        d_model: int,
        attention_method: Literal["additive", "scaled_dot_product"] = "additive",
    ):
        super().__init__()

        if attention_method not in ["additive", "scaled_dot_product"]:
            raise ValueError(f"{attention_method} is not valid attention_method")

        self.d_q = d_q
        self.d_model = d_model
        self.attention_method = attention_method
        self.wq = nn.LSTMCell(d_q, d_model)
        self.wq_state: tuple[Tensor, Tensor] | None = None
        self.wk = nn.Linear(d_k, d_model)

        self.attention_logits: ScaledDotProductAttentionLogits | AdditiveAttentionLogits
        if attention_method == "scaled_dot_product":
            self.attention_logits = ScaledDotProductAttentionLogits()
        else:
            self.v = nn.Parameter(torch.empty(1, d_model))
            nn.init.kaiming_uniform_(self.v)
            self.attention_logits = AdditiveAttentionLogits()

    def reset_wq_state(self) -> None:
        self.wq_state = None

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        mask: Tensor | None,
        clip: float | None = 10.0,
        temp: float = 1.0,
    ) -> Tensor:
        if self.wq_state is None:
            h0 = torch.zeros(q.shape[0], self.d_model, device=q.device, dtype=q.dtype)
            c0 = torch.zeros(q.shape[0], self.d_model, device=q.device, dtype=q.dtype)
            self.wq_state = (h0, c0)
        q, self.wq_state = self.wq(q, self.wq_state)
        q = q.unsqueeze(1)
        k = self.wk(k)
        if self.attention_method == "scaled_dot_product":
            pointer_logits = self.attention_logits(
                q, k, mask=mask, clip=clip, temp=temp
            )
        else:
            pointer_logits = self.attention_logits(
                q, k, self.v, mask=mask, clip=clip, temp=temp
            )
        return pointer_logits.squeeze(1)


class Mlp(nn.Module):
    def __init__(self, d_x: int, d_out: int, ds_hidden: list[int]):
        super(Mlp, self).__init__()

        self.units = ds_hidden + [d_out]
        for i in range(len(self.units)):
            next_dim = self.units[i]
            if i == 0:
                prev_dim = d_x
            else:
                prev_dim = self.units[i - 1]

            setattr(self, f"dense_{i}", nn.Linear(prev_dim, next_dim))
            setattr(self, f"relu_{i}", nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        net = x
        for i in range(len(self.units)):
            net = getattr(self, f"dense_{i}")(net)
            net = getattr(self, f"relu_{i}")(net)
        out = net
        return out


class MlpNum(nn.Module):
    def __init__(self, d_x: int, d_out: int, ds_hidden: list[int]):
        super(MlpNum, self).__init__()

        self.units = ds_hidden + [d_out]
        for i in range(len(self.units)):
            next_dim = self.units[i]
            if i == 0:
                prev_dim = d_x
            else:
                prev_dim = self.units[i - 1]

            setattr(self, f"dense_{i}", nn.Linear(prev_dim, next_dim))

    def forward(self, x: Tensor) -> Tensor:
        net = x
        for i in range(len(self.units)):
            net = getattr(self, f"dense_{i}")(net)
        out = net
        return out


def _same_pad(
    h: int, w: int, k: tuple[int, int], s: tuple[int, int], d: tuple[int, int]
) -> tuple[int, int, int, int]:
    # Remaining 1 pixel will be added to right/below (Upper rule)
    out_h = math.ceil(h / s[0])
    out_w = math.ceil(w / s[1])
    pad_h = max((out_h - 1) * s[0] + (k[0] - 1) * d[0] + 1 - h, 0)
    pad_w = max((out_w - 1) * s[1] + (k[1] - 1) * d[1] + 1 - w, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return (left, right, top, bottom)


class Cnn(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        filters: list[int] = [32, 64, 128, 128],
        kernels: list[int] = [1, 4, 4, 4],
        strides: list[int] = [1, 2, 2, 2],
    ):
        super(Cnn, self).__init__()

        self.filters = filters

        for i in range(len(filters)):
            if i > 0:
                in_channels = filters[i - 1]
            setattr(
                self,
                f"conv2d_{i}",
                nn.Conv2d(
                    in_channels,
                    filters[i],
                    kernels[i],
                    stride=strides[i],
                    padding=0,
                ),
            )
            setattr(self, f"relu_{i}", nn.ReLU())

    def forward(self, inputs: Tensor) -> Tensor:
        net = inputs
        h, w = net.shape[-2:]
        for i in range(len(self.filters)):
            conv: nn.Conv2d = getattr(self, f"conv2d_{i}")
            k = _pair(conv.kernel_size)
            s = _pair(conv.stride)
            d = _pair(conv.dilation)
            pad = _same_pad(h, w, k, s, d)
            if any(pad):
                net = nn.functional.pad(net, pad)
            net = conv(net)

            net = getattr(self, f"relu_{i}")(net)
        out = net
        return out


class ResCnn(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        filters: list[int] = [128, 128, 128, 128],
        kernels: list[int] = [3, 3, 3, 3],
    ):
        super(ResCnn, self).__init__()

        self.filters = filters

        for i in range(len(filters)):
            if i > 0:
                in_channels = filters[i - 1]
            setattr(
                self,
                f"conv2d_{i}",
                nn.Conv2d(in_channels, filters[i], kernels[i], padding="same"),
            )
            setattr(self, f"relu_{i}", nn.ReLU())

    def forward(self, inputs: Tensor) -> Tensor:
        net = inputs
        for i in range(len(self.filters)):
            net = getattr(self, f"conv2d_{i}")(net)
            if i == 0:
                residual = net
            else:
                residual += net
            net = getattr(self, f"relu_{i}")(residual)
        out = net
        return out
