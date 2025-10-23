# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose

from .config import LoraConfig
from .dora import DoraConv2dLayer, DoraLinearLayer


class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("lora_ga"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.lora_ga_init(adapter_name)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("lora_ns"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.lora_ns_init(adapter_name)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def lora_ga_init(self, adapter_name):
        def get_float_weight(model: torch.nn.Module):
            model: torch.nn.Linear

            device = model.weight.device
            in_features = model.in_features
            with torch.no_grad():
                I = torch.eye(in_features).to(device)
                w = model(I)
                if hasattr(model, "bias") and isinstance(model.bias, torch.Tensor):
                    w -= model.bias
                w = torch.transpose(w, 0, 1)
            w.requires_grad = model.weight.requires_grad
            return w
        
        if "grad" not in self.kwargs.keys():
            return

        base_layer = self.get_base_layer()
        weight = self.get_base_layer().weight
        device = weight.device
        dtype = weight.dtype
        quant_flag = False
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            """
            for quantized model, it is needed to get the floating point weights through forward, 
            which may take 1-2 minutes (7bmodel, all linear)
            """
            quant_flag = True
            weight = get_float_weight(base_layer)
            dtype = weight.dtype
        grad = self.kwargs["grad"].to(device).to(torch.float32)
        weight = weight.to(torch.float32)
        lora_r = self.r[adapter_name]
        init_config = self.kwargs["peft_config"]
        try:
            U, S, V = torch.svd_lowrank(
                grad.float(), q=min(4 * lora_r, min(grad.shape)),
                niter=4
            )
            V = V.T
        except Exception as e:
            raise ValueError("error from torch.svd_lowrank")
        # set direction
        if init_config.direction == "ArBr":
            B = U[:, 0: 2 * lora_r: 2]
            A = V[1: 2 * lora_r: 2, :]
        elif init_config.direction == "A2rBr":
            B = U[:, :lora_r]
            A = V[lora_r: 2 * lora_r, :]
        elif init_config.direction == "ArB2r":
            B = U[:, lora_r: 2 * lora_r]
            A = V[:lora_r, :]
        elif init_config.direction == "random":
            import random
            random_list = random.sample(range(2 * lora_r), 2 * lora_r)
            indexes_A = random_list[0:lora_r]
            indexes_B = random_list[lora_r:2 * lora_r]
            print(f"indexes_A={indexes_A}")
            print(f"indexes_B={indexes_B}")
            B = U[:, indexes_B]
            A = V[indexes_A, :]
        scaling_factor = self.scaling["default"]
        if init_config.scale == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config.scale == "unit":
            # Because A,B is orthogonal, do not need to scale
            pass
        elif init_config.scale == "stable":
            m, n = grad.shape  # m: feature_out, n: feature_in
            # the scale of output is only related to the feature_out
            gamma = init_config.stable_gamma
            B = B * m ** 0.25 / gamma ** 0.5
            A = A * m ** 0.25 / gamma ** 0.5
        elif init_config.scale == "weightS":
            _, S, _ = torch.svd_lowrank(weight.data.float(), q=4 * lora_r, niter=4)
            S = S / self.scaling["default"]
            avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s

        offset = B @ A
        # Training type
        # consider dtype not in init_config
        if not hasattr(init_config, "dtype"):
            pass
        elif init_config.dtype == "bf16":
            A = A.to(torch.bfloat16)
            B = B.to(torch.bfloat16)
        elif init_config.dtype == "fp32":
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        scaling_factor = self.scaling["default"]
        offset *= scaling_factor
        if hasattr(init_config, "norm_clip") and init_config.norm_clip:
            # for numerical stability, offset's largest value must be less then weight's largest value
            ratio = torch.max(torch.abs(weight.data)) / torch.max(
                torch.abs(offset)
            )
            if ratio < 1:
                offset *= ratio
                A *= ratio ** 0.5
                B *= ratio ** 0.5

        weight.data -= offset

        self.lora_A[adapter_name].weight.data = A.contiguous()
        self.lora_B[adapter_name].weight.data = B.contiguous()
        if not quant_flag:
            weight = weight.data
            weight = weight.to(dtype)
            self.get_base_layer().weight.data = weight
        else:
            has_bias = True if base_layer.bias is not None else False
            float_linear = torch.nn.Linear(base_layer.in_features, base_layer.out_features, has_bias)
            if has_bias and isinstance(base_layer.bias.data, torch.Tensor):
                float_linear.bias.data = base_layer.bias.data
            float_linear.weight.data = weight.data
            import bitsandbytes
            if isinstance(base_layer, bitsandbytes.nn.Linear8bitLt):
                new_base_layer = type(base_layer)(base_layer.in_features, base_layer.out_features, has_bias,
                                                  has_fp16_weights=False)
            else:
                new_base_layer = type(base_layer)(base_layer.in_features, base_layer.out_features, has_bias, )
            new_base_layer.load_state_dict(float_linear.state_dict())
            new_base_layer.to(device)
            base_layer.__dict__.update(new_base_layer.__dict__)
            del new_base_layer

    def lora_ns_init(self, adapter_name):
        """
        Direct-rank NS-based LoRA initialization (no 2r 'direction' split).
        Constructs B ∈ R^{d_out×r}, A ∈ R^{r×d_in} directly from the NS polar factor
        Q ≈ G (G^T G)^(-1/2), using a randomized range finder with target rank r.

        Steps:
        1) Obtain grad G = dL/dW (from a single calibration backward).
        2) Q = ns_polar(G)  (no SVD on G).
        3) Randomized range finder on Q to get V_r (right basis, d_in×r),
            then U_r = Q @ V_r (d_out×r).
        4) B_init = scaled U_r; A_init = scaled V_r^T.
        5) Freeze-compensation: W ← W - offset, where offset = B_init @ A_init
            (then LoRA forward adds it back, keeping initial forward invariant).

        Notes:
        - For Conv2d, reshape to 2D before calling this routine (consistent with LoRA-GA usage).
        - Quantized base layers are handled by materializing float weights (as in LoRA-GA code).
        """

        # ---------- helpers ----------
        def get_float_weight(model: torch.nn.Module):
            # Materialize float weight for quantized linear via forward(I) and removing bias
            model: torch.nn.Linear
            device = model.weight.device
            in_features = model.in_features
            with torch.no_grad():
                I = torch.eye(in_features, device=device, dtype=torch.float32)
                w = model(I)  # [out, in]
                if hasattr(model, "bias") and isinstance(model.bias, torch.Tensor):
                    w = w - model.bias
                w = w.transpose(0, 1).contiguous()  # [in, out] -> we will keep weight as [d_out, d_in] below
            w.requires_grad = model.weight.requires_grad
            return w

        @torch.no_grad()
        def ns_polar(G: torch.Tensor, steps: int = 5, eps: float = 1e-6) -> torch.Tensor:
            """
            Approximate polar factor Q ≈ G (G^T G)^(-1/2) via Newton–Schulz iteration.
            Returns Q with Q^T Q ≈ I (direction-only; singular values flattened to ~1).
            """
            dtype = torch.float32
            G = G.to(dtype)
            fro = torch.linalg.norm(G, ord='fro') + eps
            X = G / fro

            # Improve conditioning for tall matrices by transposing during iteration
            transposed = X.size(0) > X.size(1)
            if transposed:
                X = X.t()

            I = torch.eye(X.size(1), device=X.device, dtype=X.dtype)
            for _ in range(steps):
                XtX = X.t() @ X
                X = 0.5 * X @ (3.0 * I - XtX)

            if transposed:
                X = X.t()
            return X  # [d_out, d_in], approximately orthonormal columns (or rows if tall)

        @torch.no_grad()
        def randomized_right_basis(Q: torch.Tensor, r: int, oversample: int = 4):
            """
            Return V_r ∈ R^{d_in×r} as an orthonormal right basis of rank r from Q (no SVD on G).
            Steps:
            1) Y = Q^T @ Ω with Ω ∈ R^{d_out×(r+p)}
            2) Ṽ = orth(Y) via QR
            3) Truncate to r columns: V_r = Ṽ[:, :r]
            Then U_r can be formed as U_r = Q @ V_r.
            """
            d_out, d_in = Q.shape
            k = min(r + oversample, min(d_out, d_in))
            Omega = torch.randn(d_out, k, device=Q.device, dtype=Q.dtype)
            Y = Q.t() @ Omega                   # [d_in, k]
            V_tilde, _ = torch.linalg.qr(Y, mode='reduced')  # [d_in, k]
            return V_tilde[:, :r]               # [d_in, r]

        # ---------- main ----------
        if "grad" not in self.kwargs.keys():
            return
        base_layer = self.get_base_layer()
        weight_param = base_layer.weight
        device = weight_param.device
        orig_dtype = weight_param.dtype

        # Quantized handling: materialize float weights if needed (mirrors LoRA-GA code path)
        quant_flag = False
        if orig_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            quant_flag = True
            weight = get_float_weight(base_layer).to(torch.float32)
        else:
            weight = weight_param.data.to(torch.float32)

        # Gradient from calibration backward
        G = self.kwargs["grad"].to(device).to(torch.float32)  # [d_out, d_in]
        if G is None:
            return

        r = self.r[adapter_name]
        init_config = self.kwargs["peft_config"]

        # NS / randomized params (defaults if not provided)
        ns_steps = getattr(init_config, "ns_steps", 5)
        ns_eps = getattr(init_config, "ns_eps", 1e-6)
        oversample = getattr(init_config, "oversample", 4)

        # 1) Q = ns_polar(G)  (no SVD on G)
        Q = ns_polar(G, steps=ns_steps, eps=ns_eps)  # [d_out, d_in]

        # 2) Direct-rank extraction: build V_r (right basis), then U_r = Q @ V_r
        V_r = randomized_right_basis(Q, r=r, oversample=oversample)   # [d_in, r]
        U_r = Q @ V_r                                                # [d_out, r]

        # 3) Form A,B directly (no 2r 'direction' split)
        B = U_r.contiguous()                 # [d_out, r]
        A = V_r.t().contiguous()             # [r, d_in]

        # 4) Scaling strategies (reuse LoRA-GA naming for compatibility)
        scaling_factor = self.scaling["default"]  # same global multiplier used in LoRA-GA
        scale_mode = getattr(init_config, "scale", "stable")  # {'gd','unit','stable','weightS'}

        if scale_mode == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif scale_mode == "unit":
            # Q is approx orthonormal; keep unit-like columns
            pass
        elif scale_mode == "stable":
            # Forward-scale stability heuristic depending only on d_out (mirrors LoRA-GA spirit)
            m, n = G.shape
            gamma = getattr(init_config, "stable_gamma", 1.0)
            coef = (m ** 0.25) / (gamma ** 0.5 + 1e-12)
            B = B * coef
            A = A * coef
        elif scale_mode == "weightS":
            # Optional mild scaling using weight spectrum (small-rank SVD on weight only; not on G)
            try:
                _, S_w, _ = torch.svd_lowrank(weight.float(), q=min(4 * r, min(weight.shape)), niter=2)
                S_w = S_w / scaling_factor
                avg_s = torch.sqrt(S_w[:r]).mean().to(A.device)
                B = B * avg_s
                A = A * avg_s
            except Exception:
                pass

        # 5) Build offset and optional elementwise clipping
        offset = (B @ A)  # [d_out, d_in]
        if getattr(init_config, "norm_clip", False):
            max_w = torch.max(torch.abs(weight))
            max_off = torch.max(torch.abs(offset))
            ratio = max_w / (max_off + 1e-12)
            if ratio < 1:
                offset = offset * ratio
                A = A * (ratio ** 0.5)
                B = B * (ratio ** 0.5)

        # Apply global scaling factor used in your codebase (keeps compatibility with LoRA-GA)
        offset = offset * scaling_factor

        # Dtype for adapters
        if hasattr(init_config, "dtype"):
            if init_config.dtype == "bf16":
                A = A.to(torch.bfloat16)
                B = B.to(torch.bfloat16)
            elif init_config.dtype == "fp32":
                A = A.to(torch.float32)
                B = B.to(torch.float32)

        # Freeze-compensation: keep initial forward invariant
        weight = weight - offset  # fp32 buffer

        # Write back LoRA params
        self.lora_A[adapter_name].weight.data = A.contiguous()
        self.lora_B[adapter_name].weight.data = B.contiguous()

        # Write back base weight (quantized vs float)
        if not quant_flag:
            self.get_base_layer().weight.data = weight.to(orig_dtype).contiguous()
        else:
            base = base_layer
            has_bias = (base.bias is not None and isinstance(base.bias.data, torch.Tensor))
            float_linear = nn.Linear(base.in_features, base.out_features, has_bias)
            if has_bias:
                float_linear.bias.data = base.bias.data
            float_linear.weight.data = weight.contiguous()

            try:
                import bitsandbytes
                if isinstance(base, bitsandbytes.nn.Linear8bitLt):
                    new_base = type(base)(base.in_features, base.out_features, has_bias, has_fp16_weights=False)
                else:
                    new_base = type(base)(base.in_features, base.out_features, has_bias)
            except Exception:
                new_base = type(base)(base.in_features, base.out_features, has_bias)

            new_base.load_state_dict(float_linear.state_dict())
            new_base.to(device)
            base.__dict__.update(new_base.__dict__)
            del new_base    


    def lora_ns_init_backup(self, adapter_name):
        """
        NS-based LoRA initialization:
        1) Use a single calibration backward to read grad G of base weight.
        2) Compute polar factor Q ≈ G (G^T G)^(-1/2) via Newton–Schulz (no SVD).
        3) Build a rank-2r right/left orthonormal basis from Q using randomized range finder + QR.
        4) Choose r columns for B and r rows for A according to init_config.direction (same API as LoRA-GA).
        5) Apply scaling (gd/unit/stable/weightS compatible choices) and freeze-compensation W -= offset.
        Notes:
        - Convs should already be reshaped by caller if needed (consistent with your LoRA codebase).
        - Quantized weights: mirrors the LoRA-GA path by materializing float weights when necessary.
        """

        # ---------- helpers ----------
        def get_float_weight(model: torch.nn.Module):
            model: torch.nn.Linear
            device = model.weight.device
            in_features = model.in_features
            with torch.no_grad():
                I = torch.eye(in_features, device=device, dtype=torch.float32)
                w = model(I)  # [out, in]
                if hasattr(model, "bias") and isinstance(model.bias, torch.Tensor):
                    w = w - model.bias  # remove bias effect
                w = w.transpose(0, 1).contiguous()  # [in, out] -> expect [out, in] later; we keep consistency below
            w.requires_grad = model.weight.requires_grad
            return w

        @torch.no_grad()
        def ns_polar(G: torch.Tensor, steps: int = 5, eps: float = 1e-6) -> torch.Tensor:
            """
            Compute polar factor Q ≈ G (G^T G)^(-1/2) via classic Newton–Schulz iteration.
            Returns Q with Q^T Q ≈ I (direction only; singular values flattened).
            """
            dtype = torch.float32  # accumulate in fp32 for stability
            G = G.to(dtype)
            fro = torch.linalg.norm(G, ord='fro') + eps
            X = G / fro

            # If tall matrix, transpose for better conditioning, then transpose back
            transposed = X.size(0) > X.size(1)
            if transposed:
                X = X.t()

            I = torch.eye(X.size(1), device=X.device, dtype=X.dtype)
            for _ in range(steps):
                XtX = X.t() @ X
                X = 0.5 * X @ (3.0 * I - XtX)

            if transposed:
                X = X.t()
            return X  # approximately orthogonal

        @torch.no_grad()
        def randomized_right_left_basis(Q: torch.Tensor, k: int, oversample: int = 4):
            """
            Build right/left orthonormal bases (V_tilde in R^{d_in x k}, U_tilde in R^{d_out x k})
            from Q using randomized range finder WITHOUT SVD on G:
            1) sample Omega in R^{d_out x (k+p)}
            2) Y = Q^T @ Omega in R^{d_in x (k+p)}
            3) V_tilde = orth(Y), U_tilde = Q @ V_tilde
            """
            d_out, d_in = Q.shape
            k_eff = min(k + oversample, min(d_out, d_in))
            Omega = torch.randn(d_out, k_eff, device=Q.device, dtype=Q.dtype)
            Y = Q.t() @ Omega                              # [d_in, k_eff]
            V_tilde, _ = torch.linalg.qr(Y, mode='reduced')  # [d_in, k_eff]
            U_tilde = (Q @ V_tilde)                        # [d_out, k_eff]
            return U_tilde[:, :k], V_tilde[:, :k]          # [d_out,k], [d_in,k]

        # ---------- main ----------
        if "grad" not in self.kwargs:
            return

        base_layer = self.get_base_layer()
        weight = base_layer.weight
        device = weight.device
        orig_dtype = weight.dtype

        # Handle quantized weights in the same way as LoRA-GA
        quant_flag = False
        if orig_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            quant_flag = True
            weight = get_float_weight(base_layer).to(torch.float32)
        else:
            weight = weight.data.to(torch.float32)

        grad = self.kwargs["grad"].to(device).to(torch.float32)  # [d_out, d_in]
        if grad is None:
            return

        lora_r = self.r[adapter_name]
        init_config = self.kwargs["peft_config"]

        # Parameters for NS and randomized basis (with sane defaults if not provided)
        ns_steps = getattr(init_config, "ns_steps", 5)
        ns_eps = getattr(init_config, "ns_eps", 1e-6)
        oversample = getattr(init_config, "oversample", 4)

        # 1) Polar factor (no SVD on G)
        Q = ns_polar(grad, steps=ns_steps, eps=ns_eps)  # [d_out, d_in], Q^T Q ≈ I

        # 2) Build a 2r subspace basis to emulate LoRA-GA "direction" options
        two_r = min(2 * lora_r, min(Q.shape))
        U_basis, V_basis = randomized_right_left_basis(Q, k=two_r, oversample=oversample)  # [d_out,2r], [d_in,2r]

        # 3) Direction selection (mirror LoRA-GA API)
        direction = getattr(init_config, "direction", "A2rBr")
        if direction == "ArBr":
            # B uses even indices, A uses odd indices
            B = U_basis[:, 0:2 * lora_r:2]                    # [d_out, r]
            A = V_basis[:, 1:2 * lora_r:2].t().contiguous()   # [r, d_in]
        elif direction == "A2rBr":
            B = U_basis[:, :lora_r]                           # [d_out, r]
            A = V_basis[:, lora_r:2 * lora_r].t().contiguous()# [r, d_in]
        elif direction == "ArB2r":
            B = U_basis[:, lora_r:2 * lora_r]                 # [d_out, r]
            A = V_basis[:, :lora_r].t().contiguous()          # [r, d_in]
        elif direction == "random":
            import random
            idx = list(range(two_r))
            random.shuffle(idx)
            idx_A = idx[:lora_r]
            idx_B = idx[lora_r:2 * lora_r]
            B = U_basis[:, idx_B]
            A = V_basis[:, idx_A].t().contiguous()
        else:
            # default fallback
            B = U_basis[:, :lora_r]
            A = V_basis[:, lora_r:2 * lora_r].t().contiguous()

        # 4) Scaling strategies (compatible names with LoRA-GA)
        scaling_factor = self.scaling["default"]  # same meaning as in LoRA-GA code
        scale_mode = getattr(init_config, "scale", "stable")  # {gd, unit, stable, weightS}

        if scale_mode == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif scale_mode == "unit":
            # Q is approximately orthogonal, A/B already have unit-like columns
            pass
        elif scale_mode == "stable":
            # Forward-scale stable (depends only on d_out), mirroring LoRA-GA's rule of thumb
            m, n = grad.shape  # m=d_out, n=d_in
            gamma = getattr(init_config, "stable_gamma", 1.0)
            coef = (m ** 0.25) / (gamma ** 0.5 + 1e-12)
            B = B * coef
            A = A * coef
        elif scale_mode == "weightS":
            # Use weight spectrum as a mild scale reference (low-rank approx without heavy SVD)
            # Here we avoid SVD on G, but for weight we allow a small-rank SVD like original if desired.
            try:
                # Note: small-rank SVD on weight for scaling only (q=4r like LoRA-GA)
                _, S_w, _ = torch.svd_lowrank(weight.float(), q=min(4 * lora_r, min(weight.shape)), niter=2)
                S_w = S_w / scaling_factor
                avg_s = torch.sqrt(S_w[:lora_r]).mean().to(A.device)
                B = B * avg_s
                A = A * avg_s
            except Exception:
                pass  # fall back silently

        # 5) Offset and dtype handling
        offset = (B @ A)  # [d_out, d_in]
        # Optional norm-clip (same logic as LoRA-GA)
        if getattr(init_config, "norm_clip", False):
            max_w = torch.max(torch.abs(weight))
            max_off = torch.max(torch.abs(offset))
            ratio = (max_w / (max_off + 1e-12))
            if ratio < 1:
                offset = offset * ratio
                A = A * (ratio ** 0.5)
                B = B * (ratio ** 0.5)

        # Apply global scaling factor used in your codebase
        offset = offset * scaling_factor

        # Dtype cast for adapters
        if hasattr(init_config, "dtype"):
            if init_config.dtype == "bf16":
                A = A.to(torch.bfloat16)
                B = B.to(torch.bfloat16)
            elif init_config.dtype == "fp32":
                A = A.to(torch.float32)
                B = B.to(torch.float32)

        # Freeze-compensation: adjust base weight to keep initial forward unchanged
        weight = weight - offset  # fp32 buffer

        # Write back to module and LoRA params
        self.lora_A[adapter_name].weight.data = A.contiguous()
        self.lora_B[adapter_name].weight.data = B.contiguous()

        if not quant_flag:
            # Cast back to original dtype and store
            self.get_base_layer().weight.data = weight.to(orig_dtype).contiguous()
        else:
            # Replace quantized base layer with a new one holding the updated float weights, mirroring LoRA-GA flow
            base = base_layer
            has_bias = (base.bias is not None and isinstance(base.bias.data, torch.Tensor))
            float_linear = nn.Linear(base.in_features, base.out_features, has_bias)
            if has_bias:
                float_linear.bias.data = base.bias.data
            float_linear.weight.data = weight.contiguous()

            try:
                import bitsandbytes
                if isinstance(base, bitsandbytes.nn.Linear8bitLt):
                    new_base = type(base)(base.in_features, base.out_features, has_bias, has_fp16_weights=False)
                else:
                    new_base = type(base)(base.in_features, base.out_features, has_bias)
            except Exception:
                new_base = type(base)(base.in_features, base.out_features, has_bias)

            new_base.load_state_dict(float_linear.state_dict())
            new_base.to(device)
            base.__dict__.update(new_base.__dict__)
            del new_base

    def olora_init(self, adapter_name):
        dtype = self.get_base_layer().weight.dtype
        if dtype in [torch.int8, torch.uint8]:
            weight_tensor = dequantize_module_weight(self.get_base_layer())
        elif dtype in [torch.float32, torch.float16, torch.bfloat16]:
            weight_tensor = self.get_base_layer().weight
        else:
            raise TypeError(f"Unsupported data type for the base layer. Got {dtype}.")

        scale_factor = self.scaling[adapter_name]
        r = self.r[adapter_name]
        weight_tensor = weight_tensor.to(torch.float32)
        Q, R = torch.linalg.qr(weight_tensor.data)

        Qr, Rr = Q[:, :r], R[:r]

        self.lora_A[adapter_name].weight.data = Rr.contiguous()
        self.lora_B[adapter_name].weight.data = Qr.contiguous()

        weight_tensor.data -= scale_factor * self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight
        weight_tensor = weight_tensor.to(dtype)
        self.get_base_layer().weight.data = weight_tensor

    def pissa_init(self, adapter_name, init_lora_weights):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        if init_lora_weights == "pissa":
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vr = V[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[: self.r[adapter_name]]
        elif len(init_lora_weights.split("_niter_")) == 2:
            Vr, Sr, Ur = svd_lowrank(
                weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
            )
            Sr /= self.scaling[adapter_name]
            Uhr = Ur.t()
        else:
            raise ValueError(
                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
            )

        lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
        lora_B = Vr @ torch.diag(torch.sqrt(Sr))
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

    def loftq_init(self, adapter_name):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def dora_init(self, adapter_name: str) -> None:
        if not self.lora_magnitude_vector:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(self, "fan_in_fan_out", False))
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        place_on_cpu = self.ephemeral_gpu_offload and (lora_A.device.type == "cpu" or lora_B.device.type == "cpu")
        if self.ephemeral_gpu_offload:
            if lora_A.device.type == "cuda":
                lora_B = lora_B.to(lora_A.device)
            else:
                if lora_B.device.type != "cuda":
                    lora_B = lora_B.to("cuda")
                lora_A = lora_A.to(lora_B.device)
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling, place_on_cpu=place_on_cpu
        )
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        unique_adapters = set(self.active_adapters)
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Embedding(nn.Module, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights + self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = base_layer.weight.data + self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result = result + (after_A @ embedding_B) * scaling
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(nn.Module, LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def dora_init(self, adapter_name: str) -> None:
        if self.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraConv2dLayer(fan_in_fan_out=False)
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling)
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        orig_weights = dora_factor.view(-1, 1, 1, 1) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(base_layer.weight, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        new_weight = dora_factor.view(-1, 1, 1, 1) * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1, 1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                F.conv2d(
                    weight_A.permute(1, 0, 2, 3),
                    weight_B,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
