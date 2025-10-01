"""VRAM calculation utilities for LLM hosting."""


def calculate_kv_cache_vram(
    input_tokens: int,
    output_tokens: int,
    num_layers: int,
    hidden_dim: int,
    bytes_per_param: int,
    batch_size: int = 1
) -> float:
    """
    Calculate VRAM required for KV cache.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        num_layers: Number of transformer layers
        hidden_dim: Hidden dimension size
        bytes_per_param: Bytes per parameter (1 for FP8, 2 for FP16)
        batch_size: Batch size (default: 1)

    Returns:
        VRAM required in GB
    """
    # KV cache VRAM = batch_size * 2 * (input + output) * layers * hidden_dim * bytes_per_param / (1024^3)
    kv_cache_vram = (
        batch_size * 2 * (input_tokens + output_tokens) *
        num_layers * hidden_dim * bytes_per_param / (1024**3)
    )
    return kv_cache_vram


def calculate_min_vram(
    model_vram: float,
    input_tokens: int,
    output_tokens: int,
    num_layers: int,
    hidden_dim: int,
    bytes_per_param: int,
    batch_size: int = 1
) -> float:
    """
    Calculate minimum VRAM required for a given precision.

    Args:
        model_vram: Base model VRAM (GB)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        num_layers: Number of transformer layers
        hidden_dim: Hidden dimension size
        bytes_per_param: Bytes per parameter (1 for FP8, 2 for FP16)
        batch_size: Batch size (default: 1)

    Returns:
        Minimum VRAM required in GB
    """
    kv_cache = calculate_kv_cache_vram(
        input_tokens, output_tokens, num_layers, hidden_dim,
        bytes_per_param, batch_size
    )
    return model_vram + kv_cache


def calculate_min_vram_required(
    model_vram_fp8: float,
    model_vram_fp16: float,
    input_tokens: int,
    output_tokens: int,
    num_layers: int,
    hidden_dim: int,
    batch_size: int = 1
) -> tuple[float, float]:
    """
    Calculate minimum VRAM required for both FP8 and FP16.

    Args:
        model_vram_fp8: Base model VRAM in FP8 (GB)
        model_vram_fp16: Base model VRAM in FP16 (GB)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        num_layers: Number of transformer layers
        hidden_dim: Hidden dimension size
        batch_size: Batch size (default: 1)

    Returns:
        Tuple of (min_vram_fp8, min_vram_fp16) in GB
    """
    min_vram_fp8 = calculate_min_vram(
        model_vram_fp8, input_tokens, output_tokens,
        num_layers, hidden_dim, bytes_per_param=1, batch_size=batch_size
    )
    min_vram_fp16 = calculate_min_vram(
        model_vram_fp16, input_tokens, output_tokens,
        num_layers, hidden_dim, bytes_per_param=2, batch_size=batch_size
    )

    return min_vram_fp8, min_vram_fp16
