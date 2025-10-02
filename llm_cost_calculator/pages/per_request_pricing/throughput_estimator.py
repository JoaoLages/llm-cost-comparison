"""Throughput estimation utilities for LLM inference."""


def estimate_tokens_per_sec(num_params: float) -> float:
    """
    Estimate throughput in tokens per second based on model size.

    Based on throughput benchmarks:
    - Small models (7B params): ~50-100 tokens/sec
    - Medium models (13-70B params): ~20-50 tokens/sec
    - Large models (70B+ params): ~5-20 tokens/sec

    Args:
        num_params: Number of parameters in billions

    Returns:
        Estimated tokens per second
    """
    if num_params < 10:  # Small models (7B)
        return 75  # midpoint of 50-100
    elif num_params < 70:  # Medium models (13-70B)
        return 35  # midpoint of 20-50
    else:  # Large models (70B+)
        return 12  # midpoint of 5-20


def calculate_execution_time(
    num_requests: int,
    input_tokens: int,
    output_tokens: int,
    tokens_per_sec: float,
    batch_size: int = 1,
    conservative_buffer_hours: float = 1.0
) -> float:
    """
    Calculate execution time in hours for processing requests.

    Args:
        num_requests: Total number of requests
        input_tokens: Input tokens per request
        output_tokens: Output tokens per request
        tokens_per_sec: Model throughput in tokens/sec
        batch_size: Batch size for parallel processing
        conservative_buffer_hours: Additional buffer time in hours

    Returns:
        Total execution time in hours
    """
    # Calculate number of batches needed
    num_batches = (num_requests + batch_size - 1) // batch_size  # Ceiling division

    # Time per batch = (input_tokens + output_tokens) / tokens_per_sec
    time_per_batch_seconds = (input_tokens + output_tokens) / tokens_per_sec
    total_execution_seconds = num_batches * time_per_batch_seconds
    total_execution_hours = total_execution_seconds / 3600

    # Add conservative buffer
    return total_execution_hours + conservative_buffer_hours
