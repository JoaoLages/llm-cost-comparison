# LLM Cost Calculator - Code Structure

This document describes the refactored code structure of the LLM Cost Calculator.

## Module Organization

The codebase has been refactored following best practices with clear separation of concerns:

### Core Modules

#### `main.py`
- **Purpose**: Application entry point
- **Responsibilities**:
  - Streamlit app configuration
  - Page routing
  - Application orchestration

#### `pages.py`
- **Purpose**: Streamlit page implementations
- **Contains**:
  - `always_on_hosting_page()`: Always-on hosting cost comparison
  - `per_request_pricing_page()`: Per-request pricing comparison
- **Responsibilities**: UI rendering and user interaction

#### `cost_calculator.py`
- **Purpose**: Cost calculation logic
- **Contains**:
  - `calculate_paid_api_costs()`: Calculate costs for paid APIs
  - `calculate_opensource_costs()`: Calculate costs for open-source models
  - `find_optimal_gpu_config()`: Find best GPU configuration for a model
- **Responsibilities**: Business logic for cost calculations

#### `vram_calculator.py`
- **Purpose**: VRAM requirement calculations
- **Contains**:
  - `calculate_kv_cache_vram()`: Calculate KV cache memory requirements
  - `calculate_min_vram_required()`: Calculate VRAM for FP8 and FP16
- **Responsibilities**: Memory requirement calculations for LLM inference

#### `throughput_estimator.py`
- **Purpose**: Performance estimation utilities
- **Contains**:
  - `estimate_tokens_per_sec()`: Estimate model throughput
  - `calculate_execution_time()`: Calculate execution time for batch processing
- **Responsibilities**: Performance modeling and time estimation

#### `data_loader.py`
- **Purpose**: Data loading and preprocessing
- **Contains**:
  - `load_spreadsheet()`: Load Excel data with caching
  - `prepare_performance_scores()`: Process LMArena scores
  - `prepare_pricing_dataframe()`: Prepare pricing data
- **Responsibilities**: Data I/O and preprocessing

## Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **DRY (Don't Repeat Yourself)**: Common calculations are extracted into reusable functions
3. **Type Hints**: All functions include type hints for better code documentation
4. **Docstrings**: Comprehensive docstrings following Google style
5. **Testability**: Pure functions with minimal side effects make testing easier
6. **Caching**: Expensive operations (data loading) are cached using Streamlit's `@st.cache_data`

## Function Flow

### Always-on Hosting Flow
```
main.py
  └─> pages.always_on_hosting_page()
       ├─> vram_calculator.calculate_min_vram_required()
       └─> data_loader.prepare_pricing_dataframe()
```

### Per-Request Pricing Flow
```
main.py
  └─> pages.per_request_pricing_page()
       ├─> data_loader.prepare_performance_scores()
       ├─> data_loader.prepare_pricing_dataframe()
       ├─> cost_calculator.calculate_paid_api_costs()
       └─> cost_calculator.calculate_opensource_costs()
            ├─> cost_calculator.find_optimal_gpu_config()
            ├─> vram_calculator.calculate_min_vram_fp8()
            └─> throughput_estimator.calculate_execution_time()
```

## Key Improvements

1. **Eliminated Code Duplication**: VRAM calculation logic is now centralized
2. **Better Maintainability**: Changes to calculation logic only need to be made in one place
3. **Improved Readability**: Each function has a clear purpose and is properly documented
4. **Easier Testing**: Pure functions can be tested independently
5. **Scalability**: New features can be added by extending existing modules or adding new ones

## Usage

The application can be run using:
```bash
streamlit run llm_cost_calculator/main.py
```

Or imported as a package:

```python
from llm_cost_calculator.vram_calculator import calculate_min_vram_required
from llm_cost_calculator.cost_calculator import calculate_paid_api_costs
```
