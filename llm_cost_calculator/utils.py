def get_pricing_policies():
    return (
        "**Pricing policy differences:**\n"
        "- **On-Demand**: Pay only for actual usage, can suspend anytime\n"
        "- **Spot**: Cheaper but bills in full-hour increments (rounded up) and be terminated by provider with only 2-minutes notice\n"
        "- **Subscription**: Pay for full month regardless of usage"
    )