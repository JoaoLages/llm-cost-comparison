"""Pages module containing different calculator pages."""

from llm_cost_calculator.pages.always_on_hosting import always_on_hosting_page
from llm_cost_calculator.pages.per_request_pricing import per_request_pricing_page

__all__ = [
    "always_on_hosting_page",
    "per_request_pricing_page",
]
