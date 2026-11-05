"""The ratios shown on the Score page, grouped like an analyst reads a balance sheet.

Only 16 of the model's 95 inputs are editable; the rest stay at the chosen preset's values.
Values are the dataset's own 0-1 normalised ratios, not raw accounting numbers.
"""
import streamlit as st

from ui.components import load_presets

GROUPS = {
    "Leverage": [
        ("Debt ratio %", "Debt ratio", "Total liabilities / total assets. Net worth/Assets is set to 1 − this."),
        ("Borrowing dependency", "Borrowing dependency", "Reliance on borrowed funds."),
        ("Current Liability to Assets", "Current liabilities / assets", "Short-term obligations per unit of assets."),
        ("Liability to Equity", "Liabilities / equity", "Leverage relative to the equity cushion."),
    ],
    "Profitability": [
        ("ROA(A) before interest and % after tax", "Return on assets (A)", "≈0.5 is break-even in this data."),
        ("Net Income to Total Assets", "Net income / assets", ""),
        ("Retained Earnings to Total Assets", "Retained earnings / assets", "Accumulated profits (Altman X2)."),
        ("Operating Gross Margin", "Operating gross margin", ""),
    ],
    "Liquidity": [
        ("Current Ratio", "Current ratio", "Current assets / current liabilities (normalised)."),
        ("Quick Ratio", "Quick ratio", "Like the current ratio, excluding inventory."),
        ("Cash/Total Assets", "Cash / assets", ""),
        ("Working Capital to Total Assets", "Working capital / assets", "Altman X1."),
    ],
    "Cash flow": [
        ("CFO to Assets", "Operating cash flow / assets", ""),
        ("Cash Flow to Total Assets", "Cash flow / assets", ""),
        ("Cash Flow to Liability", "Cash flow / liabilities", ""),
        ("Cash flow rate", "Cash flow rate", ""),
    ],
}
LINKED = ("Net worth/Assets", "Debt ratio %")  # exact identity in the data: Net worth/Assets = 1 - Debt ratio %


def widget_key(col: str) -> str:
    return f"in::{col}"


def load_firm(label: str) -> None:
    """Replace the firm being edited with a preset and push its values into the input widgets."""
    firm = dict(load_presets()[label])
    st.session_state.firm, st.session_state.firm_label = firm, label
    for feats in GROUPS.values():
        for col, _, _ in feats:
            st.session_state[widget_key(col)] = float(firm[col])


def current_firm() -> dict:
    """The preset's 95 ratios with the edited ones applied and the linked ratio kept consistent."""
    firm = dict(st.session_state.firm)
    for feats in GROUPS.values():
        for col, _, _ in feats:
            firm[col] = float(st.session_state.get(widget_key(col), firm[col]))
    firm[LINKED[0]] = 1.0 - firm[LINKED[1]]
    return firm


def render_inputs() -> None:
    for group, feats in GROUPS.items():
        with st.expander(group, expanded=group == "Leverage"):
            for col, label, help_text in feats:
                key = widget_key(col)
                st.session_state.setdefault(key, float(st.session_state.firm[col]))
                st.number_input(label, key=key, min_value=0.0, step=0.001, format="%.4f", help=help_text or None)
