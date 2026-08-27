"""Multi-agent trading council for futures markets.

Structure follows TradingAgents (Xiao et al., arXiv:2412.20138): specialist analysts
feed a bull-vs-bear debate, a trader forms a plan, and a risk manager delivers the
final verdict. That decomposition is the genuinely useful idea in that paper -- an
adversarial debate surfaces the counter-case instead of letting one pass of
reasoning rationalize a hunch.

Rebuilt here rather than used off the shelf, for reasons established by actually
running theirs first (see brief.py's docstring for the diagnosed failure):

1. Their pipeline hands analysts raw CSVs and asks the model to do arithmetic. With
   a local 8B model this produced a pandas tutorial in place of a technical
   analysis, and downstream agents -- left with no price anchor -- hallucinated
   levels 15% from the real price. Here, all numerical work happens in verified
   Python (brief.py) and the model receives finished figures in prose.
2. Their tooling is equity-shaped: `get_balance_sheet` / `get_income_statement` /
   StockTwits / insider transactions are meaningless or 404 for `CL=F`, and their
   FRED path needs an API key this project doesn't use (macro.py reads the keyless
   public CSV endpoint). Roughly half their analyst roster contributes nothing on
   futures.
3. This project already has real, point-in-time-correct macro, fundamentals, and
   CFTC positioning data that their framework has no slot for at all.

The result is that the council reasons over the SAME information set as Models 4-9,
which makes "does an LLM council beat the trained models?" an honest comparison
rather than a confounded one.

Every agent is instructed to quote only prices from the brief. The output is still
verified programmatically afterward (paper_trade_agent.py's price-anchor guard) --
instructions reduce hallucination, they do not prove its absence, and this project
does not trust an LLM's self-report on that.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = os.environ.get("COUNCIL_MODEL", "qwen3:8b")
DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

_GROUNDING = (
    "GROUNDING RULES (violating these makes your analysis worthless):\n"
    "1. The brief contains the only correct current price. Every level you name must come from it "
    "or be computed from it. Never recall a price from memory.\n"
    "2. Do not invent news, data, or events that are not in the brief.\n"
    "3. If the evidence is genuinely mixed, say so. A confident wrong call is worse than an honest 'unclear'.\n"
    "4. Be concise and specific. Cite the actual numbers you are reasoning from.\n"
)

ROLES = {
    "technical": (
        "You are a futures technical analyst. Analyse trend, momentum, volatility and position "
        "within the recent range. Weigh the normalized trend scores across the three horizons and note "
        "whether they agree or conflict. State what the price action alone implies: bullish, bearish, or unclear."
    ),
    "macro": (
        "You are a macro/fundamental analyst for commodity and rates futures. Interpret the supply, demand, "
        "inventory, production and dollar-index figures in the brief. Explain the causal mechanism linking them "
        "to this contract's price, and state whether fundamentals lean bullish, bearish, or unclear."
    ),
    "positioning": (
        "You are a positioning analyst specialising in CFTC Commitment of Traders data. Speculators trend-follow "
        "and are most often wrong at extremes; commercial hedgers are the better-informed cohort. Percentile "
        "readings above 90 or below 10 signal crowding and unwind risk. Say whether positioning is a contrarian "
        "warning, a confirmation, or neutral -- and note explicitly if it is mid-range and therefore uninformative."
    ),
    "news": (
        "You are a news analyst for futures markets. Identify only the headlines genuinely relevant to this "
        "contract's supply, demand, or macro drivers, and ignore unrelated equity/company noise. Distinguish "
        "what is already priced in from what is new. If the headlines are not materially informative, say so "
        "plainly rather than manufacturing a narrative."
    ),
}


@dataclass
class CouncilResult:
    ticker: str
    price: float
    action: str
    confidence: str
    reasoning: str
    analyst_reports: dict = field(default_factory=dict)
    bull_case: str = ""
    bear_case: str = ""
    risk_review: str = ""
    raw_final: str = ""


class TradingCouncil:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, temperature: float = 0.2, max_tokens: int = 700):
        self.llm = ChatOpenAI(
            model=model, base_url=base_url, api_key="ollama",
            temperature=temperature, max_tokens=max_tokens, timeout=600,
        )
        self.model = model

    def _ask(self, system: str, user: str) -> str:
        resp = self.llm.invoke([SystemMessage(content=system + "\n\n" + _GROUNDING), HumanMessage(content=user)])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        # Qwen-family models emit chain-of-thought in <think> blocks; strip so the
        # stored reasoning is the actual argument, not the scratchpad.
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def run(self, brief: dict, verbose: bool = True) -> CouncilResult:
        text, price, ticker = brief["text"], brief["price"], brief["facts"]["ticker"]

        reports = {}
        for role, system in ROLES.items():
            if verbose:
                print(f"  [{role} analyst] thinking...")
            reports[role] = self._ask(system, f"{text}\n\nGive your {role} assessment (max 150 words).")

        analyst_block = "\n\n".join(f"### {r.upper()} ANALYST\n{v}" for r, v in reports.items())

        if verbose:
            print("  [bull researcher] building the case to buy...")
        bull = self._ask(
            "You are a bull researcher. Make the strongest evidence-based case for a LONG position. "
            "Use only facts from the brief and the analyst reports. If the bullish case is genuinely weak, admit it.",
            f"{text}\n\n{analyst_block}\n\nMake the bull case (max 180 words).",
        )

        if verbose:
            print("  [bear researcher] rebutting...")
        bear = self._ask(
            "You are a bear researcher. Make the strongest evidence-based case for a SHORT position and rebut "
            "the bull's specific claims. Use only facts from the brief. If the bearish case is genuinely weak, admit it.",
            f"{text}\n\n{analyst_block}\n\nBULL CASE:\n{bull}\n\nMake the bear case and rebut the bull (max 180 words).",
        )

        if verbose:
            print("  [risk manager] final verdict...")
        final = self._ask(
            "You are the risk manager and final decision maker for a futures trading desk. Weigh both cases and "
            "decide. Prefer HOLD when the evidence conflicts -- capital preservation beats forcing a trade, and "
            "most days genuinely have no edge. Respond in EXACTLY this format:\n"
            "ACTION: <BUY|SELL|HOLD>\n"
            "CONFIDENCE: <LOW|MEDIUM|HIGH>\n"
            "KEY_LEVEL: <a price from the brief that would invalidate this view>\n"
            "REASONING: <3-4 sentences citing specific numbers>",
            f"{text}\n\n{analyst_block}\n\nBULL CASE:\n{bull}\n\nBEAR CASE:\n{bear}\n\nGive your final decision.",
        )

        action = self._extract(final, "ACTION", r"\b(BUY|SELL|HOLD)\b", "HOLD")
        confidence = self._extract(final, "CONFIDENCE", r"\b(LOW|MEDIUM|HIGH)\b", "LOW")

        return CouncilResult(
            ticker=ticker, price=price, action=action, confidence=confidence,
            reasoning=final, analyst_reports=reports, bull_case=bull, bear_case=bear, raw_final=final,
        )

    @staticmethod
    def _extract(text: str, label: str, pattern: str, default: str) -> str:
        """Prefer the labelled line; fall back to a bare pattern match. Returns
        `default` if neither is found -- callers treat a defaulted action as
        low-trust rather than a real decision."""
        m = re.search(rf"{label}\s*[:\-]\s*\**\s*([A-Za-z]+)", text, re.IGNORECASE)
        if m and re.fullmatch(pattern, m.group(1).upper()):
            return m.group(1).upper()
        m2 = re.search(pattern, text.upper())
        return m2.group(1) if m2 else default


def to_dict(r: CouncilResult) -> dict:
    return {
        "ticker": r.ticker, "price": r.price, "action": r.action, "confidence": r.confidence,
        "reasoning": r.reasoning, "analyst_reports": r.analyst_reports,
        "bull_case": r.bull_case, "bear_case": r.bear_case,
    }


if __name__ == "__main__":
    import sys

    from app.agent.brief import build_brief

    tk = sys.argv[1] if len(sys.argv) > 1 else "CL=F"
    b = build_brief(tk)
    res = TradingCouncil().run(b)
    print(json.dumps(to_dict(res), indent=2)[:3000])
