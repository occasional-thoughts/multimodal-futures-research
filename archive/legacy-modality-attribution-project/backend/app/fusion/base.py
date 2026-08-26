from dataclasses import dataclass


@dataclass
class FusionOutput:
    """A fused feature matrix whose columns stay traceable to their source modality.

    This traceability is the fix for the attention-fusion attribution problem: every
    fusion variant must expose which output columns are technical-derived and which
    are news-derived, so downstream SHAP attribution can sum |SHAP| per group on the
    same basis regardless of which fusion strategy produced the columns.
    """

    matrix: "any"
    feature_names: list
    tech_group: list  # column indices belonging to the technical modality
    news_group: list  # column indices belonging to the news modality
