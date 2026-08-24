"""
AutoScientist: The LLM-augmented meta-cognitive layer.

Ingests experiment logs + KnowledgeBase, proposes novel hypotheses,
performs high-level reasoning, symbolic analysis, and generates
intelligent experiment batches.

Distinct from Scientist (execution engine):
  - Scientist executes experiments reliably.
  - AutoScientist decides *what* to execute and *why*.
"""

from computronium.autoscientist.bridge import AutoScientistBridge, ExperimentProposal
from computronium.autoscientist.campaign import (
    AutoScientistCampaign,
    CampaignCheckpointer,
    CampaignDatabase,
    CampaignState,
    IterationRecord,
    create_campaign,
    list_branches,
    list_campaigns,
)
from computronium.autoscientist.counterfactual import (
    BetaScheduleCounterfactuals,
    Counterfactual,
    CounterfactualBatch,
    CounterfactualGenerator,
    generate_what_if_report,
)
from computronium.autoscientist.literature import (
    ArxivClient,
    ArxivPaper,
    LiteratureCache,
    LiteratureRetriever,
    LiteratureSearchResult,
)
from computronium.autoscientist.local_llm import (
    LlamaCppBackend,
    LocalLLMBackend,
    LocalLLMHypothesisGenerator,
    OllamaBackend,
    TransformersBackend,
    VLLMBackend,
    create_local_llm,
    get_recommended_local_model,
)
from computronium.autoscientist.proposer import ExperimentProposer
from computronium.autoscientist.reasoner import (
    Hypothesis,
    HypothesisReasoner,
    LLMHypothesisGenerator,
)

__all__ = [
    "ArxivClient",
    "ArxivPaper",
    "AutoScientistBridge",
    "AutoScientistCampaign",
    "BetaScheduleCounterfactuals",
    "CampaignCheckpointer",
    "CampaignDatabase",
    "CampaignState",
    "Counterfactual",
    "CounterfactualBatch",
    "CounterfactualGenerator",
    "ExperimentProposal",
    "ExperimentProposer",
    "Hypothesis",
    "HypothesisReasoner",
    "IterationRecord",
    "LLMHypothesisGenerator",
    "LiteratureCache",
    "LiteratureRetriever",
    "LiteratureSearchResult",
    "LlamaCppBackend",
    "LocalLLMBackend",
    "LocalLLMHypothesisGenerator",
    "OllamaBackend",
    "TransformersBackend",
    "VLLMBackend",
    "create_campaign",
    "create_local_llm",
    "generate_what_if_report",
    "get_recommended_local_model",
    "list_branches",
    "list_campaigns",
]
