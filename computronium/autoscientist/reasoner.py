"""
HypothesisReasoner: High-level meta-cognitive reasoning.

Analyses experiment history and KnowledgeBase to generate hypotheses.
Supports both rule-based reasoning and optional LLM integration.
"""

import json
from dataclasses import dataclass, field
from enum import Enum

from computronium.core.logging import get_logger
from computronium.knowledge import KnowledgeBase, KnowledgeEntry

logger = get_logger()


class ReasoningTemplate(str, Enum):  # ruff: ignore[replace-str-enum]
    """Chain-of-thought reasoning templates."""

    FAILURE_ANALYSIS = "failure_analysis"
    TRANSFER_REASONING = "transfer_reasoning"
    COMPOSITION = "composition"
    HYPOTHESIS_REFINEMENT = "hypothesis_refinement"
    EXPERIMENTAL_DESIGN = "experimental_design"


@dataclass(frozen=True, slots=True)
class ReasoningChain:
    """Structured chain-of-thought reasoning."""

    template: ReasoningTemplate
    steps: list[str]
    conclusion: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A scientific hypothesis about what might work and why."""

    statement: str
    confidence: float = 0.5
    supporting_evidence: list[str] = field(default_factory=list)
    proposed_model: str | None = None
    proposed_task: str | None = None
    proposed_propagator: str | None = None
    reasoning_chain: list[str] = field(default_factory=list)
    source: str = "rule-based"  # "rule-based" or "llm"
    reasoning_template: ReasoningTemplate | None = None
    structured_reasoning: ReasoningChain | None = None


class HypothesisReasoner:
    """
    Generates hypotheses from experiment data and knowledge.

    Two modes:
    1. Rule-based: deterministic reasoning from known patterns
    2. LLM-augmented: uses language model for creative hypotheses (optional)

    Supports chain-of-thought templates for structured reasoning:
    - Failure Analysis: "Why did X fail? Root cause → hypothesis → fix"
    - Transfer Reasoning: "What transfers from domain A to B? Evidence?"
    - Composition: "Combine X + Y → novel algorithm Z"
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        llm_backend: str | None = None,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.llm_backend = llm_backend
        self._hypotheses: list[Hypothesis] = []
        self._reasoning_chains: list[ReasoningChain] = []

    def generate_hypotheses(
        self,
        recent_results: list[dict[str, object]] | None = None,
    ) -> list[Hypothesis]:
        """
        Generate hypotheses based on recent results and knowledge.

        Args:
            recent_results: Optional list of recent experiment metrics.

        Returns:
            List of generated hypotheses.
        """
        hypotheses = []

        # Rule-based hypotheses
        hypotheses.extend(self._cross_domain_transfer_hypotheses(recent_results))
        hypotheses.extend(self._bio_accuracy_tradeoff_hypotheses(recent_results))

        # LLM-augmented hypotheses (if enabled)
        if self.llm_backend:
            try:
                hypotheses.extend(self._llm_hypotheses(recent_results))
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("LLM hypothesis generation failed: %s", e)

        self._hypotheses.extend(hypotheses)
        logger.info("Generated %s hypotheses", len(hypotheses))
        return hypotheses

    def _cross_domain_transfer_hypotheses(
        self,
        recent_results: list[dict[str, object]] | None = None,
    ) -> list[Hypothesis]:
        """Hypothesis: transfer successful methods across domains."""
        hypotheses = []
        if not recent_results:
            return hypotheses

        successful_propagators = set()
        for r in recent_results:
            if r.get("val_accuracy", 0) > 0.6:
                model = r.get("model", "")
                if r.get("task") in ["mnist", "cifar10", "fashion_mnist"]:  # ruff: ignore[literal-membership]
                    successful_propagators.add(model)

        for prop in successful_propagators:
            hypotheses.append(
                Hypothesis(
                    statement=f"{prop} works on vision; should transfer to language",
                    confidence=0.6,
                    supporting_evidence=[
                        f"{prop} achieved {r.get('val_accuracy', 0):.2f} on vision"
                        for r in (recent_results or [])
                        if r.get("model") == prop
                    ],
                    proposed_model=prop,
                    proposed_task="tiny_shakespeare",
                    reasoning_chain=[
                        "Local learning rules are domain-agnostic",
                        "Success on vision suggests general-purpose credit assignment",
                    ],
                    source="rule-based",
                    reasoning_template=ReasoningTemplate.TRANSFER_REASONING,
                )
            )
        return hypotheses

    def _bio_accuracy_tradeoff_hypotheses(
        self,
        recent_results: list[dict[str, object]] | None = None,
    ) -> list[Hypothesis]:
        """Hypothesis: hybrid models balance bio-plausibility and accuracy."""
        hypotheses = []
        if not recent_results:
            return hypotheses

        for r in recent_results:
            acc = r.get("val_accuracy", 0)
            if acc < 0.5 and r.get("bio_score", 0) > 0.8:
                hypotheses.append(
                    Hypothesis(
                        statement=(
                            f"{r.get('model')} has high bio-plausibility "
                            f"but low accuracy ({acc:.2f}); "
                            "hybrid with backprop head may improve"
                        ),
                        confidence=0.7,
                        proposed_model=r.get("model"),
                        proposed_task=r.get("task"),
                        reasoning_chain=[
                            "Pure local learning may underfit complex patterns",
                            "Adding a global backprop head provides error signal",
                            "Hybrid models retain partial bio-plausibility",
                        ],
                        source="rule-based",
                        reasoning_template=ReasoningTemplate.COMPOSITION,
                    )
                )
        return hypotheses

    def _llm_hypotheses(
        self,
        recent_results: list[dict[str, object]] | None = None,
    ) -> list[Hypothesis]:
        """
        Generate hypotheses using LLM (optional, local-first).

        Uses the LLM to suggest novel experiment ideas based on
        knowledge base patterns and recent results.
        """
        hypotheses = []

        insights = self.analyze_knowledge_base() if self.knowledge_base else []

        context = []
        if recent_results:
            context.append("Recent Results:")
            for r in recent_results[:5]:
                context.append(
                    f"  - {r.get('model', 'unknown')} on "
                    f"{r.get('task', 'unknown')}: {r.get('val_accuracy', 0):.2f}"
                )
        if insights:
            context.append("Knowledge Base Insights:")
            for i in insights[:5]:
                context.append(f"  - {i}")

        prompt = "\n".join(context)
        if not prompt:
            return hypotheses

        try:
            generator = LLMHypothesisGenerator(backend=self.llm_backend)
            llm_hypotheses = generator.generate(prompt)
            hypotheses.extend(llm_hypotheses)
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Could not initialize LLM backend: %s", e)

        return hypotheses

    def analyze_knowledge_base(self) -> list[str]:
        """Analyze KnowledgeBase for patterns and insights."""
        insights = []
        if not self.knowledge_base:
            return insights

        entries = self.knowledge_base.query(limit=50)
        if not entries:
            return insights

        model_perf = {}
        for entry in entries:
            if isinstance(entry, KnowledgeEntry):
                metrics = entry.metrics or {}
                model = entry.model_family or "unknown"
            else:
                metrics = entry.get("metrics", {})
                config = entry.get("config", {})
                model = (
                    config.get("model", "unknown")
                    if isinstance(metrics, dict)
                    else "unknown"
                )
            acc = metrics.get("val_accuracy", 0) if isinstance(metrics, dict) else 0
            if model not in model_perf:
                model_perf[model] = []
            model_perf[model].append(acc)

        for model, accs in model_perf.items():
            if accs:
                mean_acc = sum(accs) / len(accs)
                insights.append(
                    f"{model}: mean accuracy {mean_acc:.3f} across {len(accs)} runs"
                )

        return insights

    def get_top_hypotheses(self, n: int = 5) -> list[Hypothesis]:
        """Get the highest-confidence hypotheses."""
        sorted_hypotheses = sorted(
            self._hypotheses, key=lambda h: h.confidence, reverse=True
        )
        return sorted_hypotheses[:n]

    # ============================================================
    # Chain-of-Thought Templates
    # ============================================================

    def failure_analysis(
        self,
        failed_experiment: dict[str, object],
        context: list[str] | None = None,
    ) -> ReasoningChain:
        """
        Failure Analysis Template: "Why did X fail? Root cause → hypothesis → fix"

        Analyzes a failed experiment and generates structured reasoning
        about the root cause and potential fixes.
        """
        model = failed_experiment.get("model", "unknown")
        task = failed_experiment.get("task", "unknown")
        error = failed_experiment.get("error", "unknown error")
        accuracy = failed_experiment.get("val_accuracy", 0.0)

        steps = [
            f"Observed: {model} on {task} achieved {accuracy:.3f} accuracy",
            f"Error: {error}",
            "Step 1: Categorize failure mode",
        ]

        # Categorize failure
        if (
            "cuda" in str(error).lower()
            or "oom" in str(error).lower()
            or "memory" in str(error).lower()
        ):
            failure_category = "Resource Exhaustion"
            steps.append("  → Category: Resource Exhaustion (OOM/CUDA)")
            steps.append("  → Root cause: Model too large for GPU memory")
            hypothesis = "Reduce model size, use gradient checkpointing, or enable mixed precision"
            fix = (
                "Try: smaller hidden_dim, gradient_accumulation_steps > 1, or fp16/bf16"
            )
        elif "nan" in str(error).lower() or "inf" in str(error).lower():
            failure_category = "Numerical Instability"
            steps.append("  → Category: Numerical Instability (NaN/Inf)")
            steps.append("  → Root cause: Exploding gradients or unstable dynamics")
            hypothesis = "Reduce learning rate, add gradient clipping, or increase settling steps"
            fix = "Try: lr * 0.1, gradient_clip=1.0, or max_steps * 2"
        elif accuracy < 0.1:
            failure_category = "Complete Training Failure"
            steps.append("  → Category: Complete Training Failure (near-random)")
            steps.append(
                "  → Root cause: Architecture/algorithm mismatch or hyperparameter issue"
            )
            hypothesis = (
                "Algorithm incompatible with task or critical hyperparameter wrong"
            )
            fix = "Try: different algorithm family, check beta/gamma, verify implementation"
        elif accuracy < 0.3:
            failure_category = "Underfitting"
            steps.append("  → Category: Underfitting (low accuracy)")
            steps.append(
                "  → Root cause: Insufficient capacity or wrong inductive bias"
            )
            hypothesis = "Model capacity too low or learning rule cannot solve task"
            fix = "Try: increase hidden_dim/num_layers, or switch to more expressive algorithm"
        else:
            failure_category = "Suboptimal Performance"
            steps.append("  → Category: Suboptimal Performance")
            steps.append("  → Root cause: Hyperparameter suboptimality")
            hypothesis = (
                "Hyperparameters not optimal for this task/algorithm combination"
            )
            fix = "Try: learning rate sweep, beta sweep, or architecture search"

        steps.extend([
            f"Step 2: Identified category = {failure_category}",
            f"Step 3: Generated hypothesis: {hypothesis}",
            f"Step 4: Proposed fix: {fix}",
        ])

        evidence = [f"Accuracy: {accuracy:.3f}", f"Error: {error}"]
        if context:
            evidence.extend(context)

        chain = ReasoningChain(
            template=ReasoningTemplate.FAILURE_ANALYSIS,
            steps=steps,
            conclusion=hypothesis,
            confidence=0.75,
            evidence=evidence,
            assumptions=[
                "Error message is accurate",
                "Failure mode categorization is correct",
            ],
        )

        self._reasoning_chains.append(chain)
        return chain

    def transfer_reasoning(  # ruff: ignore[complex-structure]
        self,
        source_domain: str,
        target_domain: str,
        source_results: list[dict[str, object]],
    ) -> ReasoningChain:
        """
        Transfer Reasoning Template: "What transfers from domain A to B? Evidence?"

        Analyzes what knowledge from a source domain can transfer
        to a target domain based on algorithm properties and empirical results.
        """
        steps = [
            f"Source domain: {source_domain}",
            f"Target domain: {target_domain}",
            "Step 1: Identify successful algorithms in source domain",
        ]

        # Find successful algorithms
        successful = []
        for r in source_results:
            if r.get("val_accuracy", 0) > 0.5:
                successful.append({
                    "model": r.get("model", "unknown"),
                    "accuracy": r.get("val_accuracy", 0),
                    "bio_score": r.get("bio_score", 0),
                    "task": r.get("task", "unknown"),
                })

        if not successful:
            steps.append("  → No successful algorithms found in source domain")
            return ReasoningChain(
                template=ReasoningTemplate.TRANSFER_REASONING,
                steps=steps,
                conclusion="Insufficient evidence for transfer",
                confidence=0.1,
                evidence=[],
            )

        steps.append(f"  → Found {len(successful)} successful algorithms")
        for s in successful[:3]:
            steps.append(
                f"    - {s['model']}: {s['accuracy']:.3f} on {s['task']} (bio: {s['bio_score']:.2f})"
            )

        steps.append("Step 2: Analyze algorithm properties for transferability")
        transferable = []
        for s in successful:
            model = s["model"]
            # Check algorithm family properties
            if (
                "eqprop" in model.lower()
                or "fa" in model.lower()
                or "hebbian" in model.lower()
            ):
                transferable.append(model)
                steps.append(f"  → {model}: Local learning rule - likely transferable")

        steps.append("Step 3: Consider domain-specific adaptations")
        adaptations = []
        if "vision" in source_domain.lower() and "language" in target_domain.lower():
            adaptations.append(
                "Language needs sequential processing - check if algorithm supports RNN/Transformer"
            )
            adaptations.append(
                "Vocabulary size vs pixel count - adjust input/output dimensions"
            )
        elif "vision" in source_domain.lower() and "rl" in target_domain.lower():
            adaptations.append(
                "RL needs credit assignment over time - check temporal credit assignment capability"
            )
            adaptations.append("Reward sparsity - may need different beta schedule")

        steps.append("  → Required adaptations:")
        for a in adaptations:
            steps.append(f"    - {a}")

        # Generate conclusion
        if transferable:
            conclusion = f"Transfer {', '.join(transferable[:2])} to {target_domain} with adaptations: {', '.join(adaptations[:2])}"
            confidence = 0.7
        else:
            conclusion = f"No directly transferable algorithms; design new algorithm for {target_domain}"
            confidence = 0.3

        evidence = [
            f"Source: {source_domain}",
            f"Target: {target_domain}",
            f"Successful: {len(successful)}",
        ]
        chain = ReasoningChain(
            template=ReasoningTemplate.TRANSFER_REASONING,
            steps=steps,
            conclusion=conclusion,
            confidence=confidence,
            evidence=evidence,
            assumptions=[
                "Algorithm properties are domain-agnostic",
                "Empirical results generalize across domains",
            ],
        )

        self._reasoning_chains.append(chain)
        return chain

    def composition(
        self,
        algorithm_a: str,
        algorithm_b: str,
        goal: str = "novel algorithm",
    ) -> ReasoningChain:
        """
        Composition Template: "Combine X + Y → novel algorithm Z"

        Proposes novel algorithm combinations by composing two existing
        algorithms and reasoning about the resulting properties.
        """
        steps = [
            f"Algorithm A: {algorithm_a}",
            f"Algorithm B: {algorithm_b}",
            f"Goal: {goal}",
            "Step 1: Decompose algorithms into components",
        ]

        # Known algorithm properties
        algo_props = {
            "eqprop": {
                "credit": "equilibrium",
                "memory": "O(1)",
                "local": True,
                "settling": True,
            },
            "fa": {
                "credit": "random_feedback",
                "memory": "O(1)",
                "local": True,
                "settling": False,
            },
            "tp": {
                "credit": "target_prop",
                "memory": "O(L)",
                "local": True,
                "settling": True,
            },
            "pc": {
                "credit": "prediction_error",
                "memory": "O(L)",
                "local": True,
                "settling": True,
            },
            "hebbian": {
                "credit": "correlation",
                "memory": "O(1)",
                "local": True,
                "settling": False,
            },
            "snn": {
                "credit": "spike_timing",
                "memory": "O(T)",
                "local": True,
                "settling": True,
            },
            "ff": {
                "credit": "goodness",
                "memory": "O(1)",
                "local": True,
                "settling": False,
            },
            "pepita": {
                "credit": "error_feedback",
                "memory": "O(1)",
                "local": True,
                "settling": False,
            },
            "mep": {
                "credit": "spectral",
                "memory": "O(L)",
                "local": True,
                "settling": True,
            },
            "backprop": {
                "credit": "gradient",
                "memory": "O(T)",
                "local": False,
                "settling": False,
            },
        }

        props_a = algo_props.get(algorithm_a.lower(), {})
        props_b = algo_props.get(algorithm_b.lower(), {})

        steps.append(f"  → {algorithm_a}: {props_a}")
        steps.append(f"  → {algorithm_b}: {props_b}")

        steps.append("Step 2: Identify compatible and complementary properties")

        # Find compatible properties
        compatible = []
        complementary = []
        for key in ["credit", "memory", "local", "settling"]:
            if key in props_a and key in props_b:
                if props_a[key] == props_b[key]:
                    compatible.append(f"{key}={props_a[key]}")
                else:
                    complementary.append(f"{key}: {props_a[key]} vs {props_b[key]}")

        steps.append(f"  → Compatible: {compatible}")
        steps.append(f"  → Complementary: {complementary}")

        steps.append("Step 3: Propose composition patterns")

        patterns = []

        # Pattern 1: Hybrid credit assignment
        if (
            "credit" in props_a
            and "credit" in props_b
            and props_a["credit"] != props_b["credit"]
        ):
            patterns.append(
                f"Hybrid credit: {props_a['credit']} (body) + {props_b['credit']} (head)"
            )

        # Pattern 2: Memory-accuracy tradeoff
        if props_a.get("memory") == "O(1)" and props_b.get("memory") != "O(1)":
            patterns.append(
                f"Use {algorithm_a} for memory efficiency, {algorithm_b} for accuracy"
            )

        # Pattern 3: Settling dynamics
        if props_a.get("settling") and not props_b.get("settling"):
            patterns.append(f"Add {algorithm_a}'s settling dynamics to {algorithm_b}")

        # Pattern 4: Locality preservation
        if props_a.get("local") and props_b.get("local"):
            patterns.append("Full local learning preserved")

        steps.append("  → Composition patterns:")
        for p in patterns:
            steps.append(f"    - {p}")

        steps.append("Step 4: Synthesize novel algorithm proposal")

        if patterns:
            novel_name = f"{algorithm_a}_{algorithm_b}_hybrid"
            conclusion = f"Proposed: {novel_name} - {'; '.join(patterns[:2])}"
            confidence = 0.65
        else:
            novel_name = f"{algorithm_a}_{algorithm_b}_combo"
            conclusion = f"Proposed: {novel_name} - experimental combination needed"
            confidence = 0.4

        evidence = [f"A: {algorithm_a}", f"B: {algorithm_b}"]
        chain = ReasoningChain(
            template=ReasoningTemplate.COMPOSITION,
            steps=steps,
            conclusion=conclusion,
            confidence=confidence,
            evidence=evidence,
            assumptions=[
                "Algorithm properties are composable",
                "No negative interference between components",
            ],
        )

        self._reasoning_chains.append(chain)
        return chain

    def hypothesis_refinement(
        self,
        hypothesis: Hypothesis,
        counterevidence: list[str] | None = None,
    ) -> ReasoningChain:
        """
        Hypothesis Refinement Template: Refine a hypothesis based on new evidence.

        Uses counterfactual reasoning to strengthen or weaken a hypothesis.
        """
        steps = [
            f"Original hypothesis: {hypothesis.statement}",
            f"Original confidence: {hypothesis.confidence:.2f}",
            "Step 1: Evaluate supporting evidence",
        ]

        for ev in hypothesis.supporting_evidence:
            steps.append(f"  + {ev}")

        steps.append("Step 2: Evaluate counterevidence")
        if counterevidence:
            for ce in counterevidence:
                steps.append(f"  - {ce}")
        else:
            steps.append("  (no counterevidence provided)")

        steps.append("Step 3: Identify assumptions to test")
        assumptions = (
            hypothesis.structured_reasoning.assumptions
            if hypothesis.structured_reasoning
            else []
        )
        for a in assumptions:
            steps.append(f"  ? {a}")

        steps.append("Step 4: Refine hypothesis")

        # Simple refinement logic
        new_confidence = hypothesis.confidence
        if counterevidence:
            new_confidence = max(
                0.1, hypothesis.confidence - 0.1 * len(counterevidence)
            )

        if new_confidence < 0.3:
            conclusion = f"Hypothesis weakened. Consider alternative: {hypothesis.proposed_model or 'different approach'}"
        elif new_confidence > 0.7:
            conclusion = f"Hypothesis strengthened. Proceed with: {hypothesis.proposed_model} on {hypothesis.proposed_task}"
        else:
            conclusion = f"Hypothesis uncertain ({new_confidence:.2f}). Design targeted experiment."

        chain = ReasoningChain(
            template=ReasoningTemplate.HYPOTHESIS_REFINEMENT,
            steps=steps,
            conclusion=conclusion,
            confidence=new_confidence,
            evidence=hypothesis.supporting_evidence + (counterevidence or []),
            assumptions=assumptions,
        )

        self._reasoning_chains.append(chain)
        return chain

    def experimental_design(  # ruff: ignore[complex-structure]
        self,
        research_question: str,
        available_algorithms: list[str],
        available_tasks: list[str],
        constraints: dict | None = None,
    ) -> ReasoningChain:
        """
        Experimental Design Template: Design experiments to answer a research question.

        Generates a structured experimental plan with controls, variables, and success criteria.
        """
        steps = [
            f"Research Question: {research_question}",
            "Step 1: Decompose into testable sub-questions",
        ]

        # Generate sub-questions
        sub_questions = []
        if "compare" in research_question.lower():
            sub_questions.append("Which algorithm performs best on which task?")
            sub_questions.append("What are the accuracy/memory/speed tradeoffs?")
        elif "scale" in research_question.lower():
            sub_questions.append("How does performance scale with model size?")
            sub_questions.append("What is the critical depth for local learning?")
        elif "transfer" in research_question.lower():
            sub_questions.append("Does algorithm A transfer from task X to task Y?")
            sub_questions.append("What fine-tuning is needed?")
        else:
            sub_questions.append(research_question)

        for i, sq in enumerate(sub_questions, 1):
            steps.append(f"  SQ{i}: {sq}")

        steps.append("Step 2: Define experimental variables")
        variables = {
            "independent": ["algorithm", "task", "model_size"],
            "dependent": ["val_accuracy", "training_time", "memory_usage", "bio_score"],
            "controlled": ["seed", "optimizer", "epochs", "batch_size"],
        }

        if constraints:
            for k, v in constraints.items():
                variables.setdefault("controlled", []).append(f"{k}={v}")

        steps.append(f"  Variables: {variables}")

        steps.append("Step 3: Design factorial experiment")
        n_algos = len(available_algorithms)
        n_tasks = len(available_tasks)
        n_conditions = n_algos * n_tasks
        steps.append(
            f"  Full factorial: {n_algos} algorithms × {n_tasks} tasks = {n_conditions} conditions"
        )

        if n_conditions > 50:
            steps.append(
                "  → Too many conditions; use fractional factorial or Bayesian optimization"
            )

        steps.append("Step 4: Define success criteria")
        criteria = [
            "Statistical significance (p < 0.05) for main effects",
            "Effect size (Cohen's d > 0.5) for practical significance",
            "Reproducibility across ≥3 seeds",
        ]
        for c in criteria:
            steps.append(f"  ✓ {c}")

        steps.append("Step 5: Plan analysis")
        analyses = [
            "ANOVA for algorithm × task interaction",
            "Post-hoc pairwise comparisons with BH correction",
            "Scaling law fits (power law) for model size effects",
            "Pareto frontier for accuracy vs compute tradeoffs",
        ]
        for a in analyses:
            steps.append(f"  → {a}")

        conclusion = (
            f"Designed {n_conditions}-condition experiment to test: {research_question}"
        )
        if n_conditions > 50:
            conclusion += " (will use fractional design)"

        chain = ReasoningChain(
            template=ReasoningTemplate.EXPERIMENTAL_DESIGN,
            steps=steps,
            conclusion=conclusion,
            confidence=0.8,
            evidence=[
                f"Algorithms: {available_algorithms}",
                f"Tasks: {available_tasks}",
            ],
            assumptions=[
                "Algorithms are correctly implemented",
                "Tasks are well-defined",
            ],
        )

        self._reasoning_chains.append(chain)
        return chain

    def get_reasoning_history(self) -> list[ReasoningChain]:
        """Get all executed reasoning chains."""
        return self._reasoning_chains

    def clear_reasoning_history(self) -> None:
        """Clear reasoning history."""
        self._reasoning_chains.clear()


class LLMHypothesisGenerator:
    """
    Optional LLM-powered hypothesis generator for novel experiment ideas.

    Supports local-first backends:
    - 'openai': OpenAI API (requires API key)
    - 'local': Local model via llama.cpp or similar
    """

    def __init__(self, backend: str = "openai", api_key: str | None = None):
        self.backend = backend
        self.api_key = api_key

    def generate(self, context: str) -> list[Hypothesis]:
        """
        Generate hypotheses from context using LLM.

        Args:
            context: Text context with experiment results and insights.

        Returns:
            List of hypotheses suggested by the LLM.
        """
        if self.backend == "openai" and self.api_key:
            return self._generate_openai(context)
        return self._fallback_hypotheses(context)

    def _generate_openai(self, context: str) -> list[Hypothesis]:
        """Generate using OpenAI API."""
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a scientific research assistant. "
                            "Suggest novel experiments based on the "
                            "provided context. Return JSON with hypothesis "
                            "statements and confidence scores."
                        ),
                    },
                    {"role": "user", "content": context},
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                hypotheses = []
                for item in data.get("hypotheses", []):
                    hypotheses.append(
                        Hypothesis(
                            statement=item.get("statement", ""),
                            confidence=item.get("confidence", 0.5),
                            proposed_model=item.get("model"),
                            proposed_task=item.get("task"),
                            proposed_propagator=item.get("propagator"),
                            reasoning_chain=item.get("reasoning", []),
                            source="llm",
                        )
                    )
                return hypotheses
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("OpenAI hypothesis generation failed: %s", e)

        return []

    def _fallback_hypotheses(self, context: str) -> list[Hypothesis]:
        """Fallback when LLM is unavailable."""
        return [
            Hypothesis(
                statement="Try alternative architectures on underexplored tasks",
                confidence=0.3,
                source="rule-based",
                reasoning_chain=["LLM backend unavailable, using fallback heuristic"],
            ),
        ]


__all__ = [
    "Hypothesis",
    "HypothesisReasoner",
    "LLMHypothesisGenerator",
    "ReasoningChain",
    "ReasoningTemplate",
]
