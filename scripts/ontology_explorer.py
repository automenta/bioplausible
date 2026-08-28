#!/usr/bin/env python
"""6-D Ontology Explorer: Interactive visualization of the computronium 6-D design space.

Usage:
    uv run scripts/ontology_explorer.py
    # Opens NiceGUI at http://localhost:8080

Features:
- Click axes to select primitives for each of the 6 dimensions
- Shows valid/invalid combinations in real-time
- Generates config YAML or Python code
- Links to relevant papers/benchmarks
"""

from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Any

from nicegui import ui

# Import computronium components for validation
from computronium.core.ontology import SystemConfig
from computronium.core.joint.transition import PlasticityConfig


# ----------------------------------------------------------------------
# 6-D Ontology Definition
# ----------------------------------------------------------------------

ONTOLOGY = {
    "substrate": {
        "label": "Substrate (S)",
        "description": "Physical state space constraints",
        "options": {
            "digital": {
                "label": "Digital",
                "description": "Standard floating-point computation",
                "params": {"precision": "float32", "noise_level": 0.0, "sparsity": 0.0},
                "papers": ["von Neumann (1945)", "Standard DL"],
            },
            "memristive": {
                "label": "Memristive",
                "description": "Analog conductance-based weights",
                "params": {"precision": "int8", "noise_level": 0.05, "sparsity": 0.1},
                "papers": ["Strukov et al. Nature 2008", "Yao et al. Nature 2020"],
            },
            "neuromorphic": {
                "label": "Neuromorphic",
                "description": "Event-driven sparse computation",
                "params": {
                    "precision": "float16",
                    "noise_level": 0.01,
                    "sparsity": 0.95,
                },
                "papers": ["Davies et al. IEEE 2018", "Merolla et al. Science 2014"],
            },
            "optical": {
                "label": "Optical",
                "description": "Light-based matrix multiplication",
                "params": {
                    "precision": "float32",
                    "noise_level": 0.01,
                    "sparsity": 0.0,
                },
                "papers": [
                    "Shen et al. Nature Photonics 2017",
                    "Feldmann et al. Nature 2021",
                ],
            },
            "quantum": {
                "label": "Quantum",
                "description": "Quantum circuit state space",
                "params": {
                    "precision": "complex64",
                    "noise_level": 0.0,
                    "sparsity": 0.0,
                },
                "papers": ["Schuld et al. PRL 2019", "Killoran et al. Nature 2019"],
            },
            "sparse": {
                "label": "Sparse",
                "description": "Structured sparsity constraints",
                "params": {"precision": "float32", "noise_level": 0.0, "sparsity": 0.8},
                "papers": ["Frankle & Carbin ICLR 2019", "Evci et al. NeurIPS 2020"],
            },
            "ternary": {
                "label": "Ternary",
                "description": "Ternary weight quantization {-1, 0, 1}",
                "params": {"precision": "float32", "noise_level": 0.0, "sparsity": 0.0},
                "papers": ["Li et al. ICLR 2017", "Zhu et al. ICLR 2017"],
            },
        },
    },
    "geometry": {
        "label": "Geometry (G)",
        "description": "Topology & routing",
        "options": {
            "feedforward": {
                "label": "Feedforward",
                "description": "Layered DAG, no recurrence",
                "params": {"topology_type": "feedforward", "hidden_dims": [256, 128]},
                "papers": ["LeCun et al. Nature 2015", "Goodfellow et al. 2016"],
            },
            "recurrent": {
                "label": "Recurrent",
                "description": "Feedback connections, settling dynamics",
                "params": {"topology_type": "recurrent", "hidden_dims": [256]},
                "papers": ["Hopfield 1982", "Scellier & Bengio 2017"],
            },
            "tile_mesh": {
                "label": "Tile Mesh",
                "description": "Modular tiled architecture",
                "params": {
                    "topology_type": "tile_mesh",
                    "neurons_per_tile": 64,
                    "tiles_per_layer": 4,
                },
                "papers": ["Kaiser et al. ICML 2018", "Riquelme et al. 2021"],
            },
        },
    },
    "dynamics": {
        "label": "State Dynamics (D)",
        "description": "Forward evolution & settling",
        "options": {
            "instantaneous": {
                "label": "Instantaneous",
                "description": "Single forward pass, no settling",
                "params": {"dynamics_type": "instantaneous"},
                "papers": ["Standard feedforward"],
            },
            "energy_minimization": {
                "label": "Energy Minimization",
                "description": "Iterative settling to energy minimum",
                "params": {
                    "dynamics_type": "energy_minimization",
                    "max_steps": 20,
                    "beta": 0.5,
                },
                "papers": ["Hopfield 1982", "Scellier & Bengio 2017 (EqProp)"],
            },
            "predictive_settling": {
                "label": "Predictive Settling",
                "description": "Predictive coding style settling",
                "params": {
                    "dynamics_type": "predictive_settling",
                    "max_steps": 20,
                    "beta": 0.5,
                },
                "papers": ["Rao & Ballard 1999", "Whittington & Bogacz 2019"],
            },
            "spike_integration": {
                "label": "Spike Integration",
                "description": "Leaky integrate-and-fire dynamics",
                "params": {"dynamics_type": "spike_integration", "max_steps": 50},
                "papers": ["Gerstner et al. 2014", "Bellec et al. NeurIPS 2018"],
            },
            "diffusion": {
                "label": "Diffusion",
                "description": "Continuous-time diffusion dynamics",
                "params": {"dynamics_type": "diffusion", "max_steps": 100},
                "papers": ["Sohl-Dickstein et al. ICML 2015", "Song et al. ICLR 2021"],
            },
        },
    },
    "plasticity": {
        "label": "Plasticity/MetaDynamics (M)",
        "description": "Intra-episode state evolution",
        "options": {
            "null": {
                "label": "Null (Zero-Extension)",
                "description": "No plasticity, ψ constant (5-D equivalence)",
                "params": {"type": "null"},
                "papers": ["Zero-Extension Theorem (this work)"],
            },
            "routing": {
                "label": "Routing",
                "description": "State-dependent pathway gating",
                "params": {"type": "routing", "gate_dim": 64, "temperature": 1.0},
                "papers": ["Rosenbaum et al. NeurIPS 2019", "Pathak et al. ICLR 2023"],
            },
            "fast_weights": {
                "label": "Fast Weights",
                "description": "Episode-local associative memory",
                "params": {"type": "fast_weights", "fast_weight_dim": 512},
                "papers": ["Ba et al. NeurIPS 2016", "Schlag et al. ICML 2021"],
            },
            "substrate_coupled": {
                "label": "Substrate Coupled",
                "description": "Reuse substrate adapters as plasticity",
                "params": {"type": "substrate_coupled"},
                "papers": ["Gallese & Micali 2024 (this work)"],
            },
            "rule_state": {
                "label": "Rule State",
                "description": "Operator selection via ψ (Z3)",
                "params": {"type": "rule_state", "num_operators": 8},
                "papers": ["Kirsch & Schmidhuber 2021", "Z3 paper"],
            },
        },
    },
    "credit": {
        "label": "Credit Assignment (C)",
        "description": "Error routing & pseudo-gradients",
        "options": {
            "backprop": {
                "label": "Backprop (Gradient)",
                "description": "Exact gradient via autograd",
                "params": {"credit_type": "gradient"},
                "papers": ["Rumelhart et al. 1986", "Baydin et al. 2018"],
            },
            "thermo": {
                "label": "Thermodynamic Contrast",
                "description": "Free energy difference (EqProp)",
                "params": {"credit_type": "thermodynamic_contrast", "beta": 0.5},
                "papers": ["Scellier & Bengio 2017", "Laborieux et al. 2021"],
            },
            "random_projections": {
                "label": "Random Projections (FA)",
                "description": "Fixed random feedback matrices",
                "params": {"credit_type": "random_projections", "feedback_scale": 0.01},
                "papers": ["Lillicrap et al. 2016", "Bartunov et al. 2018"],
            },
            "local_goodness": {
                "label": "Local Goodness",
                "description": "Layer-wise goodness functions",
                "params": {"credit_type": "local_goodness"},
                "papers": ["Hinton 2022 (Forward-Forward)", "Peña et al. 2023"],
            },
            "temporal_trace": {
                "label": "Temporal Trace",
                "description": "Eligibility traces for credit",
                "params": {"credit_type": "temporal_trace"},
                "papers": ["Sutton & Barto 2018", "Bellec et al. 2020"],
            },
        },
    },
    "update": {
        "label": "Parameter Update (U)",
        "description": "Optimization rule",
        "options": {
            "euclidean": {
                "label": "Euclidean (SGD)",
                "description": "Standard gradient descent",
                "params": {"update_type": "euclidean", "step_size": 0.01},
                "papers": ["Robbins & Monro 1951", "Bottou 2010"],
            },
            "riemannian_orthogonal": {
                "label": "Riemannian Orthogonal",
                "description": "Orthogonal manifold optimization",
                "params": {"update_type": "riemannian_orthogonal"},
                "papers": ["Absil et al. 2009", "Lezcano-Casado & Martinez-Rubio 2019"],
            },
            "spectral_constrained": {
                "label": "Spectral Constrained",
                "description": "Spectral norm constrained updates",
                "params": {"update_type": "spectral_constrained"},
                "papers": ["Yoshida & Miyato 2017", "Sedghi et al. 2019"],
            },
            "natural_gradient": {
                "label": "Natural Gradient",
                "description": "Fisher information metric",
                "params": {"update_type": "natural_gradient"},
                "papers": ["Amari 1998", "Martens 2020"],
            },
            "elastic_consolidation": {
                "label": "Elastic Consolidation",
                "description": "EWC-style regularization",
                "params": {"update_type": "elastic_consolidation"},
                "papers": ["Kirkpatrick et al. PNAS 2017", "Schwarz et al. 2018"],
            },
        },
    },
}


# ----------------------------------------------------------------------
# Validation Rules (Cross-axis constraints)
# ----------------------------------------------------------------------

VALIDATION_RULES = [
    # Energy minimization requires recurrent or tile geometry
    {
        "if": {"dynamics": "energy_minimization"},
        "then": {"geometry": ["recurrent", "tile_mesh"]},
        "message": "Energy minimization requires recurrent or tile geometry",
    },
    # Predictive settling requires recurrent or tile geometry
    {
        "if": {"dynamics": "predictive_settling"},
        "then": {"geometry": ["recurrent", "tile_mesh"]},
        "message": "Predictive settling requires recurrent or tile geometry",
    },
    # Spike integration requires neuromorphic substrate
    {
        "if": {"dynamics": "spike_integration"},
        "then": {"substrate": ["neuromorphic"]},
        "message": "Spike integration works best with neuromorphic substrate",
    },
    # Thermodynamic contrast requires energy minimization or predictive settling
    {
        "if": {"credit": "thermo"},
        "then": {"dynamics": ["energy_minimization", "predictive_settling"]},
        "message": "Thermodynamic contrast requires energy minimization or predictive settling dynamics",
    },
    # Random projections (FA) requires feedforward or recurrent
    {
        "if": {"credit": "random_projections"},
        "then": {"geometry": ["feedforward", "recurrent"]},
        "message": "Feedback alignment works with feedforward or recurrent geometry",
    },
    # Quantum substrate requires quantum-compatible components
    {
        "if": {"substrate": "quantum"},
        "then": {"geometry": ["feedforward", "recurrent"]},
        "message": "Quantum substrate works with standard geometries",
    },
]


def validate_combination(selection: dict[str, str]) -> list[str]:
    """Validate a 6-D combination against cross-axis rules.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []
    for rule in VALIDATION_RULES:
        if all(selection.get(k) == v for k, v in rule["if"].items()):
            then_key = list(rule["then"].keys())[0]
            then_values = rule["then"][then_key]
            if selection.get(then_key) not in then_values:
                errors.append(rule["message"])
    return errors


# ----------------------------------------------------------------------
# Code Generation
# ----------------------------------------------------------------------


def generate_python_code(selection: dict[str, str]) -> str:
    """Generate Python code for the selected 6-D coordinate."""
    coord_str = "/".join([
        selection["substrate"],
        selection["geometry"],
        selection["dynamics"],
        selection["plasticity"],
        selection["credit"],
        selection["update"],
    ])

    # Map credit names
    credit_map = {
        "backprop": "BackpropCredit",
        "thermo": "ThermodynamicContrast",
        "random_projections": "RandomProjectionsCredit",
        "local_goodness": "LocalGoodnessCredit",
        "temporal_trace": "TemporalTraceCredit",
    }

    # Map update names
    update_map = {
        "euclidean": "EuclideanUpdate",
        "riemannian_orthogonal": "RiemannianOrthogonalUpdate",
        "spectral_constrained": "SpectralConstrainedUpdate",
        "natural_gradient": "NaturalGradientUpdate",
        "elastic_consolidation": "ElasticConsolidationUpdate",
    }

    # Map plasticity
    plasticity_map = {
        "null": "NullPlasticity",
        "routing": "RoutingPlasticity",
        "fast_weights": "FastWeightPlasticity",
        "substrate_coupled": "SubstrateCoupledPlasticity",
        "rule_state": "RuleStatePlasticity",
    }

    code = f'''"""
Generated 6-D Joint System Configuration
Coordinate: {coord_str}
"""

import torch
from computronium.core.ontology import (
    DigitalSubstrate, SubstrateConfig,
    FeedforwardGeometry, RecurrentGeometry, GeometryConfig,
    InstantaneousDynamics, EnergyMinimizationDynamics, PredictiveSettlingDynamics, StateDynamicsConfig,
    BackpropCredit, ThermodynamicContrast, RandomProjectionsCredit, CreditAssignmentConfig,
    EuclideanUpdate, ParameterUpdateConfig,
)
from computronium.core.plasticity.routing import RoutingPlasticity
from computronium.core.plasticity.fast_weights import FastWeightPlasticity
from computronium.core.joint.transition import NullPlasticity, PlasticityConfig
from computronium.core.system_trainer import compose_joint_system


# Create components
substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))

'''

    # Geometry
    geom = ONTOLOGY["geometry"]["options"][selection["geometry"]]
    if selection["geometry"] == "feedforward":
        code += f"""geometry = FeedforwardGeometry(
    GeometryConfig.feedforward(
        input_dim=784,
        output_dim=10,
        hidden_dims={geom["params"].get("hidden_dims", [256, 128])},
        init_scale=0.1,
    )
)
"""
    elif selection["geometry"] == "recurrent":
        code += f"""geometry = RecurrentGeometry(
    GeometryConfig.recurrent(
        input_dim=784,
        output_dim=10,
        hidden_dims={geom["params"].get("hidden_dims", [256])},
        init_scale=0.1,
    ),
    hidden_dim={geom["params"].get("hidden_dims", [256])[0]},
)
"""
    else:
        code += f"""# Tile mesh geometry not shown for brevity
geometry = FeedforwardGeometry(
    GeometryConfig.feedforward(input_dim=784, output_dim=10, hidden_dims=[256, 128])
)
"""

    # Dynamics
    dyn = ONTOLOGY["dynamics"]["options"][selection["dynamics"]]
    if selection["dynamics"] == "instantaneous":
        code += (
            "dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())\n"
        )
    elif selection["dynamics"] == "energy_minimization":
        params = dyn["params"]
        code += f"dynamics = EnergyMinimizationDynamics(StateDynamicsConfig.energy_minimization(max_steps={params.get('max_steps', 20)}, beta={params.get('beta', 0.5)}, step_size=0.1))\n"
    elif selection["dynamics"] == "predictive_settling":
        params = dyn["params"]
        code += f"dynamics = PredictiveSettlingDynamics(StateDynamicsConfig.predictive_settling(max_steps={params.get('max_steps', 20)}, step_size=0.1))\n"

    # Plasticity
    plas = ONTOLOGY["plasticity"]["options"][selection["plasticity"]]
    if selection["plasticity"] == "null":
        code += "plasticity = NullPlasticity()\n"
    elif selection["plasticity"] == "routing":
        params = plas["params"]
        code += f"plasticity = RoutingPlasticity(gate_dim={params.get('gate_dim', 64)}, temperature={params.get('temperature', 1.0)})\n"
    elif selection["plasticity"] == "fast_weights":
        params = plas["params"]
        code += f"plasticity = FastWeightPlasticity(fast_weight_dim={params.get('fast_weight_dim', 512)})\n"
    else:
        code += f"# {selection['plasticity']} plasticity - see docs\n"
        code += "plasticity = NullPlasticity()  # placeholder\n"

    # Credit
    cred = ONTOLOGY["credit"]["options"][selection["credit"]]
    if selection["credit"] == "backprop":
        code += "credit = BackpropCredit(CreditAssignmentConfig.gradient())\n"
    elif selection["credit"] == "thermo":
        params = cred["params"]
        code += f"credit = ThermodynamicContrast(CreditAssignmentConfig.thermodynamic_contrast(beta={params.get('beta', 0.5)}))\n"
    elif selection["credit"] == "random_projections":
        params = cred["params"]
        code += f"credit = RandomProjectionsCredit(CreditAssignmentConfig.random_projections(feedback_scale={params.get('feedback_scale', 0.01)}))\n"
    else:
        code += f"# {selection['credit']} credit - see docs\n"
        code += "credit = BackpropCredit(CreditAssignmentConfig.gradient())  # placeholder\n"

    # Update
    upd = ONTOLOGY["update"]["options"][selection["update"]]
    if selection["update"] == "euclidean":
        params = upd["params"]
        code += f"update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size={params.get('step_size', 0.01)}))\n"
    else:
        code += f"# {selection['update']} update - see docs\n"
        code += "update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01))  # placeholder\n"

    code += f"""
# Compose joint system
system = compose_joint_system(
    substrate=substrate,
    geometry=geometry,
    dynamics=dynamics,
    plasticity=plasticity,
    credit=credit,
    update=update,
)

# Train
x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))
metrics = system.train_step(x, y)
print(f"Loss: {{metrics['loss']:.4f}}, Energy: {{metrics['energy']:.4f}}")
"""
    return code


def generate_yaml_config(selection: dict[str, str]) -> str:
    """Generate YAML config for the selected 6-D coordinate."""
    config = {
        "schema_version": "2.0",
        "coordinate": "/".join([
            selection["substrate"],
            selection["geometry"],
            selection["dynamics"],
            selection["plasticity"],
            selection["credit"],
            selection["update"],
        ]),
        "substrate": ONTOLOGY["substrate"]["options"][selection["substrate"]]["params"],
        "geometry": ONTOLOGY["geometry"]["options"][selection["geometry"]]["params"],
        "dynamics": ONTOLOGY["dynamics"]["options"][selection["dynamics"]]["params"],
        "plasticity": ONTOLOGY["plasticity"]["options"][selection["plasticity"]][
            "params"
        ],
        "credit": ONTOLOGY["credit"]["options"][selection["credit"]]["params"],
        "update": ONTOLOGY["update"]["options"][selection["update"]]["params"],
    }
    return yaml.dump(config, sort_keys=False)


# ----------------------------------------------------------------------
# NiceGUI Application
# ----------------------------------------------------------------------


class OntologyExplorer:
    """Interactive 6-D Ontology Explorer."""

    def __init__(self):
        self.selection = {
            dim: list(opts["options"].keys())[0] for dim, opts in ONTOLOGY.items()
        }
        self.build_ui()

    def build_ui(self):
        """Build the NiceGUI interface."""
        ui.page_title("6-D Ontology Explorer")

        with ui.header().classes("bg-primary text-white"):
            ui.label("🧬 6-D Ontology Explorer").classes("text-h4 q-px-md")
            ui.label("S ⊗ G ⊗ D ⊗ M ⊗ C ⊗ U").classes("text-caption q-px-md")

        with ui.row().classes("w-full h-[calc(100vh-60px)] no-wrap"):
            # Left panel: Dimension selectors
            with (
                ui
                .column()
                .classes("w-1/3 p-4 gap-4 overflow-auto")
                .style("min-width: 350px;")
            ):
                ui.label("Select Primitives").classes("text-h6")

                self.dimension_cards = {}
                for dim_key, dim_info in ONTOLOGY.items():
                    with ui.card().classes("w-full") as card:
                        ui.label(dim_info["label"]).classes("text-subtitle1 font-bold")
                        ui.label(dim_info["description"]).classes(
                            "text-caption text-grey-7"
                        )

                        with ui.column().classes("w-full gap-1"):
                            for opt_key, opt_info in dim_info["options"].items():
                                is_selected = self.selection[dim_key] == opt_key
                                btn = (
                                    ui
                                    .button(
                                        opt_info["label"],
                                        on_click=lambda k=opt_key, d=dim_key: (
                                            self.select_option(d, k)
                                        ),
                                    )
                                    .classes("w-full justify-start")
                                    .props(
                                        f"outline={'true' if not is_selected else 'false'}"
                                    )
                                    .style(
                                        "background-color: #e3f2fd;"
                                        if is_selected
                                        else ""
                                    )
                                )
                                if is_selected:
                                    btn.props("color=primary")
                                with btn:
                                    ui.tooltip(opt_info["description"])

                        self.dimension_cards[dim_key] = card

                # Validation status
                self.validation_label = ui.label("").classes("text-body1 mt-4")
                self.update_validation()

            # Right panel: Details and output
            with ui.column().classes("w-2/3 p-4 gap-4 overflow-auto"):
                # Current coordinate display
                with ui.card().classes("w-full"):
                    ui.label("Current Coordinate").classes("text-h6")
                    self.coordinate_display = ui.label("").classes(
                        "text-h5 font-mono text-primary"
                    )

                # Papers/References
                with ui.card().classes("w-full"):
                    ui.label("Relevant Papers").classes("text-h6")
                    self.papers_container = ui.column().classes("w-full gap-1")

                # Generated code
                with ui.expansion("Generated Python Code", icon="code").classes(
                    "w-full"
                ):
                    self.python_code = (
                        ui.code("").classes("w-full").style("min-height: 300px;")
                    )

                # Generated YAML
                with ui.expansion("Generated YAML Config", icon="description").classes(
                    "w-full"
                ):
                    self.yaml_code = (
                        ui.code("").classes("w-full").style("min-height: 200px;")
                    )

                # Actions
                with ui.row().classes("gap-2"):
                    ui.button(
                        "Copy Python",
                        icon="content_copy",
                        on_click=lambda: ui.copy_to_clipboard(self.python_code.content),
                    ).props("color=primary")
                    ui.button(
                        "Copy YAML",
                        icon="content_copy",
                        on_click=lambda: ui.copy_to_clipboard(self.yaml_code.content),
                    ).props("color=primary")
                    ui.button(
                        "Save Config", icon="save", on_click=self.save_config
                    ).props("color=secondary")
                    ui.button(
                        "Run Quick Test",
                        icon="play_arrow",
                        on_click=self.run_quick_test,
                    ).props("color=positive")

                self.output_log = ui.log().classes("w-full").style("height: 200px;")

        # Initial render
        self.render_selection()

    def select_option(self, dimension: str, option: str):
        """Handle option selection."""
        self.selection[dimension] = option
        self.render_selection()

    def render_selection(self):
        """Update UI based on current selection."""
        # Update button styles
        for dim_key, card in self.dimension_cards.items():
            for child in card.default_slot.children:
                if hasattr(child, "default_slot"):
                    for btn in child.default_slot.children:
                        if hasattr(btn, "_props"):
                            opt_key = None
                            for ok, oi in ONTOLOGY[dim_key]["options"].items():
                                if btn.text == oi["label"]:
                                    opt_key = ok
                                    break
                            if opt_key:
                                is_sel = self.selection[dim_key] == opt_key
                                btn.props(
                                    f"outline={'true' if not is_sel else 'false'}"
                                )
                                if is_sel:
                                    btn.props("color=primary")
                                else:
                                    btn.props("color=")

        # Update coordinate display
        coord_str = "/".join([self.selection[k] for k in ONTOLOGY.keys()])
        self.coordinate_display.set_text(coord_str)

        # Update validation
        self.update_validation()

        # Update papers
        self.update_papers()

        # Update generated code
        self.python_code.set_content(generate_python_code(self.selection))
        self.yaml_code.set_content(generate_yaml_config(self.selection))

    def update_validation(self):
        """Update validation status."""
        errors = validate_combination(self.selection)
        if errors:
            self.validation_label.set_text("⚠️ " + "; ".join(errors))
            self.validation_label.classes("text-negative")
        else:
            self.validation_label.set_text("✅ Valid combination")
            self.validation_label.classes("text-positive")

    def update_papers(self):
        """Update papers list."""
        self.papers_container.clear()
        with self.papers_container:
            for dim_key, opt_key in self.selection.items():
                opt = ONTOLOGY[dim_key]["options"][opt_key]
                if opt.get("papers"):
                    with ui.row().classes("gap-2 items-center"):
                        ui.label(f"{ONTOLOGY[dim_key]['label']}:").classes(
                            "font-bold text-sm"
                        )
                        for paper in opt["papers"]:
                            ui.label(paper).classes("text-sm text-grey-7")

    def save_config(self):
        """Save YAML config to file."""
        yaml_content = generate_yaml_config(self.selection)
        coord_str = "/".join([self.selection[k] for k in ONTOLOGY.keys()])
        filename = f"config_{coord_str.replace('/', '_')}.yaml"
        Path(filename).write_text(yaml_content)
        self.output_log.push(f"Saved config to {filename}")
        ui.notify(f"Saved to {filename}", type="positive")

    async def run_quick_test(self):
        """Run a quick training test with the selected configuration."""
        self.output_log.push("Starting quick test...")
        try:
            # Use the lab inspect-state command as a test
            import subprocess

            coord_str = "/".join([self.selection[k] for k in ONTOLOGY.keys()])
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "biopl",
                    "lab",
                    "inspect-state",
                    "--coordinate",
                    coord_str,
                    "--task",
                    "mnist",
                    "--steps",
                    "3",
                    "--hidden-dim",
                    "64",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                self.output_log.push("✅ Quick test passed!")
                self.output_log.push(result.stdout[-500:])
            else:
                self.output_log.push(f"❌ Quick test failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            self.output_log.push("⏱️ Quick test timed out")
        except Exception as e:
            self.output_log.push(f"❌ Error: {e}")


def main():
    """Main entry point."""
    explorer = OntologyExplorer()
    ui.run(host="0.0.0.0", port=8080, title="6-D Ontology Explorer", reload=False)


if __name__ == "__main__":
    main()
