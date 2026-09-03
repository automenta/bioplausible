"""
AutoScientist Human-in-the-Loop Dashboard

Provides a web-based interface for:
- Viewing campaign progress and experiment proposals
- Approving/rejecting hypotheses
- Real-time metrics streaming via WebSocket
- Hypothesis annotation and linking to literature/KB
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nicegui import Client

try:
    from nicegui import ui

    HAS_NICEGUI = True
except ImportError:
    ui = None
    HAS_NICEGUI = False

from computronium.autoscientist.campaign import (
    AutoScientistCampaign,
    create_campaign,
    list_branches,
)
from computronium.autoscientist.reasoner import (
    HypothesisReasoner,
)
from computronium.core.logging import get_logger
from computronium.knowledge import KnowledgeBase

if TYPE_CHECKING:
    from nicegui import Client

logger = get_logger()

__all__ = [
    "AutoScientistDashboard",  # ruff: ignore[undefined-export]
    "DashboardConfig",
    "DashboardServer",
    "create_dashboard",
    "main",
    "run_dashboard",
]


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Configuration for the dashboard server."""

    host: str = "0.0.0.0"  # ruff: ignore[hardcoded-bind-all-interfaces]
    port: int = 8080
    campaign_dir: str = "autoscientist_campaigns"
    knowledge_base_path: str | None = None
    enable_websockets: bool = True
    theme: str = "light"  # "light" or "dark"
    title: str = "AutoScientist Dashboard"


class DashboardState:
    """Shared state for the dashboard."""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.campaign: AutoScientistCampaign | None = None
        self.reasoner: HypothesisReasoner | None = None
        self.knowledge_base: KnowledgeBase | None = None
        self.connected_clients: set = set()
        self.pending_proposals: list[dict] = []
        self.approved_proposals: list[dict] = []
        self.rejected_proposals: list[dict] = []
        self.current_iteration: int = 0
        self.metrics_history: list[dict] = []

    def initialize(self):
        """Initialize campaign, reasoner, and knowledge base."""
        self.campaign = create_campaign(
            output_dir=self.config.campaign_dir,
            branch="main",
            resume=True,
        )

        self.knowledge_base = KnowledgeBase()
        self.reasoner = HypothesisReasoner(knowledge_base=self.knowledge_base)

        # Load pending proposals from campaign
        self._load_campaign_state()

    def _load_campaign_state(self):
        """Load state from campaign database."""
        if not self.campaign:
            return

        history = self.campaign.get_history()
        if history:
            latest = history[-1]
            self.current_iteration = latest.iteration
            for p in latest.proposals:
                proposal = {
                    "id": str(uuid.uuid4())[:8],
                    "model": p["model"],
                    "task": p["task"],
                    "hypothesis": p["hypothesis"],
                    "propagator": p.get("propagator"),
                    "priority": p.get("priority", 0),
                    "status": "pending",
                    "timestamp": latest.timestamp,
                }
                self.pending_proposals.append(proposal)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected WebSocket clients."""
        if not self.config.enable_websockets:
            return

        message_str = json.dumps(message)
        disconnected = set()
        for ws in self.connected_clients:
            try:
                await ws.send_text(message_str)
            except Exception:
                disconnected.add(ws)

        self.connected_clients -= disconnected

    def approve_proposal(self, proposal_id: str, annotator: str = "human") -> bool:
        """Approve a pending proposal."""
        for i, p in enumerate(self.pending_proposals):
            if p["id"] == proposal_id:
                p["status"] = "approved"
                p["approved_by"] = annotator
                p["approved_at"] = datetime.now().isoformat()
                approved = self.pending_proposals.pop(i)
                self.approved_proposals.append(approved)
                return True
        return False

    def reject_proposal(
        self, proposal_id: str, annotator: str = "human", reason: str = ""
    ) -> bool:
        """Reject a pending proposal."""
        for i, p in enumerate(self.pending_proposals):
            if p["id"] == proposal_id:
                p["status"] = "rejected"
                p["rejected_by"] = annotator
                p["rejected_at"] = datetime.now().isoformat()
                p["rejection_reason"] = reason
                rejected = self.pending_proposals.pop(i)
                self.rejected_proposals.append(rejected)
                return True
        return False

    def add_annotation(self, proposal_id: str, annotation: dict) -> bool:
        """Add annotation to a proposal."""
        for p in self.pending_proposals:
            if p["id"] == proposal_id:
                p.setdefault("annotations", []).append({
                    **annotation,
                    "timestamp": datetime.now().isoformat(),
                })
                return True
        for p in self.approved_proposals:
            if p["id"] == proposal_id:
                p.setdefault("annotations", []).append({
                    **annotation,
                    "timestamp": datetime.now().isoformat(),
                })
                return True
        return False

    def link_to_literature(
        self, proposal_id: str, paper_id: str, notes: str = ""
    ) -> bool:
        """Link proposal to literature/KB entry."""
        return self.add_annotation(
            proposal_id,
            {
                "type": "literature_link",
                "paper_id": paper_id,
                "notes": notes,
            },
        )

    def link_to_kb(self, proposal_id: str, kb_entry_id: str, notes: str = "") -> bool:
        """Link proposal to KnowledgeBase entry."""
        return self.add_annotation(
            proposal_id,
            {
                "type": "kb_link",
                "kb_entry_id": kb_entry_id,
                "notes": notes,
            },
        )

    def get_summary(self) -> dict:
        """Get dashboard summary."""
        return {
            "campaign": self.campaign.get_summary() if self.campaign else {},
            "pending_count": len(self.pending_proposals),
            "approved_count": len(self.approved_proposals),
            "rejected_count": len(self.rejected_proposals),
            "current_iteration": self.current_iteration,
            "kb_stats": self.knowledge_base.get_stats() if self.knowledge_base else {},
        }


async def run_dashboard(config: DashboardConfig | None = None):
    """
    Run the AutoScientist dashboard server.

    Uses NiceGUI if available, otherwise falls back to a simple FastAPI server.
    """
    config = config or DashboardConfig()
    state = DashboardState(config)
    state.initialize()

    try:
        from nicegui import Client, app, ui  # ruff: ignore[unused-import]

        return await _run_nicegui_dashboard(state, config)
    except ImportError:
        logger.warning("NiceGUI not available, using FastAPI fallback")
        return await _run_fastapi_dashboard(state, config)


async def _run_nicegui_dashboard(state: DashboardState, config: DashboardConfig):  # ruff: ignore[complex-structure, too-many-statements, unused-async]
    """Run dashboard with NiceGUI."""
    from nicegui import app, ui

    @ui.page("/")
    async def index(client: Client):
        """Main dashboard page."""
        ui.colors(
            primary="#2563eb",
            secondary="#64748b",
            accent="#10b981",
            positive="#10b981",
            negative="#ef4444",
            warning="#f59e0b",
            info="#3b82f6",
        )

        with ui.header().classes("bg-primary text-white"):
            ui.label(config.title).classes("text-xl font-bold")
            with ui.row().classes("w-full justify-end items-center gap-4"):
                ui.label(f"Iteration: {state.current_iteration}").classes("text-sm")
                ui.label(f"Pending: {len(state.pending_proposals)}").classes(
                    "text-sm bg-yellow-100 text-yellow-800 px-2 py-1 rounded"
                )
                ui.label(f"Approved: {len(state.approved_proposals)}").classes(
                    "text-sm bg-green-100 text-green-800 px-2 py-1 rounded"
                )

        with ui.tabs().classes("w-full") as tabs:
            overview_tab = ui.tab("Overview")
            proposals_tab = ui.tab("Proposals")
            hypotheses_tab = ui.tab("Hypotheses")
            campaigns_tab = ui.tab("Campaigns")
            kb_tab = ui.tab("Knowledge Base")
            settings_tab = ui.tab("Settings")

        with ui.tab_panels(tabs, value=overview_tab).classes("w-full p-4"):
            with ui.tab_panel(overview_tab):
                await _render_overview(state)

            with ui.tab_panel(proposals_tab):
                await _render_proposals(state)

            with ui.tab_panel(hypotheses_tab):
                await _render_hypotheses(state)

            with ui.tab_panel(campaigns_tab):
                await _render_campaigns(state)

            with ui.tab_panel(kb_tab):
                await _render_knowledge_base(state)

            with ui.tab_panel(settings_tab):
                await _render_settings(state)

        # WebSocket endpoint for real-time updates
        if config.enable_websockets:

            @app.websocket("/ws")
            async def websocket_endpoint(websocket):
                await websocket.accept()
                state.connected_clients.add(websocket)
                try:
                    while True:
                        data = await websocket.receive_text()
                        msg = json.loads(data)
                        if msg.get("type") == "ping":
                            await websocket.send_text(json.dumps({"type": "pong"}))
                except Exception:  # ruff: ignore[try-except-pass]
                    pass
                finally:
                    state.connected_clients.discard(websocket)

    async def _render_overview(state: DashboardState):  # ruff: ignore[unused-async]
        """Render overview tab."""
        with ui.row().classes("w-full gap-4"):
            # Campaign summary card
            with ui.card().classes("flex-1"):
                ui.label("Campaign Summary").classes("text-lg font-semibold mb-2")
                summary = state.get_summary()
                camp = summary.get("campaign", {})
                with ui.column().classes("gap-1"):
                    ui.label(f"Campaign ID: {camp.get('campaign_id', 'N/A')}")
                    ui.label(f"Branch: {camp.get('branch_name', 'N/A')}")
                    ui.label(f"Total Experiments: {camp.get('total_experiments', 0)}")
                    ui.label(f"Best Accuracy: {camp.get('best_accuracy', 0):.4f}")

            # Quick stats
            with ui.card().classes("flex-1"):
                ui.label("Quick Stats").classes("text-lg font-semibold mb-2")
                with ui.row().classes("w-full gap-4"):
                    _stat_card(
                        "Pending Proposals", len(state.pending_proposals), "warning"
                    )
                    _stat_card("Approved", len(state.approved_proposals), "positive")
                    _stat_card("Rejected", len(state.rejected_proposals), "negative")
                    _stat_card("Iteration", state.current_iteration, "info")

            # Recent metrics chart placeholder
            with ui.card().classes("w-full"):
                ui.label("Recent Performance").classes("text-lg font-semibold mb-2")
                ui.label("(Metrics chart - connect to campaign data)").classes(
                    "text-gray-500"
                )

    async def _render_proposals(state: DashboardState):  # ruff: ignore[unused-async]
        """Render proposals tab with approval controls."""
        ui.label("Experiment Proposals").classes("text-xl font-semibold mb-4")

        # Pending proposals table
        ui.label("Pending Review").classes("text-lg font-medium mb-2")
        if state.pending_proposals:
            columns = [
                {"name": "id", "label": "ID", "field": "id"},
                {"name": "model", "label": "Model", "field": "model"},
                {"name": "task", "label": "Task", "field": "task"},
                {"name": "hypothesis", "label": "Hypothesis", "field": "hypothesis"},
                {"name": "propagator", "label": "Propagator", "field": "propagator"},
                {"name": "priority", "label": "Priority", "field": "priority"},
                {"name": "actions", "label": "Actions", "field": "actions"},
            ]
            rows = []
            for p in state.pending_proposals:
                rows.append({**p, "actions": p["id"]})

            table = ui.table(columns=columns, rows=rows, row_key="id").classes("w-full")

            # Add action buttons via slots
            table.add_slot(
                "body-cell-actions",
                """
                <q-td :props="props">
                    <q-btn size="sm" color="positive" label="Approve"
                        @click="$parent.$emit('approve', props.row.id)" />
                    <q-btn size="sm" color="negative" label="Reject"
                        @click="$parent.$emit('reject', props.row.id)" />
                    <q-btn size="sm" color="primary" label="Annotate"
                        @click="$parent.$emit('annotate', props.row.id)" />
                </q-td>
            """,
            )

            table.on("approve", lambda e: _handle_approve(state, e.args))
            table.on("reject", lambda e: _handle_reject(state, e.args))
            table.on("annotate", lambda e: _handle_annotate(state, e.args))
        else:
            ui.label("No pending proposals").classes("text-gray-500")

        # Approved proposals
        ui.label("Approved").classes("text-lg font-medium mt-6 mb-2")
        if state.approved_proposals:
            for p in state.approved_proposals[-10:]:
                with ui.card().classes("w-full mb-2"):
                    ui.label(f"{p['model']} on {p['task']} - {p['hypothesis'][:80]}...")
                    ui.label(
                        f"Approved by {p.get('approved_by', 'unknown')} at {p.get('approved_at', 'N/A')}"
                    ).classes("text-sm text-gray-500")
        else:
            ui.label("No approved proposals yet").classes("text-gray-500")

    async def _render_hypotheses(state: DashboardState):  # ruff: ignore[unused-async]
        """Render hypotheses tab."""
        ui.label("Generated Hypotheses").classes("text-xl font-semibold mb-4")

        if state.reasoner:  # ruff: ignore[too-many-nested-blocks]
            hypotheses = state.reasoner.get_top_hypotheses(20)
            for h in hypotheses:
                with ui.card().classes("w-full mb-2"):  # ruff: ignore[multiple-with-statements]
                    with ui.row().classes("w-full items-start gap-4"):
                        with ui.column().classes("flex-1"):
                            ui.label(h.statement).classes("font-medium")
                            if h.proposed_model:
                                ui.label(f"Model: {h.proposed_model}").classes(
                                    "text-sm text-gray-600"
                                )
                            if h.proposed_task:
                                ui.label(f"Task: {h.proposed_task}").classes(
                                    "text-sm text-gray-600"
                                )
                            if h.reasoning_chain:
                                with ui.expansion("Reasoning").classes("w-full"):
                                    for step in h.reasoning_chain:
                                        ui.label(f"• {step}").classes("text-sm")
                        with ui.column().classes("items-end"):
                            ui.label(f"Confidence: {h.confidence:.0%}").classes(
                                "text-lg font-bold "
                                + (
                                    "text-green-600"
                                    if h.confidence > 0.7
                                    else "text-yellow-600"
                                    if h.confidence > 0.4
                                    else "text-red-600"
                                )
                            )
                            ui.label(f"Source: {h.source}").classes(
                                "text-xs text-gray-500"
                            )
        else:
            ui.label("No reasoner available").classes("text-gray-500")

        # Reasoning chains
        if state.reasoner and state.reasoner.get_reasoning_history():
            ui.label("Reasoning Chains").classes("text-lg font-semibold mt-6 mb-2")
            for chain in state.reasoner.get_reasoning_history()[-5:]:
                with ui.expansion(
                    f"{chain.template.value}: {chain.conclusion[:60]}..."
                ).classes("w-full"):
                    for step in chain.steps:
                        ui.label(step).classes("text-sm font-mono")
                    ui.label(f"Confidence: {chain.confidence:.0%}").classes("text-sm")

    async def _render_campaigns(state: DashboardState):  # ruff: ignore[unused-async]
        """Render campaigns tab."""
        ui.label("Campaign Management").classes("text-xl font-semibold mb-4")

        if state.campaign:
            summary = state.campaign.get_summary()
            with ui.card().classes("w-full mb-4"):
                ui.label("Current Campaign").classes("text-lg font-semibold mb-2")
                with ui.row().classes("w-full gap-8"):
                    _info_card("Campaign ID", summary.get("campaign_id", "N/A"))
                    _info_card("Branch", summary.get("branch_name", "N/A"))
                    _info_card("Iterations", str(summary.get("iterations", 0)))
                    _info_card(
                        "Total Experiments", str(summary.get("total_experiments", 0))
                    )
                    _info_card("Completed", str(summary.get("completed", 0)))
                    _info_card(
                        "Best Accuracy", f"{summary.get('best_accuracy', 0):.4f}"
                    )

            # Branch management
            ui.label("Branch Operations").classes("text-lg font-semibold mt-6 mb-2")
            with ui.row().classes("gap-4"):
                new_branch = ui.input("New branch name").classes("w-64")
                ui.button(
                    "Create Branch",
                    on_click=lambda: _create_branch(state, new_branch.value),
                )
                ui.button("List Branches", on_click=lambda: _list_branches(state))

        # Iteration history
        ui.label("Iteration History").classes("text-lg font-semibold mt-6 mb-2")
        if state.campaign:  # ruff: ignore[too-many-nested-blocks]
            history = state.campaign.get_history()
            for it in history[-10:]:
                with ui.card().classes("w-full mb-2"):
                    ui.label(f"Iteration {it.iteration} - {it.timestamp}")
                    ui.label(
                        f"Proposals: {it.n_proposals} | Completed: {it.n_completed} | Failed: {it.n_failed}"
                    ).classes("text-sm")
                    if it.insights:
                        with ui.expansion("Insights").classes("w-full"):
                            for ins in it.insights[:3]:
                                ui.label(f"• {ins}").classes("text-sm")

    async def _render_knowledge_base(state: DashboardState):  # ruff: ignore[unused-async]
        """Render knowledge base tab."""
        ui.label("Knowledge Base").classes("text-xl font-semibold mb-4")

        if state.knowledge_base:
            stats = state.knowledge_base.get_stats()
            with ui.row().classes("w-full gap-4"):
                _stat_card("Total Entries", stats.get("total_entries", 0), "info")
                _stat_card("Experiments", stats.get("total_experiments", 0), "positive")
                _stat_card("Models", len(stats.get("by_model_family", {})), "warning")
                _stat_card("Vector Index", stats.get("vector_index_size", 0), "info")

            # Search
            ui.label("Semantic Search").classes("text-lg font-semibold mt-6 mb-2")
            search_input = ui.input(
                "Search query", placeholder="e.g., equilibrium propagation scaling"
            ).classes("w-96")
            ui.button("Search", on_click=lambda: _search_kb(state, search_input.value))

    async def _render_settings(state: DashboardState):  # ruff: ignore[unused-async]
        """Render settings tab."""
        ui.label("Settings").classes("text-xl font-semibold mb-4")

        with ui.card().classes("w-full max-w-2xl"):
            ui.label("Campaign Settings").classes("text-lg font-semibold mb-4")
            ui.checkbox(
                "Human approval gate",
                value=state.campaign.human_approval_gate if state.campaign else False,
            )
            ui.number(
                "Max concurrent experiments",
                value=state.campaign.max_concurrent if state.campaign else 1,
                min=1,
                max=10,
            )
            ui.number("Checkpoint interval", value=5, min=1, max=100)

            ui.separator().classes("my-4")

            ui.label("Dashboard Settings").classes("text-lg font-semibold mb-4")
            ui.select("Theme", ["light", "dark"], value=config.theme)
            ui.number("Port", value=config.port, min=1024, max=65535)
            ui.checkbox("Enable WebSockets", value=config.enable_websockets)

            ui.button(
                "Save Settings",
                color="primary",
                on_click=lambda: ui.notify("Settings saved"),
            )

    ui.run(
        host=config.host,
        port=config.port,
        title=config.title,
        favicon="🧬",
        dark=config.theme == "dark",
    )


def _stat_card(label: str, value: float, color: str):
    """Render a stat card."""
    with ui.card().classes("flex-1 min-w-[120px] p-4"):
        ui.label(str(value)).classes(f"text-3xl font-bold text-{color}-600")
        ui.label(label).classes("text-sm text-gray-500")


def _info_card(label: str, value: str):
    """Render an info card."""
    with ui.column().classes("gap-1"):
        ui.label(value).classes("text-lg font-semibold")
        ui.label(label).classes("text-sm text-gray-500")


async def _handle_approve(state: DashboardState, proposal_id: str):
    """Handle proposal approval."""
    if state.approve_proposal(proposal_id):
        await state.broadcast({"type": "proposal_approved", "proposal_id": proposal_id})
        ui.notify(f"Proposal {proposal_id} approved", type="positive")
    else:
        ui.notify("Proposal not found", type="negative")


async def _handle_reject(state: DashboardState, proposal_id: str):  # ruff: ignore[unused-async]
    """Handle proposal rejection."""
    # Show dialog for rejection reason
    with ui.dialog() as dialog, ui.card():
        ui.label("Reject Proposal").classes("text-lg font-semibold")
        reason = ui.textarea("Reason (optional)").classes("w-full")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dialog.close)
            ui.button(
                "Reject",
                color="negative",
                on_click=lambda: _confirm_reject(
                    state, proposal_id, reason.value, dialog
                ),
            )
    dialog.open()


async def _confirm_reject(state: DashboardState, proposal_id: str, reason: str, dialog):
    """Confirm rejection."""
    if state.reject_proposal(proposal_id, reason=reason):
        await state.broadcast({
            "type": "proposal_rejected",
            "proposal_id": proposal_id,
            "reason": reason,
        })
        ui.notify(f"Proposal {proposal_id} rejected", type="warning")
    dialog.close()


async def _handle_annotate(state: DashboardState, proposal_id: str):  # ruff: ignore[unused-async]
    """Handle proposal annotation."""
    with ui.dialog() as dialog, ui.card():
        ui.label("Add Annotation").classes("text-lg font-semibold")
        annotation_type = ui.select(
            "Type",
            ["note", "literature_link", "kb_link", "concern", "suggestion"],
            value="note",
        )
        content = ui.textarea("Content").classes("w-full")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dialog.close)
            ui.button(
                "Save",
                on_click=lambda: _save_annotation(
                    state, proposal_id, annotation_type.value, content.value, dialog
                ),
            )
    dialog.open()


async def _save_annotation(  # ruff: ignore[unused-async]
    state: DashboardState, proposal_id: str, ann_type: str, content: str, dialog
):
    """Save annotation."""
    if state.add_annotation(proposal_id, {"type": ann_type, "content": content}):
        ui.notify("Annotation added", type="positive")
    dialog.close()


def _create_branch(state: DashboardState, branch_name: str):
    """Create new campaign branch."""
    if branch_name and state.campaign:
        try:
            AutoScientistCampaign.create_branch(
                source_branch="main",
                new_branch=branch_name,
                db_path=state.campaign.db_path,
                knowledge_base=state.knowledge_base,
                output_dir=state.config.campaign_dir,
            )
            ui.notify(f"Branch '{branch_name}' created", type="positive")
        except Exception as e:
            ui.notify(f"Failed to create branch: {e}", type="negative")


def _list_branches(state: DashboardState):
    """List campaign branches."""
    if state.campaign:
        branches = list_branches(state.campaign.db_path)
        ui.notify(f"Branches: {', '.join(branches)}", type="info")


def _search_kb(state: DashboardState, query: str):
    """Search knowledge base."""
    if query and state.knowledge_base:
        results = state.knowledge_base.search(query, k=10)
        if results:
            ui.notify(f"Found {len(results)} results", type="info")
            for entry, score in results[:5]:
                ui.notify(
                    f"[{entry.model_family}] {entry.finding} ({score:.2f})", type="info"
                )
        else:
            ui.notify("No results found", type="warning")


def _run_fastapi_dashboard(state: DashboardState, config: DashboardConfig):  # ruff: ignore[complex-structure]
    """Run dashboard with FastAPI + simple HTML (fallback)."""
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        import uvicorn
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles  # ruff: ignore[unused-import]

        app = FastAPI(title=config.title)

        @app.get("/")
        async def root():
            return HTMLResponse(_get_dashboard_html(state, config))

        @app.get("/api/summary")
        async def api_summary():
            return state.get_summary()

        @app.get("/api/proposals")
        async def api_proposals():
            return {
                "pending": state.pending_proposals,
                "approved": state.approved_proposals,
                "rejected": state.rejected_proposals,
            }

        @app.post("/api/proposals/{proposal_id}/approve")
        async def api_approve(proposal_id: str):
            if state.approve_proposal(proposal_id):
                await state.broadcast({
                    "type": "proposal_approved",
                    "proposal_id": proposal_id,
                })
                return {"success": True}
            return {"success": False, "error": "Not found"}

        @app.post("/api/proposals/{proposal_id}/reject")
        async def api_reject(proposal_id: str, reason: str = ""):
            if state.reject_proposal(proposal_id, reason=reason):
                await state.broadcast({
                    "type": "proposal_rejected",
                    "proposal_id": proposal_id,
                    "reason": reason,
                })
                return {"success": True}
            return {"success": False, "error": "Not found"}

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            state.connected_clients.add(websocket)
            try:
                while True:
                    data = await websocket.receive_text()  # ruff: ignore[unused-variable]
            except WebSocketDisconnect:
                state.connected_clients.discard(websocket)

        logger.info(f"Starting FastAPI dashboard on {config.host}:{config.port}")
        uvicorn.run(app, host=config.host, port=config.port)

    except ImportError:
        logger.error(  # ruff: ignore[error-instead-of-exception]
            "Neither NiceGUI nor FastAPI available. Install with: pip install nicegui or pip install fastapi uvicorn"
        )
        raise


def _get_dashboard_html(state: DashboardState, config: DashboardConfig) -> str:
    """Generate basic HTML dashboard."""
    summary = state.get_summary()
    camp = summary.get("campaign", {})

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{config.title}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8fafc; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #2563eb; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: #2563eb; }}
        .stat-label {{ color: #64748b; font-size: 0.875rem; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px; }}
        .proposal {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }}
        .proposal-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
        .proposal-actions {{ display: flex; gap: 8px; }}
        .btn {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-weight: 500; }}
        .btn-approve {{ background: #10b981; color: white; }}
        .btn-reject {{ background: #ef4444; color: white; }}
        .btn-annotate {{ background: #3b82f6; color: white; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f8fafc; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{config.title}</h1>
            <p>Campaign: {camp.get("campaign_id", "N/A")} | Branch: {camp.get("branch_name", "N/A")} | Iteration: {state.current_iteration}</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(state.pending_proposals)}</div>
                <div class="stat-label">Pending Proposals</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #10b981;">{len(state.approved_proposals)}</div>
                <div class="stat-label">Approved</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #ef4444;">{len(state.rejected_proposals)}</div>
                <div class="stat-label">Rejected</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{state.current_iteration}</div>
                <div class="stat-label">Current Iteration</div>
            </div>
        </div>

        <div class="card">
            <h2>Pending Proposals</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th><th>Model</th><th>Task</th><th>Hypothesis</th><th>Propagator</th><th>Priority</th><th>Actions</th>
                    </tr>
                </thead>
                <tbody>
"""

    for p in state.pending_proposals:
        html += f"""
                    <tr>
                        <td>{p["id"]}</td>
                        <td>{p["model"]}</td>
                        <td>{p["task"]}</td>
                        <td>{p["hypothesis"][:80]}...</td>
                        <td>{p.get("propagator", "N/A")}</td>
                        <td>{p.get("priority", 0)}</td>
                        <td>
                            <button class="btn btn-approve" onclick="approve('{p["id"]}')">Approve</button>
                            <button class="btn btn-reject" onclick="reject('{p["id"]}')">Reject</button>
                            <button class="btn btn-annotate" onclick="annotate('{p["id"]}')">Annotate</button>
                        </td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Approved Proposals</h2>
"""

    for p in state.approved_proposals[-10:]:
        html += f"""
            <div class="proposal">
                <div class="proposal-header">
                    <strong>{p["model"]} on {p["task"]}</strong>
                    <span style="color: #10b981;">✓ Approved by {p.get("approved_by", "unknown")}</span>
                </div>
                <p>{p["hypothesis"]}</p>
            </div>
"""

    html += """
        </div>
    </div>

    <script>
        async function approve(id) {
            const res = await fetch('/api/proposals/' + id + '/approve', {method: 'POST'});
            if (res.ok) location.reload();
        }
        async function reject(id) {
            const reason = prompt('Rejection reason (optional):');
            const res = await fetch('/api/proposals/' + id + '/reject', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({reason: reason || ''})
            });
            if (res.ok) location.reload();
        }
        function annotate(id) {
            alert('Annotation feature requires NiceGUI. Install: pip install nicegui');
        }
    </script>
</body>
</html>
"""
    return html


class DashboardServer:
    """Async dashboard server wrapper."""

    def __init__(self, config: DashboardConfig | None = None):
        self.config = config or DashboardConfig()
        self.state = DashboardState(self.config)
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start the dashboard server."""
        self.state.initialize()
        self._task = asyncio.create_task(run_dashboard(self.config))
        await self._task

    async def stop(self):
        """Stop the dashboard server."""
        if self._task:
            self._task.cancel()
            try:  # ruff: ignore[suppressible-exception]
                await self._task
            except asyncio.CancelledError:
                pass


def create_dashboard(
    campaign_dir: str = "autoscientist_campaigns",
    host: str = "0.0.0.0",  # ruff: ignore[hardcoded-bind-all-interfaces]
    port: int = 8080,
    **kwargs,
) -> DashboardServer:
    """Factory function to create a dashboard server."""
    config = DashboardConfig(
        host=host,
        port=port,
        campaign_dir=campaign_dir,
        **kwargs,
    )
    return DashboardServer(config)


def main():
    """CLI entry point for running the dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="AutoScientist Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")  # ruff: ignore[hardcoded-bind-all-interfaces]
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument(
        "--campaign-dir", default="autoscientist_campaigns", help="Campaign directory"
    )
    parser.add_argument(
        "--theme", choices=["light", "dark"], default="light", help="UI theme"
    )

    args = parser.parse_args()

    config = DashboardConfig(
        host=args.host,
        port=args.port,
        campaign_dir=args.campaign_dir,
        theme=args.theme,
    )

    asyncio.run(run_dashboard(config))


if __name__ == "__main__":
    main()
