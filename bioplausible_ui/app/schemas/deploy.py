from bioplausible_ui.core.schema import TabSchema, WidgetDef, ActionDef
from bioplausible_ui.core.widgets.task_selector import TaskSelector
from bioplausible_ui.core.widgets.model_selector import ModelSelector
from bioplausible_ui.core.widgets.export_format_selector import ExportFormatSelector
from bioplausible_ui.core.widgets.run_selector import RunSelector

DEPLOY_TAB_SCHEMA = TabSchema(
    name="Deploy",
    widgets=[
        WidgetDef("run_selector", RunSelector),
        WidgetDef("format_selector", ExportFormatSelector),
        # Optional: Keep these for "New Model" export if needed, or remove.
        # Removing task/model selector simplifies flow: Deploy = Deploy trained run.
        # But if user wants to export architecture without training, they are stuck.
        # Let's keep them as "Alternative" or separate section?
        # TabSchema doesn't support sections well yet.
        # Let's rely on RunSelector for now as it's the primary "Exploit" use case.
    ],
    actions=[
        ActionDef("export", "📦", "_export_model", style="primary"),
        ActionDef("serve", "🚀", "_serve_model", style="success"),
        ActionDef("refresh", "🔄", "_refresh_runs", style="secondary"),
    ],
    plots=[]
)
