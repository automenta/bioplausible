from bioplausible_ui.core.schema import TabSchema, WidgetDef, ActionDef
from bioplausible_ui.core.widgets.task_selector import TaskSelector
from bioplausible_ui.core.widgets.model_selector import ModelSelector
from bioplausible_ui.core.widgets.export_format_selector import ExportFormatSelector

DEPLOY_TAB_SCHEMA = TabSchema(
    name="Deploy",
    widgets=[
        WidgetDef("task_selector", TaskSelector),
        WidgetDef("model_selector", ModelSelector, bindings={"task": "@task_selector.value"}),
        WidgetDef("format_selector", ExportFormatSelector),
    ],
    actions=[
        ActionDef("export", "📦", "_export_model", style="primary"),
        ActionDef("serve", "🚀", "_serve_model", style="success"),
    ],
    plots=[]
)
