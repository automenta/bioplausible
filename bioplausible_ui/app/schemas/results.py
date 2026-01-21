from bioplausible_ui.core.schema import TabSchema, WidgetDef, ActionDef
from bioplausible_ui.core.widgets.results_table import ResultsTable

RESULTS_TAB_SCHEMA = TabSchema(
    name="Results",
    widgets=[
        WidgetDef("results_table", ResultsTable),
    ],
    actions=[
        ActionDef("refresh", "🔄", "_refresh_results"),
        ActionDef("export", "📤 Export", "_export_run"),
        ActionDef("import", "📥 Import", "_import_run"),
        ActionDef("delete", "🗑", "_delete_run", style="danger"),
    ],
    plots=[]
)
