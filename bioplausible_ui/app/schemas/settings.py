from bioplausible_ui.core.schema import TabSchema, WidgetDef, ActionDef
from bioplausible_ui.core.widgets.hyperparam_editor import HyperparamEditor

SETTINGS_TAB_SCHEMA = TabSchema(
    name="Settings",
    widgets=[
        WidgetDef("preferences", HyperparamEditor, params={"defaults": {"theme": "dark", "auto_save": True}}),
    ],
    actions=[
        ActionDef("save", "💾", "_save_settings", style="primary"),
        ActionDef("reset", "🔄", "_reset_settings"),
    ],
    plots=[]
)
