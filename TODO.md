# Bioplausible UI Redesign - Complete Plan

> **Goal**: Replace `bioplausible_ui/` with an intuitive, testable, component-based dual-app architecture using metaprogramming abstractions.

---

## Design Principles

1. **Backend-first** - All training logic testable without PyQt
2. **Metaprogramming** - Generate UI from declarative schemas
3. **Zero duplication** - Abstract common patterns into base classes
4. **Complete coverage** - Support ALL current functionality
5. **Separation of concerns** - Production UI vs experimental tools

---

## Current Functionality Inventory

### Main Dashboard Tabs (11 total)
- **Vision Tab** (722 LOC) - Image tasks with Dream/Oracle/Alignment/Robustness/Cube tools
- **LM Tab** (477 LOC) - Language modeling with text generation
- **RL Tab** (392 LOC) - Reinforcement learning (CartPole)
- **Diffusion Tab** (308 LOC) - Diffusion models
- **Discovery Tab** (458 LOC) - Architecture search space visualization + P2P network
- **Micro scope Tab** (343 LOC) - Live dynamics visualization
- **P2P Tab** (435 LOC) - Distributed training network
- **Deploy Tab** (261 LOC) - ONNX/TorchScript export + inference server
- **Benchmarks Tab** (342 LOC) - Validation track runner
- **Console Tab** (127 LOC) - System diagnostics + Python REPL

### Separate Apps
- **Hyperopt Dashboard** (951 LOC) - Standalone hyperparameter search with Pareto plots

---

## Architecture: Dual Apps + Metaprogramming

### Metaprogramming Abstractions

#### 1. Declarative Tab Schema
```python
# bioplausible_ui/core/schema.py
@dataclass
class TabSchema:
    """Declarative tab definition."""
    name: str
    task: str  # "vision", "lm", "rl", "diffusion"
    widgets: List[WidgetDef]
    actions: List[ActionDef]
    plots: List[PlotDef]

# Example usage:
TRAIN_TAB_SCHEMA = TabSchema(
    name="Train",
    task="configurable",
    widgets=[
        WidgetDef("task_selector", TaskSelector),
        WidgetDef("dataset_picker", DatasetPicker, task="@task_selector.value"),
        WidgetDef("model_selector", ModelSelector, task="@task_selector.value")
,
        WidgetDef("hyperparam_editor", HyperparamEditor, model="@model_selector.value", visible="@advanced_mode"),
    ],
    actions=[
        ActionDef("start_training", icon="▶", callback="_start_training"),
        ActionDef("stop_training", icon="⏹", callback="_stop_training"),
    ],
    plots=[
        PlotDef("loss", xlabel="Epoch", ylabel="Loss"),
        PlotDef("accuracy", xlabel="Epoch", ylabel="Accuracy"),
    ]
)
```

#### 2. Metaclass for Auto-wiring
```python
# bioplausible_ui/core/base.py
class TabMeta(type(QWidget)):
    """Metaclass that auto-wires signals/slots from schema."""
    
    def __new__(mcs, name, bases, dct):
        if 'SCHEMA' in dct:
            schema = dct['SCHEMA']
            # Auto-generate __init__ from schema
            dct['__init__'] = mcs._generate_init(schema)
            # Auto-generate widget accessor properties
            for widget_def in schema.widgets:
                dct[widget_def.name] = property(lambda self, w=widget_def.name: self._widgets[w])
        
        return super().__new__(mcs, name, bases, dct)
    
    @staticmethod
    def _generate_init(schema):
        def __init__(self, parent=None):
            QWidget.__init__(self, parent)
            self._widgets = {}
            self._build_from_schema(schema)
        return __init__

class BaseTab(QWidget, metaclass=TabMeta):
    """Base class for all tabs - UI built from schema."""
    
    SCHEMA = None  # Override in subclasses
    
    def _build_from_schema(self, schema):
        """Build UI from declarative schema."""
        layout = QVBoxLayout(self)
        
        # Create widgets
        for widget_def in schema.widgets:
            widget = widget_def.widget_class(**self._resolve_params(widget_def.params))
            self._widgets[widget_def.name] = widget
            
            # Auto-connect signals if specified
            if widget_def.signal:
                signal = getattr(widget, widget_def.signal)
                callback = getattr(self, widget_def.callback)
                signal.connect(callback)
            
            layout.addWidget(widget)
        
        # Create action buttons
        if schema.actions:
            buttons = QHBoxLayout()
            for action_def in schema.actions:
                btn = QPushButton(action_def.icon + " " + action_def.name)
                btn.clicked.connect(getattr(self, action_def.callback))
                buttons.addWidget(btn)
            layout.addLayout(buttons)
        
        # Create plots
        for plot_def in schema.plots:
            plot = PlotWidget()
            plot.setLabel('bottom', plot_def.xlabel)
            plot.setLabel('left', plot_def.ylabel)
            self._widgets[f"plot_{plot_def.name}"] = plot
            layout.addWidget(plot)
```

#### 3. Registry Pattern for Tools
```python
# bioplausible_ui/lab/registry.py
class ToolRegistry:
    """Registry for lab analysis tools."""
    _tools = {}
    
    @classmethod
    def register(cls, name, requires=None):
        """Decorator to register analysis tools."""
        def decorator(tool_class):
            cls._tools[name] = {
                'class': tool_class,
                'requires': requires or []  # Model capabilities required
            }
            return tool_class
        return decorator
    
    @classmethod
    def get_compatible_tools(cls, model):
        """Get tools compatible with given model."""
        spec = get_model_spec(model)
        compatible = []
        for name, info in cls._tools.items():
            if all(getattr(spec, f"supports_{req}", False) for req in info['requires']):
                compatible.append(name)
        return compatible

# Usage:
@ToolRegistry.register("microscope", requires=["dynamics_tracking"])
class MicroscopeTool(BaseTool):
    """Live dynamics visualization."""
    pass

@ToolRegistry.register("dreaming", requires=["dreaming"])
class DreamingTool(BaseTool):
    """Network inversion."""
    pass
```

---

## Implementation

### Phase 1: Backend Pipeline

**Create**: `bioplausible/pipeline/`

**Key Innovation**: Event-driven architecture
```python
class TrainingSession:
    """Generator-based training with events."""
    
    def start(self) -> Generator[Event, None, None]:
        """Yield events: ProgressEvent, MetricsEvent, CompletedEvent."""
        for epoch in range(self.config.epochs):
            metrics = self.trainer.train_epoch()
            yield ProgressEvent(epoch=epoch, metrics=metrics)
        yield CompletedEvent(final_metrics=self.get_metrics())
```

**Benefits**:
- ✅ No polling, pure push
- ✅ Natural pause/resume
- ✅ Testable without mocks

---

### Phase 2: Core Abstractions

**Create**: `bioplausible_ui/core/`

#### Meta-Components (8 base widgets)
| Component | Abstraction Level | Used By |
|-----------|-------------------|---------|
| `BaseConfigWidget` | Abstract | Subclassed by TaskSelector, DatasetPicker, ModelSelector |
| `BaseDynamicForm` | Abstract | Subclassed by HyperparamEditor |
| `BasePresetWidget` | Abstract | Subclassed by PresetSelector |
| `BaseMetricsWidget` | Abstract | Subclassed by MetricsDisplay |
| `BasePlotWidget` | Concrete | Used directly everywhere |
| `BaseProgressWidget` | Concrete | Used directly in all training tabs |
| `BaseControlPanel` | Abstract | Subclassed by all control panels |
| `BaseAnalysisPanel` | Abstract | Subclassed by analysis tools |

**Code Reduction**:
- Before: 8 independent widgets × 100 LOC = 800 LOC
- After: 3 abstract base classes × 50 LOC + 5 concrete implementations × 20 LOC = 250 LOC
- **Savings**: 69%

---

### Phase 3: Main App (biopl)

**Directory**:
```
bioplausible_ui/app/
├── main.py              # Entry point
├── window.py            # AppMainWindow (uses TabRegistry)
├── schemas/             # Declarative tab schemas
│   ├── train.py
│   ├── compare.py
│   ├── search.py
│   ├── results.py
│   └── settings.py
└── tabs/
    ├── __init__.py
    ├── train_tab.py     # SCHEMA = TRAIN_TAB_SCHEMA
    ├── compare_tab.py   # SCHEMA = COMPARE_TAB_SCHEMA
    ├── search_tab.py    # SCHEMA = SEARCH_TAB_SCHEMA
    ├── results_tab.py   # SCHEMA = RESULTS_TAB_SCHEMA
    └── settings_tab.py  # SCHEMA = SETTINGS_TAB_SCHEMA
```

**Tab Implementation** (using metaprogramming):
```python
# bioplausible_ui/app/tabs/train_tab.py
from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.app.schemas.train import TRAIN_TAB_SCHEMA

class TrainTab(BaseTab):
    """Training tab - UI auto-generated from schema."""
    
    SCHEMA = TRAIN_TAB_SCHEMA
    
    # Only need to implement callbacks - UI is automatic!
    def _start_training(self):
        config = TrainingConfig(
            task=self.task_selector.get_task(),
            dataset=self.dataset_picker.get_dataset(),
            model=self.model_selector.get_selected_model(),
            **self.hyperparam_editor.get_values()
        )
        self.bridge = SessionBridge(config)
        self.bridge.progress_updated.connect(self._on_progress)
        self.bridge.start()
    
    def _stop_training(self):
        self.bridge.stop()
    
    def _on_progress(self, epoch, metrics):
        self.plot_loss.add_point(epoch, metrics['loss'])
        self.plot_accuracy.add_point(epoch, metrics['accuracy'])
```

**Code Reduction**:
- Before: 722 LOC (vision_tab.py)
- After: ~100 LOC (30 LOC schema + 70 LOC callbacks)
- **Savings**: 86%

---

### Phase 4: Lab App (biopl-lab)

**Directory**:
```
bioplausible_ui/lab/
├── main.py              # Entry point + CLI arg parsing
├── window.py            # LabMainWindow (uses ToolRegistry)
├── registry.py          # ToolRegistry metaclass
└── tools/
    ├── __init__.py
    ├── base.py          # BaseTool abstract class
    ├── microscope.py    # @ToolRegistry.register("microscope")
    ├── dreaming.py      # @ToolRegistry.register("dreaming")
    ├── oracle.py        # @ToolRegistry.register("oracle")
    ├── alignment.py     # @ToolRegistry.register("alignment")
    ├── robustness.py    # @ToolRegistry.register("robustness")
    ├── cube_viz.py      # @ToolRegistry.register("cube_viz")
    └── p2p_grid.py      # @ToolRegistry.register("p2p_grid")
```

**Auto-Discovery Pattern**:
```python
# bioplausible_ui/lab/window.py
class LabMainWindow(QMainWindow):
    def __init__(self, model_path=None):
        super().__init__()
        self.model = None
        self.tabs = QTabWidget()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path):
        self.model = torch.load(path)
        spec = get_model_spec(self.model)
        
        # Auto-populate tabs based on model capabilities
        compatible_tools = ToolRegistry.get_compatible_tools(spec)
        
        for tool_name in compatible_tools:
            tool_class = ToolRegistry.get_tool(tool_name)
            tool_instance = tool_class(model=self.model)
            self.tabs.addTab(tool_instance, tool_instance.ICON + " " + tool_name)
```

**Benefits**:
- ✅ Tools automatically appear/disappear based on model capabilities
- ✅ New tools added by simply decorating with `@ToolRegistry.register`
- ✅ Zero manual tab management

---

### Phase 5: Additional Features

#### Benchmarks Integration
```python
# bioplausible_ui/app/tabs/benchmarks_tab.py
class BenchmarksTab(BaseTab):
    """Validation track runner."""
    
    SCHEMA = BenchmarksTabSchema(
        widgets=[
            WidgetDef("track_table", TrackTable),
            WidgetDef("filter_input", QLineEdit, placeholder="Filter tracks..."),
        ],
        actions=[
            ActionDef("run_selected", "Run Selected", "_run_selected"),
            ActionDef("run_all", "Run All", "_run_all"),
        ]
    )
    
    def _run_selected(self):
        selected_tracks = self.track_table.get_selected()
        worker = BenchmarkWorker(selected_tracks)
        worker.progress.connect(self._on_progress)
        worker.start()
```

#### Deploy Tools
```python
# bioplausible_ui/app/tabs/deploy_tab.py
class DeployTab(BaseTab):
    """Model export and deployment."""
    
    SCHEMA = DeployTabSchema(
        widgets=[
            WidgetDef("format_selector", QComboBox, items=["ONNX", "TorchScript", "TFLite"]),
            WidgetDef("server_toggle", QCheckBox, text="Run Inference Server"),
        ],
        actions=[
            ActionDef("export", "📥 Export", "_export_model"),
            ActionDef("serve", "🌐 Start Server", "_start_server"),
        ]
    )
```

#### Console/Diagnostics
```python
# bioplausible_ui/app/tabs/console_tab.py
class ConsoleTab(BaseTab):
    """System diagnostics + Python REPL."""
    
    SCHEMA = ConsoleTabSchema(
        widgets=[
            WidgetDef("console_output", QTextEdit, readonly=True),
            WidgetDef("command_input", QLineEdit),
        ],
        actions=[
            ActionDef("check_status", "🩺 Diagnostics", "_check_status"),
        ]
    )
    
    COMMANDS = {
        "!status": "_check_status",
        "!cuda": "_check_cuda",
        "!model": "_show_model",
    }
```

---

## Complete Feature Matrix

| Feature | Current Location | New Location | Implementation |
|---------|-----------------|--------------|----------------|
| **Training** |
| Vision training | vision_tab.py | biopl/tabs/train_tab.py | Schema-based |
| LM training | lm_tab.py | biopl/tabs/train_tab.py | Schema-based |
| RL training | rl_tab.py | biopl/tabs/train_tab.py | Schema-based |
| Diffusion training | diffusion_tab.py | biopl/tabs/train_tab.py | Schema-based |
| **Comparison** |
| Algorithm comparison | N/A | biopl/tabs/compare_tab.py | New feature |
| Benchmark tracks | benchmarks_tab.py | biopl/tabs/benchmarks_tab.py | Migrated |
| **Optimization** |
| Hyperparameter search | hyperopt_dashboard.py | biopl/tabs/search_tab.py | Integrated |
| Architecture search (P2P) | p2p_tab.py + discovery_tab.py | biopl/tabs/search_tab.py | Unified |
| **Analysis Tools** |
| Live dynamics | microscope_tab.py | biopl-lab/tools/microscope.py | Registry-based |
| Dreaming | vision_specialized_components.py | biopl-lab/tools/dreaming.py | Registry-based |
| Oracle | vision_specialized_components.py | biopl-lab/tools/oracle.py | Registry-based |
| Alignment | vision_specialized_components.py | biopl-lab/tools/alignment.py | Registry-based |
| Robustness | vision_specialized_components.py | biopl-lab/tools/robustness.py | Registry-based |
| Cube visualization | vision_specialized_components.py | biopl-lab/tools/cube_viz.py | Registry-based |
| P2P network viz | discovery_tab.py | biopl-lab/tools/p2p_grid.py | Registry-based |
| **Deployment** |
| ONNX export | deploy_tab.py | biopl/tabs/deploy_tab.py | Schema-based |
| TorchScript export | deploy_tab.py | biopl/tabs/deploy_tab.py | Schema-based |
| Inference server | deploy_tab.py | biopl/tabs/deploy_tab.py | Schema-based |
| **Utilities** |
| Console/REPL | console_tab.py | biopl/tabs/console_tab.py | Schema-based |
| System diagnostics | console_tab.py | biopl/tabs/console_tab.py | Command registry |
| Results browser | N/A | biopl/tabs/results_tab.py | New feature |
| Settings | dashboard.py | biopl/tabs/settings_tab.py | Extracted |

---

## Testing Strategy

### Metaprogramming Tests
```python
# bioplausible_ui/tests/test_schema.py
def test_tab_generation_from_schema():
    """Test schema-based tab generation."""
    schema = TabSchema(
        name="Test",
        widgets=[WidgetDef("selector", QComboBox)],
        actions=[ActionDef("test", "Test", "_test")]
    )
    
    tab = BaseTab()
    tab.SCHEMA = schema
    tab._build_from_schema(schema)
    
    assert hasattr(tab, 'selector')
    assert tab.selector.count() == 0
```

### Component Tests
```python
# bioplausible_ui/tests/test_components.py
def test_model_selector_auto_filters(qtbot):
    """Test auto-filtering based on task."""
    selector = ModelSelector(task="vision")
    qtbot.addWidget(selector)
    
    # Family change triggers auto-population
    selector.family_combo.setCurrentText("EqProp")
    
    models = [selector.model_combo.itemText(i) 
              for i in range(selector.model_combo.count())]
    assert all("EqProp" in m or "Conv" in m for m in models)
```

### Integration Tests
```python
# bioplausible_ui/tests/test_app.py
def test_full_training_workflow(qtbot, mocker):
    """Test complete training workflow."""
    window = AppMainWindow()
    train_tab = window.tabs.widget(0)  # Train tab
    
    # Mock backend
    mock_session = mocker.patch("bioplausible.pipeline.TrainingSession")
    
    # Use UI
    train_tab.dataset_picker.set_dataset("mnist")
    train_tab.start_btn.click()
    
    assert mock_session.called
```

---

## Changelog

### Removed
- ❌ `dashboard.py` (1883 LOC) - Replaced by `app/window.py`
- ❌ 11 individual tab files (~4000 LOC) - Replaced by 5 schema-based tabs
- ❌ `hyperopt_dashboard.py` (951 LOC) - Integrated into `app/tabs/search_tab.py`
- ❌ `vision_specialized_components.py` (685 LOC) - Split into `lab/tools/`

### Added
- ✅ `bioplausible/pipeline/` - Backend training abstraction
- ✅ `bioplausible_ui/core/schema.py` - Declarative UI schemas
- ✅ `bioplausible_ui/core/base.py` - Metaclass for auto-wiring
- ✅ `bioplausible_ui/lab/registry.py` - Tool auto-discovery
- ✅ `bioplausible_ui/app/` - New main app
- ✅ `bioplausible_ui/lab/` - New lab app

### Code Metrics
| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Main window | 1883 | ~150 | 92% |
| Vision tab | 722 | ~100 | 86% |
| LM tab | 477 | (shared train_tab) | 100% |
| Hyperopt | 951 | ~250 | 74% |
| Total UI code | ~8000 | ~2000 | 75% |
| **Unique widgets** | ~20 | **8** | **60%** |

---

## Implementation Checklist

### Phase 1: Backend Pipeline
- [ ] `bioplausible/pipeline/session.py`
- [ ] `bioplausible/pipeline/config.py`
- [ ] `bioplausible/pipeline/events.py`
- [ ] Integration tests

### Phase 2: Core Abstractions
- [ ] `core/schema.py` - Schema definitions
- [ ] `core/base.py` - Metaclasses
- [ ] `core/widgets/` - 8 base components
- [ ] `core/bridge.py` - Qt adapter
- [ ] Component tests

### Phase 3: Main App
- [ ] `app/window.py`
- [ ] `app/schemas/` - 5 tab schemas
- [ ] `app/tabs/` - 5 tab implementations
- [ ] UI integration tests

### Phase 4: Lab App
- [ ] `lab/window.py`
- [ ] `lab/registry.py`
- [ ] `lab/tools/base.py`
- [ ] `lab/tools/` - 7 tool implementations
- [ ] Tool integration tests

### Phase 5: Migration
- [ ] Update `pyproject.toml` entry points
- [ ] Archive old code
- [ ] Update README
- [ ] Full regression test

---

## Entry Points

```toml
[project.scripts]
biopl = "bioplausible_ui.app.main:main"
biopl-lab = "bioplausible_ui.lab.main:main"
eqprop-dashboard = "bioplausible_ui.app.main:main"  # backwards compat
```

---

## Key Innovations

1. **Schema-Driven UI** - Tabs defined declaratively, UI auto-generated
2. **Metaclass Magic** - Auto-wiring of signals/slots from schema
3. **Tool Registry** - Lab tools auto-discover based on model capabilities
4. **Event Streaming** - Generator-based training, no polling
5. **Zero Duplication** - Abstract base classes, concrete implementations

**Result**: 75% code reduction while supporting 100% of current functionality.
