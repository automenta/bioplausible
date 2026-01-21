"""
Training utilities for Bioplausible Trainer UI.

This module contains common training functionality that can be reused across different tabs.
"""

from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING
import torch
from torch.utils.data import DataLoader
from PyQt6.QtWidgets import QWidget, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar, QLabel, QMessageBox
from PyQt6.QtCore import pyqtSignal

if TYPE_CHECKING:
    from bioplausible.models.registry import ModelSpec
    from bioplausible_ui_old.worker import TrainingWorker, RLWorker


def create_vision_model_and_loader(vis_tab: QWidget) -> Tuple[Optional[Any], Optional[DataLoader]]:
    """
    Create vision model and data loader.

    Args:
        vis_tab: Vision tab instance containing UI controls

    Returns:
        Tuple of (model, train_loader) or (None, None) if creation fails
    """
    from bioplausible.datasets import get_vision_dataset
    from torch.utils.data import DataLoader
    from bioplausible.models.registry import get_model_spec
    from bioplausible.models.factory import create_model

    # Get dataset
    dataset_name = vis_tab.vis_dataset_combo.currentText().lower().replace('-', '_')

    # Determine flattening based on model type
    model_name = vis_tab.vis_model_combo.currentText()
    try:
        spec = get_model_spec(model_name)
        use_flatten = spec.model_type != "modern_conv_eqprop"
    except Exception as e:
        _log_error(f"Could not get model spec for {model_name}: {e}", vis_tab)
        use_flatten = True

    try:
        train_data = get_vision_dataset(dataset_name, train=True, flatten=use_flatten)
        train_loader = DataLoader(train_data, batch_size=vis_tab.vis_batch_spin.value(), shuffle=True)
    except Exception as e:
        _log_error(f"Could not load dataset {dataset_name}: {e}", vis_tab)
        return None, None

    # Create model
    hidden = vis_tab.vis_hidden_spin.value()

    try:
        spec = get_model_spec(model_name)

        # Determine input_dim based on dataset
        if 'MNIST' in vis_tab.vis_dataset_combo.currentText():
            input_dim = 784 if use_flatten else 1
        else:  # CIFAR-10
            input_dim = 3072 if use_flatten else 3

        # Map combo text to internal string
        grad_text = vis_tab.vis_grad_combo.currentText()
        if "BPTT" in grad_text:
            grad_method = "bptt"
        elif "Equilibrium" in grad_text:
            grad_method = "equilibrium"
        elif "Contrastive" in grad_text:
            grad_method = "contrastive"
        else:
            grad_method = "bptt"

        model = create_model(
            spec=spec,
            input_dim=input_dim,
            output_dim=10,
            hidden_dim=hidden,
            device="cuda" if torch.cuda.is_available() else "cpu",
            task_type="vision",
            gradient_method=grad_method
        )

        # Update step if spin box is used
        if hasattr(model, 'max_steps'):
            model.max_steps = vis_tab.vis_steps_spin.value()
        elif hasattr(model, 'eq_steps'):
            model.eq_steps = vis_tab.vis_steps_spin.value()

    except Exception as e:
        error_msg = f"Could not create {model_name}: {e}"
        _log_error(error_msg, vis_tab)
        QMessageBox.warning(vis_tab, "Model Creation Failed", error_msg)
        return None, None

    return model, train_loader


def create_lm_model_and_loader(lm_tab: QWidget) -> Tuple[Optional[Any], Optional[DataLoader]]:
    """
    Create language model and data loader.

    Args:
        lm_tab: LM tab instance containing UI controls

    Returns:
        Tuple of (model, train_loader) or (None, None) if creation fails
    """
    from bioplausible.datasets import get_lm_dataset
    from torch.utils.data import DataLoader
    from bioplausible.models.registry import get_model_spec
    from bioplausible.models.factory import create_model

    model_name = lm_tab.lm_model_combo.currentText()

    try:
        # Get dataset
        dataset_name = lm_tab.lm_dataset_combo.currentText()
        seq_len = lm_tab.lm_seqlen_spin.value()

        dataset = get_lm_dataset(dataset_name, seq_len=seq_len, split='train')
        vocab_size = dataset.vocab_size if hasattr(dataset, 'vocab_size') else 256

        spec = get_model_spec(model_name)

        model = create_model(
            spec=spec,
            input_dim=None,  # Uses embedding
            output_dim=vocab_size,
            hidden_dim=lm_tab.lm_hidden_spin.value(),
            num_layers=lm_tab.lm_layers_spin.value(),
            device="cuda" if torch.cuda.is_available() else "cpu",
            task_type="lm"
        )

        # Apply steps
        if hasattr(model, 'max_steps'):
            model.max_steps = lm_tab.lm_steps_spin.value()
        elif hasattr(model, 'eq_steps'):
            model.eq_steps = lm_tab.lm_steps_spin.value()

        train_loader = DataLoader(dataset, batch_size=lm_tab.lm_batch_spin.value(), shuffle=True)
        return model, train_loader

    except Exception as e:
        error_msg = f"Failed to create LM model: {e}"
        _log_error(error_msg, lm_tab)
        import traceback
        traceback.print_exc()
        return None, None


def create_diffusion_model_and_loader(diff_tab: QWidget) -> Tuple[Optional[Any], Optional[DataLoader]]:
    """
    Create diffusion model and data loader.

    Args:
        diff_tab: Diffusion tab instance containing UI controls

    Returns:
        Tuple of (model, train_loader) or (None, None) if creation fails
    """
    from bioplausible.datasets import get_vision_dataset
    from torch.utils.data import DataLoader
    import torch
    from bioplausible.models.registry import get_model_spec
    from bioplausible.models.factory import create_model

    try:
        # Dataset (MNIST) - must be unflattened [C, H, W]
        train_data = get_vision_dataset('mnist', train=True, flatten=False)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

        # Model
        spec = get_model_spec("EqProp Diffusion")
        hidden = diff_tab.hidden_spin.value()

        model = create_model(
            spec=spec,
            input_dim=1,  # Channels
            output_dim=1,
            hidden_dim=hidden,
            device="cuda" if torch.cuda.is_available() else "cpu",
            task_type="vision"
        )
        return model, train_loader
    except Exception as e:
        error_msg = f"Failed to create Diffusion model: {e}"
        _log_error(error_msg, diff_tab)
        import traceback
        traceback.print_exc()
        return None, None


def start_training_common(
    model: Any,
    train_loader: DataLoader,
    tab_instance: QWidget,
    tab_name: str,
    hyperparam_widgets: Optional[Dict[str, Any]] = None,
    microscope_check: Optional[Any] = None,
    compile_check: Optional[Any] = None,
    kernel_check: Optional[Any] = None,
    epochs_spin: Optional[Any] = None,
    lr_spin: Optional[Any] = None,
    progress_bar: Optional[QProgressBar] = None,
    status_label: Optional[QLabel] = None,
    worker_class: Any = None
) -> Optional[Any]:
    """
    Common method to start training for different model types.

    Args:
        model: The model to train
        train_loader: Training data loader
        tab_instance: Tab instance containing UI controls
        tab_name: Name of the tab (for UI updates)
        hyperparam_widgets: Dictionary of hyperparameter widgets
        microscope_check: Microscope checkbox widget
        compile_check: Compile checkbox widget
        kernel_check: Kernel checkbox widget
        epochs_spin: Epochs spin box
        lr_spin: Learning rate spin box
        progress_bar: Progress bar widget
        status_label: Status label widget
        worker_class: Class of the worker to use

    Returns:
        Training worker instance or None if failed
    """
    from bioplausible_ui_old.generation import count_parameters, format_parameter_count
    from bioplausible_ui_old.worker import TrainingWorker
    import time

    # Set default worker class if not provided
    if worker_class is None:
        worker_class = TrainingWorker

    # Clear history
    parent = tab_instance.parent()
    if parent:
        if hasattr(parent, 'loss_history'):
            parent.loss_history.clear()
        if hasattr(parent, 'acc_history'):
            parent.acc_history.clear()
        if hasattr(parent, 'lipschitz_history'):
            parent.lipschitz_history.clear()

    # Get hyperparameters
    hyperparams = {}
    if hyperparam_widgets:
        hyperparams = _get_current_hyperparams(hyperparam_widgets)

    # Update parameter count
    param_label_attr = f'{tab_name}_param_label'
    if hasattr(tab_instance, param_label_attr):
        param_label = getattr(tab_instance, param_label_attr)
        count = count_parameters(model)
        param_label.setText(f"Parameters: {format_parameter_count(count)}")

    # Determine microscope interval
    micro_interval = 0
    if microscope_check:
        micro_interval = 1 if microscope_check.isChecked() else 0

    # Determine compile and kernel options
    use_compile = compile_check.isChecked() if compile_check else False
    use_kernel = kernel_check.isChecked() if kernel_check else False

    # Create and start worker
    worker = worker_class(
        model,
        train_loader,
        epochs=epochs_spin.value() if epochs_spin else 10,
        lr=lr_spin.value() if lr_spin else 0.001,
        use_compile=use_compile,
        use_kernel=use_kernel,
        hyperparams=hyperparams,
        microscope_interval=micro_interval,
    )

    # Connect signals
    worker.progress.connect(lambda m: _on_progress(m, tab_instance, progress_bar, status_label))
    worker.finished.connect(lambda r: _on_finished(r, tab_instance, status_label))
    worker.error.connect(lambda e: _on_error(e, tab_instance, status_label))

    # Update UI
    train_btn_attr = f'{tab_name}_train_btn'
    stop_btn_attr = f'{tab_name}_stop_btn'
    pause_btn_attr = f'{tab_name}_pause_btn'

    if hasattr(tab_instance, train_btn_attr):
        train_btn = getattr(tab_instance, train_btn_attr)
        train_btn.setEnabled(False)
    if hasattr(tab_instance, stop_btn_attr):
        stop_btn = getattr(tab_instance, stop_btn_attr)
        stop_btn.setEnabled(True)
    if hasattr(tab_instance, pause_btn_attr):
        pause_btn = getattr(tab_instance, pause_btn_attr)
        pause_btn.setEnabled(True)
        pause_btn.clicked.connect(lambda: _toggle_pause(tab_instance))

    if progress_bar and epochs_spin:
        progress_bar.setMaximum(epochs_spin.value())
        progress_bar.setValue(0)

    # Set start time
    if parent and hasattr(parent, 'start_time'):
        parent.start_time = time.time()

    model_name = getattr(tab_instance, f'{tab_name}_model_combo').currentText()
    dataset_combo_attr = f'{tab_name}_dataset_combo'
    if status_label:
        dataset_name = getattr(tab_instance, dataset_combo_attr).currentText() if hasattr(tab_instance, dataset_combo_attr) else ""
        if dataset_name:
            status_label.setText(f"Training {model_name} on {dataset_name}...")
        else:
            status_label.setText(f"Training {model_name}...")
        status_label.setStyleSheet("color: #00ff88; padding: 5px; font-weight: bold;")

    # Start timer and worker
    if parent and hasattr(parent, 'plot_timer'):
        parent.plot_timer.start(100)
    worker.start()

    return worker


def _get_current_hyperparams(widgets: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract current values from hyperparameter widgets.

    Args:
        widgets: Dictionary of hyperparameter widgets

    Returns:
        Dictionary mapping parameter names to current values
    """
    hyperparams = {}
    for name, widget in widgets.items():
        if name.endswith('_label'):
            continue
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            hyperparams[name] = widget.value()
        elif isinstance(widget, QCheckBox):
            hyperparams[name] = widget.isChecked()
    return hyperparams


def _on_progress(metrics: Dict[str, Any], tab_instance: QWidget, progress_bar: Optional[QProgressBar], status_label: Optional[QLabel]):
    """
    Handle training progress update.

    Args:
        metrics: Training metrics dictionary
        tab_instance: Tab instance
        progress_bar: Progress bar widget
        status_label: Status label widget
    """
    # Update history if parent has it
    parent = tab_instance.parent()
    if parent:
        if hasattr(parent, 'loss_history'):
            parent.loss_history.append(metrics['loss'])
        if hasattr(parent, 'acc_history'):
            parent.acc_history.append(metrics['accuracy'])
        if hasattr(parent, 'lipschitz_history'):
            parent.lipschitz_history.append(metrics['lipschitz'])

    # Update progress bar
    if progress_bar:
        progress_bar.setValue(metrics['epoch'])

    # Calculate ETA and update status
    if parent and hasattr(parent, 'start_time') and parent.start_time:
        import time
        elapsed = time.time() - parent.start_time
        if metrics['epoch'] > 0:
            speed = metrics['epoch'] / elapsed
            remaining = metrics['total_epochs'] - metrics['epoch']
            eta_seconds = remaining / speed if speed > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            eta_label_attr = f'{tab_instance.__class__.__name__.lower().replace("tab", "")}_eta_label'
            if hasattr(tab_instance, eta_label_attr):
                eta_label = getattr(tab_instance, eta_label_attr)
                eta_label.setText(f"ETA: {eta_str} | Speed: {speed:.2f} ep/s")

    # Update labels if they exist
    tab_prefix = tab_instance.__class__.__name__.lower().replace("tab", "")
    acc_label_attr = f'{tab_prefix}_acc_label'
    loss_label_attr = f'{tab_prefix}_loss_label'
    lip_label_attr = f'{tab_prefix}_lip_label'

    if hasattr(tab_instance, acc_label_attr):
        acc_label = getattr(tab_instance, acc_label_attr)
        acc_label.setText(f"{metrics['accuracy']:.1%}")
    if hasattr(tab_instance, loss_label_attr):
        loss_label = getattr(tab_instance, loss_label_attr)
        loss_label.setText(f"{metrics['loss']:.4f}")
    if hasattr(tab_instance, lip_label_attr):
        lip_label = getattr(tab_instance, lip_label_attr)
        lip_label.setText(f"{metrics['lipschitz']:.4f}")

    if status_label:
        status_label.setText(
            f"Epoch {metrics['epoch']}/{metrics['total_epochs']} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.1%} | "
            f"L: {metrics['lipschitz']:.4f}"
        )


def _on_finished(result: Dict[str, Any], tab_instance: QWidget, status_label: Optional[QLabel]):
    """
    Handle training completion.

    Args:
        result: Result dictionary
        tab_instance: Tab instance
        status_label: Status label widget
    """
    parent = tab_instance.parent()

    # Stop timer if parent has it
    if parent and hasattr(parent, 'plot_timer'):
        parent.plot_timer.stop()

    # Reset UI
    _reset_training_ui(tab_instance)

    if result.get('success'):
        if status_label:
            status_label.setText(f"✓ Training complete! ({result['epochs_completed']} epochs)")
            status_label.setStyleSheet("color: #00ff88; padding: 5px; font-weight: bold;")
    else:
        if status_label:
            status_label.setText("Training stopped.")
            status_label.setStyleSheet("color: #ffaa00; padding: 5px;")


def _on_error(error: str, tab_instance: QWidget, status_label: Optional[QLabel]):
    """
    Handle training error.

    Args:
        error: Error message
        tab_instance: Tab instance
        status_label: Status label widget
    """
    parent = tab_instance.parent()

    # Stop timer if parent has it
    if parent and hasattr(parent, 'plot_timer'):
        parent.plot_timer.stop()

    # Reset UI
    _reset_training_ui(tab_instance)

    if status_label:
        status_label.setText("Training error!")
        status_label.setStyleSheet("color: #ff5588; padding: 5px; font-weight: bold;")

    QMessageBox.critical(tab_instance, "Training Error", error)


def _reset_training_ui(tab_instance: QWidget):
    """
    Reset UI state after training stops.

    Args:
        tab_instance: Tab instance
    """
    tab_name = ""
    if hasattr(tab_instance, 'vis_train_btn'):
        tab_name = 'vis'
    elif hasattr(tab_instance, 'lm_train_btn'):
        tab_name = 'lm'
    elif hasattr(tab_instance, 'rl_train_btn'):
        tab_name = 'rl'
    elif hasattr(tab_instance, 'diff_train_btn'):
        tab_name = 'diff'

    if tab_name:
        train_btn_attr = f'{tab_name}_train_btn'
        stop_btn_attr = f'{tab_name}_stop_btn'
        pause_btn_attr = f'{tab_name}_pause_btn'

        if hasattr(tab_instance, train_btn_attr):
            getattr(tab_instance, train_btn_attr).setEnabled(True)
        if hasattr(tab_instance, stop_btn_attr):
            getattr(tab_instance, stop_btn_attr).setEnabled(False)
        if hasattr(tab_instance, pause_btn_attr):
            pause_btn = getattr(tab_instance, pause_btn_attr)
            pause_btn.setEnabled(False)
            pause_btn.setChecked(False)


def _toggle_pause(tab_instance: QWidget):
    """
    Pause/Resume training.

    Args:
        tab_instance: Tab instance
    """
    parent = tab_instance.parent()
    if not parent or not hasattr(parent, 'worker'):
        return

    # Determine which pause button to use based on tab
    is_paused = False
    if hasattr(tab_instance, 'vis_pause_btn'):
        is_paused = tab_instance.vis_pause_btn.isChecked()
    elif hasattr(tab_instance, 'lm_pause_btn'):
        is_paused = tab_instance.lm_pause_btn.isChecked()

    if is_paused:
        parent.worker.pause()
        status_label = getattr(parent, 'status_label', None)
        if status_label:
            status_label.setText("Training paused.")
            status_label.setStyleSheet("color: #f1c40f; padding: 5px; font-weight: bold;")
    else:
        parent.worker.resume()
        status_label = getattr(parent, 'status_label', None)
        if status_label:
            status_label.setText("Training resumed.")
            status_label.setStyleSheet("color: #00ff88; padding: 5px; font-weight: bold;")


def _log_error(message: str, tab_instance: QWidget):
    """
    Log an error message.

    Args:
        message: Error message to log
        tab_instance: Tab instance
    """
    import logging
    logging.error(message)
    _append_log(f"ERROR: {message}", tab_instance)


def _log_warning(message: str, tab_instance: QWidget):
    """
    Log a warning message.

    Args:
        message: Warning message to log
        tab_instance: Tab instance
    """
    import logging
    logging.warning(message)
    _append_log(f"WARNING: {message}", tab_instance)


def _log_info(message: str, tab_instance: QWidget):
    """
    Log an info message.

    Args:
        message: Info message to log
        tab_instance: Tab instance
    """
    import logging
    logging.info(message)
    _append_log(f"INFO: {message}", tab_instance)


def _append_log(message: str, tab_instance: QWidget):
    """
    Append a message to the console log.

    Args:
        message: Message to append
        tab_instance: Tab instance
    """
    parent = tab_instance.parent()
    if parent and hasattr(parent, 'console_log'):
        import time
        timestamp = time.strftime("%H:%M:%S")
        if not message.startswith("["):  # Avoid double timestamping if already timestamped
            parent.console_log.append(f"[{timestamp}] {message}")
        else:
            parent.console_log.append(message)

    # Also append to the ConsoleTab if initialized
    if parent and hasattr(parent, 'console_tab'):
        parent.console_tab.append_log(message)