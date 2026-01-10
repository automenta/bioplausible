"""
Dark Cyberpunk Theme for EqProp Trainer

Neon accents, glassmorphism, and futuristic aesthetics.
"""

# Color constants for maintainability
THEME_COLORS = {
    'background': '#0a0a0f',
    'background_alt': '#0d0d15',
    'background_group': 'rgba(20, 20, 40, 0.7)',
    'background_group_alt': 'rgba(15, 15, 25, 0.95)',
    'border': '#2a2a4f',
    'border_alt': '#1a1a2e',
    'text_primary': '#e0e0e0',
    'text_secondary': '#c0c0d0',
    'text_accent': '#00ffff',
    'text_accent_alt': '#00ff88',
    'neon_cyan': '#00ffff',
    'neon_green': '#00ff88',
    'neon_pink': '#ff5588',
    'neon_orange': '#ffaa00',
    'neon_purple': '#aa88ff',
    'neon_magenta': '#ff88ff',
    'neon_red': '#ff6666',
    'neon_green_alt': '#66ff66',
    'button_primary': '#2a2a4f',
    'button_primary_alt': '#3a3a6f',
    'button_train': '#00aa88',
    'button_train_alt': '#00ccaa',
    'button_stop': '#cc3355',
    'button_stop_alt': '#992244',
    'slider_handle': '#00ffff',
    'slider_handle_alt': '#0088aa',
    'progress_fill': '#00aacc',
    'progress_fill_alt': '#00ffff',
    'checkbox_checked': '#00aacc',
    'checkbox_border': '#3a3a5f',
}

CYBERPUNK_DARK = f"""
/* === Global === */
QWidget {{
    background-color: {THEME_COLORS['background']};
    color: {THEME_COLORS['text_primary']};
    font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
    font-size: 13px;
}}

/* === Main Window === */
QMainWindow {{
    background-color: {THEME_COLORS['background']};
}}

/* === Tab Widget === */
QTabWidget::pane {{
    border: 1px solid {THEME_COLORS['border_alt']};
    background-color: {THEME_COLORS['background_group_alt']};
    border-radius: 8px;
}}

QTabBar::tab {{
    background-color: #12121f;
    color: #808090;
    padding: 12px 24px;
    border: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
    font-weight: 600;
}}

QTabBar::tab:selected {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1e1e3f, stop:1 #14142a);
    color: {THEME_COLORS['neon_cyan']};
    border-bottom: 2px solid {THEME_COLORS['neon_cyan']};
}}

QTabBar::tab:hover:!selected {{
    background-color: #1a1a35;
    color: #a0a0b0;
}}

/* === Group Boxes === */
QGroupBox {{
    background-color: {THEME_COLORS['background_group']};
    border: 1px solid {THEME_COLORS['border']};
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 10px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 0 8px;
    color: {THEME_COLORS['neon_cyan']};
    font-size: 14px;
}}

/* === Buttons === */
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {THEME_COLORS['button_primary']}, stop:1 #1a1a35);
    color: {THEME_COLORS['text_primary']};
    border: 1px solid #3a3a5f;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    min-width: 100px;
}}

QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {THEME_COLORS['button_primary_alt']}, stop:1 {THEME_COLORS['button_primary']});
    border-color: {THEME_COLORS['neon_cyan']};
}}

QPushButton:pressed {{
    background: #1a1a35;
}}

QPushButton#trainButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {THEME_COLORS['button_train']}, stop:1 #006655);
    color: white;
    font-size: 16px;
    padding: 15px 40px;
    border: none;
}}

QPushButton#trainButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {THEME_COLORS['button_train_alt']}, stop:1 #008877);
    box-shadow: 0 0 20px rgba(0, 255, 200, 0.4);
}}

QPushButton#stopButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {THEME_COLORS['button_stop']}, stop:1 {THEME_COLORS['button_stop_alt']});
    color: white;
}}

/* === Sliders === */
QSlider::groove:horizontal {{
    height: 6px;
    background: #1a1a35;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    width: 18px;
    height: 18px;
    margin: -6px 0;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {THEME_COLORS['slider_handle']}, stop:1 {THEME_COLORS['slider_handle_alt']});
    border-radius: 9px;
    border: 2px solid #00aacc;
}}

QSlider::handle:horizontal:hover {{
    background: {THEME_COLORS['slider_handle']};
    border-color: {THEME_COLORS['slider_handle']};
}}

QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00aacc, stop:1 {THEME_COLORS['slider_handle']});
    border-radius: 3px;
}}

/* === Spin Boxes === */
QSpinBox, QDoubleSpinBox {{
    background-color: #12121f;
    border: 1px solid {THEME_COLORS['border']};
    border-radius: 6px;
    padding: 6px 10px;
    color: {THEME_COLORS['text_primary']};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {THEME_COLORS['neon_cyan']};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: #1a1a35;
    border: none;
    width: 20px;
}}

/* === Combo Boxes === */
QComboBox {{
    background-color: #12121f;
    border: 1px solid {THEME_COLORS['border']};
    border-radius: 6px;
    padding: 8px 12px;
    color: {THEME_COLORS['text_primary']};
    min-width: 150px;
}}

QComboBox:hover {{
    border-color: {THEME_COLORS['neon_cyan']};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox QAbstractItemView {{
    background-color: #12121f;
    border: 1px solid {THEME_COLORS['border']};
    selection-background-color: #00aacc;
    color: {THEME_COLORS['text_primary']};
}}

/* === Progress Bar === */
QProgressBar {{
    background-color: #12121f;
    border: 1px solid {THEME_COLORS['border']};
    border-radius: 6px;
    height: 20px;
    text-align: center;
    color: white;
    font-weight: bold;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {THEME_COLORS['progress_fill']}, stop:0.5 {THEME_COLORS['progress_fill_alt']}, stop:1 {THEME_COLORS['progress_fill']});
    border-radius: 5px;
}}

/* === Text Areas === */
QTextEdit, QPlainTextEdit {{
    background-color: {THEME_COLORS['background_alt']};
    border: 1px solid #252540;
    border-radius: 8px;
    color: {THEME_COLORS['text_accent_alt']};
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
    padding: 10px;
}}

/* === Labels === */
QLabel {{
    color: {THEME_COLORS['text_secondary']};
}}

QLabel#headerLabel {{
    font-size: 24px;
    font-weight: bold;
    color: {THEME_COLORS['neon_cyan']};
}}

QLabel#metricLabel {{
    font-size: 18px;
    font-weight: bold;
    color: {THEME_COLORS['neon_pink']};
}}

/* === Scrollbars === */
QScrollBar:vertical {{
    background-color: {THEME_COLORS['background']};
    width: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {THEME_COLORS['button_primary']};
    border-radius: 6px;
    min-height: 40px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {THEME_COLORS['button_primary_alt']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* === Splitter === */
QSplitter::handle {{
    background-color: #1a1a35;
}}

QSplitter::handle:horizontal {{
    width: 3px;
}}

QSplitter::handle:vertical {{
    height: 3px;
}}

/* === Check Boxes === */
QCheckBox {{
    spacing: 8px;
    color: {THEME_COLORS['text_secondary']};
}}

QCheckBox::indicator {{
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 2px solid {THEME_COLORS['checkbox_border']};
    background-color: #12121f;
}}

QCheckBox::indicator:checked {{
    background-color: {THEME_COLORS['checkbox_checked']};
    border-color: {THEME_COLORS['neon_cyan']};
}}

QCheckBox::indicator:hover {{
    border-color: {THEME_COLORS['neon_cyan']};
}}
"""

# Neon glow colors for plots
PLOT_COLORS = {
    'loss': THEME_COLORS['neon_pink'],       # Pink
    'accuracy': THEME_COLORS['neon_green'],   # Green
    'perplexity': THEME_COLORS['neon_orange'], # Orange
    'lipschitz': THEME_COLORS['neon_cyan'],  # Cyan
    'memory': THEME_COLORS['neon_purple'],     # Purple
    'gradient': THEME_COLORS['neon_magenta'],   # Magenta
    'backprop': THEME_COLORS['neon_red'],   # Red
    'eqprop': THEME_COLORS['neon_green_alt'],     # Green
}

# Animation durations (ms)
ANIMATION = {
    'button_glow': 300,
    'progress': 500,
    'plot_update': 50,
}
