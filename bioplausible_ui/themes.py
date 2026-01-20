"""
Themes for Bioplausible Trainer

Includes Dark Cyberpunk and Professional Light themes.
"""

# === DARK THEME COLORS ===
DARK_THEME_COLORS = {
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
    'button_reset': '#aa7722',
    'button_reset_alt': '#cc9933',
    'slider_handle': '#00ffff',
    'slider_handle_alt': '#0088aa',
    'progress_fill': '#00aacc',
    'progress_fill_alt': '#00ffff',
    'checkbox_checked': '#00aacc',
    'checkbox_border': '#3a3a5f',
}

# === LIGHT THEME COLORS ===
LIGHT_THEME_COLORS = {
    'background': '#f5f5f7',
    'background_alt': '#ffffff',
    'background_group': 'rgba(240, 240, 245, 0.7)',
    'background_group_alt': 'rgba(255, 255, 255, 0.95)',
    'border': '#d0d0d8',
    'border_alt': '#e0e0e5',
    'text_primary': '#1d1d1f',
    'text_secondary': '#515154',
    'text_accent': '#0066cc',
    'text_accent_alt': '#008844',
    'neon_cyan': '#0077aa',
    'neon_green': '#009944',
    'neon_pink': '#cc3355',
    'neon_orange': '#dd7700',
    'neon_purple': '#8844cc',
    'neon_magenta': '#cc33cc',
    'neon_red': '#dd4444',
    'neon_green_alt': '#44aa44',
    'button_primary': '#e0e0e5',
    'button_primary_alt': '#d0d0d5',
    'button_train': '#28a745',
    'button_train_alt': '#218838',
    'button_stop': '#dc3545',
    'button_stop_alt': '#c82333',
    'button_reset': '#ffc107',
    'button_reset_alt': '#e0a800',
    'slider_handle': '#007aff',
    'slider_handle_alt': '#0051a8',
    'progress_fill': '#007aff',
    'progress_fill_alt': '#5ac8fa',
    'checkbox_checked': '#007aff',
    'checkbox_border': '#c0c0c5',
}

# Backwards compatibility
THEME_COLORS = DARK_THEME_COLORS

def create_theme(colors):
    return f"""
/* === Global === */
QWidget {{
    background-color: {colors['background']};
    color: {colors['text_primary']};
    font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
    font-size: 13px;
}}

/* === Main Window === */
QMainWindow {{
    background-color: {colors['background']};
}}

/* === Tab Widget === */
QTabWidget::pane {{
    border: 1px solid {colors['border_alt']};
    background-color: {colors['background_group_alt']};
    border-radius: 8px;
}}

QTabBar::tab {{
    background-color: {colors['background_alt']};
    color: {colors['text_secondary']};
    padding: 12px 24px;
    border: 1px solid {colors['border_alt']};
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
    font-weight: 600;
}}

QTabBar::tab:selected {{
    background: {colors['background_group_alt']};
    color: {colors['text_accent']};
    border-bottom: 2px solid {colors['text_accent']};
}}

QTabBar::tab:hover:!selected {{
    background-color: {colors['background_group']};
    color: {colors['text_primary']};
}}

/* === Group Boxes === */
QGroupBox {{
    background-color: {colors['background_group']};
    border: 1px solid {colors['border']};
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
    color: {colors['text_accent']};
    font-size: 14px;
}}

/* === Buttons === */
QPushButton {{
    background-color: {colors['button_primary']};
    color: {colors['text_primary']};
    border: 1px solid {colors['border']};
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    min-width: 100px;
}}

QPushButton:hover {{
    background-color: {colors['button_primary_alt']};
    border-color: {colors['text_accent']};
}}

QPushButton:pressed {{
    background-color: {colors['border']};
}}

QPushButton#trainButton {{
    background-color: {colors['button_train']};
    color: white;
    font-size: 16px;
    padding: 15px 40px;
    border: none;
}}

QPushButton#trainButton:hover {{
    background-color: {colors['button_train_alt']};
    box-shadow: 0 0 10px rgba(40, 167, 69, 0.4);
}}

QPushButton#stopButton {{
    background-color: {colors['button_stop']};
    color: white;
}}

QPushButton#stopButton:hover {{
    background-color: {colors['button_stop_alt']};
}}

QPushButton#resetButton {{
    background-color: {colors['button_reset']};
    color: white;
    min-width: 60px;
    padding: 10px;
}}

QPushButton#resetButton:hover {{
    background-color: {colors['button_reset_alt']};
}}

/* === Sliders === */
QSlider::groove:horizontal {{
    height: 6px;
    background: {colors['border_alt']};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    width: 18px;
    height: 18px;
    margin: -6px 0;
    background: {colors['slider_handle']};
    border-radius: 9px;
    border: 2px solid {colors['background_alt']};
}}

QSlider::handle:horizontal:hover {{
    background: {colors['slider_handle_alt']};
}}

QSlider::sub-page:horizontal {{
    background: {colors['slider_handle']};
    border-radius: 3px;
}}

/* === Spin Boxes === */
QSpinBox, QDoubleSpinBox {{
    background-color: {colors['background_alt']};
    border: 1px solid {colors['border']};
    border-radius: 6px;
    padding: 6px 10px;
    color: {colors['text_primary']};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {colors['text_accent']};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {colors['button_primary']};
    border: none;
    width: 20px;
}}

/* === Combo Boxes === */
QComboBox {{
    background-color: {colors['background_alt']};
    border: 1px solid {colors['border']};
    border-radius: 6px;
    padding: 8px 12px;
    color: {colors['text_primary']};
    min-width: 150px;
}}

QComboBox:hover {{
    border-color: {colors['text_accent']};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox QAbstractItemView {{
    background-color: {colors['background_alt']};
    border: 1px solid {colors['border']};
    selection-background-color: {colors['text_accent']};
    color: {colors['text_primary']};
}}

/* === Progress Bar === */
QProgressBar {{
    background-color: {colors['background_alt']};
    border: 1px solid {colors['border']};
    border-radius: 6px;
    height: 20px;
    text-align: center;
    color: {colors['text_primary']};
    font-weight: bold;
}}

QProgressBar::chunk {{
    background-color: {colors['progress_fill']};
    border-radius: 5px;
}}

/* === Text Areas === */
QTextEdit, QPlainTextEdit {{
    background-color: {colors['background_alt']};
    border: 1px solid {colors['border']};
    border-radius: 8px;
    color: {colors['text_primary']};
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
    padding: 10px;
}}

/* === Labels === */
QLabel {{
    color: {colors['text_secondary']};
}}

QLabel#headerLabel {{
    font-size: 24px;
    font-weight: bold;
    color: {colors['text_accent']};
}}

QLabel#metricLabel {{
    font-size: 18px;
    font-weight: bold;
    color: {colors['neon_pink']};
}}

/* === Scrollbars === */
QScrollBar:vertical {{
    background-color: {colors['background']};
    width: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {colors['button_primary']};
    border-radius: 6px;
    min-height: 40px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {colors['button_primary_alt']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* === Splitter === */
QSplitter::handle {{
    background-color: {colors['border']};
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
    color: {colors['text_secondary']};
}}

QCheckBox::indicator {{
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 2px solid {colors['checkbox_border']};
    background-color: {colors['background_alt']};
}}

QCheckBox::indicator:checked {{
    background-color: {colors['checkbox_checked']};
    border-color: {colors['text_accent']};
}}

QCheckBox::indicator:hover {{
    border-color: {colors['text_accent']};
}}
"""

CYBERPUNK_DARK = create_theme(DARK_THEME_COLORS)
LIGHT_THEME = create_theme(LIGHT_THEME_COLORS)

# Neon glow colors for plots - Dark theme default, but can be used in Light too
PLOT_COLORS = {
    'loss': DARK_THEME_COLORS['neon_pink'],
    'accuracy': DARK_THEME_COLORS['neon_green'],
    'perplexity': DARK_THEME_COLORS['neon_orange'],
    'lipschitz': DARK_THEME_COLORS['neon_cyan'],
    'memory': DARK_THEME_COLORS['neon_purple'],
    'gradient': DARK_THEME_COLORS['neon_magenta'],
    'backprop': DARK_THEME_COLORS['neon_red'],
    'eqprop': DARK_THEME_COLORS['neon_green_alt'],
}

LIGHT_PLOT_COLORS = {
    'loss': LIGHT_THEME_COLORS['neon_pink'],
    'accuracy': LIGHT_THEME_COLORS['neon_green'],
    'perplexity': LIGHT_THEME_COLORS['neon_orange'],
    'lipschitz': LIGHT_THEME_COLORS['neon_cyan'],
    'memory': LIGHT_THEME_COLORS['neon_purple'],
    'gradient': LIGHT_THEME_COLORS['neon_magenta'],
    'backprop': LIGHT_THEME_COLORS['neon_red'],
    'eqprop': LIGHT_THEME_COLORS['neon_green_alt'],
}

# Animation durations (ms)
ANIMATION = {
    'button_glow': 300,
    'progress': 500,
    'plot_update': 50,
}
