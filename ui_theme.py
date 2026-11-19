"""BIFB-inspired desktop workspace; all analysis widgets keep their original signals."""
from pathlib import Path

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication, QBoxLayout, QButtonGroup, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLayout, QLineEdit, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)
from shiboken6 import delete

from ui_home import HomePage


ROOT = Path(__file__).resolve().parent
ACCENT = "#bd2030"
ICON_PATHS = {
    "home": '<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1z"/>',
    "bubbles": '<circle cx="8" cy="9" r="4"/><circle cx="17" cy="16" r="5"/><circle cx="18" cy="5" r="2"/><circle cx="5" cy="19" r="2"/>',
    "tracking": '<rect x="3" y="3" width="18" height="18" rx="5"/><circle cx="9" cy="9" r="2"/><path d="m5 18 5-5 3 3 3-4 3 3"/>',
    "holdup": '<path d="M5 3h14M8 3v7l-4 8a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3l-4-8V3M7 14h10"/><circle cx="11" cy="17" r="1"/>',
    "warning": '<path d="M3 12h4l3-7 4 14 3-7h4"/>',
    "play": '<path d="m9 5 10 7-10 7z"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v10H3z"/>',
    "file": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9zM14 3v6h6M8 14h8M8 17h5"/>',
}


def icon(name, color=ACCENT, size=24):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
           f'fill="none" stroke="{color}" stroke-width="1.65" '
           'stroke-linecap="round" stroke-linejoin="round">'
           + ICON_PATHS[name] + '</svg>')
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    QSvgRenderer(QByteArray(svg.encode())).render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)


def _label(text, role, parent=None):
    widget = QLabel(text, parent)
    widget.setProperty("role", role)
    return widget


def _scroll(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area


def _metric(title, value):
    card = QFrame()
    card.setProperty("role", "metric")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(14, 11, 14, 11)
    layout.setSpacing(5)
    title.setProperty("role", "metricTitle")
    title.setWordWrap(True)
    value.setProperty("role", "metricValue")
    value.setMinimumWidth(0)
    value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    value.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(title)
    layout.addWidget(value)
    return card


def _metric_grid(group, columns, preserve_layout=False):
    """Rehouse a result form without replacing labels used by worker callbacks."""
    form = group.layout()
    pairs = []
    for row in range(form.rowCount()):
        title = form.itemAt(row, QFormLayout.LabelRole)
        value = form.itemAt(row, QFormLayout.FieldRole)
        if title and value:
            pairs.append((title.widget(), value.widget()))
    original_group = group
    if preserve_layout:
        group = QGroupBox(original_group.title())
    for title, value in pairs:
        title.setParent(group)
        value.setParent(group)
    if preserve_layout:
        original_group.hide()
    else:
        delete(form)
    grid = QGridLayout(group)
    grid.setContentsMargins(0, 6, 0, 0)
    grid.setSpacing(8)
    for index, (title, value) in enumerate(pairs):
        grid.addWidget(_metric(title, value), index // columns, index % columns)
    for column in range(columns):
        grid.setColumnStretch(column, 1)
    group.setProperty("role", "metricGroup")
    return group


def _stack_form_labels(form):
    """Make long localized labels and paths fit a narrow configuration panel."""
    if not isinstance(form, QFormLayout):
        return
    form.setRowWrapPolicy(QFormLayout.WrapAllRows)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setVerticalSpacing(7)
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.LabelRole)
        if item and isinstance(item.widget(), QLabel):
            item.widget().setWordWrap(True)
            item.widget().setProperty("role", "fieldLabel")


def _tracking_layout(controller, page):
    w = controller.window
    original_page = page
    page = QWidget()
    controller.workflow_tabs.removeTab(0)
    controller.workflow_tabs.insertTab(0, page, "Bubble Tracking")
    original_page.setParent(w)
    original_page.hide()
    # Detach only UI containers. Object names, form controls and connections survive.
    for widget in (w.tabWidget, w.previewGroup,
                   w.startBtn, w.stopBtn, w.statusLabel, w.progressLabel,
                   w.progressBar, w.processingFpsLabel, w.frameLatencyLabel):
        widget.setParent(page)
    w.headerTitle.hide()
    w.headerHint.hide()
    w.centerSplitter.hide()
    root = QVBoxLayout(page)
    root.setContentsMargins(0, 0, 0, 0)
    splitter = QSplitter(Qt.Horizontal)
    splitter.setObjectName("trackingWorkspaceSplitter")
    splitter.setChildrenCollapsible(False)
    root.addWidget(splitter)

    # Keep the Designer tabs; add files as the first setup page.
    files_page = QWidget()
    files_layout = QVBoxLayout(files_page)
    files_layout.setContentsMargins(0, 0, 0, 0)
    w.tabWidget.insertTab(0, _scroll(files_page), "Files")
    w.tabWidget.setCurrentIndex(0)
    w.tabWidget.setMinimumWidth(300)
    w.tabWidget.setMaximumWidth(460)
    w.tabWidget.setObjectName("configurationTabs")
    w.paramsColumnsLayout.setDirection(QBoxLayout.TopToBottom)
    w.paramsScrollArea.setFrameShape(QFrame.NoFrame)
    w.paramsScrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    w.paramsColumnsLayout.setContentsMargins(0, 0, 0, 0)
    w.paramsColumnsLayout.setSpacing(12)
    w.paramsColumnsLayout.addStretch()
    for group in (w.detectionGroup, w.motionGroup):
        group.setMinimumWidth(0)
        _stack_form_labels(group.findChild(QFormLayout))
    # Move file labels above each path. Existing browse buttons retain their slots.
    files_group = QGroupBox(w.filesGroup.title())
    files_layout.addWidget(files_group)
    files_layout.addStretch()
    rows = [
        (w.modelLabel, w.modelEdit, (w.modelBrowseBtn,)),
        (w.inputLabel, w.inputEdit, (w.inputFileBtn, w.inputFolderBtn)),
        (w.outputDirLabel, w.outputDirEdit, (w.outputDirBrowseBtn,)),
        (w.outputVideoLabel, w.outputVideoEdit, (w.outputVideoSaveBtn,)),
        (w.outputCsvLabel, w.outputCsvEdit, (w.outputCsvSaveBtn,)),
    ]
    for row, (title, edit, buttons) in enumerate(rows):
        title.setParent(files_group)
        edit.setParent(files_group)
        for button in buttons:
            button.setParent(files_group)
    form = QVBoxLayout(files_group)
    form.setContentsMargins(14, 22, 14, 16)
    form.setSpacing(9)
    for title, edit, buttons in rows:
        title.setProperty("role", "fieldLabel")
        form.addWidget(title)
        form.addWidget(edit)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        for button in buttons:
            button.setIcon(icon("folder" if button is w.inputFolderBtn or button is w.outputDirBrowseBtn else "file", "#847780", 16))
            actions.addWidget(button)
        actions.addStretch()
        form.addLayout(actions)
        form.addSpacing(6)
    splitter.addWidget(w.tabWidget)

    results = QWidget()
    result_layout = QVBoxLayout(results)
    result_layout.setContentsMargins(0, 0, 0, 0)
    result_layout.setSpacing(12)
    result_layout.addWidget(_metric_grid(w.statsGroup, 4, preserve_layout=True))
    w.previewLabel.setMinimumSize(280, 180)
    w.previewLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    w.previewLabel.setProperty("role", "preview")
    w.previewLayout.setContentsMargins(12, 22, 12, 12)
    result_layout.addWidget(w.previewGroup, 1)

    run_panel = QFrame()
    run_panel.setProperty("role", "runPanel")
    run_layout = QVBoxLayout(run_panel)
    run_layout.setContentsMargins(16, 12, 16, 12)
    run_layout.setSpacing(9)
    actions = QHBoxLayout()
    w.statusLabel.setProperty("role", "status")
    actions.addWidget(w.statusLabel)
    actions.addStretch()
    actions.addWidget(w.progressLabel)
    actions.addWidget(w.startBtn)
    actions.addWidget(w.stopBtn)
    run_layout.addLayout(actions)
    run_layout.addWidget(w.progressBar)
    timings = QHBoxLayout()
    timings.addWidget(w.processingFpsLabel)
    timings.addStretch()
    timings.addWidget(w.frameLatencyLabel)
    run_layout.addLayout(timings)
    result_layout.addWidget(run_panel)
    splitter.addWidget(results)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([340, 800])


def _analysis_run_panel(page, gas):
    """Keep the run controls visible while the configuration panel scrolls."""
    if gas:
        page.start_button.parentWidget().hide()
    footer = QFrame()
    footer.setProperty("role", "runPanel")
    layout = QVBoxLayout(footer)
    layout.setContentsMargins(14, 10, 14, 10)
    layout.setSpacing(8)
    row = QHBoxLayout()
    if not gas:
        confirmation = page.time_confirm.parentWidget()
        confirmation.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row.addWidget(confirmation, 1)
    else:
        page.status_label.setProperty("role", "status")
        row.addWidget(page.status_label, 1)
    if not gas:
        row.addWidget(page.open_output_button)
    row.addWidget(page.start_button)
    row.addWidget(page.stop_button)
    layout.addLayout(row)
    progress = QHBoxLayout()
    if gas:
        progress.addWidget(page.progress_label)
    else:
        page.status_label.setProperty("role", "muted")
        progress.addWidget(page.status_label)
    progress.addWidget(page.progress_bar, 1)
    layout.addLayout(progress)
    page.layout().addWidget(footer)


def _polish_analysis_page(page, gas=False):
    root = page.layout()
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(10)
    intro = root.itemAt(0).widget()
    intro.setProperty("role", "pageNote")
    for area in page.findChildren(QScrollArea):
        area.setFrameShape(QFrame.NoFrame)
        area.setMinimumWidth(300)
        area.setMaximumWidth(460)
        area.widget().layout().setContentsMargins(0, 0, 8, 0)
    for form in page.findChildren(QFormLayout):
        _stack_form_labels(form)
    # Path rows wrap buttons below their input, including longer English labels.
    path_edits = (page.model_edit, page.input_folder_edit, page.overlay_folder_edit,
                  page.output_csv_edit) if gas else tuple(
                      field for field in page.fields.values() if isinstance(field, QLineEdit)
                      and isinstance(field.parentWidget().layout(), QHBoxLayout))
    for edit in path_edits:
        field = edit.parentWidget()
        old = field.layout()
        controls = [old.itemAt(index).widget() for index in range(old.count())]
        if not controls or any(widget is None for widget in controls):
            continue
        delete(old)
        column = QVBoxLayout(field)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        edit.setMinimumWidth(0)
        column.addWidget(edit)
        actions = QHBoxLayout()
        for button in controls[1:]:
            button.setMinimumWidth(0)
            button.setIcon(icon("folder", "#937d85", 15))
            actions.addWidget(button)
        actions.addStretch()
        column.addLayout(actions)
    for splitter in page.findChildren(QSplitter):
        splitter.setSizes([350, 780])
    page.preview_label.setStyleSheet("")
    page.preview_label.setProperty("role", "preview")
    page.preview_label.setMinimumSize(240, 100)
    page.preview_label.setWordWrap(True)
    page.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    if gas:
        page.setObjectName("gasHoldupPage")
        # A two-by-two ROI grid fits the same width as the other form fields.
        roi_grid = page.roi_x_spin.parentWidget().layout()
        roi_pairs = [(roi_grid.itemAtPosition(0, col).widget(),
                      roi_grid.itemAtPosition(1, col).widget()) for col in range(4)]
        for title, spin in roi_pairs:
            roi_grid.removeWidget(title)
            roi_grid.removeWidget(spin)
        for index, (title, spin) in enumerate(roi_pairs):
            roi_grid.addWidget(title, 2 * (index // 2), index % 2)
            roi_grid.addWidget(spin, 2 * (index // 2) + 1, index % 2)
        results = page.current_frame_value.parentWidget()
        _metric_grid(results, 3)
    else:
        page.event_table.setMinimumHeight(85)
        page.event_table.setMaximumHeight(115)
        page.event_table.setAlternatingRowColors(True)
        page.event_table.verticalHeader().hide()
        page.overall_label.setStyleSheet("font-size: 14px; font-weight: 600; padding: 2px;")
        for label in page.reason_labels.values():
            label.setProperty("role", "muted")
            label.parentWidget().setMinimumHeight(190)
    splitter = page.findChild(QSplitter)
    right = splitter.widget(1)
    right.setParent(None)
    right_scroll = _scroll(right)
    right_scroll.setMinimumWidth(360)
    splitter.addWidget(right_scroll)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([350, 780])
    _analysis_run_panel(page, gas)


class ResearchWorkspace(QWidget):
    """Navigation shell around the existing, independently testable workflows."""

    def __init__(self, controller, tracking_page, language_panel):
        super().__init__()
        self.setObjectName("researchWorkspace")
        self.language = controller.language
        self.tabs = controller.workflow_tabs
        self.tabs.tabBar().hide()
        self.tabs.setObjectName("workflowTabs")
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("workspaceSidebar")
        sidebar.setFixedWidth(224)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 30, 18, 22)
        side.setSpacing(10)
        brand = QHBoxLayout()
        mark = QLabel()
        mark.setPixmap(icon("bubbles", ACCENT, 36).pixmap(36, 36))
        brand.addWidget(mark)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(1)
        brand_text.addWidget(_label("BIFB", "brand"))
        brand_text.addWidget(_label("BUBBLE ANALYZER", "brandCaption"))
        brand.addLayout(brand_text)
        brand.addStretch()
        side.addLayout(brand)
        side.addSpacing(30)
        side.addWidget(_label("RESEARCH WORKSPACE", "sectionCaption"))
        side.addSpacing(6)
        self.nav_group = QButtonGroup(self)
        self.nav_buttons = {}
        for index, name, graphic in (
            (3, "Home", "home"), (0, "Bubble Tracking", "tracking"),
            (1, "Gas Holdup", "holdup"), (2, "Flow State Warning", "warning"),
        ):
            button = QPushButton(name)
            button.setProperty("role", "navigation")
            button.setCheckable(True)
            button.setIcon(icon(graphic, "#937d85", 21))
            button.setIconSize(QSize(21, 21))
            button.setCursor(Qt.PointingHandCursor)
            self.language.set_text(button, name)
            self.nav_group.addButton(button, index)
            self.nav_buttons[index] = button
            side.addWidget(button)
        self.nav_group.idClicked.connect(self.tabs.setCurrentIndex)
        side.addStretch()
        note = _label("Local analysis workspace", "sidebarNote")
        note.setWordWrap(True)
        self.language.set_text(note, "Local analysis workspace")
        side.addWidget(note)
        line = QFrame()
        line.setProperty("role", "sidebarRule")
        line.setFixedHeight(1)
        side.addWidget(line)
        side.addWidget(_label("YOLOTracker  /  DESKTOP", "brandCaption"))
        layout.addWidget(sidebar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        topbar = QFrame()
        topbar.setObjectName("workspaceTopbar")
        top = QHBoxLayout(topbar)
        top.setContentsMargins(26, 12, 26, 12)
        breadcrumb = _label("Research / Bubble analysis", "muted")
        self.language.set_text(breadcrumb, "Research / Bubble analysis")
        top.addWidget(breadcrumb)
        top.addStretch()
        local_badge = _label("●  Desktop workspace", "localBadge")
        self.language.set_text(local_badge, "●  Desktop workspace")
        top.addWidget(local_badge)
        top.addSpacing(18)
        top.addWidget(language_panel)
        content_layout.addWidget(topbar)

        hero = QFrame()
        hero.setObjectName("workspaceHero")
        heading = QHBoxLayout(hero)
        heading.setContentsMargins(28, 20, 28, 19)
        text = QVBoxLayout()
        text.setSpacing(7)
        text.addWidget(_label("MULTIPHASE FLOW · EXPERIMENT ANALYSIS", "eyebrow"))
        self.title = _label("", "pageTitle")
        self.subtitle = _label("", "subtitle")
        self.subtitle.setWordWrap(True)
        text.addWidget(self.title)
        text.addWidget(self.subtitle)
        heading.addLayout(text, 1)
        self.page_number = _label("01 / 03", "pageNumber")
        heading.addWidget(self.page_number)
        content_layout.addWidget(hero)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(24, 14, 24, 20)
        body_layout.addWidget(self.tabs)
        content_layout.addWidget(body, 1)
        layout.addWidget(content, 1)
        self.tabs.currentChanged.connect(self._refresh)
        self.language.changed.connect(self._refresh)
        self._refresh()

    def _refresh(self, *_args):
        index = max(self.tabs.currentIndex(), 0)
        titles = ("Bubble Tracking", "Gas Holdup", "Flow State Warning", "Home")
        subtitles = (
            "Detect bubbles, follow trajectories, and measure experimental motion.",
            "Estimate spherical bubble volume fraction with calibrated scale and view depth.",
            "Compare against a normal baseline and review sustained changes.",
            "One workspace for bubble motion, gas holdup, and flow-state changes.",
        )
        self.language.set_text(self.title, titles[index])
        self.language.set_text(self.subtitle, subtitles[index])
        self.page_number.setText(self.language.text("Overview") if index == 3 else f"0{index + 1} / 03")
        self.nav_buttons[index].setChecked(True)
        for number, graphic in enumerate(("tracking", "holdup", "warning", "home")):
            self.nav_buttons[number].setIcon(icon(graphic, ACCENT if number == index else "#937d85", 21))


def install_workspace(controller, tracking_page, language_panel):
    window = controller.window
    controller.setParent(window)
    # The Designer file and two analysis pages contain old local styles.
    for widget in window.findChildren(QWidget):
        widget.setStyleSheet("")
    _tracking_layout(controller, tracking_page)
    _polish_analysis_page(controller.gas_holdup_widget, gas=True)
    _polish_analysis_page(controller.flow_warning_widget)
    controller.home_page = HomePage(controller.language, {
        name: icon(name, ACCENT, 28) for name in ("tracking", "holdup", "warning")})
    controller.home_page.workflow_requested.connect(controller.workflow_tabs.setCurrentIndex)
    controller.workflow_tabs.addTab(controller.home_page, "Home")
    window.takeCentralWidget()
    shell = ResearchWorkspace(controller, tracking_page, language_panel)
    window.setCentralWidget(shell)
    window.setMinimumSize(1100, 720)
    screen = QApplication.primaryScreen().availableGeometry()
    window.resize(min(1480, screen.width() - 48), min(960, screen.height() - 64))
    for button in (window.startBtn, controller.gas_holdup_widget.start_button,
                   controller.flow_warning_widget.start_button):
        button.setProperty("role", "primary")
        button.setIcon(icon("play", "#ffffff", 17))
    for button in (window.stopBtn, controller.gas_holdup_widget.stop_button,
                   controller.flow_warning_widget.stop_button):
        button.setProperty("role", "stop")
        button.setIcon(icon("stop", "#9e6872", 16))
    for button in window.findChildren(QPushButton):
        button.setCursor(Qt.PointingHandCursor)
    for group in window.findChildren(QGroupBox):
        group.layout().setSizeConstraint(QLayout.SetDefaultConstraint)
    window.setWindowIcon(icon("bubbles", ACCENT, 32))
    stylesheet = (ROOT / "ui_theme.qss").read_text(encoding="utf-8-sig")
    window.setStyleSheet(stylesheet.replace("__ASSETS__", (ROOT / "ui_assets").as_posix()))
    app = QApplication.instance()
    if not app.property("researchWorkspaceStyleInstalled"):
        app.setStyle("Fusion")
        app.setProperty("researchWorkspaceStyleInstalled", True)
    font = QFont("Microsoft YaHei UI", 9)
    app.setFont(font)
    palette = QPalette()
    for role, color in (
        (QPalette.Window, "#faf9fb"), (QPalette.Base, "#ffffff"),
        (QPalette.AlternateBase, "#fcf7f9"), (QPalette.Text, "#34323b"),
        (QPalette.WindowText, "#34323b"), (QPalette.Button, "#ffffff"),
        (QPalette.ButtonText, "#34323b"), (QPalette.Highlight, ACCENT),
        (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#ffffff"),
        (QPalette.ToolTipText, "#34323b"),
    ):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    for edit, placeholder in (
        (window.modelEdit, "Choose a .pt or .onnx model"),
        (window.inputEdit, "Choose a video, image, or image folder"),
        (window.outputDirEdit, "Choose where to save analysis results"),
        (controller.gas_holdup_widget.model_edit, "Choose a .pt or .onnx model"),
        (controller.gas_holdup_widget.input_folder_edit, "Choose a folder of experiment images"),
    ):
        controller.language.set_property(edit, "placeholderText", placeholder)
    controller.workspace = shell
    return shell
