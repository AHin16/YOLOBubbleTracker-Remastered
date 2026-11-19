"""Start page for choosing an experimental analysis workflow."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)


class HomePage(QScrollArea):
    workflow_requested = Signal(int)

    def __init__(self, language, icons, parent=None):
        super().__init__(parent)
        self.language = language
        self.workflow_buttons = {}
        self.setObjectName("analysisHome")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.language.set_property(self, "accessibleName", "Analysis Home")
        body = QWidget()
        self.setWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)
        root.addWidget(self._text("Choose your analysis", "homeHeading"))
        introduction = self._text(
            "Start with your experiment images. Choose the measurements you need.", "homeBody")
        root.addWidget(introduction)

        columns = QHBoxLayout()
        columns.setSpacing(20)
        workflows = QVBoxLayout()
        workflows.setSpacing(12)
        entries = (
            (0, "tracking", "Bubble Tracking",
             "Follow individual bubbles and measure their motion over time.",
             "Video / image sequence · Trajectories, velocity, motion CSV",
             "Open Tracking", "Open Bubble Tracking"),
            (1, "holdup", "Gas Holdup",
             "Estimate gas volume fraction from equivalent spherical bubbles.",
             "Image folder · Volume fraction, area fraction, bubble size",
             "Open Measurement", "Open Gas Holdup"),
            (2, "warning", "Flow State Warning",
             "Compare a test sequence with normal conditions and replay changes.",
             "Baseline + test sequence · Appearance, large bubbles, motion",
             "Open Warning", "Open Flow State Warning"),
        )
        for entry in entries:
            workflows.addWidget(self._workflow(entry, icons))
        columns.addLayout(workflows, 3)
        columns.addWidget(self._guide(), 2)
        root.addLayout(columns)
        footer = self._text(
            "Your settings stay with each workflow. Return here anytime from the sidebar.", "homeMeta")
        root.addWidget(footer)
        root.addStretch()

    def _text(self, text, role):
        label = QLabel()
        label.setProperty("role", role)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        label.setSizePolicy(policy)
        self.language.set_text(label, text)
        return label

    def _workflow(self, entry, icons):
        index, graphic, title, description, details, action, accessible = entry
        card = QFrame()
        card.setProperty("role", "homeWorkflow")
        row = QHBoxLayout(card)
        row.setContentsMargins(20, 16, 20, 16)
        row.setSpacing(16)
        graphic_label = QLabel()
        graphic_label.setPixmap(icons[graphic].pixmap(28, 28))
        graphic_label.setFixedSize(36, 40)
        row.addWidget(graphic_label, 0, Qt.AlignTop)
        content = QVBoxLayout()
        content.setSpacing(6)
        heading = QHBoxLayout()
        heading.addWidget(self._text(title, "homeWorkflowTitle"), 1)
        number = QLabel(f"0{index+1}")
        number.setProperty("role", "homeNumber")
        heading.addWidget(number)
        content.addLayout(heading)
        content.addWidget(self._text(description, "homeBody"))
        content.addWidget(self._text(details, "homeMeta"))
        actions = QHBoxLayout()
        actions.addStretch()
        button = QPushButton()
        button.setProperty("role", "homeAction")
        button.setFocusPolicy(Qt.StrongFocus)
        button.setCursor(Qt.PointingHandCursor)
        self.language.set_text(button, action)
        self.language.set_property(button, "accessibleName", accessible)
        button.clicked.connect(lambda _checked=False, target=index: self.workflow_requested.emit(target))
        self.workflow_buttons[index] = button
        actions.addWidget(button)
        content.addLayout(actions)
        row.addLayout(content, 1)
        return card

    def _guide(self):
        panel = QFrame()
        panel.setProperty("role", "homeGuide")
        panel.setMinimumWidth(240)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        layout.addWidget(self._text("A clear path from images to results", "homeGuideTitle"))
        steps = (
            ("Prepare your images", "Choose your segmentation model and the experiment data to analyze."),
            ("Set the measurement scale", "Check the image scale, ROI, acquisition rate, or view depth for your workflow."),
            ("Review and export", "Inspect overlays and measurements, then save the evidence and CSV results."),
        )
        for index, (title, description) in enumerate(steps, 1):
            row = QHBoxLayout()
            row.setSpacing(12)
            number = QLabel(str(index))
            number.setProperty("role", "homeStep")
            number.setAlignment(Qt.AlignCenter)
            number.setFixedSize(28, 28)
            row.addWidget(number, 0, Qt.AlignTop)
            text = QVBoxLayout()
            text.setSpacing(6)
            text.addWidget(self._text(title, "homeStepTitle"))
            text.addWidget(self._text(description, "homeBody"))
            row.addLayout(text, 1)
            layout.addLayout(row)
        layout.addStretch()
        method = QFrame()
        method.setProperty("role", "homeMethod")
        note = QVBoxLayout(method)
        note.setContentsMargins(12, 12, 12, 12)
        note.setSpacing(6)
        note.addWidget(self._text("Gas holdup measurement", "homeStepTitle"))
        note.addWidget(self._text(
            "The spherical volume estimate uses calibrated lengths and your actual view depth. "
            "Projected area fraction is reported separately.", "homeBody"))
        layout.addWidget(method)
        return panel
