import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow

import bubble_tracker_ui_qt


class HomePageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def controller(self, state=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "state.json"
        if state:
            path.write_text(json.dumps(state), encoding="utf-8")
        state_patch = patch.object(bubble_tracker_ui_qt, "UI_STATE_PATH", str(path))
        state_patch.start()
        self.addCleanup(state_patch.stop)
        with patch.object(QMainWindow, "show"):
            controller = bubble_tracker_ui_qt.BubbleTrackerQt()
        self.addCleanup(controller.window.deleteLater)
        self.addCleanup(controller.window.close)
        return controller

    def test_startup_opens_home_with_existing_settings(self):
        controller = self.controller({"frame_rate": 172, "gas_holdup": {"view_depth": 2.5}})
        self.assertIs(controller.workflow_tabs.currentWidget(), controller.home_page)
        self.assertEqual(controller.window.fpsSpin.value(), 172)
        self.assertEqual(controller.gas_holdup_widget.view_depth_spin.value(), 2.5)
        self.assertEqual(controller.workflow_tabs.count(), 4)

    def test_home_entries_and_sidebar_preserve_configuration(self):
        controller = self.controller()
        before = controller._serialize_state()
        home_index = controller.workflow_tabs.indexOf(controller.home_page)
        for index, button in controller.home_page.workflow_buttons.items():
            self.assertNotEqual(button.focusPolicy(), Qt.NoFocus)
            self.assertTrue(button.accessibleName())
            button.click()
            self.assertEqual(controller.workflow_tabs.currentIndex(), index)
            self.assertTrue(controller.workspace.nav_buttons[index].isChecked())
            controller.workspace.nav_buttons[home_index].click()
            self.assertIs(controller.workflow_tabs.currentWidget(), controller.home_page)
        self.assertEqual(controller._serialize_state(), before)
        self.assertIsNone(controller.worker_thread)
        self.assertFalse(controller.gas_holdup_widget.is_running())
        self.assertFalse(controller.flow_warning_widget.is_running())

    def test_home_follows_language_switch_and_remains_the_active_page(self):
        controller = self.controller()
        controller.language.set_language("en")
        self.assertIs(controller.workflow_tabs.currentWidget(), controller.home_page)
        self.assertEqual(controller.workspace.title.text(), "Home")
        self.assertEqual(controller.home_page.workflow_buttons[1].accessibleName(), "Open Gas Holdup")
        controller.language.set_language("zh_CN")
        self.assertEqual(controller.workspace.title.text(), "首页")
        self.assertEqual(controller.home_page.workflow_buttons[1].accessibleName(), "进入气含率")


if __name__ == "__main__":
    unittest.main()
