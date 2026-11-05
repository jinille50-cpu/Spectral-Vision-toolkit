"""Raman MC Studio 실행: python run.py"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["QT_API"] = "pyside6"            # matplotlib Qt 백엔드를 PySide6 로

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")

from PySide6 import QtWidgets  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raman_mc_studio import theme  # noqa: E402


def main():
    app = QtWidgets.QApplication(sys.argv)
    theme.apply_theme(app)
    from raman_mc_studio.app import MainWindow
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
