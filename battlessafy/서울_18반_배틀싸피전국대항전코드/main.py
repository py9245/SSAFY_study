from __future__ import annotations

import multiprocessing

from battlessafygui.gui import run_gui


def main() -> None:
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # 이미 설정돼 있는 경우 무시
        pass
    run_gui()


if __name__ == "__main__":
    main()

