"""Tkinter 기반 간단한 BattleSSAFY 시뮬레이터 GUI."""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional

from .engine import GameEngine
from .orchestrator import BotSpec, TurnOrchestrator
from . import maps

DEFAULT_BOT_PATHS = {
    "Attacker1": "Attacker1.py",
    "Player1": "Player1_서울18_신건하.py",
    "Attacker2": "Attacker2.py",
    "Player2": "Player2_서울18_박유신.py",
    "Attacker3": "Attacker3.py",
    "Player3": "Player3_서울18_이명재.py",
}

TURN_ORDER = [
    ("Attacker1", "A1", "attackers"),
    ("Player1", "B1", "defenders"),
    ("Attacker2", "A2", "attackers"),
    ("Player2", "B2", "defenders"),
    ("Attacker3", "A3", "attackers"),
    ("Player3", "B3", "defenders"),
]



TERRAIN_STYLES = {
    "G": ("#e8f5e9", "#1b5e20", ""),
    "S": ("#fff5d7", "#795548", ""),
    "W": ("#d0e7ff", "#0d47a1", ""),
    "T": ("#8bc34a", "#1b5e20", "T"),
    "R": ("#cfd8dc", "#37474f", ""),
    "F": ("#ffe082", "#e65100", "F"),
}

UNIT_STYLES = {
    "A1": ("#ff8a80", "#ffffff", None),
    "A2": ("#ff5252", "#ffffff", None),
    "A3": ("#ff1744", "#ffffff", None),
    "B1": ("#82b1ff", "#ffffff", None),
    "B2": ("#448aff", "#ffffff", None),
    "B3": ("#2979ff", "#ffffff", None),
    "AH": ("#d32f2f", "#ffffff", "AH"),
    "BH": ("#1976d2", "#ffffff", "BH"),
}

DEFAULT_STYLE = ("#f0f0f0", "#000000", None)


class BotConfigRow(tk.Frame):
    def __init__(self, master: tk.Misc, label: str, default_path: str):
        super().__init__(master)
        self.label = label
        tk.Label(self, text=label, width=12, anchor="w").pack(side=tk.LEFT)
        self.var = tk.StringVar(value=default_path)
        self.entry = tk.Entry(self, textvariable=self.var, width=50)
        self.entry.pack(side=tk.LEFT, padx=4)
        tk.Button(self, text="Browse", command=self.browse).pack(side=tk.LEFT)

    def browse(self) -> None:
        path = filedialog.askopenfilename(title=f"Select {self.label}", filetypes=[("Python", "*.py"), ("All", "*.*")])
        if path:
            self.var.set(path)

    def get_path(self) -> str:
        return self.var.get()


class GameGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BattleSSAFY GUI Simulator")
        self.geometry("1200x800")

        self.map_text: Optional[str] = None
        self.engine: Optional[GameEngine] = None
        self.orchestrator: Optional[TurnOrchestrator] = None
        self.running = False
        self.after_id: Optional[str] = None
        self.grid_labels: List[List[tk.Label]] = []

        self._build_widgets()
        self._load_default_map()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_widgets(self) -> None:
        config_frame = tk.Frame(self)
        config_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        tk.Button(config_frame, text="Load Map", command=self.load_map).pack(side=tk.LEFT)
        self.map_label = tk.Label(config_frame, text="map: (not loaded)")
        self.map_label.pack(side=tk.LEFT, padx=8)

        self.bot_rows: Dict[str, BotConfigRow] = {}
        bot_frame = tk.Frame(self)
        bot_frame.pack(side=tk.TOP, fill=tk.X, padx=8)
        for name, default in DEFAULT_BOT_PATHS.items():
            absolute_default = os.path.abspath(default) if os.path.exists(default) else default
            row = BotConfigRow(bot_frame, name, absolute_default)
            row.pack(fill=tk.X, pady=2)
            self.bot_rows[name] = row

        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        tk.Button(control_frame, text="Start", command=self.start_simulation).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Step", command=self.step_simulation).pack(side=tk.LEFT, padx=4)
        tk.Button(control_frame, text="Run", command=self.run_simulation).pack(side=tk.LEFT, padx=4)
        tk.Button(control_frame, text="Pause", command=self.pause_simulation).pack(side=tk.LEFT, padx=4)
        tk.Button(control_frame, text="Stop", command=self.stop_simulation).pack(side=tk.LEFT, padx=4)

        self.status_label = tk.Label(control_frame, text="Idle")
        self.status_label.pack(side=tk.LEFT, padx=16)

        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.grid_frame = tk.Frame(main_frame)
        self.grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        info_frame = tk.Frame(main_frame)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8)
        tk.Label(info_frame, text="Unit Stats").pack(anchor="w")
        self.info_text = tk.Text(info_frame, width=40, height=40)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.configure(state=tk.DISABLED)

    def _load_default_map(self) -> None:
        default_path = "map.txt"
        if os.path.exists(default_path):
            self._set_map_from_file(default_path)

    def load_map(self) -> None:
        path = filedialog.askopenfilename(title="Select Map", filetypes=[("Map", "*.txt"), ("All", "*.*")])
        if path:
            self._set_map_from_file(path)

    def _set_map_from_file(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            maps.parse_map_text(text)  # 검증
        except Exception as exc:
            messagebox.showerror("Map Load Error", str(exc))
            return
        self.map_text = text
        self.map_label.config(text=f"map: {os.path.basename(path)}")
        self._build_grid_from_map(text)

    def _build_grid_from_map(self, text: str) -> None:
        parsed = maps.parse_map_text(text)
        for row in self.grid_labels:
            for label in row:
                label.destroy()
        self.grid_labels.clear()
        for r in range(parsed.height):
            label_row: List[tk.Label] = []
            for c in range(parsed.width):
                lbl = tk.Label(self.grid_frame, text="", width=4, height=2, relief=tk.RIDGE, borderwidth=1)
                lbl.grid(row=r, column=c, sticky="nsew")
                lbl.config(bg=DEFAULT_STYLE[0], fg=DEFAULT_STYLE[1])
                label_row.append(lbl)
            self.grid_labels.append(label_row)
        for r in range(parsed.height):
            self.grid_frame.grid_rowconfigure(r, weight=1)
        for c in range(parsed.width):
            self.grid_frame.grid_columnconfigure(c, weight=1)

    def _gather_bot_specs(self) -> List[BotSpec]:
        specs: List[BotSpec] = []
        for name, tank_id, team in TURN_ORDER:
            row = self.bot_rows[name]
            path = row.get_path()
            if not os.path.isfile(path):
                raise FileNotFoundError(f"봇 파일을 찾을 수 없습니다: {path}")
            specs.append(BotSpec(name=name, path=path, tank_id=tank_id, team=team))
        return specs

    def start_simulation(self) -> None:
        if self.orchestrator is not None:
            messagebox.showinfo("Info", "이미 시뮬레이션이 실행 중입니다.")
            return
        if not self.map_text:
            messagebox.showwarning("Warning", "먼저 맵을 로드하세요.")
            return
        try:
            engine = GameEngine.from_text(self.map_text)
            specs = self._gather_bot_specs()
            orchestrator = TurnOrchestrator(engine, specs)
            orchestrator.start()
        except Exception as exc:
            messagebox.showerror("Start Failed", str(exc))
            return
        self.engine = engine
        self.orchestrator = orchestrator
        self.status_label.config(text="Running (paused)")
        self.update_display(self.engine.get_snapshot())

    def step_simulation(self) -> None:
        if not self.orchestrator:
            messagebox.showwarning("Warning", "먼저 시뮬레이션을 시작하세요.")
            return
        try:
            snapshot = self.orchestrator.step()
        except Exception as exc:
            messagebox.showerror("Step Error", str(exc))
            self.stop_simulation()
            return
        self.update_display(snapshot)

    def run_simulation(self) -> None:
        if not self.orchestrator:
            messagebox.showwarning("Warning", "먼저 시뮬레이션을 시작하세요.")
            return
        if self.running:
            return
        self.running = True
        self.status_label.config(text="Running")
        self._schedule_next_step()

    def _schedule_next_step(self) -> None:
        if not self.running:
            return
        try:
            snapshot = self.orchestrator.step() if self.orchestrator else None
        except Exception as exc:
            messagebox.showerror("Run Error", str(exc))
            self.stop_simulation()
            return
        if snapshot:
            self.update_display(snapshot)
        self.after_id = self.after(150, self._schedule_next_step)

    def pause_simulation(self) -> None:
        self.running = False
        self.status_label.config(text="Paused" if self.orchestrator else "Idle")
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None

    def stop_simulation(self) -> None:
        self.pause_simulation()
        if self.orchestrator:
            try:
                self.orchestrator.stop()
            except Exception:
                pass
        self.orchestrator = None
        self.engine = None
        self.status_label.config(text="Stopped")


    def _style_cell(self, label: tk.Label, token: str) -> None:
        style = UNIT_STYLES.get(token)
        if style is None:
            style = TERRAIN_STYLES.get(token, DEFAULT_STYLE)
        bg, fg, text_override = style
        display_text = token if text_override is None else text_override
        label.config(text=display_text, bg=bg, fg=fg)


    def update_display(self, snapshot: Dict[str, object]) -> None:
        base_grid = snapshot["base_grid"]
        display_grid = [row[:] for row in base_grid]
        for gid, info in snapshot["tanks"].items():
            if info["alive"] and info["position"]:
                r, c = info["position"]
                display_grid[r][c] = gid
        for gid, info in snapshot["turrets"].items():
            if info["alive"] and info["position"]:
                r, c = info["position"]
                display_grid[r][c] = gid

        for r, row in enumerate(display_grid):
            for c, token in enumerate(row):
                if r < len(self.grid_labels) and c < len(self.grid_labels[r]):
                    self._style_cell(self.grid_labels[r][c], token)

        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, f"Turn: {snapshot['turn']}\n\n")
        self.info_text.insert(tk.END, "Tanks:\n")
        for gid, info in snapshot["tanks"].items():
            self.info_text.insert(
                tk.END,
                f" {gid}: HP={info['hp']}, Ammo={info['normal_ammo']}, Mega={info['mega_ammo']}, YellowCards={info.get('yellow_cards', 0)}, Turns={info.get('turn_count', 0)}, Pos={info['position']}\n",
            )
        self.info_text.insert(tk.END, "\nTurrets:\n")
        for gid, info in snapshot["turrets"].items():
            self.info_text.insert(
                tk.END,
                f" {gid}: HP={info['hp']}, Pos={info['position']}\n",
            )
        self.info_text.configure(state=tk.DISABLED)

    def on_close(self) -> None:
        self.stop_simulation()
        self.destroy()


def run_gui() -> None:
    app = GameGUI()
    app.mainloop()

