Product Requirements Document (PRD)
“BattleSSAFY GUI Simulator” — run 6 existing Python bots with a lightweight, local GUI
TL;DR for Claude Code

Build a stand-alone desktop app (Python) that:

Loads and runs six existing Python files (unchanged) in a fixed turn order.

Emulates the old “main program” so the bots’ current init/submit/receive socket API still “works” without the legacy executable.

Simulates the game loop (grid map, movement, shooting, decoding at supply) and renders a GUI (grid, HUD, logs, controls).

Lets me iterate quickly: start/stop, step, speed, reload code on save, run batch simulations, and overwrite bots’ files in place (keep file structure/entrypoints).

Keeps game rules & I/O format compatible with our current code so I can later copy the improved logic back into the original repos.

1) Goals & Non-Goals
Goals

Run without the legacy BattleSSAFY executable.
Provide a drop-in compatibility layer so the six player scripts execute unmodified.

Faithful simulation.
Implement rules required by the current bots:

Terrain: G, W, S, T, F and entities M, H, allies, E1/E2/E3, X.

Actions: R A/D A/L A/U A (move), R F/D F/L F/U F (shoot), R F M/... (mega shoot), "G <decoded_text>" (supply).

Shot range 3, first hit wins, T can be destroyed, default damages: normal 30, mega 70.

Exact turn orchestration.
Run order must be:
Attacker1 → Player1 → Attacker2 → Player2 → Attacker3 → Player3
(1,3,5 are one team; 2,4,6 the other).

GUI for fast iteration.
Start/stop, step, set speed, show grid & HP/ammo, show last commands & logs, load maps, seed RNG, save replays.

File-compatible.
Keep bots’ filenames/entrypoints unchanged (so I can overwrite later).

Non-Goals

Not a networked multiplayer game.

No need to ship installers; a single python -m battlesaf_gui is fine.

No fancy 3D; simple 2D grid is enough.

2) Files & Inputs

Six bot files will be provided (examples):

Attacker1.py, Attacker2.py, Attacker3.py

Player1_서울18_신건하.py, Player2_서울18_박유신.py, Player3_서울18_이명재.py

Map input format (keep identical to slides / the existing bots):
First line: H W allies enemies codes
Then H rows of tiles (space-separated cell tokens).
Then ally lines, enemy lines, then optional cipher lines.
(The GUI must be able to paste/load this text; the engine will also be able to export the current state back into this text format.)

Optional: a Map Builder panel that writes the same text format.

3) Key Game Rules (engine must implement)

Keep these as config defaults so we can tweak without code changes.

Board: H×W grid.

Terrain cells:

G grass: passable.

W water: impassable for movement. (Still counts as “transparent” for shots; the current bots’ can_attack allow W along line of fire.)

S sand/swamp: passable (optionally HP penalty only in battle mode; default no penalty).

T tree: impassable to move, destroyable (1 normal shot destroys one T).

F supply: stand adjacent to respond "G <decoded>" to gain 1 Mega (M).

Entities:

M = my tank (per bot).

H / X = allied/enemy turrets (bases).

E1, E2, E3 = enemy tanks (other team).

Other ally tank IDs appear as non-reserved letters (the bots already detect them).

Actions per turn (one command string from each bot):

Move: "R A" | "D A" | "L A" | "U A" (right/down/left/up 1 tile)

Shoot: "R F" | ... (normal), "R F M" | ... (mega)

Decode: "G <PLAINTEXT>" when adjacent to F

Wait: "S" (engine will treat as no-op)

Shooting:

Range: up to 3 tiles in a straight line.

Line-of-fire must contain only G/S/W until the first hit.

First object hit stops the projectile:

If first hit is T: destroy that T (no further travel).

If it is a tank or turret: apply damage and stop.

Damage: normal 30, mega 70.

Movement:

Can move into G (always), cannot enter W.

Entering T is not allowed (must shoot first).

Forbidden safety cells (around H depending on corner) must be enforced as in bots.

Win/End conditions (configurable):

Stop after N turns or when one turret HP ≤ 0.

Ammo:

Normal shells: unlimited in “battle mode” examples; otherwise initialize from ally line (keep both options).

Mega shells: gained by successful decode at F.

Decode:

A bot adjacent to F may issue "G <decoded_text>".

If the text matches engine’s expected plaintext for the cipher: award +1 mega shell (idempotent per F instance).

Cipher type used currently: Caesar; engine will store the expected plaintext (we can also pass through unvalidated if needed for compatibility).

4) Compatibility Layer (critical)

We must let the existing six Python files run unmodified even though they currently:

Use socket to connect to 127.0.0.1:8747.

Call init(nickname) → submit(command) → receive() in a loop.

Expect the engine to send a single string each turn containing the whole game state text (same format as above).

Plan:

Provide an in-process fake socket that conforms to Python’s socket.socket API used by the bots:

Implement .connect(), .send(), .recv(), .close().

Route bytes to an internal Message Bus owned by the GUI engine.

When a bot calls init(nickname), our fake socket registers that bot with the engine and immediately returns the initial game state text via .recv().

Each bot turn:

Bot calls submit("<COMMAND>") → our fake socket posts the command to the engine’s inbox and blocks until the engine sends the next state text.

Engine processes turns in the fixed order; after resolving the bot’s action and any consequences, it serializes the updated state text and replies to that bot (and updates GUI).

We will monkey-patch socket.socket for the bot processes (or run them in isolated interpreters with sitecustomize), so we don’t modify bot source files.

Isolation Model:

Start each bot in a separate Python subprocess (preferred) to avoid global monkey-patch clashes and allow termination/reload.

Provide a small bootstrap that:

Sets PYTHONPATH to include compat_socket.py (monkey-patch).

Executes the bot file (runpy.run_path).

Engine keeps per-bot queues (bidirectional) to send/receive strings.

5) Turn Orchestration

Fixed order per full round:

Attacker1.py

Player1_서울18_신건하.py

Attacker2.py

Player2_서울18_박유신.py

Attacker3.py

Player3_서울18_이명재.py

Engine loop (pseudo):

while not end_condition:
    for bot in ordered_bots:
        # 1) deliver current state text to this bot (already given at last recv)
        cmd = await bot.read_command(timeout=TURN_TIMEOUT)
        validated = validate_and_normalize(cmd)  # default to "S" on error

        apply_action(bot_id, validated, state)  # movement/shots/decodes
        resolve_hits_and_destruction(state)
        clamp_HP_and_check_base(state)

        # Send updated state text back to THIS bot only
        bot.send_state(serialize_state(state))

    # After 6 actions, it’s one “round” – repaint GUI, log, maybe tick timers


Notes:

Conflicts (two tanks moving into same cell) can’t happen because moves are sequential in this fixed order.

Shooting uses current board after prior actions in the same round.

6) GUI Requirements

Tech: Tkinter or PySide6 (PyQt) — pick the quickest you can ship.

Panels

Grid Board

Draw H×W tiles with colors/icons: G/W/S/T/F and entities M/H/E1/E2/E3/X.

Animate the acting bot’s tile for the current step.

Controls

Load map (paste text / open file).

Choose six files (file pickers with defaults prefilled).

Buttons: Run / Pause / Step / Reset, Speed slider (1–20 steps/sec).

Seed input (RNG).

Batch mode: “Run N rounds, keep best log”.

HUD

Team A (Attackers): HP for E1/E2/E3 + X HP; ammo (normal/mega).

Team B (Players): HP for their tanks + H HP; ammo.

Turn Log

Per step: BotName → COMMAND → result, damage notes, decode events.

Inspector

Hover a cell: show coordinates, terrain, occupant, line-of-fire preview.

Replay

Save a compact JSON replay (seed + initial map + per-turn actions).

Load & scrub via slider.

Hot Reload

Watch the six files; when any changes, offer Restart Bots (single click).

7) Data Model
State:
  grid: List[List[Cell]]  # terrain + occupant references
  units:
    - tanks: dict[id, Tank]  # id in {"M","E1","E2","E3", plus ally labels}
    - turrets: {"H": Turret, "X": Turret}
  codes: List[Cipher]       # plaintext per F (for validation)
  turn_index: int           # 0..5 inside a round
  round_count: int
  rng_seed: int

Tank:
  id, team, r, c, hp, facing, ammo_normal, ammo_mega

Turret:
  id, r, c, hp

Cell:
  terrain: Enum("G","W","S","T","F")
  occupant: Optional[UnitId]

Config:
  max_rounds, shot_range=3, dmg_normal=30, dmg_mega=70,
  pass_through_for_shots={"G","S","W"}, move_blockers={"W","T"},
  forbidden_cells_fn(turret_pos, H, W) -> Set[(r,c)]

8) State Serialization (for bots)

Exactly match current bots’ expectations.
Use the same text format they already parse:

<height> <width> <num_allies> <num_enemies> <num_codes>
<map rows: H lines of space-separated tokens>
<ally lines: e.g., "M 100 R 1 0">
<enemy lines: e.g., "E1 10", "X 10">
<code lines: raw cipher strings>


map cells must include entity tokens at their positions (M, ally labels, E1/E2/E3, H, X).

Ammo counts appear on the M line as currently used by the bots: "M <hp> <dir> <normal> <mega>".

9) Validation & Fallbacks

If a bot sends an invalid command, treat it as "S" and log a warning.

Out-of-bounds or into W/T/forbidden: the move is ignored (or apply the bot’s own safety logic if it’s in the script).

Shooting with no ammo (for normal if limited; for mega if 0): ignore and log.

Decode when not adjacent to F: ignore and log.

10) Performance & Reliability

Target: 60+ steps/sec on a mid laptop for H,W ≤ 40.

Each bot subprocess has a turn timeout (e.g., 1–2s). On timeout, force "S" and keep sim running.

Crash isolation: if a bot process dies, show status red; allow Restart Bot.

11) Developer Tasks (Claude Code)

Project skeleton

battlessafygui/engine.py (rules, state, serializer)

battlessafygui/compat_socket.py (fake socket)

battlessafygui/orchestrator.py (subprocess mgmt, turn order, message bus)

battlessafygui/gui.py (Tkinter/PySide6 UI)

battlessafygui/replay.py

battlessafygui/maps.py (parser/builder)

main.py (entry)

Fake socket
Monkey-patch socket.socket only inside child processes:

Provide send()/recv() to pass UTF-8 strings to engine via multiprocessing.Pipe or Queue.

Serializer
Implement state_to_text(state) and text_to_state(text) matching the bots’ current parsing.

Rules engine
Implement movement, shooting (range=3, pass-through terrain), tree destruction, decode→+1 mega, forbidden cells around H when at TR (0,W-1) or BL (H-1,0).

Orchestration
Fixed order 1→6 each round. After each action, update state and send fresh text to the acting bot (as the original main program did).

GUI
Grid drawing, controls, HUD, logs, replay, hot reload, file pickers for six scripts, map loader, seed, speed.

Batch mode (optional but desired)
Run N seeds × maps; export CSV of results.

12) Test Plan

Unit tests for: serializer round-trip, LOS/shot resolution, tree destruction, forbidden cells, decode adjacency, sequential move order.

Golden tests: use a small 10×10 map from the slides and verify a deterministic sequence (seeded).

Manual: load the six current bot files; verify no source changes are required and commands flow each turn; stepping shows consistent state transitions.

13) Risks & Mitigations

Socket emulation mismatch → Keep API minimal (only methods used). Add verbose logging if bots read unexpected bytes.

Rule divergence → Expose config toggles; keep defaults matching current bot assumptions (W/S transparent for shots, etc.).

Performance with subprocesses → Use queues with small payloads (single state string per turn), and limit redraw rate at high speeds.

14) Deliverables

Runnable app: python -m battlessafygui

Source with README (how to pick the six files, load a map, run, step, replay).

Example maps and a replay sample.

Minimal test suite (pytest).

15) Acceptance Criteria

I can select the six provided bot files, load the example map text, press Run, and watch the grid update as each bot acts in the exact order: 1,2,3,4,5,6.

Bots run without modifying their source (compat socket works).

Movement/shooting/decoding behave as in the bots’ logic (tree destruction, shot range 3, first-hit, mega damage 70).

I can pause, step, change speed, and reload a modified bot file and continue.

I can export a replay and load it back.

Appendix A — LOS / Shooting Resolution

From (r,c) to direction d ∈ {R,D,L,U}, walk 1..3 tiles:

If encounter T: remove T; stop.

Else if encounter a unit (tank or turret): apply damage; stop.

Else if tile ∉ {"G","S","W"}: stop with no effect.

Else continue until range exhausted.

Appendix B — Forbidden Cell Function

If H at (0, W-1): forbid
(0,W-2),(0,W-3),(1,W-1),(1,W-2),(1,W-3),(2,W-1),(2,W-2),(2,W-3)

If H at (H-1, 0): forbid
(H-1,1),(H-1,2),(H-2,0),(H-2,1),(H-2,2),(H-3,0),(H-3,1),(H-3,2)

If anything here is unclear or you want me to prioritize the minimal viable path (engine + fake socket + very simple Tk grid), I can trim scope.


16 16 4 3 0
F G G G G T T T T T T G G G G X
G G G G G R R G G R R G E1 G G G
G G G R G R R G G R R G G G G G
G G G G G T T G G T T G G E3 G G
G G G G G T T G G T T G R G G G
G G G R G G G G G G G G G G E2 G
G G G G G T F S S F T G G G G G
G G G G G G S T T S G G R G G G
G G G R G G S T G S G G G G G G
G M G G G G F S S F T G G G G G
G G G G G G G G G G G G R G G G
M3 G G R G T T G G T T G G G G G
G G G G G T T G G T T G G G G G
G G G M2 G R R G G R R G R G G G
G G G G G R R G G R R G G G G G
H G G G G G G G G G G G G G G F
M 100 L 99 1
M2 100
M3 70
H 100
E1 100
E2 100
E3 100
X 100