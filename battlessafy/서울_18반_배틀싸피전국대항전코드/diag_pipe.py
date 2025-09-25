import time
from pathlib import Path

from battlessafygui.engine import GameEngine
from battlessafygui.orchestrator import BotSpec, TurnOrchestrator


def main() -> None:
    map_text = Path('map.txt').read_text(encoding='utf-8')
    engine = GameEngine.from_text(map_text)
    specs = [
        BotSpec(name='Attacker1', path='Attacker1.py', tank_id='A1', team='attackers'),
        BotSpec(name='Player1', path='Player1_??18_????.py', tank_id='B1', team='defenders'),
    ]
    orch = TurnOrchestrator(engine, specs)
    try:
        orch.start()
        for i in range(10):
            print('stepping', i)
            snapshot = orch.step()
            print('turn', snapshot['turn'])
            time.sleep(0.2)
    finally:
        orch.stop()


if __name__ == '__main__':
    main()
