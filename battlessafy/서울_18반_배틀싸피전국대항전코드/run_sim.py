import os
import multiprocessing

os.environ['PYTHONIOENCODING'] = 'utf-8'


def main():
    from battlessafygui.engine import GameEngine
    from battlessafygui.orchestrator import TurnOrchestrator, BotSpec

    with open('map.txt', 'r', encoding='utf-8') as f:
        map_text = f.read()
    engine = GameEngine.from_text(map_text)

    def spec(name, path, tank_id, team):
        return BotSpec(name=name, path=path, tank_id=tank_id, team=team)

    bot_specs = [
        spec('Attacker1', 'Attacker1.py', 'A1', 'attackers'),
        spec('Player1', 'Player1_서울18_신건하.py', 'B1', 'defenders'),
        spec('Attacker2', 'Attacker2.py', 'A2', 'attackers'),
        spec('Player2', 'Player2_서울18_박유신.py', 'B2', 'defenders'),
        spec('Attacker3', 'Attacker3.py', 'A3', 'attackers'),
        spec('Player3', 'Player3_서울18_이명재.py', 'B3', 'defenders'),
    ]

    orchestrator = TurnOrchestrator(engine, bot_specs, command_timeout=2.0)

    original_process = engine.process_command

    def logging_process(actor_id, command):
        print(f"EXEC {actor_id}: {command}")
        return original_process(actor_id, command)

    engine.process_command = logging_process

    orchestrator.start()

    for turn in range(30):
        snap_before = orchestrator.engine.get_snapshot()
        if orchestrator.turn_index == 3:
            data = orchestrator.engine.build_game_data('B2')
            print("--- game_data for B2 before its turn ---")
            print(data)
            print("----------------------------------------")
        snap = orchestrator.step()
        b1 = snap['tanks']['B1']['position']
        b2 = snap['tanks']['B2']['position']
        b3 = snap['tanks']['B3']['position']
        print(f"turn {turn+1}: B1={b1}, B2={b2}, B3={b3}, next_idx={orchestrator.turn_index}")

    orchestrator.stop()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
