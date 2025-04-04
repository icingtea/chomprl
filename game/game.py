import sys
import time
from stable_baselines3 import DQN
from training.environment import ChompEnv
from rich.console import Console
from typing import Tuple

console = Console()

def loading() -> None:
    with console.status("[#A0E8E3]bot move", spinner="shark"):
        time.sleep(3)

def human_action(env: ChompEnv) -> Tuple[int, str]:
    valid_actions: list[int] = env.get_valid_actions()

    move: str = console.input("make your move: ").strip()

    if len(move) < 2 or not move[0].isalpha() or not move[1:].isdigit():
        console.print("[bold #CFA8FF]invalid input. you lose.[/bold #CFA8FF]")
        sys.exit(1)

    action: str = move.lower()
    
    try:
        letter: int = ord(action[0])
        number: int = int(action[1:])
    except (ValueError, IndexError):
        console.print("[bold #CFA8FF]invalid input. you lose.[/bold #CFA8FF]")
        sys.exit(1)

    if letter - 97 >= env.NUM_COLS or number - 1 >= env.NUM_ROWS:
        console.print("[bold #CFA8FF]that move’s off the board. you lose.[/bold #CFA8FF]")
        sys.exit(1)

    action_index: int = (letter - 97) + ((number - 1) * env.NUM_COLS)

    if action_index not in valid_actions:
        env.done = True
        console.print("[bold #CFA8FF]invalid move. you lose![/bold #CFA8FF]")
        sys.exit(1)

    return action_index, move.upper()

def play() -> None:
    try: 
        model: DQN = DQN.load("model/chomp_dqn")
        env: ChompEnv = ChompEnv()

        obs, _ = env.reset()
        console.print("\n" + env.render())
        env.opponent_mode = False

        while True:
            player_action, display_move = human_action(env)
            env.update_grid(player_action)
            obs = env.grid.numpy()

            console.print(f"[#FF99CC]you played:[/#FF99CC] [#A0E8E3]{display_move}[/#A0E8E3]\n")
            console.print(env.render())

            if env.done:
                console.print("[bold #CFA8FF]game over! you lose.[/bold #CFA8FF]")
                sys.exit(0)

            loading()

            bot_action, _states = model.predict(obs, deterministic=True)
            obs, _reward, _done, _truncated, _info = env.step(bot_action)

            bot_action_number: int = (bot_action // env.NUM_COLS) + 1
            bot_action_letter: int = 97 + (bot_action % env.NUM_COLS)

            console.print(f"[#A0E8E3]the bot played:[/#A0E8E3] [#FF99CC]{chr(bot_action_letter).upper()}{bot_action_number}[/#FF99CC]\n")
            console.print(env.render())

            if env.done:
                if bot_action != env.poison:
                    console.print("[bold #A0E8E3]the bot played an invalid move!")
                    sys.exit(1)

                console.print("[bold #A0E8E3]game over! the bot loses.[/bold #A0E8E3]")
                sys.exit(0)
    except KeyboardInterrupt:
        console.print("[bold red]oops! you left.[/bold red]")
        sys.exit(1)