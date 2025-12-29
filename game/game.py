import sys
import time
from stable_baselines3 import DQN
from training.environment import ChompEnv
from rich.console import Console
from typing import Tuple

console = Console()


# spinner animation
def loading() -> None:
    with console.status("[#A0E8E3]bot move", spinner="shark"):
        time.sleep(3)


# alphanumeric to action space integer conversion
def human_action(env: ChompEnv) -> Tuple[int, str]:
    valid_actions: list[int] = env.get_valid_actions()

    move: str = console.input("make your move: ").strip()

    if (
        len(move) < 2 or not move[0].isalpha() or not move[1:].isdigit()
    ):  # filtering out invalid move formats
        console.print("[bold #CFA8FF]invalid input. you lose.[/bold #CFA8FF]")
        sys.exit(1)

    action: str = move.lower()

    try:
        letter: int = ord(action[0])
        number: int = int(action[1:])
    except (ValueError, IndexError):  # input error handling
        console.print("[bold #CFA8FF]invalid input. you lose.[/bold #CFA8FF]")
        sys.exit(1)

    if (
        letter - 97 >= env.NUM_COLS or number - 1 >= env.NUM_ROWS
    ):  # filtering out moves that are out of bounds
        console.print(
            "[bold #CFA8FF]that move’s off the board. you lose.[/bold #CFA8FF]"
        )
        sys.exit(1)

    action_index: int = (letter - 97) + ((number - 1) * env.NUM_COLS)  # 97 is a

    if (
        action_index not in valid_actions
    ):  # filtering out moves that are no longer valid
        env.done = True
        console.print("[bold #CFA8FF]invalid move. you lose![/bold #CFA8FF]")
        sys.exit(1)

    return action_index, move.upper()


# gameplay loop
def play() -> None:
    try:
        # initialization
        model: DQN = DQN.load("model/chomp_dqn")
        env: ChompEnv = ChompEnv()

        obs, _ = env.reset()
        console.print("\n" + env.render())
        env.opponent_mode = False  # no simulated opponent

        while True:
            # player move
            player_action, display_move = human_action(env)
            env.update_grid(player_action)
            obs = env.grid.numpy()

            console.print(
                f"[#FF99CC]you played:[/#FF99CC] [#A0E8E3]{display_move}[/#A0E8E3]\n"
            )
            console.print(env.render())

            if env.done:
                console.print("[bold #CFA8FF]game over! you lose.[/bold #CFA8FF]")
                sys.exit(0)

            # fake thinking
            loading()

            # bot move
            bot_action, _states = model.predict(obs, deterministic=True)
            obs, _reward, _done, _truncated, _info = env.step(bot_action)

            bot_action_number: int = (bot_action // env.NUM_COLS) + 1
            bot_action_letter: int = 97 + (bot_action % env.NUM_COLS)

            console.print(
                f"[#A0E8E3]the bot played:[/#A0E8E3] [#FF99CC]{chr(bot_action_letter).upper()}{bot_action_number}[/#FF99CC]\n"
            )
            console.print(env.render())

            if env.done:
                if bot_action != env.poison:
                    console.print(
                        "[bold #A0E8E3]the bot played an invalid move!"
                    )  # handling invalid bot moves
                    sys.exit(1)

                console.print(
                    "[bold #A0E8E3]game over! the bot loses.[/bold #A0E8E3]"
                )  # handling bot loss
                sys.exit(0)
    except KeyboardInterrupt:
        console.print("[bold red]oops! you left.[/bold red]")  # handling ctrl + c
        sys.exit(1)
