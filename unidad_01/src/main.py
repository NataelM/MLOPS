import pandas as pd
import numpy as np

def hello_world():
    print("\n" + "━" * 60)
    box_width = 58
    title = "H O L A   M U N D O"
    msg = "🚀 Hello, world! Bienvenido al entorno dev"

    print(f"\033[1;36m╔{'═' * (box_width + 2)}╗\033[0m")
    print(f"\033[1;35m║\033[0m {title.center(box_width)} \033[1;35m║\033[0m")
    print(f"\033[1;35m║\033[0m {msg.center(box_width)} \033[1;35m║\033[0m")
    print(f"\033[1;36m╚{'═' * (box_width + 2)}╝\033[0m")
    print("━" * 60 + "\n")

print(pd.__version__)
print(np.__version__)
hello_world()