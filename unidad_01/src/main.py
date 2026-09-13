import pandas as pd
import numpy as np

def hello_world():
    print("\n" + "━" * 60)
    print("\033[1;36m╔" + "═" * 24 + " H O L A   M U N D O " + "═" * 24 + "╗\033[0m")
    print("\033[1;35m║\033[0m" + " " * 4 + "\033[1;33m🚀 Hello, world! Bienvenido al entorno dev\033[0m" + " " * 4 + "\033[1;35m║\033[0m")
    print("\033[1;36m╚" + "═" * 56 + "╝\033[0m")
    print("━" * 60 + "\n")

print(pd.__version__)
print(np.__version__)
hello_world()