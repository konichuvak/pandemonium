from pathlib import Path

from dotenv import load_dotenv

from experiments.tools import RLogger

HOME_DIR = Path.home()
EXPERIMENT_DIR = Path(__file__).parent.absolute()
LIB_ROOT = EXPERIMENT_DIR.parent

env_path = EXPERIMENT_DIR / '.env'
load_dotenv(dotenv_path=env_path, verbose=True)
