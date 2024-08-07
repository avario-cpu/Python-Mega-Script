import os

from src.config.settings import PROJECT_DIR_PATH

# General project paths
TERMINAL_WINDOW_SLOTS_DB_FILE_PATH = os.path.join(
    PROJECT_DIR_PATH, "src/core/terminal_window_slots.db"
)
APPS_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "src/apps")

# Temp project paths
TEMP_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "temp")
LOG_DIR_PATH = os.path.join(TEMP_DIR_PATH, "logs")
LOCK_FILES_DIR_PATH = os.path.join(TEMP_DIR_PATH, "lock_files")
COMMON_LOGS_FILE_PATH = os.path.join(LOG_DIR_PATH, "all_logs.log")

# URLs
STREAMERBOT_WS_URL = "ws://127.0.0.1:50001/"

# Window names
SERVER_WINDOW_NAME = "MY SERVER"

# Subprocesses
STOP_SUBPROCESS_MESSAGE = "stop$subprocess"  # $ character is used to avoid accidental trigger from speech to text.
SUBPROCESSES_PORTS = {
    # list of subprocesses name and their socket server ports
    "shop_watcher": 59000,
    "pregame_phase_detector": 59001,
    "robeau": 59002,
    "synonym_adder": 59003,
}
