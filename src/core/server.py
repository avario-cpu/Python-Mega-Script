"""
Websocket server made for listening to a basic StreamDeck Websocket client
plugin. This is where the scripts will be launched from... and sometimes
communicated with, although this is the kind of complex stuff I'll worry
about later.
"""

import asyncio
import os
import subprocess

import aiosqlite
import websockets
from websockets import WebSocketServerProtocol

from src.config.settings import PROJECT_DIR_PATH, PYTHONPATH
from src.core import slots_db_handler as sdh
from src.core import terminal_window_manager_v4 as twm
from src.core.constants import (
    APPS_DIR_PATH,
    LOCK_FILES_DIR_PATH,
    STOP_SUBPROCESS_MESSAGE,
    SUBPROCESSES_PORTS,
    TERMINAL_WINDOW_SLOTS_DB_FILE_PATH,
)
from src.utils.helpers import construct_script_name
from src.utils.logging_utils import setup_logger

SCRIPT_NAME = construct_script_name(__file__)

logger = setup_logger(SCRIPT_NAME, "DEBUG")


async def manage_subprocess(message: str):
    parts = message.split(maxsplit=1)
    if len(parts) >= 2:

        target = parts[0]
        instructions = parts[1].strip()
        if target not in list(SUBPROCESSES_PORTS.keys()):
            raise ValueError(
                f" Unknown target {target} not in {list(SUBPROCESSES_PORTS.keys())}"
            )
    else:
        raise ValueError(
            "Invalid message format, must be at least two "
            "separate words: target, instructions"
        )

    if instructions == "start":
        # Open the process in a new separate cli window: this is done to be
        # able to manipulate the position of the script's terminal with the
        # terminal window manager module.

        command = (
            f'start /min cmd /c "cd /d {PROJECT_DIR_PATH}'
            f"&& set PYTHONPATH={PYTHONPATH}"
            f"&& .\\venv\\Scripts\\activate"
            f"&& cd {APPS_DIR_PATH}"
            f"&& cd {target}"
            f"&& py {target}_main.py"
        )

        print(f"Attempting to start {target}")

        subprocess.Popen(command, shell=True)

    elif instructions == "stop":
        await send_message_to_subprocess_socket(
            STOP_SUBPROCESS_MESSAGE, SUBPROCESSES_PORTS[target]
        )

    elif instructions == "unlock":
        # Check for an ACK from the subprocess SOCK server. If there is one,
        # it means the subprocess is running and should not be attempted to
        # be unlocked.
        answer = await send_message_to_subprocess_socket(
            "Check for server ACK", SUBPROCESSES_PORTS[target]
        )
        if not answer:
            lock_path = os.path.join(LOCK_FILES_DIR_PATH, target)
            print(f"Checking '{lock_path}' for lock file")
            if os.path.exists(f"{lock_path}.lock"):
                os.remove(f"{lock_path}.lock")
                print(f"Found and removed lock for {target}")
            else:
                print("No lock found here, traveler.")
        else:
            print(f"{target} seems to be running, cannot unlock")


async def manage_windows(conn: aiosqlite.Connection, message: str):
    if message == "refit":
        await twm.refit_all_windows(conn)
    elif message == "refit_server":
        await twm.bring_windows_to_foreground(conn, server=True)
    else:
        print("Invalid windows path message, does not fit any use case")


async def manage_database(conn: aiosqlite.Connection, message: str):
    if message == "free all slots":
        await sdh.free_all_slots(conn)
        await sdh.free_all_denied_slots(conn)
    else:
        print("Invalid database path message, does not fit any use")


def create_websocket_handler(conn: aiosqlite.Connection):
    async def websocket_handler(websocket: WebSocketServerProtocol, path: str):
        async for message in websocket:
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            print(f"Received: '{message}' on path: {path}")

            if path == "/subprocess":  # Path to control subprocesses
                await manage_subprocess(message)

            elif path == "/windows":  # Path to manipulate windows properties
                await manage_windows(conn, message)

            elif path == "/database":  # Path to manipulate db entries
                await manage_database(conn, message)

            elif path == "/test":  # Path to test stuff
                if message == "get windows":
                    windows_names = await twm.get_all_windows_names(conn)
                    print(f"Windows in slot DB: {windows_names}")

    return websocket_handler


async def send_message_to_subprocess_socket(
    message: str, port: int, host="localhost"
) -> str:
    """Client function to send messages to subprocesses servers"""
    try:
        reader, writer = await asyncio.open_connection(host, port)

        writer.write(message.encode("utf-8"))
        print(f"SOCK: Sent: {message}")
        data = await reader.read(1024)
        msg = data.decode("utf-8")
        print(f"SOCK: Received: {msg}")

        print("SOCK: closing connection")
        writer.close()
        await writer.wait_closed()

    except OSError as e:
        print(e)
    return msg


async def main():
    print("Welcome to the server, bro. You know what to do.")
    conn = await sdh.create_connection(TERMINAL_WINDOW_SLOTS_DB_FILE_PATH)
    await twm.manage_window(conn, twm.WinType.SERVER, "SERVER")
    if conn:
        websocket_server = await websockets.serve(
            create_websocket_handler(conn), "localhost", 50000
        )

    try:
        await asyncio.Future()
    except Exception as e:
        print(e)
    finally:
        websocket_server.close()
        await websocket_server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
