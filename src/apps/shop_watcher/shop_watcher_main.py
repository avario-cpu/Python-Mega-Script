import asyncio

import cv2 as cv

from src.apps.shop_watcher.core.constants import SECONDARY_WINDOWS
from src.apps.shop_watcher.core.shared_events import (
    mute_ssim_prints,
    secondary_windows_spawned,
)
from src.apps.shop_watcher.core.shop_watcher import ShopWatcher
from src.apps.shop_watcher.core.socket_handler import ShopWatcherHandler
from src.connection.websocket_client import WebSocketClient
from src.core import terminal_window_manager_v4 as twm
from src.core.constants import (
    STOP_SUBPROCESS_MESSAGE,
    STREAMERBOT_WS_URL,
    SUBPROCESSES_PORTS,
)
from src.core.constants import TERMINAL_WINDOW_SLOTS_DB_FILE_PATH as SLOTS_DB
from src.utils.helpers import construct_script_name, print_countdown
from src.utils.logging_utils import setup_logger
from src.utils.script_initializer import setup_script

PORT = SUBPROCESSES_PORTS["shop_watcher"]
SCRIPT_NAME = construct_script_name(__file__)

logger = setup_logger(SCRIPT_NAME, "DEBUG")


async def run_main_task(slot: int, shop_watcher: ShopWatcher):
    mute_ssim_prints.set()
    main_task = asyncio.create_task(shop_watcher.scan_for_shop_and_notify())

    await secondary_windows_spawned.wait()
    await twm.manage_secondary_windows(slot, SECONDARY_WINDOWS)
    mute_ssim_prints.clear()
    await main_task
    return None


async def main():
    socket_server_task = None
    slots_db_conn = None
    try:
        slots_db_conn, slot = await setup_script(
            SCRIPT_NAME, SLOTS_DB, SECONDARY_WINDOWS
        )
        if slot is None:
            logger.error("No slot available, exiting.")
            return

        socket_server_handler = ShopWatcherHandler(
            port=PORT, stop_message=STOP_SUBPROCESS_MESSAGE, logger=logger
        )
        socket_server_task = asyncio.create_task(
            socket_server_handler.run_socket_server()
        )

        ws_client = WebSocketClient(STREAMERBOT_WS_URL, logger)
        await ws_client.establish_connection()

        shop_watcher = ShopWatcher(logger, socket_server_handler, ws_client)

        await run_main_task(slot, shop_watcher)

    except Exception as e:
        print(f"Unexpected error of type: {type(e).__name__}: {e}")
        logger.exception(f"Unexpected error: {e}")
        raise

    finally:
        if socket_server_task:
            socket_server_task.cancel()
            await socket_server_task
        if slots_db_conn:
            await slots_db_conn.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
    print_countdown()
