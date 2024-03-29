import asyncio
import atexit
import logging
import time

import cv2 as cv
import mss
import numpy as np
import websockets
from skimage.metrics import structural_similarity as ssim
from websockets import WebSocketException

import constants
import denied_slots_db_handler as denied_sdh
import my_classes as my
import single_instance
import slots_db_handler as sdh
import terminal_window_manager_v4 as twm
import random

logger = logging.getLogger('shop_watcher')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('temp/logs/shop_watcher.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


class ShopTracker:
    def __init__(self):
        self.shop_is_currently_open = False
        self.shop_opening_time = None
        self.shop_open_duration_task = None
        self.flags = {"reacted_to_open>3": False, "reacted_to_open>15": False}

    async def reset_flags(self):
        for key in self.flags:
            self.flags[key] = False
        pass

    async def track_shop_open_duration(self, ws):
        while self.shop_is_currently_open:
            elapsed_time = round(time.time() - self.shop_opening_time)
            print(f"Shop has been open for {elapsed_time} seconds")
            if elapsed_time >= 2 and not self.flags["reacted_to_open>3"]:
                await react_to_shop_stayed_open(ws, 3)
                self.flags["reacted_to_open>3"] = True
            if elapsed_time >= 6 and not self.flags["reacted_to_open>15"]:
                await react_to_shop_stayed_open(ws, 15)
                self.flags["reacted_to_open>15"] = True
            await asyncio.sleep(1)  # Adjust the sleep time as necessary

    async def open_shop(self, ws):
        if not self.shop_is_currently_open:
            self.shop_is_currently_open = True
            self.shop_opening_time = time.time()
            self.shop_open_duration_task = asyncio.create_task(
                self.track_shop_open_duration(ws))
            await react_to_shop("opened", ws)

    async def close_shop(self, ws):
        if self.shop_is_currently_open:
            self.shop_is_currently_open = False
            if (self.shop_open_duration_task
                    and not self.shop_open_duration_task.done()):
                self.shop_open_duration_task.cancel()
                try:
                    await self.shop_open_duration_task
                except asyncio.CancelledError:
                    print("Shop open duration tracking stopped.")
            await react_to_shop("closed", ws)
            await self.reset_flags()


# Configuration


SCREEN_CAPTURE_AREA = {"left": 1883, "top": 50, "width": 37, "height": 35}
TEMPLATE_IMAGE_PATH = 'opencv/dota_shop_top_right_icon.jpg'
WEBSOCKET_URL = "ws://127.0.0.1:8080/"
SCRIPT_NAME = "dota2_shop_watcher"
SECONDARY_WINDOWS = [my.SecondaryWindow("opencv_shop_scanner", 100, 100)]

secondary_windows_have_spawned = asyncio.Event()
mute_main_loop_print_feedback = asyncio.Event()
stop_loop = asyncio.Event()


def exit_countdown():
    """Give a bit of time to read terminal exit statements"""
    for seconds in reversed(range(1, 5)):
        print("\r" + f'cmd will close in {seconds} seconds...', end="\r")
        time.sleep(1)
    exit()


async def establish_ws_connection():
    try:
        ws = await websockets.connect(WEBSOCKET_URL)
        logger.info(f"Established connection: {ws}")
        return ws
    except WebSocketException as e:
        logger.debug(f"Websocket error: {e}")
    except OSError as e:
        logger.debug(f"OS error: {e}")
    return None


async def handle_socket_client(reader, writer):
    while True:
        data = await reader.read(1024)
        if not data:
            print("Socket client disconnected")
            break
        message = data.decode()
        if message == constants.STOP_SUBPROCESS_MESSAGE:
            stop_loop.set()
        print(f"Received: {message}")
        writer.write(b"ACK from WebSocket server")
        await writer.drain()
    writer.close()


async def run_socket_server():
    server = await asyncio.start_server(handle_socket_client, 'localhost',
                                        9999)
    addr = server.sockets[0].getsockname()
    print(f"Serving on {addr}")

    try:
        await server.serve_forever()
    except asyncio.CancelledError:
        print("Socket server task was cancelled. Stopping server")
    finally:
        server.close()
        await server.wait_closed()
        print("Server closed")


async def capture_window(area):
    with mss.mss() as sct:
        img = sct.grab(area)
    return np.array(img)


async def compare_images(image_a, image_b):
    return ssim(image_a, image_b)


async def send_json_requests(ws, json_file_paths: str | list[str]):
    if isinstance(json_file_paths, str):
        json_file_paths = [json_file_paths]

    for json_file in json_file_paths:
        with open(json_file, 'r') as file:
            await ws.send(file.read())
        response = await ws.recv()
        logger.info(f"WebSocket response: {response}")


async def react_to_shop(status, ws):
    print(f"Shop just {status}")
    if status == "opened" and ws:
        await send_json_requests(
            ws, "streamerbot_ws_requests/dslr_hide.json")
    elif status == "closed" and ws:
        await send_json_requests(
            ws, "streamerbot_ws_requests/dslr_show.json", )
    pass


async def react_to_shop_stayed_open(ws, seconds):
    if seconds == 3:
        if random.randint(1, 4) == 1:
            await send_json_requests(
                ws, "streamerbot_ws_requests/brb_buying_milk_show.json")
    if seconds == 15:
        if random.randint(1, 4) == 1:
            start_time = time.time()
            while True:
                elapsed_time = time.time() - start_time + 15
                seconds_only = int(round(elapsed_time))
                formatted_time = f"{seconds_only:02d}"
                with (open("streamerbot_watched/time_with_shop_open.txt",
                           "w") as file):
                    file.write(
                        f"Bro you've been in the shop for {formatted_time} "
                        f"seconds, just buy something...")

                await asyncio.sleep(1)


async def scan_for_shop_and_notify(ws):
    shop_tracker = ShopTracker()
    template = cv.imread(TEMPLATE_IMAGE_PATH, cv.IMREAD_GRAYSCALE)

    while not stop_loop.is_set():
        frame = await capture_window(SCREEN_CAPTURE_AREA)
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        match_value = await compare_images(gray_frame, template)
        cv.imshow(SECONDARY_WINDOWS[0].name, gray_frame)
        secondary_windows_have_spawned.set()

        if cv.waitKey(1) == ord("q"):
            break
        if not mute_main_loop_print_feedback.is_set():
            print(f"SSIM: {match_value:.6f}", end="\r")

        if match_value >= 0.8:
            await shop_tracker.open_shop(ws)

        elif match_value < 0.8:
            await shop_tracker.close_shop(ws)
        await asyncio.sleep(0)
    if ws:
        await ws.close()
        cv.destroyAllWindows()
    print('loop terminated')


async def main():
    """If there are no single instance lock file, start the Dota2 shop_watcher
     module. Reposition the terminal right at launch."""
    if single_instance.lock_exists():
        slot = twm.manage_window(twm.WinType.DENIED, SCRIPT_NAME)
        atexit.register(denied_sdh.free_slot, slot)
        print("\n>>> Lock file is present: exiting... <<<")

    else:
        slot = twm.manage_window(twm.WinType.ACCEPTED,
                                 SCRIPT_NAME, SECONDARY_WINDOWS)

        single_instance.create_lock_file()
        atexit.register(single_instance.remove_lock)
        atexit.register(sdh.free_slot_named, SCRIPT_NAME)
        socket_server_task = asyncio.create_task(run_socket_server())
        mute_main_loop_print_feedback.set()  # avoid ugly lines due to caret
        # replacement print

        ws = None
        try:
            ws = await establish_ws_connection()
            main_task = asyncio.create_task(scan_for_shop_and_notify(ws))
            await secondary_windows_have_spawned.wait()
            twm.manage_secondary_windows(slot, SECONDARY_WINDOWS)
            mute_main_loop_print_feedback.clear()
            await main_task
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            socket_server_task.cancel()
            await socket_server_task
            cv.destroyAllWindows()
            if ws:
                await ws.close()


if __name__ == "__main__":
    asyncio.run(main())
    exit_countdown()
