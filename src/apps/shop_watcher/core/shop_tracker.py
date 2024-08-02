import asyncio
import random
import time
from typing import Optional

from src.connection.websocket_client import WebSocketClient
from src.core.constants import STREAMERBOT_WS_URL


class ShopTracker:
    def __init__(self, logger):
        self.shop_is_currently_open = False
        self.shop_opening_time = 0.0
        self.shop_open_duration_task = None
        self.flags = {"reacted_to_open_short": False, "reacted_to_open_long": False}
        self.logger = logger
        self.ws_client = WebSocketClient(STREAMERBOT_WS_URL, logger)

    async def initialize_ws_client(self):
        await self.ws_client.establish_connection()

    async def reset_flags(self):
        for key in self.flags:
            self.flags[key] = False

    async def track_shop_open_duration(self):
        while self.shop_is_currently_open:
            elapsed_time = round(time.time() - self.shop_opening_time)
            print(f"Shop has been open for {elapsed_time} seconds")

            if elapsed_time >= 5 and not self.flags["reacted_to_open_short"]:
                await self.react_to_shop_staying_open("short")
                self.flags["reacted_to_open_short"] = True

            if elapsed_time >= 15 and not self.flags["reacted_to_open_long"]:
                await self.react_to_shop_staying_open("long", seconds=elapsed_time)
                self.flags["reacted_to_open_long"] = True
            await asyncio.sleep(1)

    async def open_shop(self):
        if not self.shop_is_currently_open:
            self.shop_is_currently_open = True
            self.shop_opening_time = time.time()
            self.shop_open_duration_task = asyncio.create_task(
                self.track_shop_open_duration()
            )
            await self.react_to_shop("opened")

    async def close_shop(self):
        if self.shop_is_currently_open:
            self.shop_is_currently_open = False
            if self.shop_open_duration_task and not self.shop_open_duration_task.done():
                self.shop_open_duration_task.cancel()
                try:
                    await self.shop_open_duration_task
                except asyncio.CancelledError:
                    print("Shop open duration tracking stopped.")
            await self.react_to_shop("closed")
            await self.reset_flags()

    async def react_to_short_shop_opening(self):
        await self.ws_client.send_json_requests(
            "src/apps/shop_watcher/ws_requests/brb_buying_milk_show.json"
        )

    async def react_to_long_shop_opening(self, seconds):
        await self.ws_client.send_json_requests(
            "src/apps/shop_watcher/ws_requests/brb_buying_milk_hide.json"
        )
        start_time = time.time()
        while True:
            elapsed_time = (
                time.time() - start_time + seconds if seconds is not None else 0
            )
            seconds_only = int(round(elapsed_time))
            formatted_time = f"{seconds_only:02d}"
            with open("data/streamerbot_watched/time_with_shop_open.txt", "w") as file:
                file.write(
                    f"Bro you've been in the shop for {formatted_time} seconds, just buy something..."
                )

            await asyncio.sleep(1)

    async def react_to_shop_staying_open(
        self, duration: str, seconds: Optional[float] = None
    ):
        if duration == "short":
            print("rolling for a reaction to shop staying open for a short while...")
            if random.randint(1, 4) == 1:
                print("reacting !")
                await self.react_to_short_shop_opening()
            else:
                print("not reacting !")

        elif duration == "long":
            print("rolling for a reaction to shop staying open for a long while...")
            if random.randint(1, 3) == 1:
                print("reacting !")
                await self.react_to_long_shop_opening(seconds)
            else:
                print("not reacting !")

    async def react_to_shop(self, status: str):
        print(f"Shop just {status}")
        if status == "opened":
            await self.ws_client.send_json_requests(
                "src/apps/shop_watcher/ws_requests/dslr_hide.json"
            )
        elif status == "closed":
            await self.ws_client.send_json_requests(
                "src/apps/shop_watcher/ws_requests/dslr_show.json"
            )