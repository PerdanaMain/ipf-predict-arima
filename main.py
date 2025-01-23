import array
from concurrent.futures import ProcessPoolExecutor
import time
from tracemalloc import start
from model import *
from non_vibration_train import main as non_vibration_train_main
from vibration_train import main as vibration_train_main
from predict_detail import main as predict_detail
import asyncio
import logging
import schedule  # type: ignore


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training_scheduler.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


async def start_training(part):
    features_id = "9dcb7e40-ada7-43eb-baf4-2ed584233de7"
    try:
        # run sequential
        await asyncio.get_event_loop().run_in_executor(
            None, non_vibration_train_main, part[0], features_id
        )
        await asyncio.get_event_loop().run_in_executor(
            None, predict_detail, part[0]
        )

        logger.info(f"Training completed for part: {part[1]}")
    except Exception as e:
        logger.error(f"Error training part_id {part[1]}: {e}")


async def start_non_dcs_training(part):
    try:
        features_id = "9dcb7e40-ada7-43eb-baf4-2ed584233de7"
        
        await asyncio.get_event_loop().run_in_executor(
            None, vibration_train_main, part[0], features_id
        )
        
        await asyncio.get_event_loop().run_in_executor(
            None, predict_detail, part[0]
        )
        logger.info(f"Training completed for part: {part[1]}")
    except Exception as e:
        logger.error(f"Error training part_id {part[1]}: {e}")

async def train_all_parts():
    try:
        parts = get_parts()
        non_dcs = get_non_dcs_parts()
        logger.info(f"Start Training for {len(parts)} parts and {len(non_dcs)}  non dcs parts")
        logger.info("=====================================")

        tasks = [start_training(part) for part in parts]
        second_tasks = [start_non_dcs_training(part) for part in non_dcs]

        # logger.info("=====================================")
        # logger.info("Start Training for dcs parts")
        # await asyncio.gather(*tasks)
        logger.info("=====================================")
        logger.info("Start Training for non dcs parts")
        await asyncio.gather(*second_tasks)

        logger.info("All training tasks completed")
    except Exception as e:
        logger.error(f"Error in train_all_parts: {e}")


def task():
    # Menjalankan asyncio event loop
    asyncio.run(train_all_parts())


def main():
    print(f"Starting scheduler at: {datetime.now(pytz.timezone('Asia/Jakarta'))}")
    print_log(f"Starting scheduler at: {datetime.now(pytz.timezone('Asia/Jakarta'))}")

    # Schedule task setiap 1 jam
    schedule.every(12).hours.at(":00").do(task)
    # schedule.every(6).hour.at(":00").do(feature)

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            print("Scheduler stopped by user")
            print_log("Scheduler stopped by user")
            break
        except Exception as e:
            print(f"Scheduler error: {e}")
            print_log(f"Scheduler error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    # Run the async main function
    # main()
    task()
