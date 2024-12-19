from model import *
from non_vibration_train import main as non_vibration_train_main
from vibration_train import main as vibration_train_main
import asyncio
import concurrent.futures
from typing import List, Tuple
import multiprocessing as mp
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
from apscheduler.triggers.cron import CronTrigger  # type: ignore
import logging
from datetime import datetime
import pytz

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


async def train_feature(
    part_id: str,
    features_id: str,
    features_name: str,
    part_name: str,
    is_vibration: bool,
):
    """Execute training for a single feature asynchronously"""
    logger.info(f"Training {part_name} {features_name}...")

    try:
        if is_vibration:
            await asyncio.get_event_loop().run_in_executor(
                None, vibration_train_main, part_id, features_id
            )
        else:
            await asyncio.get_event_loop().run_in_executor(
                None, non_vibration_train_main, part_id, features_id
            )

        logger.info(f"Finished training {part_name} {features_name}")
        logger.info("=====================================")
    except Exception as e:
        logger.error(f"Error training {part_name} {features_name}: {str(e)}")


async def process_part(
    part: Tuple[str, str, str], vib_type_id: str, non_vibration_features: str
):
    """Process a single part with all its features"""
    part_id, part_name, part_type = part
    is_vibration = part_type == vib_type_id

    features = (get_vibration_features if is_vibration else get_non_vibration_features)(
        non_vibration_features
    )

    tasks = [
        train_feature(part_id, feat[0], feat[1], part_name, is_vibration)
        for feat in features
    ]
    await asyncio.gather(*tasks)


async def run_training():
    """Main training function to be scheduled"""
    # Constants
    VIB_TYPE_ID = "b45a04c6-e2e2-465a-ad84-ccefe0f324d2"
    NON_VIBRATION_FEATURES = "9dcb7e40-ada7-43eb-baf4-2ed584233de7"

    try:
        # Get all parts at once
        parts = get_parts()
        logger.info(f"Start Training for {len(parts)} parts...")
        logger.info("=====================================")

        # Create tasks for all parts
        tasks = [
            process_part(part, VIB_TYPE_ID, NON_VIBRATION_FEATURES) for part in parts
        ]

        # Run all tasks concurrently
        await asyncio.gather(*tasks)
        logger.info("Daily training completed successfully")

    except Exception as e:
        logger.error(f"Error in daily training: {str(e)}")


async def main():
    # Create scheduler
    scheduler = AsyncIOScheduler()

    # Schedule the training to run daily at 1 AM (adjust timezone and time as needed)
    scheduler.add_job(
        run_training,
        trigger=CronTrigger(
            hour=1,  # Run at 1 AM
            minute=0,
            timezone=pytz.timezone("Asia/Jakarta"),  # Adjust to your timezone
        ),
        id="daily_training",
        name="Daily Training Job",
        replace_existing=True,
    )

    try:
        scheduler.start()
        logger.info("Scheduler started. Training will run daily at 1 AM Jakarta time.")

        # Keep the script running
        while True:
            await asyncio.sleep(60)  # Check every minute

    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down...")
        scheduler.shutdown()
        logger.info("Scheduler shutdown complete")


if __name__ == "__main__":
    # Set up multiprocessing
    mp.set_start_method("spawn", force=True)

    # Run the async main function
    asyncio.run(main())
