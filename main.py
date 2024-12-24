import time
from model import *
from non_vibration_train import main as non_vibration_train_main
from vibration_train import main as vibration_train_main
import asyncio
from typing import Tuple
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

def task():
    asyncio.run(run_training())

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
    main()
