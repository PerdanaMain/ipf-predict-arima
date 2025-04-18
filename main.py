import time

from tomlkit import date
from model import *
from non_vibration_train import main as non_vibration_train_main
from vibration_train import main as vibration_train_main
import asyncio
import logging
import schedule  # type: ignore
from predict_detail import main as predict_detail
from flask import Flask, request, jsonify # type: ignore
from config import Config

app = Flask(__name__)


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


async def start_training(part, process_monitoring_id):
    features_id = "9dcb7e40-ada7-43eb-baf4-2ed584233de7"
    try:
        # run sequential
        await asyncio.get_event_loop().run_in_executor(
            None, non_vibration_train_main, part[0], features_id, process_monitoring_id
        )

        logger.info(f"Training completed for part: {part[1]}")
    except Exception as e:
        logger.error(f"Error training part_id {part[1]}: {e}")


async def start_non_dcs_training(part, process_monitoring_id):
    try:
        features_id = "9dcb7e40-ada7-43eb-baf4-2ed584233de7"
        
        await asyncio.get_event_loop().run_in_executor(
            None, vibration_train_main, part[0], features_id
        )
        
        logger.info(f"Training completed for part: {part[1]}")
    except Exception as e:
        logger.error(f"Error training part_id {part[1]}: {e}")

async def train_all_parts():
    try:
        parts = get_parts()
        non_dcs = get_non_dcs_parts()
        current_time = datetime.now(pytz.timezone("Asia/Jakarta"))
        
        logger.info(f"Start Training for {len(parts)} parts and {len(non_dcs)}  non dcs parts")
        logger.info("=====================================")
        
        process = create_process_monitoring(
            "ml-process",
            current_time,
            None,
            0,
            0,
            0,
            0,
            0,
            "processing"
        )

        tasks = [start_training(part, process["process_monitoring_id"]) for part in parts]
        second_tasks = [start_non_dcs_training(part, process["process_monitoring_id"]) for part in non_dcs]

        logger.info("=====================================")
        logger.info("Start Training for dcs parts")
        await asyncio.gather(*tasks)
        logger.info("=====================================")
        logger.info("Start Training for non dcs parts")
        await asyncio.gather(*second_tasks)

        current_process = get_process_monitoring(process_monitoring_id=process["process_monitoring_id"])
        disk_usage = disk_usage_count()
        total_parts = len(parts) + len(non_dcs)
        total_percent_parts = (current_process["total_data"] / total_parts) * 100
        
        print(f"Disk usage: {disk_usage}")
        print(f"Current process: {current_process}")
        update_process_monitoring(
            process_monitoring_id=process["process_monitoring_id"],
            process_monitoring_id="078f0dc3-7727-4453-94bf-2aedc357d6f4",
            end_time= datetime.now(pytz.timezone("Asia/Jakarta")),
            total_sensor=current_process["total_data"],
            data_row_count=current_process["data_row_count"],
            data_size_mb=disk_usage[0],
            disk_usage_percentage=disk_usage[2],
            sensor_data_percentage=total_percent_parts,
            status="success",
        )
        
        logger.info("All training tasks completed")
    except Exception as e:
        logger.error(f"Error in train_all_parts: {e}")


def task():
    # Menjalankan asyncio event loop
    asyncio.run(train_all_parts())


def main():
    print(f"Starting scheduler at: {datetime.now(pytz.timezone('Asia/Jakarta'))}")
    print_log(f"Starting scheduler at: {datetime.now(pytz.timezone('Asia/Jakarta'))}")

    # schedule task every day at 00:00
    schedule.every().day.at("00:00").do(task)

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

@app.route("/", methods=["GET"])
def hai():
    return jsonify({"message": "Welcome to the API!"})

@app.route("/train", methods=["GET"])
def home():
    try:
        task()
        return (
            jsonify(
                {
                    "message": f"Train completed at: {datetime.now(pytz.timezone('Asia/Jakarta'))}"
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=Config.PORT, host='0.0.0.0')
    
    # Run the async main function
    # main()
    # task()
