"""
=========================================================
Search And Rescue System
Main Application
=========================================================
"""

import sys
from pipelines.detection_pipeline import DetectionPipeline
from core.logger import logger


class SARApplication:
    def __init__(self):
        logger.info("Initializing SAR System...")
        self.pipeline = DetectionPipeline()

    def run(self):
        try:
            self.pipeline.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down gracefully...")
            self.pipeline.stop()
        except Exception as e:
            logger.error(f"Application error: {e}")
            self.pipeline.stop()


if __name__ == "__main__":
    app = SARApplication()
    app.run()
