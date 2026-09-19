"""
=========================================================
Search And Rescue System
Main Application
=========================================================
"""

from pipelines.detection_pipeline import DetectionPipeline
from core.logger import logger

class SARApplication:
    def __init__(self):
        logger.info("Initializing SAR System...")
        self.pipeline = DetectionPipeline()

    def run(self):
        self.pipeline.run()
        
if __name__ == "__main__":
    app = SARApplication()
    app.run()
