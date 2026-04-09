import os
import sys
import logging
from synergy.autonomous_engine import AutonomousTrainingEngine

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/autonomous_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.getLogger("ORIEN_MASTER")

def main():
    log.info("="*50)
    log.info("💎 ORIEN AUTONOMOUS DEEP LEARNING SYSTEM V2.0")
    log.info("="*50)
    
    try:
        engine = AutonomousTrainingEngine(
            dataset_path="training/splits", # Using the preprocessed splits
            output_dir="models/autonomous_synergy"
        )
        
        # Override config for a faster demonstration/run if needed
        # engine.config["max_epochs"] = 100
        
        engine.run_full_pipeline()
        
    except Exception as e:
        log.error(f"🛑 CRITICAL PIPELINE FAILURE: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
