import time
import logging
from lerobot.robots.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hand_read():
    try:
        config = UnitreeG1Dex3Config()
        config.is_simulation = False
        config.cameras = {} # Disable cameras for this test
        
        logger.info("Connecting to Unitree G1 Dex3...")
        robot = UnitreeG1Dex3(config)
        robot.connect()
        logger.info("Connected!")
        
        # Read a few observations
        for i in range(10):
            obs = robot.get_observation()
            
            # Check for hand data
            left_thumb = obs.get("left_hand_thumb_0_joint.q", None)
            right_thumb = obs.get("right_hand_thumb_0_joint.q", None)
            
            if left_thumb is not None:
                logger.info(f"Step {i}: Left Thumb 0={left_thumb:.4f}, Right Thumb 0={right_thumb:.4f}")
            else:
                logger.warning(f"Step {i}: Hand data missing in observation keys: {list(obs.keys())}")
            
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        if 'robot' in locals() and robot:
            robot.disconnect()

if __name__ == "__main__":
    test_hand_read()
