"""Start/status/stop the fallen-pose controller without duplicate processes."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/management'))
from manage_getup import main

if __name__ == '__main__':
    main(controller_path=ROOT/'scripts/recovery/continuous_fallen.py',
         default_output=ROOT/'outputs/recovery_fallen/continuous')
