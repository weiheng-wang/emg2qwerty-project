import sys
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

folder = sys.argv[1]
tf_files = glob.glob(f"{folder}/lightning_logs/version_0/events.out.tfevents.*")

if not tf_files:
    print(f"No TensorBoard logs found in {folder}")
    sys.exit()

for tf_file in tf_files:
    ea = EventAccumulator(tf_file)
    ea.Reload()
    tags = ea.Tags()['scalars']
    
    print(f"\n--- Final Scores for {folder} ---")
    for tag in tags:
        # We only care about CER metrics
        if 'CER' in tag or 'cer' in tag.lower():
            final_val = ea.Scalars(tag)[-1].value
            print(f"{tag}: {final_val:.2f}%")
