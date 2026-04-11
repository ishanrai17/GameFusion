import sys

# Fix paths for imports
sys.path.append('/content/GameFusion')
sys.path.append('/content/GameFusion/interaction_prediction')

# Fix argument parsing for local rank
sys.argv = [a.replace('--local-rank', '--local_rank') for a in sys.argv]

# Run GameFusion's training script
exec(open('/content/GameFusion/interaction_prediction/train.py').read())