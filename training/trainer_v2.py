from utils import *
import sys

if __name__ == "__main__":
    i = 0
    configuration['device'] = f'cuda:{i}'
    if i > 1:
        configuration['logpath'] += "/residual"
        configuration['residual'] = True
    else:
        configuration['logpath'] += "/raw"
        configuration['residual'] = False
    if i%2 == 0:
        configuration['logpath'] += "_normalized"
        configuration['normalize'] = True
    else:
        configuration['logpath'] += "_unnormalized"
        configuration['normalize'] = False
    arguments = sys.argv[1:]  
    example_id = int(arguments[0])
    os.makedirs(f"/amin/tempigli/{os.path.basename(configuration['logpath'])}", exist_ok=True)

    if f"{toi[example_id].item()}_1024" not in os.listdir(f"/amin/tempigli/{os.path.basename(configuration['logpath'])}"):
        process_action(i, toi[example_id:example_id+1])
    else:
        print(f"Skipping {toi[example_id]}")