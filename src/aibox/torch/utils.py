import yaml
from pathlib import Path
try:
    import torch
    from torchvision import transforms
except ImportError:
    print("pytorch required for these utilities")
    exit(1)



def get_device(): 
    if torch.has_cuda:
        return 'cuda'
    if torch.has_mps: 
        return torch.device('mps')
        
    return torch.device('cpu')




def rewrite_artifact_path(metadata_file, pwd, artifact_path_key):
    with open(metadata_file, "r") as f:
        y = yaml.safe_load(f)
        y[artifact_path_key] = f"file://{pwd}"
    
    with open(metadata_file, "w") as f:
        print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, f, default_flow_style=False, sort_keys=False)


def fix_mlflow_artifact_paths(mlflow_root: Path):
    
    for experiment_folder in mlflow_root.expanduser().resolve().iterdir():
        metadata_file = experiment_folder / "meta.yaml"

        # Fix experiment metadata
        if metadata_file.exists():
            rewrite_artifact_path(metadata_file, experiment_folder, artifact_path_key='artifact_location')
        for run_folder in experiment_folder.iterdir():
            metadata_file = run_folder / "meta.yaml"
            print(run_folder)
            
            # Fix run metadata
            if metadata_file.exists():
                rewrite_artifact_path(metadata_file, run_folder / "artifacts", artifact_path_key='artifact_uri')
