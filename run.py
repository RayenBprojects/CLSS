import asyncio
from pathlib import Path

import torch

from core import train_pointnetpp

results = asyncio.run(train_pointnetpp())

if results["status"]:
    model = results["model"]
    model.eval()

    save_dir = Path("./models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / "PointnetPPS3DIS.pth"
    torch.save(model.state_dict(), checkpoint_path)

    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, save_dir / "PointnetPPS3DIS.pt")

    print(f"Saved model checkpoint to {checkpoint_path}")
    print(f"Saved TorchScript export to {save_dir / 'PointnetPPS3DIS.pt'}")