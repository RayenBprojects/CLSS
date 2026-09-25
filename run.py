import torch

import asyncio
from core import train_pointnetpp

results= asyncio.run(train_pointnetpp())

if results['status']== True:
    torch.save(results['model'].state_dict(), './models/saved/PointnetPPS3DIS.pth')