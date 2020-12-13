import torch
import torch.nn.functional as F
import torchvision

import sys
import time
from tqdm import tqdm
from model import PixelSnail

TRY_CUDA = True
IMAGE_DIM = [28,28]
NB_SAMPLES = 3
NB_CLASSES = 10

device = torch.device('cuda' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
print(f"> Using device {device}")

try:
    print(f"> Loading PixelSnail from file {sys.argv[1]}")
    # model = PixelSnail(IMAGE_DIM, 256, 32, 5, 3, 2, 16, nb_out_res_block=2).to(device)
    model = PixelSnail([28, 28], 256, 32, 5, 3, 2, 16, nb_cond_res_block=2, cond_res_channel=16, nb_out_res_block=2).to(device)
    model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()
    print("> Loaded PixelSnail succesfully!")
except:
    print("! Failed to load state dict!")
    print("! Make sure model is of correct size and path is correct!")
    exit()

with torch.no_grad():
    sample = torch.zeros(NB_SAMPLES*NB_CLASSES, *IMAGE_DIM, dtype=torch.int64).to(device)
    c = torch.tensor([d for d in range(NB_CLASSES) for _ in range(NB_SAMPLES)], dtype=torch.int64).to(device)
    c = c.view(-1,1,1).expand(-1,7,7).to(device)

    pb = tqdm(total=IMAGE_DIM[0]*IMAGE_DIM[1])

    cache = {}
    for i in range(IMAGE_DIM[0]):
        for j in range(IMAGE_DIM[1]):
            pred, cache = model(sample, cache=cache, c=c)
            pred = pred.to(device)
            pred = F.softmax(pred[:, :, i, j], dim=1)
            sample[:, i, j] = torch.multinomial(pred, 1).float().squeeze()
            pb.update(1)

    save_id = int(time.time())
    torchvision.utils.save_image(sample.unsqueeze(1) / 255., f"samples/zero-{save_id}.png", nrow=NB_SAMPLES)
