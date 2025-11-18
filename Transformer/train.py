import torch
from vit_pytorch import ViT
from models.binae import BinModel
import torch.optim as optim
from einops import rearrange
import load_data
import utils as utils
from config import Configs
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get utils functions
count_psnr = utils.count_psnr
imvisualize = utils.imvisualize
load_data_func = load_data.load_datasets

def build_model(setting, image_size, patch_size):
    hyper_params = {"base": [6, 8, 768],
                    "small": [3, 4, 512],
                    "large": [12, 16, 1024]} 

    encoder_layers = hyper_params[setting][0]
    encoder_heads = hyper_params[setting][1]
    encoder_dim = hyper_params[setting][2]

    v = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=1000,
        dim=encoder_dim,
        depth=encoder_layers,
        heads=encoder_heads,
        mlp_dim=2048
    )

    model = BinModel(
        encoder=v,
        decoder_dim=encoder_dim,      
        decoder_depth=encoder_layers,
        decoder_heads=encoder_heads  
    )
    return model

def visualize(model, epoch, validloader, image_size, patch_size):
    losses = 0
    valid_loader_tqdm = tqdm(validloader, desc=f"Visualizing Epoch {epoch}", leave=False)
    for _, (valid_index, valid_in, valid_out) in enumerate(valid_loader_tqdm):
        bs = len(valid_in)
        inputs = valid_in.to(device)
        outputs = valid_out.to(device)
        with torch.no_grad():
            loss, _, pred_pixel_values = model(inputs, outputs)
            rec_patches = pred_pixel_values
            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                   p1=patch_size, p2=patch_size, h=image_size[0]//patch_size)
            for j in range(bs):
                imvisualize(inputs[j].cpu(), outputs[j].cpu(),
                            rec_images[j].cpu(), valid_index[j],
                            epoch, experiment)
            losses += loss.item()
    print('Valid loss: ', losses / len(validloader))

def valid_model(model, data_path, epoch, experiment, valid_dibco):
    global best_psnr
    global best_epoch
    print('Last best PSNR: ', best_psnr, 'Epoch: ', best_epoch)
    psnr = count_psnr(epoch, data_path, valid_data=valid_dibco, setting=experiment)
    print('Current PSNR: ', psnr)

    if psnr >= best_psnr:
        best_psnr = psnr
        best_epoch = epoch
        if not os.path.exists('./weights/'):
            os.makedirs('./weights/')
        torch.save(model.state_dict(), f'./weights/best-model_{TPS}_{valid_dibco}_{experiment}.pt')
        dellist = os.listdir(f'vis{experiment}')
        if f'epoch{epoch}' in dellist:
            dellist.remove(f'epoch{epoch}')
        for dl in dellist:
            os.system(f'rm -r vis{experiment}/{dl}')
    else:
        os.system(f'rm -r vis{experiment}/epoch{epoch}')

best_psnr = 0
best_epoch = 0

if __name__ == "__main__":
    cfg = Configs().parse()
    SPLITSIZE = cfg.split_size
    setting = cfg.vit_model_size
    TPS = cfg.vit_patch_size
    batch_size = cfg.batch_size
    valid_dibco = cfg.validation_dataset
    data_path = cfg.data_path
    patch_size = TPS
    image_size = (SPLITSIZE, SPLITSIZE)
    vis_results = True
    experiment = f'{setting}_{SPLITSIZE}_{TPS}'
    
    trainloader, validloader, _ = load_data.all_data_loader(batch_size)
    model = build_model(setting, image_size, patch_size).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, betas=(0.9, 0.95),
                            eps=1e-08, weight_decay=0.05, amsgrad=False)

    # Epoch tqdm for full training progress
    epoch_tqdm = tqdm(range(1, cfg.epochs), desc="Training Progress")
    for epoch in epoch_tqdm: 
        running_loss = 0.0
        train_loader_tqdm = tqdm(enumerate(trainloader), total=len(trainloader),
                                 desc=f"Epoch {epoch} Training", leave=False)
        for i, (train_index, train_in, train_out) in train_loader_tqdm:
            inputs = train_in.to(device)
            outputs = train_out.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(inputs, outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            show_every = max(1, int(len(trainloader) / 7))
            if i % show_every == show_every-1:
                avg_loss = running_loss / show_every
                train_loader_tqdm.set_postfix({'Train Loss': f'{avg_loss:.3f}'})
                running_loss = 0.0

        # visualize result and validate
        if vis_results:
            visualize(model, str(epoch), validloader, image_size, patch_size)
            valid_model(model, data_path, epoch, experiment, valid_dibco)

        # update overall epoch tqdm postfix
        epoch_tqdm.set_postfix({'Best PSNR': f'{best_psnr:.3f}', 'Best Epoch': best_epoch})
