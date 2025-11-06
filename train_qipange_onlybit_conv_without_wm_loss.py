import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import torch
from torch import nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.utils import adjust_learning_rate_no_warmup, set_seed
from custom_modules import CustomVideoLatentDataset, EmbeddingNet, attack, RevealNet, save_mul_video
from Adaptive_Embedding import Adaptive_Embedding, revert_order
import torch.distributed as dist 
from utils.distributed import init_distributed_mode
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import qipangeutil

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app-only-bits-conv-no-wm-loss.log",
    filemode="a" # 追加模式
)

logger = logging.getLogger("train-qipange")

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--lambda_w", type=float, default=0.001)
    parser.add_argument("--lambda_w_b", type=float, default=1.25)
    parser.add_argument("--use_process_latent", type=int, default=True)
    parser.add_argument("--use_ddp", action="store_true")

    parser.add_argument("--data_dir", default='data/Panda-70M-sampled-latent')
    parser.add_argument("--logo_dir", default='data/qipange/train')
    parser.add_argument("--config_path", default="configs/inference_t2v_512_v2.0.yaml")
    parser.add_argument("--output_dir", default="./output-only-bit-conv-no-wm-loss")
    parser.add_argument("--results_dir", default="./results-only-bits-conv-no-wm-loss")

    parser.add_argument("--log_interval", type=int, default=1) 
    parser.add_argument("--save_interval", type=int, default=10) 

    return parser

parser = get_parser()
args = parser.parse_args()

if args.use_ddp:
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    seed = args.seed * dist.get_world_size() + rank
    set_seed(seed)
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)

device_count = torch.cuda.device_count()
args.lr = args.lr * args.batch_size * device_count

dataset = CustomVideoLatentDataset(data_dir=args.data_dir, logo_dir=args.logo_dir, img_size=args.img_size, num_frames=args.num_frames)

if args.use_ddp: 
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.seed)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=8, pin_memory=True, drop_last=True)
else:
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True) 

config = OmegaConf.load(args.config_path)
ddconfig = config.model.params.first_stage_config.params.ddconfig

print('>>> Model checkpoint loading')
vae = torch.load("ckpt/vae.pth")
print('>>> Finish!')
vae.eval()
vae.to(device)
for param in vae.parameters(): 
    param.requires_grad = False

wm_embedder = EmbeddingNet(args.batch_size, vae.decoder.state_dict(), **ddconfig).to(device)
wm_extractor = RevealNet(args.batch_size).to(device)
wm_bits_extractor = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=16, stride=16, padding=0, bias=False),
            nn.ReLU(inplace=True),
            # nn.AdaptiveAvgPool2d((16, 16))
        )
wm_bits_extractor[0].weight.data = torch.ones((1, 1, 16, 16)) / 255 / 256
wm_bits_extractor.to(device)

adaptive_embedding = Adaptive_Embedding().to(device)

if args.use_ddp:
    wm_embedder = DDP(wm_embedder, device_ids=[args.gpu], find_unused_parameters=True)
    wm_extractor = DDP(wm_extractor, device_ids=[args.gpu])
    adaptive_embedding = DDP(adaptive_embedding, device_ids=[args.gpu], find_unused_parameters=True)

loss_fn = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)

optimizer = torch.optim.AdamW([*wm_embedder.parameters(), *wm_extractor.parameters(), 
                               *adaptive_embedding.parameters(), *wm_bits_extractor.parameters()],  lr=args.lr)

for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch", ncols=100):
    if args.use_ddp:
        sampler.set_epoch(epoch)

    stream = train_loader if (args.use_ddp and rank != 0) else tqdm(train_loader, desc="Batch Progress", unit="batch", leave=False, ncols=100)
    for batch_idx, (video, logo_image) in enumerate(stream):
        # 嵌入水印的视频
        video = video.to(device)  
        # 即将嵌入的水印图像
        logo_image = logo_image.to(device) 

        video_latent = video.reshape(args.batch_size * args.num_frames, 4, 40, 64)
        # 保存了一份临时的视频latent，用于后续使用
        intermediate_latent = video_latent.detach()
        video_latent = video_latent.reshape(args.batch_size, args.num_frames, 4, 40, 64) 

        patches = logo_image.unfold(2, 16, 16).unfold(3, 16, 16) 
        patches = patches.contiguous().view(args.batch_size, 3, 256, 16, 16) 

        position_encoding = torch.arange(256).unsqueeze(0).repeat(args.batch_size, 1)
        position_binary = position_encoding.unsqueeze(-1).bitwise_and(1 << torch.arange(8)).ne(0).long() 
        position_binary = torch.where(position_binary == 0, torch.tensor(-1, device=position_binary.device), position_binary) 
        position_channel = position_binary.unsqueeze(-1).expand(-1, -1, -1, 32).to(device)
        position_channel = position_channel.reshape(args.batch_size, 1, 256, 16, 16)
        patches = torch.cat([patches, position_channel.float()], dim=1) 

        # 保存一下当前带位置信息的原始logo图像
        original_patches = patches
        patches, index_map = adaptive_embedding(patches, video_latent)

        patches = patches.view(args.batch_size, 4, 8, 4, 8, 16, 16)
        patches = patches.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
        patches = patches.view(args.batch_size, 4, 8, 64, 128).permute(0, 2, 1, 3, 4)

        secret_patch = patches.reshape(-1, *patches.shape[2:])
        cover = video_latent.reshape(-1, *video_latent.shape[2:])  # [batch_size * num_frames, C, H, W]
        cover = vae.post_quant_conv(cover)  # [batch_size * num_frames, C, H, W]
        stego_patch = wm_embedder(cover, secret_patch)  # [batch_size * num_frames, C, H, W]
        reconst_video_w = stego_patch.reshape(args.batch_size, args.num_frames, *stego_patch.shape[1:])

        reconst_video = vae.decode(intermediate_latent) 
        reconst_video = reconst_video.reshape(args.batch_size, args.num_frames, 3, 320, 512)

        watermark_exact = wm_extractor(attack(reconst_video_w)) 
        watermark_exact = watermark_exact.permute(0, 2, 1, 3, 4)
        watermark_exact = watermark_exact.view(args.batch_size, 4, 8, 4, 16, 8, 16)
        watermark_exact = watermark_exact.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
        watermark_exact = watermark_exact.view(args.batch_size, 4, 256, 16, 16)

        B, C, N, H, W = watermark_exact.size()  
        out_x_2d = watermark_exact.permute(0, 2, 1, 3, 4).contiguous().view(B, N, -1)  # [B,256,4*16*16]
        original_order_out = revert_order(out_x_2d, index_map)
        watermark_and_pos = original_order_out.view(B, N, C, H, W).permute(0,2,1,3,4).contiguous()

        watermark = watermark_and_pos[:, :3, :, :, :]
        watermark = watermark.reshape(args.batch_size, 3, 16, 16, 16, 16) 
        watermark = watermark.permute(0, 1, 2, 4, 3, 5).contiguous().view(args.batch_size, 3, 256, 256)

        adjust_learning_rate_no_warmup(optimizer, len(train_loader), batch_idx, epoch, args.epochs, args.lr, 0, 1e-6)

        # 重新构造一个水印的loss function
        origin_wm_bit, str_origin_wm_bit = qipangeutil.extract(logo_image, 0)
        extracted_wm_bit, str_extracted_wm_bit = qipangeutil.extract_with_conv(watermark, wm_bits_extractor, 0.6)
        wm_bits_loss_fun = torch.nn.BCELoss(reduction='mean', reduce=True)
        loss_wm_bits = wm_bits_loss_fun(extracted_wm_bit, origin_wm_bit)/len(origin_wm_bit)
        # torch.nn.BCELoss(origin_wm_bit, extracted_wm_bit, reduce=True)/len(origin_wm_bit)
        
        loss_wm = loss_fn(original_patches, watermark_and_pos)
        loss_video = loss_fn(reconst_video_w, reconst_video)
        loss = loss_video + args.lambda_w * loss_wm + args.lambda_w_b * loss_wm_bits
        
        logger.info(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Loss_video: {loss_video.item():.4f} "
                    f"Loss_wm: {loss_wm.item():.4f} "
                    f"Loss_wm_bits: {loss_wm_bits.item():.4f} ")
        logger.info(f"Original  WM bits: {str_origin_wm_bit}")
        logger.info(f"Extracted WM bits: {str_extracted_wm_bit}") 
        if batch_idx%200==0:
            os.makedirs(args.results_dir, exist_ok=True)
            save_image(watermark, os.path.join(args.results_dir, f"watermark_epoch{epoch}_batch{batch_idx}.png"))
            save_mul_video([reconst_video, reconst_video_w, abs(reconst_video - reconst_video_w) * 5], os.path.join(args.results_dir,   
                            f"video_epoch{epoch}_batch{batch_idx}.mp4"))
        
        total_step = (epoch - 1) * len(train_loader) + batch_idx

        loss.backward()
        if total_step * device_count * args.batch_size > 10000:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=0.1)
        optimizer.step()
        optimizer.zero_grad()

        if not (args.use_ddp and rank != 0):
            if args.save_interval != 0 and total_step % args.save_interval == 0 and (epoch > 15 or total_step == 0):
                os.makedirs(args.output_dir, exist_ok=True)
                ckpt_path = os.path.join(args.output_dir, f"model_latest_{epoch}.pth")
                torch.save({
                    'wm_embedder': wm_embedder.module.state_dict() if args.use_ddp else wm_embedder.state_dict(),
                    'wm_extractor': wm_extractor.module.state_dict() if args.use_ddp else wm_extractor.state_dict(),
                    'adaptive_embedding': adaptive_embedding.module.state_dict() if args.use_ddp else adaptive_embedding.state_dict(),
                    'wm_bits_extractor': wm_bits_extractor.module.state_dict() if args.use_ddp else wm_bits_extractor.state_dict(),
                    'total_step': total_step * device_count * args.batch_size,
                    'lr_step': ((epoch - 1) * len(train_loader) + batch_idx) * (args.batch_size * device_count), 
                }, ckpt_path)
