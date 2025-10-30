import torch

def extract(image, avg = 0):
    image = image.view(-1, 256, 256)
    image = image.mean(dim=0)
    str_watermark = ''
    extract_watermark = torch.zeros(256, dtype=torch.float32, device=image.device)   
    for index in range(256):
        key = image[index//16*16:(index//16*16 + 16), index*16%256:(index*16%256 + 16)].sum()
        key = key / 256.0
        if key > avg:
            value = 1.0
            extract_watermark[index] = value
            str_watermark += '1'
        else:
            value = 0.0
            extract_watermark[index] = value
            str_watermark += '0'  
    str_watermark = hex(int(str_watermark, 2))[2:] 
    return extract_watermark, str_watermark
    
