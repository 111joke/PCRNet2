import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import os
import time
from PIL import Image
from model.mn.model import MPRNet
from dataload.RICE import TrainDataset

# ===== 参数设置 =====
dataset_path = '/home/lwt/桌面/Dataset/water/FXDataset/test/image512/'
gt_path = '/home/lwt/桌面/Dataset/water/FXDataset/test/image512/'
save_path = '/home/lwt/桌面/results/hotmap/FX/'
weight_path = '/home/lwt/桌面/GeleNet-ori/models/Fuxin/Final/Final.pth.99'
os.makedirs(save_path, exist_ok=True)

# ===== 模型加载 =====
model = MPRNet()
model.load_state_dict(torch.load(weight_path))
model.cuda()
model.eval()

# ===== 指定目标层 =====
target_layers = {
    'resnet_layer2': model.layers[1],
    'vssm_layer': model.vssm_encoder.layers[2],
    'fusion_layer': model.Fuse[2],
    'rcm_layer': model.RCM_blocks[3]
}

feature_maps = {}
gradients = {}

# ===== Hook 函数 =====
def save_gradient(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output[0]
    return hook

def forward_hook(name):
    def hook(module, input, output):
        feature_maps[name] = output
    return hook

for name, module in target_layers.items():
    module.register_forward_hook(forward_hook(name))
    module.register_backward_hook(save_gradient(name))

# ===== 预处理 =====
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== Grad-CAM 核心函数 =====
def generate_cam(feature, gradient):
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * feature, dim=1).squeeze()
    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-6
    return cam.detach().cpu().numpy()

def overlay_heatmap(cam, image):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam**0.8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return overlay

# ===== 数据加载 =====
test_loader = TrainDataset(dataset_path, gt_path, 512)
time_sum = 0

# ===== 推理主循环 =====
for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    image = image.cuda()
    image.requires_grad = True

    img_basename = os.path.splitext(name)[0]
    raw_img_path = os.path.join(dataset_path, img_basename + '.tif')
    raw_img = cv2.imread(raw_img_path)
    if raw_img is None:
        print(f" 原图读取失败: {raw_img_path}")
        continue
    raw_img = cv2.resize(raw_img, (512, 512))

    model.zero_grad()
    start = time.time()
    output, _ = model(image)
    end = time.time()
    time_sum += (end - start)

    # 最终输出也做一次 Grad-CAM
    gradients['output_layer'] = output
    feature_maps['output_layer'] = output

    loss = output.mean()
    loss.backward()

    for lname in list(target_layers.keys()) + ['output_layer']:
        try:
            cam = generate_cam(feature_maps[lname], gradients[lname])
            overlay = overlay_heatmap(cam, raw_img)
            out_path = os.path.join(save_path, f"{img_basename}_{lname}_overlay.jpg")
            cv2.imwrite(out_path, overlay)
        except Exception as e:
            print(f" {lname} 处理失败: {e}")

    print(f" 已完成：{name}")

# ===== 运行统计 =====
print(f"\n 平均耗时: {time_sum / test_loader.size:.5f}s | FPS: {test_loader.size / time_sum:.2f}")