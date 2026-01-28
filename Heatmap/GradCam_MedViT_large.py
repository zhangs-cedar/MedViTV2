# 标准库
import argparse
import glob
import os
import random
import sys

# 第三方库
import cv2
import numpy as np
import torch
from medmnist import INFO, Evaluator
from pytorch_grad_cam import (
    AblationCAM, EigenCAM, EigenGradCAM, FullGrad, GradCAM,
    GradCAMPlusPlus, GuidedBackpropReLUModel, LayerCAM, ScoreCAM, XGradCAM
)
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torchvision import datasets

# 本地模块
# 获取项目根目录（脚本所在目录的父目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Heatmap 的父目录就是项目根目录
sys.path.insert(0, project_root)
print(f"[GradCam_MedViT_large] 项目根目录: {project_root}")
from MedViT import MedViT_large, MedViT_small, MedViT_tiny
from cedar.utils import print

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device to use')

    parser.add_argument(
        '--image-path',
        type=str,
        default=None,
        help='Input image path (if None, will select one image per class from dataset)')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--dataset', type=str, default='PAD', help='Dataset to use.')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model checkpoint (if None, will use default path)')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Root directory of dataset (for custom datasets)')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    # 如果没有指定 device 或 device 为 'cpu'，但 CUDA 可用，则使用 cuda
    if args.device == 'cpu' and torch.cuda.is_available():
        print(f"[GradCam_MedViT_large] CUDA 可用，但指定使用 CPU")
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print(f"[GradCam_MedViT_large] CUDA 不可用，切换到 CPU")
        args.device = 'cpu'
    else:
        print(f"[GradCam_MedViT_large] 使用设备: {args.device}")

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_one_image_per_class(dataset_root, split='test'):
    """
    从数据集中为每个类别选择一张图像
    
    Args:
        dataset_root: 数据集根目录（如 /dataset/songzhang/SMore_dev/learning/MedViTV2/temp/fabric）
        split: 'train' 或 'test'
    
    Returns:
        list: [(class_name, image_path), ...]
    """
    print(f"[get_one_image_per_class] 从数据集选择图像: {dataset_root}, split: {split}")
    split_dir = os.path.join(dataset_root, split)
    
    if not os.path.exists(split_dir):
        raise ValueError(f"数据集目录不存在: {split_dir}")
    
    # 获取所有类别文件夹
    class_dirs = [d for d in os.listdir(split_dir) 
                  if os.path.isdir(os.path.join(split_dir, d))]
    class_dirs.sort()
    
    print(f"[get_one_image_per_class] 找到 {len(class_dirs)} 个类别")
    
    image_list = []
    # 支持的图像扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    for class_name in class_dirs:
        class_dir = os.path.join(split_dir, class_name)
        # 查找该类别下的所有图像
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(class_dir, f'*{ext}')))
            images.extend(glob.glob(os.path.join(class_dir, f'*{ext.upper()}')))
        
        if images:
            # 随机选择一张图像
            selected_image = random.choice(images)
            image_list.append((class_name, selected_image))
            print(f"[get_one_image_per_class] 类别 {class_name}: 选择图像 {os.path.basename(selected_image)}")
        else:
            print(f"[get_one_image_per_class] 警告: 类别 {class_name} 没有找到图像")
    
    print(f"[get_one_image_per_class] 总共选择了 {len(image_list)} 张图像")
    return image_list


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    # 自定义数据集的类别数量映射
    CUSTOM_DATASET_CLASSES = {
        'fabric': 27,
        'fiber': None,  # 可以从数据集动态获取
        'PAD': 6,
        'Kvasir': 8,
        'CPN': 3,
        'Fetal': 6,
        'ISIC2018': 7,
    }
    
    # 判断是 medmnist 标准数据集还是自定义数据集
    print(f"[GradCam_MedViT_large] 处理数据集: {args.dataset}")
    if args.dataset.endswith('mnist'):
        info = INFO[args.dataset]
        task = info['task']
        nb_classes = len(info['label'])
        print(f"[GradCam_MedViT_large] medmnist 数据集: {args.dataset}, 类别数: {nb_classes}")
    else:
        # 自定义数据集
        if args.dataset in CUSTOM_DATASET_CLASSES:
            nb_classes = CUSTOM_DATASET_CLASSES[args.dataset]
            if nb_classes is None:
                # 如果类别数为 None，尝试从数据集目录动态获取
                # 这里可以添加动态获取逻辑，但为了简洁，先使用固定值
                raise ValueError(f"Dataset {args.dataset} requires manual class count configuration")
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}. Please add it to CUSTOM_DATASET_CLASSES")
        print(f"[GradCam_MedViT_large] 使用自定义数据集: {args.dataset}, 类别数: {nb_classes}")

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")



    # 加载模型
    model = MedViT_large(num_classes=nb_classes)
    
    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = f'./MedViT_large_{args.dataset}.pth'
    
    print(f"[GradCam_MedViT_large] 加载模型: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(args.device)
    model.eval()
    print(f"[GradCam_MedViT_large] 模型加载完成，移动到设备: {args.device}")
    
    target_layers = [model.features[-1].norm2, model.features[-4].norm2, model.norm]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=None)

    # 确定要处理的图像列表
    if args.image_path:
        # 单张图像模式
        image_list = [('single', args.image_path)]
        print(f"[GradCam_MedViT_large] 单张图像模式: {args.image_path}")
    else:
        # 批量模式：为每个类别选择一张图像
        if args.dataset == 'fabric':
            if args.dataset_root:
                dataset_root = args.dataset_root
            else:
                dataset_root = '/dataset/songzhang/SMore_dev/learning/MedViTV2/temp/fabric'
            image_list = get_one_image_per_class(dataset_root, split='test')
            print(f"[GradCam_MedViT_large] 批量模式: 将为 {len(image_list)} 个类别生成 heatmap")
        else:
            raise ValueError(f"批量模式仅支持 fabric 数据集，或使用 --image-path 指定单张图像")

    # 创建输出目录
    output_dir = f'heatmap_{args.dataset}_{args.method}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"[GradCam_MedViT_large] 输出目录: {output_dir}")
    
    # 创建拼接图像输出目录
    if args.dataset == 'fabric':
        combined_output_dir = '/dataset/songzhang/SMore_dev/learning/MedViTV2/temp/fabric_heatmap'
    else:
        combined_output_dir = os.path.join(project_root, f'temp/{args.dataset}_heatmap')
    os.makedirs(combined_output_dir, exist_ok=True)
    print(f"[GradCam_MedViT_large] 拼接图像输出目录: {combined_output_dir}")

    # 处理每张图像
    cam.batch_size = 32
    for class_name, image_path in image_list:
        print(f"[GradCam_MedViT_large] 处理图像: {class_name} - {image_path}")
        
        if not os.path.exists(image_path):
            print(f"[GradCam_MedViT_large] 警告: 图像不存在，跳过: {image_path}")
            continue
        
        # 读取原始图像（BGR格式）
        original_img_bgr = cv2.imread(image_path, 1)
        if original_img_bgr is None:
            print(f"[GradCam_MedViT_large] 警告: 无法读取图像，跳过: {image_path}")
            continue
        
        # 转换为RGB并resize用于模型输入
        rgb_img = original_img_bgr[:, :, ::-1]  # BGR to RGB
        original_img_rgb = rgb_img.copy()  # 保存原始RGB图像用于拼接
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img_float = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img_float, mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]).to(args.device)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        targets = None

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        # 生成带heatmap的图像（已经是224x224，返回uint8格式 0-255）
        cam_image = show_cam_on_image(rgb_img_float, grayscale_cam)
        
        # 将原图也resize到224x224以便拼接（保持uint8格式）
        original_img_resized = cv2.resize(original_img_rgb, (224, 224))
        original_img_resized_uint8 = original_img_resized.astype(np.uint8)
        
        # 将原图和heatmap水平拼接（都是uint8格式，RGB）
        combined_image = np.hstack([original_img_resized_uint8, cam_image])
        
        # 保存拼接后的图像（BGR格式用于cv2.imwrite）
        combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        combined_filename = f'{class_name}_combined.jpg'
        combined_output_path = os.path.join(combined_output_dir, combined_filename)
        cv2.imwrite(combined_output_path, combined_image_bgr)
        print(f"[GradCam_MedViT_large] 保存拼接图像: {combined_output_path}")
        
        # 同时保存单独的heatmap（可选）
        output_filename = f'{class_name}_cam.jpg'
        output_path = os.path.join(output_dir, output_filename)
        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, cam_image_bgr)
        print(f"[GradCam_MedViT_large] 保存 heatmap: {output_path}")
    
    print(f"[GradCam_MedViT_large] 完成！")
    print(f"[GradCam_MedViT_large] 拼接图像已保存到: {combined_output_dir}")
    print(f"[GradCam_MedViT_large] 单独 heatmap 已保存到: {output_dir}")


#python vit_example.py --dataset pathmnist --image-path './pathmnist.png'

#tissuemnist, pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, breastmnist, bloodmnist,
    #organamnist, organcmnist, organsmnist'