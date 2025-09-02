"""
calculate_hyperbolicity_dataset_final.py

在训练数据集中计算actions_hidden_states的hyperbolicity (Chunk级别)
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import json
from tqdm import tqdm
import glob

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("/home/ouyangzl/openvla-oft/")
sys.path.insert(0, "/home/ouyangzl/openvla-oft/prismatic/extern/hf")
sys.path.append("/home/ouyangzl/openvla-oft/Hyperbolicity/manifold")  # 使用相对路径，基于当前工作目录结构
from poincare import PoincareBall

# 导入必要的模块
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from torch.utils.data import DataLoader
from scipy.spatial import distance_matrix

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import (
    DiffusionActionHead, 
    L1RegressionActionHead, 
    VAEActionHead, 
    FlowMatchingActionHead, 
    OTFlowMatchingActionHead, 
    COTFlowMatchingActionHead,
    EndToEndDiffusionActionHead  
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

# Hyperbolicity计算相关
from sklearn.decomposition import PCA
sys.path.append("/home/ouyangzl/openvla-oft/Hyperbolicity/manifold")  

@dataclass
class HyperbolicityConfig:
    """Hyperbolicity计算配置"""
    # 模型和数据路径
    vla_path: str = "/home/ouyangzl/openvla-oft/Finetune_logs_checkpoints/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--VAE--80000_chkpt"
    data_root_dir: Path = Path("/mnt/nas_ailab_434/434_dataset/modified_libero_rlds")
    dataset_name: str = "libero_spatial_no_noops"
    
    # 模型配置
    use_l1_regression: bool = False
    use_vae: bool = True
    use_diffusion: bool = False
    use_flow_matching: bool = False
    use_ot_flow_matching: bool = False
    use_end_to_end_diffusion: bool = False
    use_proprio: bool = True
    use_film: bool = False
    num_images_in_input: int = 1
    
    # 采样配置
    batch_size: int = 8
    max_batches: int = 100  # 限制处理的batch数量，避免内存溢出
    
    # Hyperbolicity计算配置（只用chunk级别）
    n_tries: int = 10  # 批量计算hyperbolicity的尝试次数
    batch_size_hyp: int = 2000  # 每次hyperbolicity计算的batch大小
    
    # 输出配置
    output_dir: str = "./hyperbolicity_results_chunk_level_L1finetuned-libero-spatial"
    save_embeddings: bool = True
    
    def __post_init__(self):
        """初始化后处理，自动生成output_dir"""
        self.output_dir = self._generate_output_dir()
    
    def _generate_output_dir(self) -> str:
        """根据vla_path和模型配置生成结构化输出路径"""
        # 解析vla_path获取基础参数
        vla_filename = os.path.basename(self.vla_path)
        main_params = vla_filename.split('--')[0].split('+')
        
        # 提取关键训练参数
        model_name = main_params[0]
        dataset = main_params[1]
        batch_size = main_params[2]
        lr = main_params[3]
        lora = main_params[4]
        dropout = main_params[5]
        
        # 确定action head类型
        action_head_type = self._get_action_head_type()
        
        # 提取训练步数
        step = self._extract_step_from_vla_path(vla_filename)
        
        # 构建路径组件
        path_components = [
            model_name, dataset, batch_size, lr, lora, dropout,
            action_head_type,
            f"proprio-{self.use_proprio}",
            f"num_images-{self.num_images_in_input}",
            f"{step}_step"
        ]
        
        # 组合完整路径
        return os.path.join(
            "/home/ouyangzl/openvla-oft/Hyperbolicity/hyperbolicity_results_chunk_level",
            "+".join(path_components)
        )
    
    def _get_action_head_type(self) -> str:
        """根据模型配置确定action head类型"""
        if self.use_l1_regression: return "L1"
        if self.use_vae: return "VAE"
        if self.use_diffusion: return "Diffusion"
        if self.use_flow_matching: return "FlowMatching"
        if self.use_ot_flow_matching: return "OTFlowMatching"
        if self.use_end_to_end_diffusion: return "EndToEndDiffusion"
        return "Unknown"
    
    def _extract_step_from_vla_path(self, vla_filename: str) -> str:
        """从vla_path提取训练步数"""
        parts = vla_filename.split('--')
        if len(parts) >= 3 and parts[-1].endswith('_chkpt'):
            return parts[-1].replace('_chkpt', '')
        return "unknown_step"


def delta_hyp(dismat):
    """
    计算距离矩阵的delta hyperbolicity值
    基于Gromov δ-hyperbolicity定义
    
    Args:
        dismat: 距离矩阵 (n_points, n_points)
        
    Returns:
        delta: hyperbolicity值
    """
    p = 0  # 选择基准点
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)

    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)


def batched_delta_hyp(X, n_tries=10, batch_size=1500):
    """
    批量计算hyperbolicity以提高稳定性
    
    Args:
        X: 嵌入向量矩阵 (n_samples, n_dims)
        n_tries: 尝试次数
        batch_size: 每次采样的batch大小
        
    Returns:
        mean_delta_rel: 相对delta的均值
        std_delta_rel: 相对delta的标准差
    """
    vals = []
    n_samples = len(X)
    
    for i in tqdm(range(n_tries), desc="Computing batched hyperbolicity"):
        # 随机采样
        if n_samples > batch_size:
            idx = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = X[idx]
        else:
            X_batch = X
        
        # 计算距离矩阵
        distmat = distance_matrix(X_batch, X_batch)
        diam = np.max(distmat)
        
        # 避免除零
        if diam > 1e-8:
            delta_rel = 2 * delta_hyp(distmat) / diam
            vals.append(delta_rel)
    
    if vals:
        return np.mean(vals), np.std(vals)
    else:
        return 0.0, 0.0


def visualize_poincare_disk(embeddings: np.ndarray, labels: List[str], save_path: str):
    """
    使用Poincare球面的指数映射将embeddings投影到Poincare盘并可视化
    
    Args:
        embeddings: 嵌入向量，shape为(n_samples, n_dims)
        labels: 每个样本的标签
        save_path: 保存路径
    """
    print(f"Visualizing {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions using Poincare exponential map")
    
    # 初始化Poincare球面
    poincare_ball = PoincareBall()
    curvature = torch.tensor(1.0)  # 设置曲率参数
    
    # 将embeddings转换为torch tensor
    embeddings_torch = torch.from_numpy(embeddings).float()
    
    # 如果维度大于2，用PCA直接降维到2维用于可视化
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_reduced = pca.fit_transform(embeddings)
        embeddings_torch = torch.from_numpy(embeddings_reduced).float()
        explained_var_ratio = pca.explained_variance_ratio_
        print(f"PCA explained variance ratio: {explained_var_ratio}")
        print(f"Total explained variance: {np.sum(explained_var_ratio):.4f}")
    elif embeddings.shape[1] == 1:
        # 如果只有1维，补一个零维度
        embeddings_torch = torch.cat([embeddings_torch, torch.zeros_like(embeddings_torch)], dim=1)
    
    # 全局归一化：除以最大norm来保持相对关系
    max_norm = torch.norm(embeddings_torch, dim=1).max()
    print(f"Original max norm: {max_norm:.4f}")
    
    # 缩放所有向量，保持相对大小关系
    scale_factor = 0.8  # 控制映射的范围，避免映射到边界
    embeddings_scaled = embeddings_torch / max_norm * scale_factor
    
    print(f"After scaling - max norm: {torch.norm(embeddings_scaled, dim=1).max():.4f}")
    print(f"After scaling - min norm: {torch.norm(embeddings_scaled, dim=1).min():.4f}")
    print(f"After scaling - mean norm: {torch.norm(embeddings_scaled, dim=1).mean():.4f}")
    
    # 使用Poincare球面的原点指数映射（expmap0）将切向量映射到Poincare盘
    with torch.no_grad():
        poincare_points = poincare_ball.expmap0(embeddings_scaled, curvature)
    
    # 转换回numpy，现在应该是2维的
    embeddings_2d = poincare_points.numpy()
    
    # 确保所有点都在单位圆内（Poincare球面的性质）
    norms = np.linalg.norm(embeddings_2d, axis=1)
    max_norm_mapped = np.max(norms)
    min_norm_mapped = np.min(norms)
    mean_norm_mapped = np.mean(norms)
    
    print(f"After Poincare mapping - max norm: {max_norm_mapped:.4f}")
    print(f"After Poincare mapping - min norm: {min_norm_mapped:.4f}")
    print(f"After Poincare mapping - mean norm: {mean_norm_mapped:.4f}")
    
    # 如果有点超出边界，进行微调（理论上不应该发生）
    if max_norm_mapped >= 1.0:
        embeddings_2d = embeddings_2d / max_norm_mapped * 0.95
        print("Warning: Some points were outside the Poincare disk, rescaled.")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # 绘制单位圆（Poincare盘边界）
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=3)
    ax.add_patch(circle)
    
    # 绘制一些辅助线（网格）
    for r in [0.25, 0.5, 0.75]:
        circle_grid = plt.Circle((0, 0), r, fill=False, color='gray', linewidth=1, alpha=0.3)
        ax.add_patch(circle_grid)
    
    # 根据标签着色
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                  c=[colors[i]], label=f'Chunk {label}', alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    
    # 添加原点标记
    ax.scatter(0, 0, c='red', s=100, marker='x', linewidth=3, label='Origin')
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('Action Chunk Hidden States in Poincare Disk\n(Exponential Map from Origin)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 添加一些统计信息
    ax.text(-1.05, -1.05, f'Total points: {len(embeddings_2d)}\nNorm range: [{min_norm_mapped:.3f}, {max_norm_mapped:.3f}]', 
           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Poincare disk visualization saved to {save_path}")
    
    # 计算并打印一些Poincare几何统计
    with torch.no_grad():
        poincare_points_torch = torch.from_numpy(embeddings_2d).float()
        
        # 计算从原点的Poincare距离
        origin = torch.zeros_like(poincare_points_torch[:1])
        poincare_distances = []
        for point in poincare_points_torch:
            dist = poincare_ball.sqdist(origin[0:1], point.unsqueeze(0), curvature).sqrt()
            poincare_distances.append(dist.item())
        
        poincare_distances = np.array(poincare_distances)
        print(f"Poincare distance statistics:")
        print(f"  Mean distance from origin: {np.mean(poincare_distances):.4f}")
        print(f"  Std distance from origin: {np.std(poincare_distances):.4f}")
        print(f"  Max distance from origin: {np.max(poincare_distances):.4f}")
        print(f"  Min distance from origin: {np.min(poincare_distances):.4f}")


def load_model_and_components(cfg: HyperbolicityConfig):
    """加载VLA模型和相关组件"""
    print("Loading VLA model and components...")
    
    # 注册模型类
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    # 加载模型
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).cuda()
    
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()
    
    print(f"VLA model loaded. LLM dimension: {vla.llm_dim}")
    
    # 加载action head
    action_head = None
    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM
        )
    elif cfg.use_vae:
        action_head = VAEActionHead(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM, latent_dim=32
        )
    elif cfg.use_flow_matching:
        action_head = FlowMatchingActionHead(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM, num_flow_steps=50
        )
    elif cfg.use_diffusion:
        action_head = DiffusionActionHead(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM, num_diffusion_steps=50
        )
    elif cfg.use_end_to_end_diffusion:
        action_head = EndToEndDiffusionActionHead(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM, num_diffusion_steps=50
        )
    
    if action_head is not None:
        # 加载checkpoint
        checkpoint_files = glob.glob(os.path.join(cfg.vla_path, "action_head--*_checkpoint.pt"))
        if checkpoint_files:
            print(f"Loading action head from {checkpoint_files[0]}")
            state_dict = torch.load(checkpoint_files[0], map_location='cpu', weights_only=True)
            # 移除DDP前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            action_head.load_state_dict(new_state_dict)
        
        action_head = action_head.to(torch.bfloat16).cuda()
        action_head.eval()
        print("Action head loaded and moved to GPU")
    
    # 加载proprio projector
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
        checkpoint_files = glob.glob(os.path.join(cfg.vla_path, "proprio_projector--*_checkpoint.pt"))
        if checkpoint_files:
            print(f"Loading proprio projector from {checkpoint_files[0]}")
            state_dict = torch.load(checkpoint_files[0], map_location='cpu', weights_only=True)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            proprio_projector.load_state_dict(new_state_dict)
        
        proprio_projector = proprio_projector.to(torch.bfloat16).cuda()
        proprio_projector.eval()
        print("Proprio projector loaded and moved to GPU")
    
    return vla, action_head, proprio_projector, processor


def extract_chunk_level_embeddings(
    vla, action_head, proprio_projector, processor, dataloader, cfg: HyperbolicityConfig
) -> Tuple[np.ndarray, List[int], List[np.ndarray]]:
    """
    从数据集中提取chunk级别的actions_hidden_states
    
    Returns:
        all_chunk_embeddings: 所有chunk的embeddings合并 (n_total_samples, hidden_dim)
        chunk_labels: 每个embedding对应的chunk标签
        embeddings_per_chunk: 按chunk分离的embeddings列表
    """
    print("Extracting chunk-level actions hidden states...")
    print(f"Expected dimensions:")
    print(f"  - NUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}")
    print(f"  - ACTION_DIM: {ACTION_DIM}")
    print(f"  - Total action tokens per sample: {NUM_ACTIONS_CHUNK * ACTION_DIM}")
    
    # 初始化存储 - 只关注chunk级别
    all_embeddings_per_chunk = [[] for _ in range(NUM_ACTIONS_CHUNK)]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if batch_idx >= cfg.max_batches:
                break
                
            # 将batch移到GPU
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # VLA前向传播
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"].to(torch.bfloat16),
                    labels=batch["labels"],
                    output_hidden_states=True,
                    proprio=batch["proprio"] if cfg.use_proprio else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    use_film=cfg.use_film,
                )
            
            # 获取action masks
            ground_truth_token_ids = batch["labels"][:, 1:]
            current_action_mask = get_current_action_mask(ground_truth_token_ids)
            next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
            all_actions_mask = current_action_mask | next_actions_mask
            
            # 计算patch数量
            NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
            if cfg.use_proprio:
                NUM_PATCHES += 1
            
            # 提取actions hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            text_hidden_states = last_hidden_states[:, NUM_PATCHES:-1]
            
            batch_size = batch["input_ids"].shape[0]
            hidden_dim = last_hidden_states.shape[-1]
            
            # 重要：确保维度正确
            actions_hidden_states = (
                text_hidden_states[all_actions_mask]
                .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)
                .to(torch.float32)  # 先转换为float32
                .cpu().numpy()      # 再转换为numpy
            )
            
            print(f"Batch {batch_idx}: actions_hidden_states shape: {actions_hidden_states.shape}")
            
            # 按chunk级别处理embeddings
            for batch_sample in range(batch_size):
                sample_embeddings = actions_hidden_states[batch_sample]  # (NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)
                
                # 重塑为 (NUM_ACTIONS_CHUNK, ACTION_DIM, hidden_dim)
                chunk_embeddings = sample_embeddings.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM, hidden_dim)
                
                # 每个chunk平均池化得到chunk级别的embedding
                for chunk_idx in range(NUM_ACTIONS_CHUNK):
                    # 对ACTION_DIM维度进行平均池化: (ACTION_DIM, hidden_dim) -> (hidden_dim,)
                    chunk_embedding = np.mean(chunk_embeddings[chunk_idx], axis=0)  # axis=0是对ACTION_DIM维度平均
                    all_embeddings_per_chunk[chunk_idx].append(chunk_embedding)
    
    # 转换为numpy数组并创建标签
    embeddings_per_chunk = []
    chunk_labels = []
    
    for chunk_idx in range(NUM_ACTIONS_CHUNK):
        if all_embeddings_per_chunk[chunk_idx]:
            embeddings = np.stack(all_embeddings_per_chunk[chunk_idx])
            embeddings_per_chunk.append(embeddings)
            chunk_labels.extend([chunk_idx] * len(embeddings))
            print(f"Chunk {chunk_idx}: {embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions")
    
    # 合并所有chunk的embeddings
    if embeddings_per_chunk:
        all_chunk_embeddings = np.vstack(embeddings_per_chunk)
        print(f"Total chunk embeddings shape: {all_chunk_embeddings.shape}")
        print(f"  - Total samples: {all_chunk_embeddings.shape[0]}")
        print(f"  - Hidden dimension: {all_chunk_embeddings.shape[1]}")
    else:
        all_chunk_embeddings = np.array([])
        
    return all_chunk_embeddings, chunk_labels, embeddings_per_chunk


def main():
    cfg = HyperbolicityConfig()
    
    # 创建输出目录
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Output directory: {cfg.output_dir}")
    
    # 加载模型和组件
    vla, action_head, proprio_projector, processor = load_model_and_components(cfg)
    
    # 创建数据集和dataloader
    print("Loading dataset...")
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=(cfg.num_images_in_input > 1),
        use_proprio=cfg.use_proprio,
    )
    
    dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=10000,
        image_aug=False,  # 不使用数据增强以保持一致性
        train=True,
    )
    
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, 
        processor.tokenizer.pad_token_id, 
        padding_side="right"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    
    print(f"Dataset loaded. Processing {cfg.max_batches} batches with batch size {cfg.batch_size}")
    
    # 提取chunk级别的actions hidden states
    all_chunk_embeddings, chunk_labels, embeddings_per_chunk = extract_chunk_level_embeddings(
        vla, action_head, proprio_projector, processor, dataloader, cfg
    )
    
    if len(all_chunk_embeddings) == 0:
        print("ERROR: No embeddings extracted!")
        return
    
    # 计算hyperbolicity
    results = {}
    
    print("\n" + "="*80)
    print("CALCULATING HYPERBOLICITY (CHUNK LEVEL)")
    print("="*80)
    
    # 1. 整体chunk hyperbolicity
    print(f"Computing overall chunk-level hyperbolicity for {all_chunk_embeddings.shape[0]} samples...")
    mean_delta, std_delta = batched_delta_hyp(
        all_chunk_embeddings, 
        n_tries=cfg.n_tries, 
        batch_size=cfg.batch_size_hyp
    )
    
    results["chunk_level_overall"] = {
        "hyperbolicity_mean": float(mean_delta),
        "hyperbolicity_std": float(std_delta),
        "n_samples": all_chunk_embeddings.shape[0],
        "n_dims": all_chunk_embeddings.shape[1],
        "calculation_method": "chunk_level_pooling",
        "pooling_strategy": "mean_over_action_dimensions"
    }
    
    print(f"✅ Overall chunk-level hyperbolicity: {mean_delta:.6f} ± {std_delta:.6f}")
    
    # 保存结果
    results_file = os.path.join(cfg.output_dir, "hyperbolicity_results_chunk_level.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    
    # 可视化到Poincare盘
    print("\nCreating Poincare disk visualization...")
    visualize_poincare_disk(
        all_chunk_embeddings,
        chunk_labels,
        os.path.join(cfg.output_dir, "poincare_visualization_chunk_level.png")
    )
    
    # 保存embeddings
    if cfg.save_embeddings:
        embeddings_file = os.path.join(cfg.output_dir, "chunk_level_embeddings.npz")
        np.savez(
            embeddings_file, 
            all_chunk_embeddings=all_chunk_embeddings,
            chunk_labels=np.array(chunk_labels),
            **{f"chunk_{i}_embeddings": emb for i, emb in enumerate(embeddings_per_chunk)}
        )
        print(f"✅ Embeddings saved to {embeddings_file}")
    
    # 创建每个chunk的hyperbolicity图表
    print("\nCreating chunk-wise hyperbolicity visualization...")
    
    chunk_indices = []
    chunk_means = []
    chunk_stds = []
    
    for chunk_idx in range(NUM_ACTIONS_CHUNK):
        key = f"chunk_{chunk_idx}"
        if key in results:
            chunk_indices.append(chunk_idx)
            chunk_means.append(results[key]["hyperbolicity_mean"])
            chunk_stds.append(results[key]["hyperbolicity_std"])
    
    if chunk_indices:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        bars = ax.bar(chunk_indices, chunk_means, yerr=chunk_stds, 
                     capsize=5, alpha=0.7, color='skyblue', 
                     edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for i, (bar, mean, std) in enumerate(zip(bars, chunk_means, chunk_stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.001,
                   f'{mean:.4f}±{std:.4f}', 
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Chunk Index', fontsize=12)
        ax.set_ylabel('Hyperbolicity (δ)', fontsize=12)
        ax.set_title('Hyperbolicity per Action Chunk\n(Mean Pooling over Action Dimensions)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(chunk_indices)
        
        # 添加整体平均线
        overall_mean = results["chunk_level_overall"]["hyperbolicity_mean"]
        ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, 
                  label=f'Overall Mean: {overall_mean:.4f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, "chunk_hyperbolicity_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Chunk comparison visualization saved")
    
    # 最终总结
    print("\n" + "="*80)
    print("HYPERBOLICITY ANALYSIS COMPLETED")
    print("="*80)
    print(f"📊 Analysis Summary:")
    print(f"   • Processing method: Chunk-level (mean pooling over action dimensions)")
    print(f"   • Input dimensions: ({NUM_ACTIONS_CHUNK}, {ACTION_DIM}, hidden_dim) -> ({NUM_ACTIONS_CHUNK}, hidden_dim)")
    print(f"   • Total samples analyzed: {all_chunk_embeddings.shape[0]}")
    print(f"   • Embedding dimension: {all_chunk_embeddings.shape[1]}")
    
    if "chunk_level_overall" in results:
        chunk_result = results["chunk_level_overall"]
        print(f"   • Overall chunk-level hyperbolicity: {chunk_result['hyperbolicity_mean']:.6f} ± {chunk_result['hyperbolicity_std']:.6f}")
    
    print(f"📁 Results saved in: {cfg.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
    
 # # 2. 每个chunk单独的hyperbolicity
    # print(f"\nComputing hyperbolicity for each individual chunk...")
    # for chunk_idx, embeddings in enumerate(embeddings_per_chunk):
    #     if len(embeddings) >= 10:  # 需要足够的样本
    #         mean_delta, std_delta = batched_delta_hyp(
    #             embeddings, 
    #             n_tries=max(5, cfg.n_tries // 2), 
    #             batch_size=min(cfg.batch_size_hyp, len(embeddings))
    #         )
            
    #         results[f"chunk_{chunk_idx}"] = {
    #             "hyperbolicity_mean": float(mean_delta),
    #             "hyperbolicity_std": float(std_delta),
    #             "n_samples": embeddings.shape[0],
    #             "n_dims": embeddings.shape[1]
    #         }
            
    #         print(f"  Chunk {chunk_idx}: {mean_delta:.6f} ± {std_delta:.6f} (n={embeddings.shape[0]})")
    #     else:
    #         print(f"  Chunk {chunk_idx}: Not enough samples ({len(embeddings)} < 10)")