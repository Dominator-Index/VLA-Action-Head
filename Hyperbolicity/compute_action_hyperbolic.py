"""
compute_action_hyperbolic.py

计算训练过程中action的hyperbolicity的插件模块
用于分析action的层级结构特性
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
from scipy.spatial import distance_matrix
from collections import defaultdict

from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
)

sys.path.append("/home/ouyangzl/openvla-oft/Hyperbolicity")  # 添加Hyperbolicity目录到路径

def delta_hyp(dismat: np.ndarray) -> float:
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


def batched_delta_hyp(X: np.ndarray, n_tries: int = 10, batch_size: int = 2000) -> Tuple[float, float]:
    """
    批量计算hyperbolicity以提高稳定性
    
    Args:
        X: action矩阵 (n_samples, action_dim)
        n_tries: 尝试次数
        batch_size: 每次采样的batch大小
        
    Returns:
        mean_delta_rel: 相对delta的均值
        std_delta_rel: 相对delta的标准差
    """
    vals = []
    n_samples = len(X)
    
    for i in range(n_tries):
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


class ActionHyperbolicityCalculator:
    """
    Action Hyperbolicity计算器
    用于在训练过程中计算和分析action的双曲度
    """
    
    def __init__(
        self,
        save_dir: str,
        max_samples: int = 5000,
        batch_size_hyp: int = 1000,
        n_tries: int = 10,
        compute_frequency: int = 5000,  # 每5000步计算一次
    ):
        """
        初始化Action Hyperbolicity计算器
        
        Args:
            save_dir: 结果保存目录
            max_samples: 最大采样action数量
            batch_size_hyp: hyperbolicity计算的batch大小
            n_tries: 批量计算hyperbolicity的尝试次数
            compute_frequency: 计算频率（训练步数）
        """
        self.save_dir = Path(save_dir)
        self.max_samples = max_samples
        self.batch_size_hyp = batch_size_hyp
        self.n_tries = n_tries
        self.compute_frequency = compute_frequency

        # 存储固定的action head状态
        self.fixed_action_head_state = None
        self.fixed_vla_state = None
        self.fixed_proprio_projector_state = None
        
        # 收集状态
        self.collected_actions = []
        self.collection_complete = False
        
        # 计算历史
        self.hyperbolicity_history = []
        
        # 当前计算周期
        self.current_cycle_start_step = 0
        self.cycles_completed = 0
        
        print(f"ActionHyperbolicityCalculator initialized:")
        print(f"  - Save directory: {self.save_dir}")
        print(f"  - Max samples: {self.max_samples}")
        print(f"  - Compute frequency: {self.compute_frequency} steps")
        print(f"  - Will use fixed action head for consistent calculations")
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
    
    def should_collect(self, step: int) -> bool:
        """检查是否应该在当前步数收集数据"""
        return step >= 0 and not self.collection_complete
    
    def should_compute(self, step: int) -> bool:
        """检查是否应该在当前步数计算hyperbolicity"""
        return (step % self.compute_frequency == 0 and 
                step > 0 and 
                self.collection_complete)
    
    def start_new_collection_cycle(self, step: int):
        """开始新一轮数据收集"""
        self.current_cycle_start_step = step
        self.collection_complete = False
        self.collected_actions = []
        print(f"Started new collection cycle at step {step}")
    
    def update_fixed_models_for_cycle(self, vla, action_head, proprio_projector, cfg_only_action_head: bool = True):
        """为当前计算周期更新固定模型状态"""
        print(f"Updating fixed models for cycle starting at step {self.current_cycle_start_step}")
        
        # 保存当前action head状态
        self.fixed_action_head_state = action_head.state_dict()
        
        # 保存proprio projector状态（如果存在）
        if proprio_projector is not None:
            self.fixed_proprio_projector_state = proprio_projector.state_dict()
        
        # 如果只训练action head，VLA是固定的
        if cfg_only_action_head:
            self.fixed_vla_state = None  # VLA已经被冻结，不需要保存
        else:
            # 如果训练整个VLA，需要保存VLA状态
            if hasattr(vla, 'module'):
                self.fixed_vla_state = vla.module.state_dict()
            else:
                self.fixed_vla_state = vla.state_dict()
        
        print(f"✅ Fixed models updated for cycle {self.cycles_completed}")
    
    def extract_predicted_actions_from_batch(
        self,
        vla,
        action_head,
        proprio_projector,
        batch,
        device_id,
        use_proprio: bool = True,
        use_film: bool = False,
        use_fixed_models: bool = True
    ) -> np.ndarray:
        """
        从batch中提取predicted actions
        使用固定的模型状态进行预测
        """
        with torch.no_grad():
            # 如果使用固定模型，临时加载固定状态
            if use_fixed_models and self.fixed_action_head_state is not None:
                # 保存当前状态
                current_action_head_state = action_head.state_dict()
                current_proprio_state = proprio_projector.state_dict() if proprio_projector is not None else None
                current_vla_state = None
                if self.fixed_vla_state is not None:
                    if hasattr(vla, 'module'):
                        current_vla_state = vla.module.state_dict()
                    else:
                        current_vla_state = vla.state_dict()
                
                # 加载固定状态
                action_head.load_state_dict(self.fixed_action_head_state)
                if proprio_projector is not None and self.fixed_proprio_projector_state is not None:
                    proprio_projector.load_state_dict(self.fixed_proprio_projector_state)
                if self.fixed_vla_state is not None:
                    if hasattr(vla, 'module'):
                        vla.module.load_state_dict(self.fixed_vla_state)
                    else:
                        vla.load_state_dict(self.fixed_vla_state)
            
            try:
                # VLA前向传播 - 修复：确保所有输入张量都在同一设备上
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"].to(device_id),  # 修复：移动到设备
                        output_hidden_states=True,
                        proprio=batch["proprio"].to(device_id) if use_proprio else None,  # 修复：移动到设备
                        proprio_projector=proprio_projector if use_proprio else None,
                        use_film=use_film,
                    )
                
                # 计算action masks - 修复：确保在正确设备上
                ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
                current_action_mask = get_current_action_mask(ground_truth_token_ids)
                next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
                
                # 修复：确保 masks 在 GPU 上
                current_action_mask = current_action_mask.to(device_id)
                next_actions_mask = next_actions_mask.to(device_id)
                all_actions_mask = current_action_mask | next_actions_mask
                
                # 计算patch数量 - 修复：处理 VLA 是否被 DDP 包装
                if hasattr(vla, 'module'):
                    # VLA 被 DDP 包装
                    vision_backbone = vla.module.vision_backbone
                else:
                    # VLA 未被 DDP 包装
                    vision_backbone = vla.vision_backbone
                
                NUM_PATCHES = vision_backbone.get_num_patches() * vision_backbone.get_num_images_in_input()
                if use_proprio:
                    NUM_PATCHES += 1
                
                # 提取actions hidden states
                last_hidden_states = output.hidden_states[-1]
                text_hidden_states = last_hidden_states[:, NUM_PATCHES:-1]
                
                batch_size = batch["input_ids"].shape[0]
                hidden_dim = last_hidden_states.shape[-1]
                
                actions_hidden_states = (
                    text_hidden_states[all_actions_mask]  # 修复：现在 all_actions_mask 在 GPU 上
                    .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)
                    .to(torch.bfloat16)
                )
                
                # 通过action head预测actions
                if hasattr(action_head.module, 'predict_action'):
                    # 对于所有支持predict_action的heads
                    predicted_actions = action_head.module.predict_action(actions_hidden_states)
                else:
                    raise ValueError(f"Action head {type(action_head.module)} does not have predict_action method")
                
                # 转换为numpy
                predicted_actions = predicted_actions.cpu().float().numpy()  # (batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM)
                
                return predicted_actions
                
            finally:
                # 恢复原始状态
                if use_fixed_models and self.fixed_action_head_state is not None:
                    action_head.load_state_dict(current_action_head_state)
                    if proprio_projector is not None and current_proprio_state is not None:
                        proprio_projector.load_state_dict(current_proprio_state)
                    if current_vla_state is not None:
                        if hasattr(vla, 'module'):
                            vla.module.load_state_dict(current_vla_state)
                        else:
                            vla.load_state_dict(current_vla_state)
    
    
    def collect_actions(
        self,
        vla,
        action_head,
        proprio_projector,
        batch,
        device_id,
        use_proprio: bool = True,
        use_film: bool = False
    ):
        """
        收集actions用于hyperbolicity计算
        """
        if not self.should_collect(self.current_cycle_start_step):
            return
        
        try:
            # 从当前batch提取actions - 使用固定模型
            actions = self.extract_predicted_actions_from_batch(
                vla, action_head, proprio_projector, batch, device_id, 
                use_proprio, use_film, use_fixed_models=True
            )
            
            # 添加到收集列表
            self.collected_actions.append(actions)
            
            # 检查是否收集够了
            total_collected = sum(len(actions) for actions in self.collected_actions)
            if total_collected >= self.max_samples:
                self.collection_complete = True
                print(f"✅ Action collection complete for cycle {self.cycles_completed}! Total collected: {total_collected}")
        
        except Exception as e:
            print(f"Warning: Failed to collect actions from batch: {e}")
    
    def compute_hyperbolicity(self, step: int) -> Optional[Dict]:
        """
        计算当前收集的actions的hyperbolicity
        """
        if not self.collection_complete or len(self.collected_actions) == 0:
            return None
        
        try:
            # 合并所有收集的actions
            all_actions = np.vstack(self.collected_actions)
            
            # 限制到最大采样数量
            if len(all_actions) > self.max_samples:
                indices = np.random.choice(len(all_actions), self.max_samples, replace=False)
                all_actions = all_actions[indices]
            
            print(f"Computing hyperbolicity for {len(all_actions)} actions at step {step} (cycle {self.cycles_completed})...")
            
            # 将actions flatten为2D
            actions_flat = all_actions.reshape(len(all_actions), -1)
            
            # 计算hyperbolicity
            mean_delta, std_delta = batched_delta_hyp(
                actions_flat, 
                n_tries=self.n_tries, 
                batch_size=self.batch_size_hyp
            )
            
            results = {
                "step": step,
                "cycle": self.cycles_completed,
                "hyperbolicity_mean": float(mean_delta),
                "hyperbolicity_std": float(std_delta),
                "n_samples": len(all_actions),
                "cycle_start_step": self.current_cycle_start_step,
            }
            
            # 保存结果
            self.hyperbolicity_history.append(results)
            self._save_results()
            
            print(f"✅ Hyperbolicity computed for cycle {self.cycles_completed}: {mean_delta:.6f} ± {std_delta:.6f}")
            
            # 完成当前周期
            self.cycles_completed += 1
            
            return results
            
        except Exception as e:
            print(f"Error computing hyperbolicity: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_results(self):
        """保存结果到文件"""
        try:
            results_file = self.save_dir / "action_hyperbolicity_history.json"
            with open(results_file, 'w') as f:
                json.dump(self.hyperbolicity_history, f, indent=2)
            
            # 保存actions（如果需要）
            if self.collected_actions:
                all_actions = np.vstack(self.collected_actions)
                actions_file = self.save_dir / "collected_actions.npz"
                np.savez(actions_file, actions=all_actions)
                
            print(f"Results saved to {self.save_dir}")
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")
    
    def get_latest_result(self) -> Optional[Dict]:
        """获取最新的hyperbolicity计算结果"""
        if self.hyperbolicity_history:
            return self.hyperbolicity_history[-1]
        return None


# 便利函数，用于在训练脚本中使用
def create_action_hyperbolicity_calculator(
    save_dir: str,
    max_samples: int = 5000,
    compute_frequency: int = 5000,
) -> ActionHyperbolicityCalculator:
    """
    创建ActionHyperbolicityCalculator实例的便利函数
    
    Args:
        save_dir: 结果保存目录
        max_samples: 最大采样action数量
        compute_frequency: 计算频率（训练步数）
    
    Returns:
        ActionHyperbolicityCalculator实例
    """
    return ActionHyperbolicityCalculator(
        save_dir=save_dir,
        max_samples=max_samples,
        compute_frequency=compute_frequency
    )


def integrate_hyperbolicity_into_training(
    calculator: ActionHyperbolicityCalculator,
    vla,
    action_head, 
    proprio_projector,
    batch,
    device_id,
    step: int,
    use_proprio: bool = True,
    use_film: bool = False,
    swanlab=None
) -> Optional[Dict]:
    """
    在训练循环中集成hyperbolicity计算的便利函数
    """
    # 初始化：第0步开始新周期
    if step == 0:
        calculator.start_new_collection_cycle(step)
        calculator.update_fixed_models_for_cycle(vla, action_head, proprio_projector)
    
    # 收集actions
    calculator.collect_actions(
        vla, action_head, proprio_projector, batch, device_id, use_proprio, use_film
    )
    
    # 检查是否应该计算hyperbolicity
    if calculator.should_compute(step):
        results = calculator.compute_hyperbolicity(step)
        
        # 开始新周期
        calculator.start_new_collection_cycle(step)
        calculator.update_fixed_models_for_cycle(vla, action_head, proprio_projector)
        
        # 记录到SwanLab
        if results and swanlab:
            swanlab.log({
                "Action Hyperbolicity/Mean": results["hyperbolicity_mean"],
                "Action Hyperbolicity/Std": results["hyperbolicity_std"],
                "Action Hyperbolicity/Samples": results["n_samples"],
                "Action Hyperbolicity/Cycle": results["cycle"],
            }, step=step)
        
        return results
    
    return None