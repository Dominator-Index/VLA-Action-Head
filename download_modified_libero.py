from huggingface_hub import snapshot_download
import os

# 设置本地目标路径
target_dir = "/home/ouyangzl/openvla-oft/modified_libero_rlds"

# 开始下载整个仓库的所有内容
print(f"📥 正在下载 HuggingFace 数据集到: {target_dir}")
snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    local_dir=target_dir,
    local_dir_use_symlinks=False  # 避免软链接，确保是完整数据拷贝
)

print("✅ 下载完成！")
