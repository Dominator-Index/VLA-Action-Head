from huggingface_hub import snapshot_download
import sys
import os

def main():
    repo_id = "openvla/openvla-7b"
    local_dir = "/data/jiangjunmin/ouyangzhuoli/openvla-7b"

    print(f"Start downloading {repo_id} to {local_dir}")

    # 创建目录（如果不存在）
    os.makedirs(local_dir, exist_ok=True)

    # 开始下载
    path = snapshot_download(
        repo_id,
        local_dir=local_dir,
        # allow_patterns=["*.bin", "*.json", "*.safetensors", "*.pt"],  # 可根据需要指定
        # use_auth_token=True,  # 如果需要认证token可以打开
    )

    print(f"Model downloaded successfully to: {path}")

if __name__ == "__main__":
    main()
