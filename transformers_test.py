# debug_flash_attn.py
import torch
import sys
import inspect
import sys
sys.path.insert(0, "/data/jiangjunmin/ouyangzhuoli/VLA-Action-Head/prismatic/extern/hf")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# 测试 Flash Attention 导入
try:
    import flash_attn
    print(f"✅ Flash Attention imported successfully: {flash_attn.__version__}")
    
    from flash_attn import flash_attn_func
    print("✅ flash_attn_func imported successfully")
except ImportError as e:
    print(f"❌ Flash Attention import failed: {e}")
    print("This might cause model loading issues")

# 测试模型加载并调试 vision_backbone
try:
    from transformers import AutoModelForVision2Seq
    print("✅ transformers imported successfully")

    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",  # 或本地模型路径
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print("✅ Model loaded successfully")
    print(f"Vision backbone type: {type(vla.vision_backbone)}")

    # 打印该类的来源文件
    cls = type(vla.vision_backbone)
    print(f"Vision backbone defined in: {inspect.getfile(cls)}")

    # 打印是否有 set_num_images_in_input
    has_method = hasattr(vla.vision_backbone, 'set_num_images_in_input')
    print(f"Has set_num_images_in_input: {has_method}")

    if has_method:
        print("✅ set_num_images_in_input method found")
        vla.vision_backbone.set_num_images_in_input(1)
        print("✅ Method called successfully")
    else:
        print("❌ set_num_images_in_input method not found")
        methods = [m for m in dir(vla.vision_backbone) if not m.startswith('_')]
        print("Available methods:", methods)

    # 直接打印类源码，确认函数实现
    try:
        print("\n=== Vision Backbone Class Source Code ===")
        print(inspect.getsource(cls))
    except OSError:
        print("⚠️ Cannot print source — possibly compiled or dynamically generated class")

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
