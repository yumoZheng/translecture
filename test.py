import os
# 强制禁用所有可能冲突的加速指令
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

try:
    from faster_whisper import WhisperModel
    print("1. 库导入成功，准备加载模型...")
    # 尝试最基础的加载方式
    model = WhisperModel("tiny", device="cpu", compute_type="float32")
    print("2. ✅ 模型加载成功！说明环境支持。")
except Exception as e:
    print(f"3. ❌ 捕获到错误: {e}")