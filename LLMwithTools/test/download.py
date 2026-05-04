from modelscope import snapshot_download

# 指定你想保存模型的本地绝对路径
local_dir = "/home/jacky/vision/voice/models/SenseVoiceSmall"

# 下载模型到本地
model_dir = snapshot_download('iic/SenseVoiceSmall', cache_dir=local_dir)
print(f"模型已成功下载并保存到: {model_dir}")