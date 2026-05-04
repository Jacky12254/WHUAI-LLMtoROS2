#!/usr/bin/env python3
"""
FunASR (SenseVoiceSmall) 语音识别模块 - 适配 LLM 工具调用
支持 44100Hz 麦克风输入，自动重采样以适配模型
"""

import pyaudio
import numpy as np
import torch
import torchaudio
import time
import re
import warnings
from funasr import AutoModel

warnings.filterwarnings("ignore")
local_dir = "/home/jacky/vision/voice/models/SenseVoiceSmall/iic/SenseVoiceSmall"

class FunASRSpeechTranscriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n{'='*60}")
        print("初始化 FunASR 语音识别系统 (SenseVoiceSmall)")
        print(f"设备: {self.device.upper()}")
        print(f"{'='*60}\n")
        
        print("正在加载模型，请稍候...")
        # self.model = AutoModel(
        #     model="iic/SenseVoiceSmall",
        #     trust_remote_code=True,
        #     device=self.device,
        #     disable_update=True
        # )
        self.model = AutoModel(
            model = local_dir,
            trust_remote_code = True,
            device = self.device,
            disable_update = True
        )
        # --- 核心音频参数 ---
        self.mic_sample_rate = 44100  
        self.asr_sample_rate = 16000  
        self.chunk_duration = 0.05    
        self.chunk_size = int(self.mic_sample_rate * self.chunk_duration)
        
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.mic_sample_rate,
            new_freq=self.asr_sample_rate
        ).to(self.device)
        
        # --- 初始化 PyAudio ---
        self.p = pyaudio.PyAudio()
        self.stream = None
        print("语音模型加载完成！")

    def process_audio(self, audio_data):
        """核心处理逻辑：转浮点 -> 降采样 -> 识别"""
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).to(self.device)
        audio_16k = self.resampler(audio_tensor)
        
        try:
            input_data = audio_16k.cpu().numpy()
            res = self.model.generate(
                input=input_data,
                language="en",  # LLM要求全程英文，这里锁定为 "en" 提高准确率
                use_itn=True,
                batch_size_s=60
            )
            
            if res and len(res) > 0:
                text = res[0].get("text", "")
                clean_text = re.sub(r'<\|.*?\|>', '', text).strip()
                return clean_text
        except Exception as e:
            print(f"识别错误: {e}")
            
        return ""

    def get_next_utterance(self, interrupt_event=None) -> str:
        """每次监听都创建全新的流，用完即毁，彻底告别底层指针冲突"""
        
        # --- 修复 1：如果底层的 PyAudio 实例被干掉了，在这里重建 ---
        if not hasattr(self, 'p') or self.p is None:
            self.p = pyaudio.PyAudio()

        # 1. 确保上一个流彻底销毁
        if self.stream is not None:
            try: 
                self.stream.stop_stream()
                self.stream.close()
            except: pass
            self.stream = None
            time.sleep(0.2) # 给底层硬件一点喘息的时间，防止读取到上一帧的爆音乱码
            
        # 2. 创建全新的流
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.mic_sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                start=True # 创建即开始录音
            )
        except Exception as e:
            print(f"麦克风初始化异常 (等待自愈): {e}")
            # --- 修复 2：遇到 -9985 死锁，直接核弹级销毁当前的 PyAudio 实例 ---
            if self.p is not None:
                try: self.p.terminate()
                except: pass
                self.p = None 
            time.sleep(0.5)
            return ""

        speech_buffer = []
        in_speech = False
        silence_start = None
        energy_threshold = 6000  # <--- 保持你的 6000
        
        try:
            while True:
                # 【打断机制】：如果主系统正在处理事情（比如视觉触发了播报），立刻销毁麦克风退出！
                if interrupt_event and interrupt_event.is_set():
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                    except: pass
                    self.stream = None
                    time.sleep(0.2) # --- 修复 3：打断退出时也要等硬件释放 ---
                    return ""

                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                energy = np.sqrt(np.mean(audio_chunk**2))
                
                if energy > energy_threshold:
                    if not in_speech:
                        in_speech = True
                        print("\r🎤 正在接收语音...", end="", flush=True)
                    speech_buffer.append(data)
                    silence_start = None  
                else:
                    if in_speech:
                        speech_buffer.append(data)
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > 0.9: 
                            audio_data = b''.join(speech_buffer)
                            
                            min_bytes = int(self.mic_sample_rate * 0.5) * 2 
                            if len(audio_data) > min_bytes:
                                text = self.process_audio(audio_data)
                                if text:
                                    print("\r" + " " * 30, end="\r") 
                                    
                                    # 【阅后即焚】：拿到结果立刻销毁，给 pygame 让路！
                                    try:
                                        self.stream.stop_stream()
                                        self.stream.close()
                                    except: pass
                                    self.stream = None
                                    time.sleep(0.2) # --- 修复 4：识别成功退出时，等待硬件释放 ---
                                    return text
                                    
                            print("\r" + " " * 30, end="\r") 
                            in_speech = False
                            speech_buffer = []
                            silence_start = None
                            
        except Exception as e:
            print(f"\n音频采集异常: {e}")
            if self.stream is not None:
                try: 
                    self.stream.stop_stream()
                    self.stream.close()
                except: pass
                self.stream = None
            return ""

    def close(self):
        """释放麦克风资源"""
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# 仅做单独测试时运行
if __name__ == "__main__":
    transcriber = FunASRSpeechTranscriber()
    try:
        while True:
            print("请说话...")
            text = transcriber.get_next_utterance()
            print(f"识别结果: {text}")
    except KeyboardInterrupt:
        transcriber.close()