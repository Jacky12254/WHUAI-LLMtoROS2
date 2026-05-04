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
#使用的是funasr的sensevoicesmall
class FunASRSpeechTranscriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n{'='*60}")
        print("初始化 FunASR 语音识别系统 (SenseVoiceSmall)")
        print(f"设备: {self.device.upper()}")
        print(f"{'='*60}\n")
        
        print("正在加载模型，请稍候...")
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
        
        # --- 【修复1】全局只初始化 PyAudio 一次 ---
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
                language="en",
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

    def _safe_close_stream(self):
        """【修复2】专门用于安全关闭流的辅助函数，捕获所有底层异常"""
        if self.stream is not None:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def get_next_utterance(self, interrupt_event=None) -> str:
        """安全监听录音"""
        
        self._safe_close_stream()
        
        # 【修复3】留出 0.1 秒的硬件喘息时间，防止和 pygame 发生锁冲突
        time.sleep(0.1)
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.mic_sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                start=True 
            )
        except Exception as e:
            print(f"麦克风初始化异常: {e}")
            # 如果硬件被锁死了，强行退让等待 1 秒，等待系统自愈
            time.sleep(1.0) 
            return ""

        speech_buffer = []
        in_speech = False
        silence_start = None
        energy_threshold = 10000  # 保持 6000 的阈值来过滤底噪
        
        try:
            while True:
                # 收到打断信号，安全退出
                if interrupt_event and interrupt_event.is_set():
                    self._safe_close_stream()
                    return ""

                # 【修复4】给 read 加异常捕获，防止 IOError 导致线程直接死锁
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                except OSError as e:
                    print(f"读取音频数据失败: {e}")
                    self._safe_close_stream()
                    time.sleep(0.5)
                    return ""

                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                energy = np.sqrt(np.mean(audio_chunk**2))
                # print({energy})
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
                                    self._safe_close_stream()
                                    return text
                                    
                            print("\r" + " " * 30, end="\r") 
                            in_speech = False
                            speech_buffer = []
                            silence_start = None
                            
        except Exception as e:
            print(f"\n音频采集异常: {e}")
            self._safe_close_stream()
            return ""

    def close(self):
        """释放麦克风资源，只在程序彻底结束时调用"""
        self._safe_close_stream()
        if self.p is not None:
            try:
                self.p.terminate()
            except Exception:
                pass
            self.p = None

if __name__ == "__main__":
    transcriber = FunASRSpeechTranscriber()
    try:
        while True:
            print("请说话...")
            text = transcriber.get_next_utterance()
            print(f"识别结果: {text}")
    except KeyboardInterrupt:
        transcriber.close()
