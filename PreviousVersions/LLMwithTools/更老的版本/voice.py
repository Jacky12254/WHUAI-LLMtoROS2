import pyaudio
from vosk import Model, KaldiRecognizer
import json

class VoskSpeechTranscriber:
    def __init__(self, model_path, rate=44100):
        """
        初始化识别器
        :param model_path: Vosk 模型所在的文件夹路径
        :param rate: 采样率，默认 44100
        """
        self.rate = rate
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.rate)
        
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start_listening(self):
        """开启麦克风流"""
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=44100
        )
        print(f"--- 麦克风已开启 (采样率: {self.rate}Hz) ---")

    def get_next_utterance(self):
        """
        阻塞直到识别出一句完整的话
        :return: 识别出的文本字符串
        """
        if self.stream is None:
            self.start_listening()

        while True:
            # 使用 exception_on_overflow=False 防止崩溃
            data = self.stream.read(44100, exception_on_overflow=False)
            
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                if text:
                    return text

    def close(self):
        """释放资源"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("--- 资源已释放 ---")

if __name__ == "__main__":
    # 替换为你自己的模型路径
    PATH = "vosk-model-small-en-us-0.15"
    
    transcriber = VoskSpeechTranscriber(PATH)
    
    try:
        print("请说话...")
        while True:
            text = transcriber.get_next_utterance()
            print(f"你说了: {text}")
            
            # 简单的退出机制
            if "stop" in text.lower() or "exit" in text.lower():
                break
    except KeyboardInterrupt:
        pass
    finally:
        transcriber.close()