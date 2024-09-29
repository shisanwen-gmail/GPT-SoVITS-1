import sys
import os
import torch

class Config:
    def __init__(self):
        global sovits_path, gpt_path, is_half, is_share
        global cnhubert_path, bert_path, pretrained_sovits_path, pretrained_gpt_path
        global exp_root, python_exec, infer_device
        global webui_port_main, webui_port_uvr5, webui_port_infer_tts, webui_port_subfix
        global api_port

        sovits_path = self.sovits_path = ""
        gpt_path = self.gpt_path = ""
        is_half = self.is_half = self._parse_env_bool("is_half", True)
        is_share = self.is_share = self._parse_env_bool("is_share", False)

        cnhubert_path = self.cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        bert_path = self.bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        pretrained_sovits_path = self.pretrained_sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
        pretrained_gpt_path = self.pretrained_gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

        exp_root = self.exp_root = "logs"
        python_exec = self.python_exec = sys.executable or "python"
        infer_device = self.infer_device = "cuda" if torch.cuda.is_available() else "cpu"

        webui_port_main = self.webui_port_main = 9874
        webui_port_uvr5 = self.webui_port_uvr5 = 9873
        webui_port_infer_tts = self.webui_port_infer_tts = 9872
        webui_port_subfix = self.webui_port_subfix = 9871

        api_port = self.api_port = 9880

        self._adjust_half_precision()

    @staticmethod
    def _parse_env_bool(key, default):
        return os.environ.get(key, str(default)).lower() == 'true'

    def _adjust_half_precision(self):
        global is_half
        if self.infer_device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            if any(gpu in gpu_name.upper() for gpu in ["16", "P40", "P10", "1060", "1070", "1080"]) and "V100" not in gpu_name.upper():
                is_half = self.is_half = False
        elif self.infer_device == "cpu":
            is_half = self.is_half = False

config = Config()
