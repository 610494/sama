import json

import numpy as np
import torch
import whisper
from datasets import load_dataset, load_from_disk
from spec_augment import spec_augment
from torch.utils.data import Dataset
from utils import add_noise
from whisper.normalizers.basic import BasicTextNormalizer


class YTTDTaigiTRSDataset(Dataset):
    def __init__(
            self,
            split,
            tokenizer,
            sample_rate,
            model_name,
            max_length, 
            spec_augment=None,
            noise_prob=0,
            noise_fn=None,
            lang='zh',
            aux_langs=None,
            aux_json_path=None,
            task='transcribe',
            pseudo_label_path=None
        ) -> None:
        super().__init__()
        
        if split == 'train':
            self.dataset = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
            print(f"主要訓練集大小: {len(self.dataset)}")
        else:
            self.dataset = load_dataset("formospeech/yttd_taigi_trs", name='test', split='train')
            print(f"主要測試集大小: {len(self.dataset)}")

        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.lang = lang
        self.task = task
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        
        # MODIFIED: aux_langs 現在是一個列表
        self.aux_langs = aux_langs if aux_langs is not None else []
        self.aux_data_lookup = {}

        # 檢查是否有任何語言需要從外部 JSON 檔案載入
        needs_external_json = any(lang != 'zh' for lang in self.aux_langs)

        if needs_external_json:
            if not aux_json_path:
                raise ValueError("若 aux_langs 包含非 'zh' 的語言，則必須提供 aux_json_path。")
            
            print(f"正在從 {aux_json_path} 載入輔助翻譯資料...")
            with open(aux_json_path, 'r', encoding='utf-8') as f:
                aux_items = json.load(f)
            
            self.aux_data_lookup = {item['id']: item for item in aux_items}
            print(f"已載入 {len(self.aux_data_lookup)} 筆輔助翻譯。")

        # [新增] 載入偽翻譯 (Scheme 5 專用)
        self.pseudo_labels = {}
        if pseudo_label_path:
            print(f"載入偽翻譯檔案: {pseudo_label_path}")
            # 如果 pseudo_label_path 是一個列表 (例如 [train.json, test.json])
            if isinstance(pseudo_label_path, list):
                 for path in pseudo_label_path:
                     with open(path, 'r', encoding='utf-8') as f:
                         self.pseudo_labels.update(json.load(f))
            else:
                 with open(pseudo_label_path, 'r', encoding='utf-8') as f:
                     self.pseudo_labels = json.load(f)
            print(f"已載入 {len(self.pseudo_labels)} 筆偽翻譯。")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        
        lang = self.lang
        item = self.dataset[id]
        item_id = item['id'] # 確保這是字串

        wav_data = item['audio']['array']
        wav_lens = len(wav_data)
        
        # [關鍵修改] 根據 task 決定主要目標文本 (target text)
        if self.task == 'translate':
            # 翻譯任務：目標是中文 (text_mandarin)
            text = item['text_mandarin']
            # 對中文做正規化 (去除標點等)
            text = self.text_normalizer(text).replace(" ", "") 
        else:
            # 聽寫任務 (ASR)：目標是台語 (text)
            text = item['text']
            text = self.text_normalizer(text).replace(" ", "")

        translations_list = []
        lang_to_field = {
            'en': 'text_english',
            'fr': 'text_french',
            'es': 'text_spanish',
            'hi': 'text_hindi'
        }

        # 遍歷所有指定的輔助語言
        for lang_code in self.aux_langs:
            auxiliary_text = ""
            if lang_code == 'zh':
                auxiliary_text = item['text_mandarin']
            elif lang_code in lang_to_field:
                item_id = item['id']
                translated_item = self.aux_data_lookup.get(item_id)
                if translated_item:
                    field_name = lang_to_field[lang_code]
                    auxiliary_text = translated_item.get(field_name, '')
        
            # 正規化後加入列表
            if auxiliary_text:
                if lang_code == 'zh':
                    normalized_text = self.text_normalizer(auxiliary_text).replace(" ", "")
                else:
                    normalized_text = self.text_normalizer(auxiliary_text)
                translations_list.append(normalized_text)
            else:
                # 如果找不到，也加入空字串以保持列表長度一致
                translations_list.append("")

        # [新增] 嘗試取得偽翻譯
        # 如果有載入 pseudo_labels 且找得到 ID，就用它；否則用空字串或原本的翻譯(視需求而定)
        pseudo_text = self.pseudo_labels.get(item_id, "")

        if np.random.rand() > self.noise_prob: 
            audio = wav_data.flatten().astype(np.float32)
        else:
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32)
        
        audio_frames = len(audio.flatten()) // 160
        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)
            
        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)

        if self.spec_augment:
            if self.spec_augment == "ls-double":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames)).T
            elif self.spec_augment == "ls-basic":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames, n_freq_mask=1, n_time_mask=1)).T
            else:
                raise NotImplementedError 

        # [關鍵修改] 根據 task 設定 SOT token
        if self.task == 'translate':
            task_token = self.tokenizer.translate
        else:
            task_token = self.tokenizer.transcribe

        # 產生 Decoder Inputs 和 Labels
        # 這裡會自動根據上面選定的 text (中文或台語) 進行 Tokenize
        dec_input_ids = [self.tokenizer.sot, 
                         self.tokenizer.special_tokens["<|{}|>".format(lang)],
                         task_token, # 使用正確的 task token
                         self.tokenizer.no_timestamps] + \
                         self.tokenizer.encode(" " + text)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "translations": translations_list,
            "wav_lens": wav_lens,
            "pseudo_translations": pseudo_text,
            "id": item['id']
        }

class HakkaSixianDataset(Dataset):
    def __init__(
            self,
            split,
            tokenizer,
            sample_rate,
            model_name,
            max_length, 
            dataset_path="./sixian_30h_processed", # 預設路徑，建議從 config 傳入
            spec_augment=None,
            noise_prob=0,
            noise_fn=None,
            lang='zh', # Whisper 這裡通常設 'zh'，因為客語用漢字
            aux_langs=None,
            aux_json_path=None, # [新增參數] 支援外部翻譯 JSON
            task='transcribe',
            pseudo_label_path=None, 
            **kwargs 
        ) -> None:
        super().__init__()
        
        # 1. 讀取從硬碟儲存的 DatasetDict
        print(f"Loading Hakka Dataset from: {dataset_path}")
        dataset_dict = load_from_disk(dataset_path)
        
        # 2. 根據 split 選擇 train 或 test
        if split == 'train':
            self.dataset = dataset_dict['train']
        else:
            self.dataset = dataset_dict['test']
            
        print(f"[{split}] 資料集大小: {len(self.dataset)}")

        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.lang = lang
        self.task = task
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        self.aux_langs = aux_langs if aux_langs is not None else []

        # [新增] 處理外部多語言翻譯資料載入
        self.aux_data_lookup = {}
        needs_external_json = any(l != 'zh' for l in self.aux_langs)

        if needs_external_json:
            if not aux_json_path:
                raise ValueError("若 aux_langs 包含非 'zh' 的語言，則必須提供 aux_json_path。")
            
            print(f"正在從 {aux_json_path} 載入客語的輔助翻譯資料...")
            with open(aux_json_path, 'r', encoding='utf-8') as f:
                aux_items = json.load(f)
            
            self.aux_data_lookup = {item['id']: item for item in aux_items}
            print(f"已載入 {len(self.aux_data_lookup)} 筆客語輔助翻譯。")

        # 載入偽翻譯 JSON
        self.pseudo_labels = {}
        if pseudo_label_path:
            print(f"Loading Pseudo Labels from: {pseudo_label_path}")
            if isinstance(pseudo_label_path, list):
                 for path in pseudo_label_path:
                     with open(path, 'r', encoding='utf-8') as f:
                         self.pseudo_labels.update(json.load(f))
            else:
                 with open(pseudo_label_path, 'r', encoding='utf-8') as f:
                     self.pseudo_labels = json.load(f)
            print(f"Loaded {len(self.pseudo_labels)} pseudo labels.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        item = self.dataset[id]
        
        # 1. 處理音訊
        wav_data = item['audio']['array']
        wav_lens = len(wav_data)

        # 2. 處理目標文本 (ASR Target)
        if self.task == 'translate':
            text = item.get('mandarin', '')
            if not text: 
                print(f"Warning: ID {id} has no mandarin text.")
                text = item['hanzi']
        else: # transcribe
            text = item['hanzi']

        text = self.text_normalizer(text).replace(" ", "")

        # 取得 ID 與偽翻譯
        item_id_str = item['id'] 
        pseudo_text = self.pseudo_labels.get(item_id_str, "")

        # 3. 處理輔助語言 (Aux/Translation) - [關鍵修改區塊]
        translations_list = []
        lang_to_field = {
            'en': 'text_english',
            'fr': 'text_french',
            'es': 'text_spanish',
            'hi': 'text_hindi'
        }

        for lang_code in self.aux_langs:
            auxiliary_text = ""
            if lang_code == 'zh':
                auxiliary_text = item.get('mandarin', '')
            elif lang_code in lang_to_field:
                translated_item = self.aux_data_lookup.get(item_id_str)
                if translated_item:
                    field_name = lang_to_field[lang_code]
                    auxiliary_text = translated_item.get(field_name, '')
            
            # 正規化後加入列表
            if auxiliary_text:
                if lang_code == 'zh':
                    normalized_text = self.text_normalizer(auxiliary_text).replace(" ", "")
                else:
                    normalized_text = self.text_normalizer(auxiliary_text)
                translations_list.append(normalized_text)
            else:
                # 若無對應翻譯則補空字串對齊長度
                translations_list.append("") 

        # 4. Noise Augmentation
        if np.random.rand() > self.noise_prob: 
            audio = wav_data.flatten().astype(np.float32)
        else:
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32)
        
        # 5. Log Mel Spectrogram
        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)
            
        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)

        # 6. Spec Augment
        audio_frames = len(audio) // 160
        if self.spec_augment:
             if self.spec_augment == "ls-double":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames)).T
             elif self.spec_augment == "ls-basic":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames, n_freq_mask=1, n_time_mask=1)).T

        # 7. Tokenizer
        if self.task == 'translate':
            task_token = self.tokenizer.translate
        else:
            task_token = self.tokenizer.transcribe

        dec_input_ids = [self.tokenizer.sot, 
                         self.tokenizer.special_tokens["<|{}|>".format(self.lang)], 
                         task_token, 
                         self.tokenizer.no_timestamps] + \
                         self.tokenizer.encode(" " + text)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "translations": translations_list,
            "wav_lens": wav_lens,
            "pseudo_translations": pseudo_text, 
            "id": str(id) 
        }