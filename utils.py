import os
import random
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as at
import numpy as np
import editdistance
from scipy.io import wavfile
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from operator import itemgetter
from typing import Iterator, Optional
from torch.utils.data import Dataset, DistributedSampler
from torch.utils.data.sampler import Sampler
import json
import re
import bitsandbytes as bnb

def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

def select_noise(noise_wavs):
    rand_indexes = np.random.randint(0, len(noise_wavs), size=1)
    noise_wav = []
    for x in rand_indexes:
        noise_wav.append(wavfile.read(noise_wavs[x])[1].astype(np.float32))
    return noise_wav[0]

def add_noise(clean_wav, noise_wavs, noise_snr=0):
    clean_wav = clean_wav.astype(np.float32)
    noise_wav = select_noise(noise_wavs)
    if type(noise_snr) == int or type(noise_snr) == float:
        snr = noise_snr
    elif type(noise_snr) == tuple:
        snr = np.random.randint(noise_snr[0], noise_snr[1]+1)
    clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
    if len(clean_wav) > len(noise_wav):
        ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
        noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
    if len(clean_wav) < len(noise_wav):
        start = 0
        noise_wav = noise_wav[start: start + len(clean_wav)]
    noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
    adjusted_noise_rms = clean_rms / (10**(snr/20))
    adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
    mixed = clean_wav + adjusted_noise_wav

    #Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
            reduction_rate = max_int16 / mixed.max(axis=0)
        else :
            reduction_rate = min_int16 / mixed.min(axis=0)
        mixed = mixed * (reduction_rate)
    mixed = mixed.astype(np.int16)
    return mixed


class whisper_collator:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, wav_lens = [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            wav_lens.append(f["wav_lens"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0)
                    for audio, audio_len in zip(input_ids, audio_lengths)]
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
                    for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
                        for e, e_len in zip(dec_input_ids, dec_input_ids_length)]  # 50257 is eot token id

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "wav_lens": wav_lens,
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

        return batch

class whisper_flamingo_collator:
    def __call__(self, features):
        input_ids, labels, dec_input_ids, translations, wav_lens, ids, pseudo_translations = [], [], [], [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            if "translations" in f:
                translations.append(f["translations"])
            wav_lens.append(f["wav_lens"])
            ids.append(f["id"])
            if "pseudo_translations" in f:
                pseudo_translations.append(f["pseudo_translations"])

        audio_lengths = [audio.shape[1] for audio in input_ids]
        max_audio_len = max(audio_lengths)
        input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "translations": translations,
            "wav_lens": wav_lens,
            "id": ids,
            "pseudo_translations": pseudo_translations,
        }
        
        # 只將數值類型的項目轉換為張量
        for key in ["input_ids", "labels", "dec_input_ids", "wav_lens"]:
            batch[key] = torch.tensor(np.array(batch[key]), requires_grad=False)
        
        return batch

def create_padding_mask(T, padding_amounts):
    """
    Creates a padding mask for a batch of B x T tensors, given padding amounts.

    Args:
        padding_amounts: A list or tensor of integers, where each element
                         specifies the amount of padding for the corresponding
                         sequence in the batch.

    Returns:
        A PyTorch tensor of shape (B, T) containing 1s for padded elements and 0s
        for non-padded elements.
    """

    padded_lens = T - torch.tensor(padding_amounts, dtype=torch.long)[:, None]  # Add a dimension for broadcasting
    mask = padded_lens <= torch.arange(T, dtype=torch.long)[None, :]  # Add a dimension for broadcasting
    return mask

def whisper_optimizer(model, cfg, t_total):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=cfg.learning_rate,
        eps=cfg.adam_epsilon
    )

    # finetune large OOM 可以用
    # optimizer = bnb.optim.AdamW8bit(
    #     optimizer_grouped_parameters,
    #     lr=cfg.learning_rate,
    #     eps=cfg.adam_epsilon,
    # )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )
    return optimizer, scheduler

def whisper_flamingo_optimizer(model, cfg, t_total):
    # x_attn = ["gated_x_attn", "attn_gate", "ff"]
    x_attn = ["gated_x_attn", "attn_gate", "ff", "text_projection", "audio_projection"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in x_attn )],
            "lr": cfg.learning_rate,
        },
    ]    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=cfg.learning_rate,
        eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )
    return optimizer, scheduler

def whisper_scheme3_optimizer(model, cfg, t_total):
    """
    Optimizer 函式，適用於方案三 (Prediction Head)。
    - 總是訓練 GCA (gated_x_attn, attn_gate, ff)
    - 總是訓練 Prediction Head (prediction_head)
    - [修正] 總是訓練 Projection Layer (text_projection)
    - 根據 cfg.freeze_encoder (bool) 決定是否訓練 Encoder (encoder)
    """
    
    # 1. 定義要訓練的模組名稱
    
    # GCA 相關參數 + [修正] 加入 text_projection
    # 因為 text_projection 是將 BERT/Prediction 映射到 ASR 空間的關鍵
    gca_modules = ["gated_x_attn", "attn_gate", "ff", "text_projection"]
    
    # 方案三新增的預測頭
    pred_head_modules = ["prediction_head"]
    
    # 將 GCA 和 預測頭 組合
    modules_to_train = gca_modules + pred_head_modules
    
    # 2. 檢查是否需要 fine-tune Encoder
    freeze_encoder = getattr(cfg, 'freeze_encoder', True) 
    
    if not freeze_encoder:
        print("Optimizer: 將 [Encoder] 參數加入訓練。")
        modules_to_train.append("encoder")
    else:
        print("Optimizer: [Encoder] 參數已凍結 (Frozen)。")

    # 3. 收集所有需要訓練的參數
    params_to_optimize = []
    trained_param_names = [] # 用於 debug 輸出

    for n, p in model.named_parameters():
        # 檢查參數是否需要梯度 (p.requires_grad)
        # 並且 參數名稱 (n) 是否包含我們任一指定的模組關鍵字
        if p.requires_grad and any(nd in n for nd in modules_to_train):
            params_to_optimize.append(p)
            trained_param_names.append(n)

    # 4. (Debug) 印出所有將被優化器更新的參數
    if not params_to_optimize:
        print("警告：Optimizer 沒有找到任何可訓練的參數。")
    else:
        print("--- Optimizer 將訓練以下參數: ---")
        # 簡單檢查一下有沒有 text_projection
        has_proj = any("text_projection" in n for n in trained_param_names)
        if has_proj:
            print(" [v] text_projection found and will be trained.")
        else:
            print(" [x] WARNING: text_projection NOT found!")
            
        print(f" 總共 {len(trained_param_names)} 個參數張量被加入優化器。")
        print("-----------------------------------")


    # 5. 建立 optimizer_grouped_parameters
    optimizer_grouped_parameters = [
        {
            "params": params_to_optimize,
            "lr": cfg.learning_rate,
        },
    ] 
    
    # 6. 建立 Optimizer 和 Scheduler (與原版相同)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=cfg.learning_rate,
        eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )
    
    return optimizer, scheduler

def whisper_adapter_optimizer(model, cfg, t_total):
    adapter = ["adapter"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(keyword in n for keyword in adapter)],
            "lr": cfg.learning_rate,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=cfg.learning_rate,
        eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )
    
    return optimizer, scheduler

def whisper_joint_optimizer(model, cfg, t_total):
    joint = ["gated_x_attn", "attn_gate", "ff", "adapter"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(keyword in n for keyword in joint)],
            "lr": cfg.learning_rate,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=cfg.learning_rate,
        eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=t_total
    )
    
    return optimizer, scheduler

def setup_logging_and_checkpoint(check_output_dir, train_name, train_id, monitor, filename):
    Path(check_output_dir).mkdir(exist_ok=True)
    
    val_checkpoint = ModelCheckpoint(
        dirpath=f"{check_output_dir}/{train_id}",
        filename=filename,
        monitor=monitor,
        mode='min',
        save_top_k=3,
        auto_insert_metric_name=False,
    )

    callback_list = [val_checkpoint,
                    LearningRateMonitor(logging_interval="step")]
    return callback_list

def wer_cer(hypo, ref):
    c_err, c_len, w_err, w_len = 0, 0, 0, 0
    for h, r in zip(hypo, ref):
        pred_words = h.split()
        pred_units = h.replace(' ', '|').replace('', ' ').split() # chars-space separated
        
        gt_words = r.split()
        gt_units = r.replace(' ', '|').replace('', ' ').split() # chars-space separated\
        c_err += editdistance.eval(pred_units, gt_units)
        c_len += len(gt_units)

        w_err += editdistance.eval(pred_words, gt_words)
        w_len += len(gt_words)
    return w_err/w_len, c_err/c_len

# https://github.com/mpc001/auto_avsr/blob/main/datamodule/samplers.py
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()

        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.sampler.set_epoch(epoch)

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)
