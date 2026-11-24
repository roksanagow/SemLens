"""Showing the cost of heavy work, and holding it back when the computer cannot afford it.

Before a model is downloaded, usages or corpora are embedded, or large corpora
are compared, the app builds a :class:`semlens.resources.Cost`, shows it, asks
for confirmation when it is large, and blocks it when it would fill the disk or
exhaust memory.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import streamlit as st
from semlens.resources import Cost, assess, format_bytes, format_duration, measure_tokens_per_second, system_resources
from app.paths import STORAGE

# Parameters, download in GB, disk space in GB. XL-LEXEME's repository has only
# the older .bin weights; transformers also saves a converted .safetensors copy.
MODEL_SIZES={
    'pierluigic/xl-lexeme':(560e6,2.2,4.5),
    'FacebookAI/xlm-roberta-large':(561e6,2.2,2.2),
    'FacebookAI/xlm-roberta-base':(279e6,1.1,1.1),
    'google-bert/bert-base-uncased':(110e6,0.4,0.4),
}


@st.cache_data(ttl=24*3600,show_spinner=False)
def _hub_model_size(model_id):
    """(parameters or None, safetensors bytes, .bin bytes) from Hugging Face, or None when it cannot be looked up."""
    try:
        from huggingface_hub import HfApi
        info=HfApi().model_info(model_id,files_metadata=True,timeout=6)
    except Exception:
        return None
    files=[(f.rfilename,f.size or 0) for f in info.siblings or []]
    safe=sum(size for name,size in files if name.endswith('.safetensors'))
    binary=sum(size for name,size in files if name.endswith('.bin') and 'pytorch_model' in name)
    params=getattr(info.safetensors,'total',None) if info.safetensors else None
    return params,safe,binary


def _folder_weights(path):
    return sum(f.stat().st_size for pattern in ('*.safetensors','*.bin','*.pt') for f in Path(path).expanduser().glob(pattern))


def device_kind():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def model_is_downloaded(model_id):
    """Whether a model's configuration and weights are already on this computer."""
    if Path(model_id).expanduser().is_dir():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    def cached(name):
        found=try_to_load_from_cache(model_id,name)
        return isinstance(found,str)
    return cached('config.json') and any(cached(name) for name in ('model.safetensors','pytorch_model.bin',
                                                                     'model.safetensors.index.json','pytorch_model.bin.index.json'))


@st.cache_data(ttl=3600,show_spinner=False)
def model_profile(model_id):
    """Size and shape of a model: parameters, hidden size, layers, and what a first use would download.

    Returns None when nothing can be found out (offline, or no such model).
    """
    from transformers import AutoConfig
    downloaded=model_is_downloaded(model_id)
    try:
        config=AutoConfig.from_pretrained(str(Path(model_id).expanduser()) if Path(model_id).expanduser().is_dir() else model_id,
                                          local_files_only=downloaded)
    except Exception:
        return None
    hidden=int(getattr(config,'hidden_size',0) or getattr(config,'d_model',0) or 768)
    layers=int(getattr(config,'num_hidden_layers',0) or 12)
    vocab=int(getattr(config,'vocab_size',0) or 30000)
    parameters=None
    download=extra=0.0
    if model_id in MODEL_SIZES:
        parameters,download_gb,disk_gb=MODEL_SIZES[model_id]
        download,extra=download_gb*1e9,(disk_gb-download_gb)*1e9
    elif Path(model_id).expanduser().is_dir():
        weights=_folder_weights(model_id)
        parameters=weights/4 if weights else None
    else:
        found=_hub_model_size(model_id)
        if found is not None:
            params,safe,binary=found
            weights=safe or binary
            parameters=params or (weights/4 if weights else None)
            download=float(weights)
            extra=float(binary) if binary and not safe else 0.0
    if not parameters:
        parameters=12*layers*hidden*hidden+vocab*hidden
    if downloaded:
        download=extra=0.0
    return {'parameters':float(parameters),'hidden':hidden,'layers':layers,'downloaded':downloaded,
            'download_bytes':float(download),'extra_disk_bytes':float(extra)}


def _speed_key(model_id,batch_size,max_length):
    return f'{model_id}|{device_kind()}|{int(batch_size)}|{int(max_length)}'


def known_speed(model_id,batch_size=16,max_length=256):
    """Tokens per second measured on this computer for this model and settings, if any."""
    return st.session_state.setdefault('_model_speeds',{}).get(_speed_key(model_id,batch_size,max_length))


def remember_speed(model_id,tokens_per_second,batch_size=16,max_length=256):
    if tokens_per_second and tokens_per_second>0:
        st.session_state.setdefault('_model_speeds',{})[_speed_key(model_id,batch_size,max_length)]=float(tokens_per_second)


def measure_speed(model,texts,batch_size=16,max_length=256,layer=-1):
    """Time the model for at most five seconds on real texts, and remember the speed."""
    speed=measure_tokens_per_second(model,list(texts)[:200],layer=layer,batch_size=batch_size,max_length=max_length,seconds=5.0)
    remember_speed(model.name,speed,batch_size,max_length)
    return speed


def cost_summary(cost):
    parts=[]
    if cost.download_bytes:
        parts.append(f'download {format_bytes(cost.download_bytes)}')
    if cost.disk_bytes:
        parts.append(f'disk {format_bytes(cost.disk_bytes)}')
    if cost.memory_bytes:
        parts.append(f'memory {format_bytes(cost.memory_bytes)}')
    if cost.seconds is not None:
        parts.append(f'time {format_duration(cost.seconds)} ('+('rough estimate' if cost.rough_time else 'measured on this computer')+')')
    return f'{cost.action}: '+(' · '.join(parts) if parts else 'small')+'.'


def cost_check(cost,key,remedy=''):
    """Show what an action costs; True when it may go ahead.

    Large actions need a tick to confirm; actions that would fill the disk or
    exceed the computer's memory are blocked, with ``remedy`` saying how to
    make them smaller.
    """
    verdict=assess(cost,system_resources(STORAGE))
    summary=cost_summary(cost)
    details=(' '+'; '.join(d.rstrip('.') for d in cost.details)+'.') if cost.details else ''
    if verdict.level=='block':
        st.error('**Not started: this could freeze the computer or fill its disk.** '+summary+' '+' '.join(verdict.reasons)
                 +(f' {remedy}' if remedy else ''))
        return False
    if verdict.level=='confirm':
        st.warning(summary+' '+' '.join(verdict.reasons)+details)
        signature=hashlib.sha256(json.dumps([cost.action,round(cost.total_disk,-8),round(cost.memory_bytes,-8),
                                             None if cost.seconds is None else round(cost.seconds,-2)]).encode()).hexdigest()[:10]
        return st.checkbox('I understand, continue',key=f'_confirm_{key}_{signature}')
    st.caption(summary+details)
    return True


def check_download(model_id):
    """Raise before a download that would fill the disk. Used by every model load."""
    profile=model_profile(model_id)
    if profile is None or profile['downloaded']:
        return
    cost=Cost(f'Downloading {model_id}',download_bytes=profile['download_bytes'],disk_bytes=profile['extra_disk_bytes'])
    verdict=assess(cost,system_resources(Path.home()))
    if verdict.level=='block':
        raise ValueError('Not downloading the model: '+' '.join(verdict.reasons)+' Free some disk space, or choose a smaller model.')
