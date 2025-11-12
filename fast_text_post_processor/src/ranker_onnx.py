from typing import List
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForMaskedLM = None

from .utils import extract_emails

def _cheap_score(s: str) -> int:
    # heuristic: reward valid email / ₹ digits / end punctuation; penalize “ at ”/“ dot ”
    score = 0
    if extract_emails(s): score += 5
    if '₹' in s and any(ch.isdigit() for ch in s): score += 3
    if s.endswith(('.', '?')): score += 1
    low = s.lower()
    if ' at ' in low or ' dot ' in low: score -= 3
    return score

class PseudoLikelihoodRanker:
    def __init__(self, model_name: str="distilbert-base-uncased",
                 onnx_path: str=None, device: str="cpu", max_length: int=24):
        self.max_length = max_length   # <= even shorter
        self.model_name = model_name
        self.onnx = None
        self.torch_model = None
        self.device = device
        self.tokenizer = None
        if onnx_path and ort is not None:
            self._init_onnx(onnx_path)
        elif AutoTokenizer is not None and AutoModelForMaskedLM is not None:
            self._init_torch()
        else:
            raise RuntimeError("Need onnxruntime or transformers/torch installed.")

    def _init_onnx(self, onnx_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.onnx = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])

    def _init_torch(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.torch_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.torch_model.eval().to(self.device)

    def _select_positions(self, text: str, offsets, L: int) -> List[int]:
        # Pick tokens whose character spans intersect special chars/digits
        special_idx = set(i for i,ch in enumerate(text) if ch in {'@','.', '₹'} or ch.isdigit())
        pos = []
        for t in range(1, max(L-1,1)):
            a,b = offsets[0,t]
            if a==b: continue
            if any(k>=a and k<b for k in special_idx):
                pos.append(t)
        # Always include first/last interior tokens for fluency
        basics = [1, max(2, L//2), max(1, L-2)]
        pos = basics + pos
        # Dedup and cap to 12 positions
        seen=set(); out=[]
        for p in pos:
            if 1<=p<=L-2 and p not in seen:
                out.append(p); seen.add(p)
            if len(out)>=12: break
        if not out:
            out = [1, max(2, L//2), max(1, L-2)]
        return out

    def _score_with_onnx(self, text: str) -> float:
        toks = self.tokenizer(
            text, return_tensors="np", truncation=True, max_length=self.max_length,
            return_offsets_mapping=True
        )
        input_ids = toks["input_ids"]; attn = toks["attention_mask"]
        offsets = toks["offset_mapping"]  # (1, L, 2)
        seq = input_ids[0]; L = int(attn[0].sum())
        positions = self._select_positions(text, offsets, L)
        mask_id = self.tokenizer.mask_token_id
        batch = np.repeat(seq[None,:], len(positions), axis=0)
        for i,pos in enumerate(positions):
            batch[i,pos] = mask_id
        batch_attn = np.repeat(attn, len(positions), axis=0)
        logits = self.onnx.run(None, {
            "input_ids": batch.astype(np.int64),
            "attention_mask": batch_attn.astype(np.int64)
        })[0]
        orig = np.repeat(input_ids, len(positions), axis=0)
        rows = np.arange(len(positions)); cols = np.array(positions)
        token_ids = orig[rows, cols]
        logits_pos = logits[rows, cols, :]
        m = logits_pos.max(axis=1, keepdims=True)
        log_probs = logits_pos - m - np.log(np.exp(logits_pos - m).sum(axis=1, keepdims=True))
        picked = log_probs[np.arange(len(rows)), token_ids]
        return float(picked.sum())

    def _score_with_torch(self, text: str) -> float:
        toks = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length,
            return_offsets_mapping=True
        ).to(self.device)
        input_ids=toks["input_ids"]; attn=toks["attention_mask"]; offsets=toks["offset_mapping"]
        seq=input_ids[0]; L=int(attn.sum())
        positions = self._select_positions(text, offsets.cpu().numpy(), L)
        batch = seq.unsqueeze(0).repeat(len(positions),1)
        for i,pos in enumerate(positions):
            batch[i,pos] = self.tokenizer.mask_token_id
        batch_attn = attn.repeat(len(positions),1)
        with torch.no_grad():
            logits = self.torch_model(input_ids=batch, attention_mask=batch_attn).logits
            orig = seq.unsqueeze(0).repeat(len(positions),1)
            rows = torch.arange(len(positions)); cols = torch.tensor(positions)
            token_ids = orig[rows, cols]
            logits_pos = logits[rows, cols, :]
            log_probs = torch.log_softmax(logits_pos, dim=-1)
            picked = log_probs[torch.arange(len(rows)), token_ids]
        return float(picked.sum().item())

    def score(self, sents: List[str]) -> List[float]:
        if self.onnx is not None:
            return [self._score_with_onnx(s) for s in sents]
        return [self._score_with_torch(s) for s in sents]

    def choose_best(self, candidates: List[str]) -> str:
        if not candidates: return ""
        if len(candidates)==1: return candidates[0]
        # cheap pre-rank
        scores = [(c, _cheap_score(c)) for c in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        c0, s0 = scores[0]
        if len(scores)==1: return c0
        c1, s1 = scores[1]
        # If clear margin ≥2, skip model
        if s0 - s1 >= 2:
            return c0
        # else, score both with PLL (max 2)
        ranked = self.score([c0, c1])
        return c0 if ranked[0] >= ranked[1] else c1
