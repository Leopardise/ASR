import json
from .rules import generate_candidates
from .ranker_onnx import PseudoLikelihoodRanker
from .utils import extract_emails

class PostProcessor:
    def __init__(self, names_lex_path: str, onnx_model_path: str=None, device: str="cpu", max_length: int=24):
        self.names_lex = [x.strip() for x in open(names_lex_path,'r',encoding='utf-8').read().splitlines() if x.strip()]
        self.ranker = PseudoLikelihoodRanker(onnx_path=onnx_model_path, device=device, max_length=max_length)

    def process_one(self, text: str) -> str:
        cands = generate_candidates(text, self.names_lex)
        if len(cands)==1 or any(extract_emails(c) for c in cands):
            best = next((c for c in cands if extract_emails(c)), cands[0])
        else:
            best = self.ranker.choose_best(cands)
        s = best.strip(); low=s.lower()
        if not s.endswith(('.', '?')) and not extract_emails(s) and s:
            first = (low.split()[0] if low.split() else '')
            s += '?' if first in ('can','shall','will','could','would','is','are','do','does','did','should',
                                  'hey','hello','what','why','how','when') else '.'
        return s

def run_file(input_path: str, output_path: str, names_lex_path: str,
             onnx_model_path: str=None, device: str="cpu", max_length: int=24):
        pp = PostProcessor(names_lex_path, onnx_model_path=onnx_model_path, device=device, max_length=max_length)
        rows = [json.loads(line) for line in open(input_path,'r',encoding='utf-8')]
        out=[]
        for r in rows:
            out.append({"id": r["id"], "text": pp.process_one(r["text"])})
        with open(output_path,'w',encoding='utf-8') as f:
            for o in out: f.write(json.dumps(o, ensure_ascii=False)+"\n")
