import re
from typing import List
from rapidfuzz import process, fuzz

EMAIL_RE = re.compile(r'[\w\.\-+]+@[\w\.-]+\.[a-z]{2,}', re.IGNORECASE)
EMAIL_TOKEN_PATTERNS = [
    (r'\b(dot)\b', '.'),
    (r'\b(underscore|under\s*score)\b', '_'),
    (r'\b(hyphen|dash)\b', '-'),
]
EMAIL_NEIGHBOR = re.compile(r'(dot|underscore|gmail|yahoo|hotmail|outlook|proton|com|co|in)', re.I)

def collapse_spelled_letters(s: str) -> str:
    tokens = s.split(); out=[]; i=0
    while i < len(tokens):
        if i+4 <= len(tokens) and all(len(t)==1 for t in tokens[i:i+5]):
            out.append(''.join(tokens[i:i+5])); i+=5
        else:
            out.append(tokens[i]); i+=1
    return ' '.join(out)

def normalize_email_tokens(s: str) -> str:
    s2 = collapse_spelled_letters(s)
    def at_repl(m):
        span = m.start()
        left = s2[max(0, span-40):span]
        right = s2[span: span+40]
        return '@' if (EMAIL_NEIGHBOR.search(left) or EMAIL_NEIGHBOR.search(right)) else m.group(0)
    s2 = re.sub(r'\b(at|@)\b', at_repl, s2, flags=re.IGNORECASE)
    for pat, rep in EMAIL_TOKEN_PATTERNS:
        s2 = re.sub(pat, rep, s2, flags=re.IGNORECASE)
    s2 = re.sub(r'\s*([@\.])\s*', r'\1', s2)
    s2 = re.sub(r'(@[^ \t\n\r]+)(gmail|yahoo|hotmail|outlook|proton)com\b', r'\1\2.com', s2, flags=re.IGNORECASE)
    s2 = re.sub(r'(@[^ \t\n\r]+)(co)(in)\b', r'\1\2.\3', s2, flags=re.IGNORECASE)
    s2 = re.sub(r'\bg\s*mail\b', 'gmail', s2, flags=re.IGNORECASE)
    return s2

NUM_WORD = {'zero':'0','oh':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
            'six':'6','seven':'7','eight':'8','nine':'9'}

def normalize_numbers_spoken(s: str) -> str:
    toks=s.split(); out=[]; i=0
    while i < len(toks):
        j=i; buf=[]
        while j < min(i+8,len(toks)):
            w=toks[j].lower()
            if w in ('double','triple') and j+1<len(toks) and toks[j+1].lower() in NUM_WORD:
                times=2 if w=='double' else 3
                buf.append(NUM_WORD[toks[j+1].lower()]*times); j+=2
            elif w in NUM_WORD:
                buf.append(NUM_WORD[w]); j+=1
            else:
                break
        if len(buf)>=2: out.append(''.join(buf)); i=j
        else: out.append(toks[i]); i+=1
    return ' '.join(out)

def indian_group_str(x: str) -> str:
    if '.' in x: intp,dec=x.split('.',1)
    else: intp,dec=x,None
    if len(intp)<=3: g=intp
    else:
        last3=intp[-3:]; rest=intp[:-3]; parts=[]
        while len(rest)>2:
            parts.insert(0,rest[-2:]); rest=rest[:-2]
        if rest: parts.insert(0,rest)
        g=','.join(parts+[last3])
    return g if dec is None else f"{g}.{dec}"

def normalize_currency(s: str) -> str:
    s=re.sub(r'\brupees\s+','₹',s,flags=re.IGNORECASE)
    def repl(m):
        raw=re.sub('[^0-9\.]','',m.group(0))
        if not raw: return m.group(0)
        return '₹'+indian_group_str(raw)
    return re.sub(r'₹\s*[0-9][0-9,\.]*', repl, s)

def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int=92) -> str:
    toks=s.split(); out=[]
    for t in toks:
        if t and t[0].isupper():
            best=process.extractOne(t,names_lex,scorer=fuzz.ratio)
            if best and best[1]>=threshold: out.append(best[0]); continue
        out.append(t)
    return ' '.join(out)

def generate_candidates(text: str, names_lex: List[str]) -> List[str]:
    from .utils import extract_emails
    base = normalize_email_tokens(text)
    cand1 = correct_names_with_lexicon(normalize_currency(normalize_numbers_spoken(base)), names_lex)
    cand2 = correct_names_with_lexicon(base, names_lex)
    cand3 = normalize_currency(normalize_numbers_spoken(text))
    cand4 = correct_names_with_lexicon(text, names_lex)
    cands = {cand1, cand2, cand3, cand4, text}
    def key_fn(s: str):
        has_email = 1 if extract_emails(s) else 0
        has_rs = 1 if '₹' in s else 0
        ends_ok = 1 if s.endswith(('.', '?')) else 0
        return (-has_email, -has_rs, -ends_ok, len(s))
    out = sorted(list(cands), key=key_fn)
    return out[:2]  # <= at most two; helps tail a lot
