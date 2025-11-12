#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, random, re
random.seed(42)

FIRST = ["Aarav","Aarohi","Aisha","Akash","Aishwarya","Alok","Anand","Ananya","Arnav",
         "Ashwin","Gurpreet","Harish","Kiran","Richa","Rupa","Siddharth","Sunita","Varun","Yashika","Shreya"]
LAST  = ["Mehta","Verma","Kapoor","Singh","Sharma","Iyer","Nair","Rao","Patel","Banerjee","Mukherjee"]
DOMAINS = ["gmail.com","yahoo.com","outlook.com","hotmail.com","proton.me"]
SPOKEN_DOT = [" dot "," dot "," dot "]
SPOKEN_AT  = [" at "," at "]
SPOKEN_US  = [" underscore "," underscore "]
SPOKEN_FIX = ["g mail","g  mail","g   mail","gmail"]
MIS_TLD   = ["gmailcom","yahoocom","outlookcom","hotmailcom","protonme"]

RUPEE_WORDS = ["rupees","rupees","rupees","₹"]
YES_ASK = ["can you","will you","could you","confirm","okay"]
PUNCT_END = [".","?"]
COUNTER = ["counter offer","counter-offer","counteroffer","counter  offer"]

# small helper to Indian-format numbers
def indian_group(n):
    s = str(n)
    if len(s) <= 3: return s
    last3 = s[-3:]
    rest = s[:-3]
    parts=[]
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest=rest[:-2]
    if rest: parts.insert(0, rest)
    return ",".join(parts+[last3])

def noisy_email(local, domain):
    # produce noisy spoken forms for email
    choice = random.randint(0,5)
    if choice == 0:
        return f"{local}{random.choice(SPOKEN_AT)}{domain.replace('.', random.choice(SPOKEN_DOT).strip())}"
    if choice == 1:
        return f"{local}{random.choice(SPOKEN_AT)}{domain}".replace(".", " dot ")
    if choice == 2:
        return f"{local.replace('_',' '+random.choice(['underscore','under score'])+' ')}{random.choice(SPOKEN_AT)}{domain}".replace(".", " dot ")
    if choice == 3:
        # missing dot before TLD
        d = domain.replace(".","")
        return f"{local} at {d}"
    if choice == 4:
        # g mail variants
        d = domain.replace("gmail","g mail")
        return f"{local} at {d}"
    return f"{local} {random.choice(['at','@'])} {domain.replace('.', ' dot ')}"

def gold_email(local, domain):
    return f"{local}@{domain}"

# turn a number into a spoken noisy phrase sometimes
def maybe_spoken_number(n):
    as_words = {
        0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine"
    }
    s = str(n)
    mode = random.randint(0,3)
    if mode == 0:
        # plain digits
        return s, f"₹{indian_group(n)}"
    if mode == 1:
        # spoken digits with double/triple/oh
        tokens=[]
        i=0
        while i < len(s):
            if i+1 < len(s) and s[i]==s[i+1] and random.random()<0.3:
                times = random.choice(["double","triple"])
                take = 2 if times=="double" else 3
                take = min(take, len(s)-i)
                tokens.append(times)
                tokens.append(as_words[int(s[i])])
                i += take
            else:
                tokens.append(as_words[int(s[i])] if s[i]!="0" else random.choice(["zero","oh"]))
                i+=1
        return " ".join(tokens), f"₹{indian_group(n)}"
    if mode == 2:
        # rupees words + digits (no commas)
        return f"{random.choice(RUPEE_WORDS)} {s}", f"₹{indian_group(n)}"
    # mixed
    return f"{random.choice(RUPEE_WORDS)} {s}", f"₹{indian_group(n)}"

def make_pair(i):
    fn = random.choice(FIRST)
    ln = random.choice(LAST)
    full_name = f"{fn} {ln}"
    local = (fn.lower()+random.choice(["", "_", "_"])+ln.lower())
    domain = random.choice(DOMAINS)

    # price proposals
    base = random.choice([999,1299,1399,1499,1799,1899,1999,2799,2999,10799,149800,145000])
    noisy_amt, gold_amt = maybe_spoken_number(base)

    # sentence type (offer vs ask)
    ask = random.choice(YES_ASK)
    co = random.choice(COUNTER)

    # NOISY
    noisy_email_str = noisy_email(local, domain)
    noisy = random.choice([
        f"{fn} im offering {noisy_amt} for this item listed at rupees {base} please confirm by email {noisy_email_str}",
        f"hi {fn.lower()} this is {ln.lower()} {ask} do {noisy_amt} instead of rupees {base} email me at {noisy_email_str}",
        f"{co} from {fn.lower()} {noisy_amt} current price rupees {base} reply at {noisy_email_str}",
        f"hello {fn} final offer is {noisy_amt} for quantity 3 reach me at {noisy_email_str}"
    ])

    # GOLD
    gold_email_str = gold_email(local, domain)
    gold = random.choice([
        f"{fn}, I'm offering {gold_amt} for this item, listed at ₹{indian_group(base)}. Please confirm by email: {gold_email_str}.",
        f"Hi {fn}, this is {ln}. {ask.capitalize()} {gold_amt} instead of ₹{indian_group(base)}? Email me at {gold_email_str}.",
        f"{co.capitalize()} from {fn}: {gold_amt}. Current price ₹{indian_group(base)}. Reply at {gold_email_str}.",
        f"Hello {fn}, final offer is {gold_amt} for quantity 3. Reach me at {gold_email_str}."
    ])

    return {"id": i, "noisy": noisy}, {"id": i, "gold": gold}

def main(n=80, out_dir="data"):
    noisy_path = f"{out_dir}/noisy_transcripts.jsonl"
    gold_path  = f"{out_dir}/gold.jsonl"

    with open(noisy_path, "w", encoding="utf-8") as fn, open(gold_path, "w", encoding="utf-8") as fg:
        for i in range(n):
            nrow, grow = make_pair(i)
            fn.write(json.dumps({"id": nrow["id"], "text": nrow["noisy"]}, ensure_ascii=False)+"\n")
            fg.write(json.dumps({"id": grow["id"], "text": grow["gold"]}, ensure_ascii=False)+"\n")
    print(f"Wrote {n} to {noisy_path} and {gold_path}")

if __name__ == "__main__":
    main(n=80, out_dir="data")
