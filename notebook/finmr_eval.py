"""
Valutazione di LLM sul task FinMR (Financial Mathematical Reasoning) del benchmark FinAuditing.

Per ogni istanza del dataset TheFinAI/FinMR il modello riceve un filing XBRL e una coppia di
domande, e deve restituire un JSON con il valore riportato (extracted_value) e quello corretto
(calculated_value). La valutazione e' un confronto numerico deterministico, con classificazione
a cascata degli errori: SER (JSON non valido), EER (estrazione errata), CER (calcolo errato).

Uso:
  1. Impostare la chiave OpenRouter nella variabile d'ambiente OPENROUTER_API_KEY.
  2. Scegliere il modello e il tetto ai token di ragionamento nella sezione CONFIG.
  3. Eseguire
"""


import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from openai import OpenAI
from datasets import load_dataset


MODEL = ""   
REASONING_MAX_TOKENS = 400      
N_ROWS = None                   
MAX_WORKERS = 6                


# chiave API 
try:
    from google.colab import userdata          # type: ignore
    OPENROUTER_API_KEY = userdata.get("OPENROUTER_API_KEY")
except Exception:
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

assert OPENROUTER_API_KEY, "Chiave non trovata: imposta OPENROUTER_API_KEY (env var o Secrets di Colab)."

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


# funzioni di scoring
def parse_model_json(text):
    """Estrae il primo oggetto JSON dalla risposta. None se non valido o mancano le chiavi."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)   # trova {...} anche se c'e' testo attorno
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict) or "extracted_value" not in obj or "calculated_value" not in obj:
        return None
    return obj


def to_number(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    try:
        v = float(s)
    except Exception:
        return None
    return -v if neg else v


def numbers_equal(a, b, tol=1e-6):
    na, nb = to_number(a), to_number(b)
    if na is None or nb is None:
        return False
    return abs(na - nb) <= tol * max(1.0, abs(nb))


def classify(pred_obj, gt_obj):
    if pred_obj is None:
        return "SER"   # struttura JSON non valida
    if not numbers_equal(pred_obj.get("extracted_value"), gt_obj.get("extracted_value")):
        return "EER"   # estrazione sbagliata
    if not numbers_equal(pred_obj.get("calculated_value"), gt_obj.get("calculated_value")):
        return "CER"   # calcolo sbagliato
    return "correct"


# valutazione
def valuta_riga(row, model_id):
    pred_obj, raw, ptok, ctok, err = None, "", 0, 0, None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": row["query"]}],
                max_tokens=REASONING_MAX_TOKENS + 600,                            
                extra_body={"reasoning": {"max_tokens": REASONING_MAX_TOKENS}},   
            )
            raw = resp.choices[0].message.content or ""
            ptok = resp.usage.prompt_tokens
            ctok = resp.usage.completion_tokens
            pred_obj = parse_model_json(raw)
            break
        except Exception as e:
            err = str(e)
            time.sleep(2 * (attempt + 1))
    gt = json.loads(row["answer"])
    label = classify(pred_obj, gt) if err is None else "SER"
    return {
        "id": row["id"], "dqc_id": row["dqc_id"], "label": label,
        "gt_extracted": gt.get("extracted_value"), "gt_calculated": gt.get("calculated_value"),
        "pred_extracted": (pred_obj or {}).get("extracted_value"),
        "pred_calculated": (pred_obj or {}).get("calculated_value"),
        "prompt_tokens": ptok, "completion_tokens": ctok,
        "error": err, "raw": raw[:500],
    }


def run_evaluation(model_id, dataset, max_workers=MAX_WORKERS, csv_path=None):
    if csv_path is None:
        csv_path = "risultati_" + model_id.replace("/", "_") + ".csv"

    done_ids = set()
    if os.path.exists(csv_path):
        done_ids = set(pd.read_csv(csv_path)["id"].tolist())
        print(f"Riprendo: {len(done_ids)} righe gia' fatte.")

    da_fare = [row for row in dataset if row["id"] not in done_ids]
    print(f"Righe da valutare: {len(da_fare)} con {max_workers} worker paralleli.")

    lock = Lock()
    fatte = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(valuta_riga, row, model_id): row["id"] for row in da_fare}
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                pd.DataFrame([rec]).to_csv(
                    csv_path, mode="a", header=not os.path.exists(csv_path), index=False
                )
                fatte += 1
                print(f"[{fatte}/{len(da_fare)}]  id={rec['id']:>3}  ->  {rec['label']}")

    return pd.read_csv(csv_path).sort_values("id").reset_index(drop=True)


def metriche(df, model_id):
    n = len(df)
    vc = df["label"].value_counts()
    correct = int(vc.get("correct", 0))
    return {
        "modello": model_id,
        "N": n,
        "Accuracy_%": round(100 * correct / n, 2),
        "SER_%": round(100 * int(vc.get("SER", 0)) / n, 2),
        "EER_%": round(100 * int(vc.get("EER", 0)) / n, 2),
        "CER_%": round(100 * int(vc.get("CER", 0)) / n, 2),
    }


def main():
    ds = load_dataset("TheFinAI/FinMR", split="test")
    subset = ds if N_ROWS is None else ds.select(range(N_ROWS))
    print(f"Modello: {MODEL} | righe: {len(subset)} | tetto ragionamento: {REASONING_MAX_TOKENS}")

    df = run_evaluation(MODEL, subset)
    print("\n=== METRICHE ===")
    for k, v in metriche(df, MODEL).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
