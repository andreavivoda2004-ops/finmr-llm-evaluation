# FinMR – Valutazione di LLM di nuova generazione sul ragionamento numerico finanziario

Codice a supporto della tesi di laurea triennale *"Ragionamento numerico su documenti finanziari
strutturati: un'analisi comparativa dei large language models di nuova generazione sul benchmark
FinMR"* (Università Cattolica del Sacro Cuore, A.A. 2025/2026).

Il progetto riproduce e aggiorna la componente **FinMR** (Financial Mathematical Reasoning) del
benchmark **FinAuditing**, valutando sette LLM recenti sulla loro capacità di estrarre e ricalcolare
correttamente i valori dei bilanci XBRL.

## Task
Per ogni istanza il modello riceve un filing XBRL e una coppia di domande, e deve restituire un
oggetto JSON con due valori: quello **riportato** (`extracted_value`) e quello **corretto**
(`calculated_value`). La valutazione è un confronto numerico deterministico, con normalizzazione dei
formati (separatori delle migliaia, valute, percentuali, parentesi per i negativi).

## Metriche
- **Accuracy**: risposte in cui entrambi i valori coincidono con quelli attesi.
- Errori scomposti a cascata: **SER** (struttura / JSON non valido), **EER** (estrazione errata),
  **CER** (calcolo errato).

## Dataset
[`TheFinAI/FinMR`](https://huggingface.co/datasets/TheFinAI/FinMR) (split `test`, 332 istanze).
Non è incluso nel repository: viene scaricato a runtime tramite la libreria `datasets`.

## Modelli valutati (via OpenRouter)
| Modello | Righe | Tetto token ragionamento |
|---|---|---|
| GPT-5.6 terra | 332 | 400 |
| GPT-5.6 luna | 332 | 400 |
| Gemini 3.6 Flash | 332 | 400 |
| DeepSeek V4 Flash 0731 | 332 | 400 |
| Qwen 3.7 Flash | 332 | 400 |
| Kimi K3 | 150 | 1200 |
| GPT-5.6 sol | 37 | – |

## Struttura
```
notebook/finmr_eval.py   script unico di valutazione (stessa pipeline per tutti i modelli:
                         basta cambiare MODEL e REASONING_MAX_TOKENS nella sezione CONFIG)
requirements.txt
```

## Come eseguire
Tutta la pipeline è nel file `notebook/finmr_eval.py`. Per valutare un modello:
1. Impostare la chiave OpenRouter nella variabile d'ambiente `OPENROUTER_API_KEY`
   (in Google Colab: aggiungila nei **Secrets** con lo stesso nome). La chiave **non** è nel codice.
2. Nella sezione **CONFIG** in cima al file scegliere `MODEL` (lo slug OpenRouter) e
   `REASONING_MAX_TOKENS` (400 per la maggior parte dei modelli, 1200 per Kimi K3).
3. Eseguire:
```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="la-tua-chiave"   # su Windows: set OPENROUTER_API_KEY=...
python notebook/finmr_eval.py
```
I risultati vengono salvati in un file `risultati_<modello>.csv` (con ripresa automatica in caso di
interruzione) e le metriche finali sono stampate a schermo.

Serve una chiave [OpenRouter](https://openrouter.ai) con credito sufficiente: il costo dipende
soprattutto dalla lunghezza dei documenti in ingresso (in media oltre 30.000 token per richiesta).

## Licenza
Rilasciato sotto licenza **MIT** (vedi `LICENSE`).

## Autore
Andrea Vivoda — tesi triennale, relatore Prof. Mirko Lucchese.
