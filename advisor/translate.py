"""Offline English -> Indian-language translation of the advice with IndicTrans2 (AI4Bharat, MIT licence).

Model: ai4bharat/indictrans2-en-indic-dist-200M (distilled, 200M parameters, 22 Indian languages), stored in
advisor/translator/. Llama 3.2 1B writes the advice in English (its strongest language); this module translates
it line by line, keeping the numbered steps and the "Problem / What to do / Prevention / Safety" layout.

The official IndicTransToolkit needs a C compiler on Windows, so the small pre/post-processing it does is
re-implemented here: punctuation normalisation, protecting numbers/doses/units from being translated, and the
"<src_lang> <tgt_lang>" tags the model expects.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "translator", "indictrans2-en-indic-dist-200M")

LANG_CODES = {
    "Hindi": "hin_Deva", "Bengali": "ben_Beng", "Telugu": "tel_Telu", "Marathi": "mar_Deva", "Tamil": "tam_Taml",
    "Gujarati": "guj_Gujr", "Kannada": "kan_Knda", "Malayalam": "mal_Mlym", "Odia": "ory_Orya",
    "Punjabi": "pan_Guru", "Assamese": "asm_Beng", "Urdu": "urd_Arab", "Nepali": "npi_Deva",
    "Konkani": "gom_Deva", "Maithili": "mai_Deva", "Sanskrit": "san_Deva", "Sindhi": "snd_Arab",
    "Kashmiri": "kas_Arab", "Dogri": "doi_Deva", "Manipuri": "mni_Beng", "Bodo": "brx_Deva", "Santali": "sat_Olck",
}
# section labels translated by hand (short, fixed words; keeps the layout stable)
LABELS = {
    "Hindi": {"Problem": "समस्या", "What to do": "क्या करें", "Prevention": "बचाव", "Safety": "सुरक्षा",
              "Note": "ध्यान दें"},
    "Kannada": {"Problem": "ಸಮಸ್ಯೆ", "What to do": "ಏನು ಮಾಡಬೇಕು", "Prevention": "ತಡೆಗಟ್ಟುವಿಕೆ",
                "Safety": "ಸುರಕ್ಷತೆ", "Note": "ಗಮನಿಸಿ"},
    "Tamil": {"Problem": "பிரச்சனை", "What to do": "என்ன செய்ய வேண்டும்", "Prevention": "தடுப்பு",
              "Safety": "பாதுகாப்பு", "Note": "குறிப்பு"},
    "Telugu": {"Problem": "సమస్య", "What to do": "ఏమి చేయాలి", "Prevention": "నివారణ", "Safety": "భద్రత",
               "Note": "గమనిక"},
    "Marathi": {"Problem": "समस्या", "What to do": "काय करावे", "Prevention": "प्रतिबंध", "Safety": "सुरक्षा",
                "Note": "टीप"},
}
# IndicTrans2 writes most Indic languages in Devanagari (script unification); convert to the native script
# afterwards, as the official IndicTransToolkit does. Value = Indic NLP Library language code of the target script.
NATIVE_SCRIPT = {"ben_Beng": "bn", "asm_Beng": "bn", "mni_Beng": "bn", "tel_Telu": "te", "tam_Taml": "ta",
                 "guj_Gujr": "gu", "kan_Knda": "kn", "mal_Mlym": "ml", "ory_Orya": "or", "pan_Guru": "pa"}
# doses, formulation codes and phone numbers must survive translation unchanged (product names are transliterated)
PROTECT = re.compile(r"(\b\d+(?:[.\-]\d+)?\s?(?:g/L|ml/L|g|ml|kg|L|ppm|cm|WP|EC|SC|SL|WG|SG)\b"
                     r"|\b1800-180-1551\b|\bKVK\b)")


with open(os.path.join(HERE, "knowledge", "crop_names.json"), encoding="utf-8") as _f:
    CROP_NAMES = {k: v for k, v in __import__("json").load(_f).items() if not k.startswith("_")}


class Translator:
    def __init__(self, model_dir=MODEL_DIR, num_beams=None):
        import torch
        from sacremoses import MosesPunctNormalizer
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True).eval()
        self.punct = MosesPunctNormalizer(lang="en")
        self.num_beams = num_beams or int(os.environ.get("KR_TRANSLATE_BEAMS", "4"))  # fewer = faster on the Pi

    def _protect(self, s, terms=(), language=None):
        """Replace doses/codes, the diagnosed disease or pest name, and crop names with placeholders the model
        copies. Crop names come back as the correct local word (the model once turned 'tomato' into 'onion')."""
        keep = []

        def sub(m, value=None):
            keep.append(value if value is not None else m.group(0))
            return f"<ID{len(keep) - 1}>"
        for t in sorted(terms, key=len, reverse=True):
            if t:
                s = re.sub(re.escape(t), sub, s, flags=re.IGNORECASE)
        for crop, names in CROP_NAMES.items():
            if language in names:
                s = re.sub(rf"\b{crop}(?:e?s)?\b", lambda m, n=names[language]: sub(m, n), s, flags=re.IGNORECASE)
        return PROTECT.sub(sub, s), keep

    @staticmethod
    def _restore(s, keep):
        # the model sometimes drops or changes the closing bracket of a placeholder ("<ID0", "<ID1′")
        for i, k in enumerate(keep):
            s = re.sub(rf"<\s*I\s*D\s*{i}\s*[>′’'`]*", lambda _m: k, s)
        return s.replace("′", "")

    def translate_sentences(self, sents, language, terms=()):
        code = LANG_CODES[language]
        prepped, keeps = [], []
        for s in sents:
            p, k = self._protect(self.punct.normalize(s.strip()), terms, language)
            prepped.append(f"eng_Latn {code} {p}")
            keeps.append(k)
        enc = self.tok(prepped, truncation=True, padding="longest", return_tensors="pt")
        with self.torch.no_grad():
            # output length limit relative to the input: stops the rare runaway "....." loop (it took 132 s once)
            limit = min(256, int(2.5 * enc["input_ids"].shape[1]) + 16)
            out = self.model.generate(**enc, use_cache=True, min_length=0, max_length=limit,
                                      num_beams=self.num_beams, num_return_sequences=1)
        dec = self.tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if code in NATIVE_SCRIPT:
            from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
            dec = [UnicodeIndicTransliterator.transliterate(d, "hi", NATIVE_SCRIPT[code]) for d in dec]
        dec = [re.sub(r"\s*([.।,!?])(?:\s*\1)+", r"\1", d) for d in dec]  # collapse runs like "....."
        return [self._restore(d.strip(), k) for d, k in zip(dec, keeps)]

    def translate_advice(self, text, language, terms=()):
        """Translate the structured advice, keeping labels, numbering and line breaks. `terms` (the diagnosed
        disease / pest name) stay in English: a word-by-word translation can mislead ('scab' -> 'scratch'),
        while the English name is what agriculture officers and input shops recognise."""
        if language == "English" or language not in LANG_CODES:
            return text
        labels = LABELS.get(language, {})
        lines, jobs = text.splitlines(), []
        parsed = []
        for line in lines:
            m = re.match(r"^\s*(Problem|What to do|Prevention|Safety|Note)\s*:\s*(.*)$", line)
            n = re.match(r"^\s*(\d+)\.\s*(.*)$", line)
            if m:
                parsed.append(("label", m.group(1), m.group(2)))
            elif n:
                parsed.append(("step", n.group(1), n.group(2)))
            else:
                parsed.append(("text", "", line))
        bodies = [p[2] for p in parsed if p[2].strip()]
        done = iter(self.translate_sentences(bodies, language, terms)) if bodies else iter(())
        out = []
        for kind, head, body in parsed:
            tr = next(done) if body.strip() else ""
            if kind == "label":
                out.append(f"{labels.get(head, head)}: {tr}".rstrip())
            elif kind == "step":
                out.append(f"{head}. {tr}")
            else:
                out.append(tr)
        return "\n".join(out)
