import re
import os
import random
import string
import streamlit as st
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-5.2"

st.set_page_config(page_title="LightLLMDissector", layout="centered")

st.markdown(
    """
    <style>
      :root { --accent:#2b6cb0; --soft:#e8f1ff; }
      .stButton>button {
        border: 1px solid var(--accent);
        background: white;
        color: var(--accent);
        border-radius: 10px;
        padding: 0.55rem 0.8rem;
        transition: 120ms ease-in-out;
      }
      .stButton>button:hover { background: var(--soft); transform: translateY(-1px); }
      textarea, input { border-radius: 10px !important; }
      .stAlert { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LightLLMDissector")
st.caption("A teaching tool to explore how LLM behaviour changes at each stage of development.")

EX_TRICK = (
    "A patient in a rural clinic in Maharashtra is newly diagnosed with HIV and has cough, fever, "
    "and weight loss suggestive of tuberculosis. When should ART be initiated?"
)
EX_LOCAL = (
    "A patient presents to a rural primary health center in Bihar with cough for 6 weeks, fever, and weight loss. "
    "What are the most likely causes and key differentials in this setting? Please answer in Hindi."
)
EX_IMPOSSIBLE = (
    "Please provide the 2025 guideline for cough management, as standardized by the National Dadi-Nani Council for "
    'WhatsApp Remedies, version "Forwarded by Mausi".'
)

st.markdown("**Ask your own question, or try one of these:**")
c1, c2, c3 = st.columns(3)

if "question_value" not in st.session_state:
    st.session_state.question_value = EX_TRICK

with c1:
    if st.button("TrickQuestion", use_container_width=True):
        st.session_state.question_value = EX_TRICK
with c2:
    if st.button("Local Relevance", use_container_width=True):
        st.session_state.question_value = EX_LOCAL
with c3:
    if st.button("Impossible Guideline", use_container_width=True):
        st.session_state.question_value = EX_IMPOSSIBLE

st.divider()

STOP = {
    "a","an","the","and","or","but","if","then","so","to","of","in","on","at","for","from","with","without",
    "is","are","was","were","be","been","being","do","does","did","can","could","should","would","will","may","might",
    "please","respond","answer","include","exact","official","approved","guideline","url",
    "hindi","english","french","swahili","kiswahili"
}

def fake_random_chars(length: int = 200) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>/?~"
    return "".join(random.choice(alphabet) for _ in range(length))

def clean_output(text: str) -> str:
    if not text:
        return ""
    t = text
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"__(.*?)__", r"\1", t)
    t = t.replace("`", "")
    t = re.sub(r"^\s*#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = t.replace("•", "-")
    t = re.sub(r"^\s*\*\s+", "- ", t, flags=re.MULTILINE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def longest_words(text: str, n: int = 12, min_len: int = 5) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9]+", text)
    out = []
    seen = set()
    for t in toks:
        tl = t.lower()
        if tl in STOP:
            continue
        if len(t) < min_len:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    out.sort(key=lambda x: (-len(x), x.lower()))
    return out[:n]

def remove_short_words_preserve_order(text: str, min_len_exclusive: int = 4) -> str:
    toks = re.findall(r"[A-Za-z0-9]+", text)
    kept = [t for t in toks if len(t) > min_len_exclusive]
    return " ".join(kept)

def build_user_message(stage_idx: int, full_question: str) -> str:
    if stage_idx == 0:
        return ""
    if stage_idx == 1:
        words = longest_words(full_question, n=12, min_len=5)
        random.shuffle(words)
        kw = " ".join(words) if words else ""
        return f"{kw}\n\nContinue writing as if this is a literary blog or novel mid-paragraph. Do not answer questions."
    if stage_idx == 2:
        filtered = remove_short_words_preserve_order(full_question, min_len_exclusive=4)
        return filtered[:1200]
    return full_question

def call_llm(system_prompt: str, user_q: str, max_tokens: int) -> str:
    if not OPENAI_API_KEY:
        return "[Error: OPENAI_API_KEY not set in environment]"
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_q},
            ],
            max_output_tokens=max_tokens,
        )
        return resp.output_text.strip()
    except Exception as e:
        return f"[Error calling model: {e}]"

GLOBAL_PREFACE = "Pedagogical demo. Research-only.\n"

STAGES = [
    {"ui": "Stage 1: Random initialization", "answer_key": "A", "max_tokens": 0, "system": GLOBAL_PREFACE},
    {"ui": "Stage 2: Pre-training", "answer_key": "B", "max_tokens": 320, "system": GLOBAL_PREFACE + """
Simulate large-scale generic internet text pre-training (next-token completion). Not instruction-following.
Write fluent English prose as if continuing a blog/novel mid-paragraph.
Do not answer questions. No lists. No clinical guidance.
Plain text only. No markdown.
""".strip()},
    {"ui": "Stage 3: Continued pre-training", "answer_key": "C", "max_tokens": 360, "system": GLOBAL_PREFACE + """
Simulate continued pre-training on biomedical and scientific literature. Still not instruction-following.
Infer a plausible hidden scientific question and continue as if finishing a PubMed abstract/review excerpt.
Formal scientific English only. Do not translate. Ignore any language cues in the input.
Do not provide dosing or step-by-step treatment.
Plain text only. No markdown.
""".strip()},
    {"ui": "Stage 4: Instruction tuning", "answer_key": "D", "max_tokens": 420, "system": GLOBAL_PREFACE + """
You are trained on instruction-response datasets.
Follow the user request where possible, but do not invent documents/guidelines/authorities.
If the requested guideline appears not to exist, say you cannot verify it; optionally provide clearly labeled
"Illustrative (hypothetical)" citation formats.
Plain text only. No markdown.
Put any disclaimer as the final line starting with "Note:".
""".strip()},
    {"ui": "Stage 5: Safety tuning", "answer_key": "E", "max_tokens": 280, "system": GLOBAL_PREFACE + """
You are trained on safety and policy-aligned examples.
3-6 bullets, each starting with "- ".
Avoid invented sources. Avoid dosing and step-by-step treatment.
Put any disclaimer as the final line starting with "Note:".
Plain text only. No markdown.
""".strip()},
    {"ui": "Stage 6: Real-world alignment", "answer_key": "F", "max_tokens": 340, "system": GLOBAL_PREFACE + """
You are trained on domain- and locale-specific real-world data for clinical settings.
Respond in the same language as the user.
6-8 bullets, each starting with "- ".
Include one bullet that begins exactly: "What to validate locally:"
Avoid invented sources. Avoid dosing and step-by-step treatment.
Put any disclaimer as the final line starting with "Note:".
Plain text only. No markdown.
""".strip()},
]

QUIZ_BANK = [
    ("A", "Untrained (no data)"),
    ("B", "Large-scale generic internet text"),
    ("C", "Biomedical and scientific literature"),
    ("D", "Instruction-response datasets"),
    ("E", "Safety and policy-aligned examples"),
    ("F", "Domain- and locale-specific real-world data"),
]

if "stage_idx" not in st.session_state:
    st.session_state.stage_idx = 0
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "quiz_correct" not in st.session_state:
    st.session_state.quiz_correct = False
if "quiz_order" not in st.session_state:
    st.session_state.quiz_order = []
if "quiz_choice" not in st.session_state:
    st.session_state.quiz_choice = None
if "checked" not in st.session_state:
    st.session_state.checked = False

stage = STAGES[st.session_state.stage_idx]

st.markdown("**Model stage**")
st.markdown(f"### {stage['ui']}")

# Small back link under the stage title (hidden on Stage 1)
if st.session_state.stage_idx > 0:
    if st.button("← Back", key="back_link"):
        st.session_state.stage_idx -= 1
        st.session_state.last_answer = ""
        st.session_state.quiz_correct = False
        st.session_state.quiz_choice = None
        st.session_state.checked = False
        st.rerun()

question = st.text_area("Question", height=150, value=st.session_state.question_value)
st.session_state.question_value = question


if st.button("Send", use_container_width=True):
    st.session_state.quiz_correct = False
    st.session_state.quiz_choice = None
    st.session_state.checked = False
    order = QUIZ_BANK.copy()
    random.shuffle(order)
    st.session_state.quiz_order = order

    with st.spinner("Running model..."):
        if st.session_state.stage_idx == 0:
            st.session_state.last_answer = fake_random_chars(200)
        else:
            user_payload = build_user_message(st.session_state.stage_idx, question)
            st.session_state.last_answer = clean_output(
                call_llm(stage["system"], user_payload, stage["max_tokens"])
            )

if st.session_state.last_answer:
    st.text_area("Model response", value=st.session_state.last_answer, height=320, disabled=True)

    st.markdown("### Quick quiz")
    st.write("Based on behaviour, what kind of training data most likely shaped this model?")

    option_texts = [t for _, t in st.session_state.quiz_order]
    st.session_state.quiz_choice = st.radio("Select one", option_texts, label_visibility="collapsed")

    nav1, nav2, nav3 = st.columns(3)

    with nav2:
        if st.button("Check answer", use_container_width=True):
            st.session_state.checked = True
            chosen_key = None
            for k, t in st.session_state.quiz_order:
                if t == st.session_state.quiz_choice:
                    chosen_key = k
                    break
            st.session_state.quiz_correct = (chosen_key == stage["answer_key"])

    with nav3:
        if st.button("Next stage", use_container_width=True, disabled=(not st.session_state.quiz_correct)):
            st.session_state.stage_idx = min(st.session_state.stage_idx + 1, len(STAGES) - 1)
            st.session_state.last_answer = ""
            st.session_state.quiz_correct = False
            st.session_state.quiz_choice = None
            st.session_state.checked = False
            st.rerun()

    if st.session_state.checked:
        if st.session_state.quiz_correct:
            st.success("Correct. Next stage unlocked.")
        else:
            st.markdown(
                "<div style='color:#1f77b4; font-weight:700;'>Incorrect, try again to unlock the next stage.</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Select an option and click Check answer.")

st.divider()
st.caption("Made by LiGHT (Laboratory for Intelligent Global Health & Humanitarian Response Technologies) • EPFL • Harvard • Ashoka University (Koita Centre for Digital Health) • C4IR Rwanda")
