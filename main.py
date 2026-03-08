import streamlit as st
import os
import base64
import streamlit.components.v1 as components

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ABSLI AI Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================================================
# 👇 ADD AS MANY PDFs AS YOU WANT HERE
# ================================================================
PDF_FILES = [
    "ABSLI_Group_Bima_Yojana_V04_Brochure_Web_Version_1c032b927b.pdf",
    "ABSLI_Assured_Savings_Plan_Brochure_Web_Version_V08_a623a971ac.pdf",
    "ABSLI_GROUP_VALUE_PLUS_PLAN_LEAFLET_96bcc3e1d1.pdf",
    "ABSLI_Monthly_Income_Plan_Brochure_cbf1c4cf6b.pdf",
    "ABSLI_Param_Suraksha_V01_Leaflet_Web_Version_6798874360.pdf",
    "ABSLI_Super_Term_Plan_V01_Brochure_8b9b57ef0d.pdf",
    "life_insurance_complete_guide_updated.pdf"
]
# ================================================================

DB_DIR = "chroma_db"

# ---------------- LOAD LOGO ----------------

def get_logo_base64(path="logo.png"):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

logo_b64 = get_logo_base64()
logo_img = (
    f'<img src="data:image/png;base64,{logo_b64}" style="height:44px;object-fit:contain;" />'
    if logo_b64 else
    '<span style="color:#C8102E;font-weight:900;font-size:18px;">ABSLI</span>'
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    footer { display:none!important; }
    section[data-testid="stSidebar"] { display:none!important; }

    .stApp,
    .main,
    .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stVerticalBlock"],
    [data-testid="stElementContainer"] {
        padding: 0 !important;
        margin: 0 !important;
        gap: 0 !important;
    }
    .stApp { background: #f5f5f5 !important; font-family: 'DM Sans', sans-serif !important; }

    /* ── NUCLEAR OPTION: force ALL elements in bottom area white ── */
    [data-testid="stBottom"],
    [data-testid="stBottom"] *,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] *,
    .stBottom,
    .stBottom * {
        background: #ffffff !important;
        background-color: #ffffff !important;
        background-image: none !important;
    }

    [data-testid="stBottom"] {
        border-top: 3px solid #C8102E !important;
    }

    [data-testid="stChatInput"],
    [data-testid="stChatInputContainer"] {
        background: #ffffff !important;
        border: 2.5px solid #C8102E !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        outline: none !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputContainer"] textarea {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        font-size: 14px !important;
        caret-color: #C8102E !important;
        font-family: 'DM Sans', sans-serif !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] button,
    [data-testid="stChatInputContainer"] button {
        background: #C8102E !important;
        background-color: #C8102E !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] button svg path,
    [data-testid="stChatInputContainer"] button svg path {
        fill: white !important;
        stroke: white !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInputContainer"] textarea::placeholder {
        color: #999 !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        padding: 10px 14px !important;
        margin: 6px 0 !important;
        border-radius: 16px !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #C8102E, #a00d24) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) span {
        color: #ffffff !important; font-size: 14px !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #ffffff !important;
        border: 1px solid #efefef !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) div,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) span {
        color: #1a1a1a !important; font-size: 14px !important;
    }

    /* Typing dots */
    .typing-indicator { display:flex; align-items:center; gap:5px; padding:6px 0; }
    .typing-indicator span {
        width:9px; height:9px; background:#C8102E; border-radius:50%;
        display:inline-block; animation:tdots 1.2s infinite ease-in-out; opacity:0.7;
    }
    .typing-indicator span:nth-child(1){animation-delay:0s}
    .typing-indicator span:nth-child(2){animation-delay:0.2s}
    .typing-indicator span:nth-child(3){animation-delay:0.4s}
    @keyframes tdots{0%,60%,100%{transform:translateY(0);opacity:0.4}30%{transform:translateY(-7px);opacity:1}}
    .streaming-cursor::after{content:'|';animation:cblink 0.7s infinite;color:#C8102E;margin-left:2px}
    @keyframes cblink{0%,100%{opacity:1}50%{opacity:0}}
</style>

<script>
function styleBottomBar() {
    const darkBgs = [
        'rgb(14, 17, 23)', 'rgb(0, 0, 0)', 'rgb(38, 39, 48)',
        'rgb(26, 28, 36)', 'rgb(17, 17, 17)', 'rgb(30, 30, 30)',
        'rgb(11, 11, 11)', 'rgb(20, 20, 20)', 'rgb(33, 33, 33)', 'rgb(15, 15, 15)'
    ];
    document.querySelectorAll('*').forEach(el => {
        const style = window.getComputedStyle(el);
        const bg = style.backgroundColor;
        const match = bg.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (match) {
            const r = parseInt(match[1]), g = parseInt(match[2]), b = parseInt(match[3]);
            if (r < 60 && g < 60 && b < 60) {
                el.style.setProperty('background', '#ffffff', 'important');
                el.style.setProperty('background-color', '#ffffff', 'important');
            }
        }
    });

    const bottom = document.querySelector('[data-testid="stBottom"]');
    if (bottom) {
        let el = bottom;
        while (el && el !== document.body) {
            el.style.setProperty('background', '#ffffff', 'important');
            el.style.setProperty('background-color', '#ffffff', 'important');
            el = el.parentElement;
        }
        bottom.style.setProperty('border-top', '3px solid #C8102E', 'important');
        bottom.style.setProperty('padding', '12px 24px', 'important');

        bottom.querySelectorAll('*').forEach(el => {
            const tag = el.tagName.toLowerCase();
            el.style.setProperty('background', '#ffffff', 'important');
            el.style.setProperty('background-color', '#ffffff', 'important');
            if (tag === 'textarea') {
                el.style.setProperty('color', '#1a1a1a', 'important');
                el.style.setProperty('font-size', '14px', 'important');
                el.style.setProperty('caret-color', '#C8102E', 'important');
            }
            const testid = el.getAttribute('data-testid');
            if (testid === 'stChatInput' || testid === 'stChatInputContainer') {
                el.style.setProperty('border', '2.5px solid #C8102E', 'important');
                el.style.setProperty('border-radius', '12px', 'important');
                el.style.setProperty('box-shadow', 'none', 'important');
            }
            if (tag === 'button') {
                el.style.setProperty('background', '#C8102E', 'important');
                el.style.setProperty('background-color', '#C8102E', 'important');
                el.style.setProperty('border-radius', '8px', 'important');
                el.querySelectorAll('svg path').forEach(p => {
                    p.style.setProperty('fill', 'white', 'important');
                    p.style.setProperty('stroke', 'white', 'important');
                });
            }
        });
    }
}

styleBottomBar();
[100, 300, 700, 1500, 3000].forEach(t => setTimeout(styleBottomBar, t));
const observer = new MutationObserver(() => styleBottomBar());
observer.observe(document.body, { childList: true, subtree: true, attributes: true });
</script>
""", unsafe_allow_html=True)

# ---------------- NAVBAR + CARDS ----------------

components.html(f"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'DM Sans',sans-serif; background:transparent; }}
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

  /* ── NAVBAR ── */
  .navbar {{
    background:#C8102E;
    padding:0 20px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    height:64px;
    width:100%;
    box-shadow:0 3px 12px rgba(0,0,0,0.2);
  }}
  .logo-wrap {{
    background:white;
    border-radius:6px;
    padding:5px 10px;
    display:flex;
    align-items:center;
    flex-shrink:0;
  }}
  .nav-links {{
    display:flex;
    align-items:center;
    gap:2px;
    flex:1;
    justify-content:center;
  }}
  .nav-links a {{
    color:white;
    text-decoration:none;
    font-size:13px;
    font-weight:500;
    padding:7px 10px;
    border-radius:6px;
    white-space:nowrap;
  }}
  .nav-links a:hover {{ background:rgba(255,255,255,0.15); }}
  .nav-actions {{ display:flex; align-items:center; gap:8px; flex-shrink:0; }}
  .btn-pay {{
    background:transparent;
    color:white;
    border:2px solid white;
    padding:6px 14px;
    border-radius:6px;
    font-size:12px;
    font-weight:700;
    cursor:pointer;
    white-space:nowrap;
  }}
  .btn-login {{
    background:white;
    color:#C8102E;
    border:none;
    padding:6px 14px;
    border-radius:6px;
    font-size:12px;
    font-weight:700;
    cursor:pointer;
  }}

  /* ── BODY ── */
  .body {{ padding:16px 20px 14px 20px; max-width:1400px; margin:0 auto; }}
  .section-title {{
    font-size:15px;
    font-weight:700;
    color:#1a1a1a;
    margin-bottom:14px;
    display:flex;
    align-items:center;
    gap:8px;
  }}
  .section-title span {{
    display:inline-block;
    width:4px;
    height:18px;
    background:#C8102E;
    border-radius:2px;
  }}

  /* ── PLANS GRID ── */
  .plans-grid {{
    display:grid;
    grid-template-columns:repeat(7, 1fr);
    gap:10px;
    margin-bottom:16px;
  }}

  .plan-card {{
    background:white;
    border-radius:12px;
    border:1.5px solid #f0f0f0;
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
    padding:14px 12px;
    display:flex;
    flex-direction:column;
    gap:6px;
    cursor:pointer;
    transition:box-shadow 0.2s, transform 0.2s, border-color 0.2s;
    position:relative;
    overflow:hidden;
  }}
  .plan-card::before {{
    content:'';
    position:absolute;
    top:0; left:0; right:0;
    height:3px;
    background:#C8102E;
    border-radius:12px 12px 0 0;
    transform:scaleX(0);
    transition:transform 0.2s;
  }}
  .plan-card:hover {{
    box-shadow:0 8px 28px rgba(200,16,46,0.13);
    transform:translateY(-3px);
    border-color:#f5c0c8;
  }}
  .plan-card:hover::before {{ transform:scaleX(1); }}

  .plan-icon {{ font-size:22px; margin-bottom:2px; }}
  .plan-name {{ font-size:12px; font-weight:700; color:#1a1a1a; line-height:1.3; }}
  .plan-desc {{ font-size:11px; color:#777; line-height:1.5; flex:1; }}
  .plan-badges {{ display:flex; flex-direction:column; gap:4px; margin-top:4px; }}
  .badge {{
    font-size:10px;
    font-weight:600;
    padding:3px 7px;
    border-radius:20px;
    display:inline-flex;
    align-items:center;
    gap:3px;
    width:fit-content;
  }}
  .badge-ideal {{ background:#fff0f2; color:#C8102E; }}
  .badge-maturity {{ background:#f0faf4; color:#1a7a3f; }}
  .badge-no-maturity {{ background:#f5f5f5; color:#888; }}

  .intro-box {{
    background:white;
    padding:14px 18px;
    border-radius:14px;
    border:1px solid #e5e5e5;
    color:#222;
    font-size:14px;
    line-height:1.6;
    box-shadow:0 2px 8px rgba(0,0,0,0.04);
  }}

  /* ── TABLET: 4 cols ── */
  @media (max-width: 900px) {{
    .plans-grid {{ grid-template-columns:repeat(4, 1fr); gap:10px; }}
    .nav-links {{ display:none; }}
  }}

  /* ── MOBILE: 2 cols ── */
  @media (max-width: 600px) {{
    .navbar {{ padding:0 12px; height:52px; }}
    .btn-pay {{ display:none; }}
    .body {{ padding:12px 10px 10px 10px; }}
    .plans-grid {{
      grid-template-columns:repeat(2, 1fr);
      gap:8px;
    }}
    .plan-card {{
      padding:12px 10px;
      gap:5px;
      border-radius:10px;
    }}
    .plan-icon {{ font-size:20px; }}
    .plan-name {{ font-size:12px; }}
    .plan-desc {{ font-size:10.5px; }}
    .badge {{ font-size:9px; padding:2px 6px; }}
    .section-title {{ font-size:13px; margin-bottom:10px; }}
    .intro-box {{ font-size:13px; padding:11px 13px; border-radius:10px; }}
  }}

  /* ── VERY SMALL ── */
  @media (max-width: 360px) {{
    .plans-grid {{ gap:6px; }}
    .plan-card {{ padding:10px 8px; }}
    .plan-name {{ font-size:11px; }}
    .plan-desc {{ font-size:10px; }}
  }}
</style>
</head>
<body>

<div class="navbar">
  <div class="logo-wrap">{logo_img}</div>
  <div class="nav-links">
    <a href="#">All Insurance &#9660;</a>
    <a href="#">Articles &#9660;</a>
    <a href="#">Where Do I? &#9660;</a>
    <a href="#">Term Insurance</a>
    <a href="#">Manage My Policy</a>
    <a href="#">Her Insurance</a>
  </div>
  <div class="nav-actions">
    <span style="color:white;font-size:18px;cursor:pointer;padding:0 6px;">&#128269;</span>
    <button class="btn-pay">PAY PREMIUM</button>
    <button class="btn-login">LOGIN</button>
  </div>
</div>

<div class="body">
  <div class="section-title"><span></span> Compare Life Insurance Plans</div>
  <div class="plans-grid">

    <div class="plan-card">
      <div class="plan-icon">🛡️</div>
      <div class="plan-name">Term Insurance</div>
      <div class="plan-desc">High coverage at low premiums. Pure protection, no savings.</div>
      <div class="plan-badges">
        <span class="badge badge-ideal">👤 Protection seekers</span>
        <span class="badge badge-no-maturity">✗ No maturity benefit</span>
      </div>
    </div>

    <div class="plan-card">
      <div class="plan-icon">🏛️</div>
      <div class="plan-name">Whole Life Insurance</div>
      <div class="plan-desc">Lifetime coverage with savings component and cash value growth.</div>
      <div class="plan-badges">
        <span class="badge badge-ideal">👤 Long-term planners</span>
        <span class="badge badge-maturity">✓ Cash value payout</span>
      </div>
    </div>

    <div class="plan-card">
      <div class="plan-icon">💰</div>
      <div class="plan-name">Endowment Plan</div>
      <div class="plan-desc">Life cover + savings with guaranteed payouts at maturity.</div>
      <div class="plan-badges">
        <span class="badge badge-ideal">👤 Risk-averse investors</span>
        <span class="badge badge-maturity">✓ Guaranteed payout</span>
      </div>
    </div>

    <div class="plan-card">
      <div class="plan-icon">📈</div>
      <div class="plan-name">ULIP</div>
      <div class="plan-desc">Life cover + market-linked investments. Returns based on fund performance.</div>
      <div class="plan-badges">
        <span class="badge badge-ideal">👤 Market investors</span>
        <span class="badge badge-maturity">✓ Fund-linked returns</span>
      </div>
    </div>

    <div class="plan-card">
      <div class="plan-icon">👶</div>
      <div class="plan-name">Child Plan</div>
      <div class="plan-desc">Financial support for key milestones in your child's future.</div>
      <div class="plan-badges">
        <span class="badge badge-ideal">👤 Parents</span>
        <span class="badge badge-maturity">✓ Milestone payouts</span>
      </div>
    </div>

    <div class="plan-card">
      <div class="plan-icon">🧓</div>
      <div class="plan-name">Pension Plan</div>
      <div class="plan-desc">Build a retirement corpus with regular annuity post-retirement.</div>
      <div class="plan-badges">
        <span class="badge badge-ideal">👤 Retirement planners</span>
        <span class="badge badge-maturity">✓ Annuity payments</span>
      </div>
    </div>

    <div class="plan-card">
      <div class="plan-icon">💵</div>
      <div class="plan-name">Money-Back Plan</div>
      <div class="plan-desc">Periodic payouts during the policy term plus a maturity benefit.</div>
      <div class="plan-badges">
        <span class="badge badge-ideal">👤 Liquidity seekers</span>
        <span class="badge badge-maturity">✓ Periodic + maturity</span>
      </div>
    </div>

  </div>
  <div class="intro-box">
    Hello! I am your <b>Aditya Birla Capital Life Insurance</b> assistant.
    How can I help you understand your insurance options today?
  </div>
</div>

</body>
</html>
""", height=580, scrolling=False)

# ---------------- RAG ----------------

@st.cache_resource
def load_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(DB_DIR):
        vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        all_chunks = []
        for pdf_path in PDF_FILES:
            if not os.path.exists(pdf_path):
                st.warning(f"PDF not found: {pdf_path}")
                continue
            loader   = PyPDFLoader(pdf_path)
            docs     = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
            chunks   = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source_file"] = os.path.basename(pdf_path)
            all_chunks.extend(chunks)
        if not all_chunks:
            st.error("No PDF documents loaded.")
            st.stop()
        vectordb = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        vectordb.persist()

    llm       = ChatOllama(model="phi3:mini", temperature=0)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    template  = """
You are a helpful insurance assistant for Aditya Birla Sun Life Insurance (ABSLI).
Use the context below to answer clearly and in detail.
- Give a complete answer if the info is present.
- Share related info if exact answer isn't found.
- Only say not found if truly nothing is relevant.

Context: {context}
Question: {question}
Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

chain = load_chain()

# ---------------- CHAT ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(
            f"<div style='color:#1a1a1a;font-size:14px;line-height:1.7;'>{msg['content']}</div>"
            if msg["role"] == "assistant" else msg["content"],
            unsafe_allow_html=True
        )

if question := st.chat_input("Type your query about life insurance..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("""
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
            <div style="color:#C8102E;font-size:12px;margin-top:4px;font-weight:500;">
                &#128196; Reviewing policy documents...
            </div>
        """, unsafe_allow_html=True)

        full_response = ""
        for chunk in chain.stream(question):
            full_response += chunk
            message_placeholder.markdown(
                f"<div style='color:#1a1a1a;font-size:14px;line-height:1.7;' "
                f"class='streaming-cursor'>{full_response}</div>",
                unsafe_allow_html=True
            )
        message_placeholder.markdown(
            f"<div style='color:#1a1a1a;font-size:14px;line-height:1.7;'>{full_response}</div>",
            unsafe_allow_html=True
        )

    st.session_state.messages.append({"role": "assistant", "content": full_response})