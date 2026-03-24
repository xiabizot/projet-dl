"""Cell.IA -- Agent IA Classification de Tissus Colorectaux
Projet universitaire — DU Sorbonne Data Analytics 2026"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import base64
import io
from st_clickable_images import clickable_images

sys.path.insert(0, str(Path(__file__).parent / 'src'))

# =============================================
# AUTO-DOWNLOAD MODELS FROM GOOGLE DRIVE
# =============================================
def download_models_if_needed():
    """Download models/embeddings from Google Drive if not present locally."""
    base = Path(__file__).parent / 'data'
    models_dir = base / 'models'
    emb_dir = base / 'embeddings'
    models_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    files_needed = {
        models_dir / 'NB3_cnn_model_v1.pth': None,
        models_dir / 'resnet_finetune_model.pth': None,
        models_dir / 'NB3_cnn_preds.pkl': None,
        models_dir / 'resnet_preds.pkl': None,
        emb_dir / 'cnn_test_embeddings.npy': None,
        emb_dir / 'cnn_test_images.npy': None,
        emb_dir / 'cnn_test_labels.npy': None,
    }

    missing = [f for f in files_needed if not f.exists()]
    if missing:
        try:
            import gdown
            folder_url = 'https://drive.google.com/drive/folders/12hEISvotV0l6FJQ5oPgYPwVNUdEquVIu'
            tmp_dir = base / '_tmp_gdrive'
            gdown.download_folder(folder_url, quiet=True, use_cookies=False, remaining_ok=True, output=str(tmp_dir))
            import shutil
            for f in tmp_dir.iterdir():
                if 'embeddings' in f.name or 'test' in f.name:
                    dest = emb_dir / f.name
                else:
                    dest = models_dir / f.name
                shutil.move(str(f), str(dest))
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            st.warning(f"Impossible de telecharger les modeles : {e}")

download_models_if_needed()

from agent_pathmnist import (
    run_agent, load_models, grad_cam, get_recommendations, explain_with_claude,
    CLASSES, NORM_MEAN, NORM_STD, N_CLASSES, DATA_DIR,
)

# =============================================
# CONFIG
# =============================================
st.set_page_config(
    page_title='Cell.IA',
    page_icon='C',
    layout='centered',
    initial_sidebar_state='collapsed',
)

# =============================================
# PASTEL PALETTE
# =============================================
P = {
    'rose': '#d4688a',
    'bleu': '#5a9ec0',
    'vert': '#5aaa78',
    'jaune': '#c0a850',
    'violet': '#8a70b8',
    'orange': '#c89060',
    'bg': '#ffffff',
    'card': '#f8f9fc',
    'border': '#e0e4ec',
    'text': '#1a1a2e',
    'dim': '#8890a8',
    'white': '#ffffff',
}

CLS_COLOR = ['#c0a850','#5a9ec0','#8a70b8','#5aaa78','#c89060','#d4688a','#5a9ec0','#8a70b8','#c04050']

# =============================================
# CSS
# =============================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {{
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

.block-container {{
    padding-top: 2.5rem !important;
    max-width: 860px;
}}

/* Kill default Streamlit header */
header[data-testid="stHeader"] {{
    background: {P['bg']} !important;
    backdrop-filter: none !important;
    border-bottom: none !important;
}}

/* Hide sidebar completely */
section[data-testid="stSidebar"],
button[data-testid="stSidebarCollapseButton"] {{ display: none !important; }}

/* Hide hamburger menu */
button[kind="header"] {{ display: none !important; }}

/* Tabs */
div[data-testid="stTabs"] > div[role="tablist"] {{
    gap: 0;
    border-bottom: 1px solid {P['border']};
    margin-bottom: 16px;
}}
div[data-testid="stTabs"] button[role="tab"] {{
    flex: 1;
    background: none !important;
    color: {P['dim']} !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 10px 8px !important;
    letter-spacing: 0.02em;
}}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    color: {P['rose']} !important;
    border-bottom-color: {P['rose']} !important;
    font-weight: 600 !important;
}}

/* All buttons */
.stButton > button {{
    border-radius: 4px !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
    border: 1px solid {P['border']} !important;
    background: {P['card']} !important;
    color: {P['text']} !important;
    transition: all 0.15s;
    min-height: 122px !important;
}}
.stButton > button:hover {{
    border-color: {P['rose']} !important;
    color: {P['rose']} !important;
}}


/* Center images */
div[data-testid="stImage"] {{
    display: flex !important;
    justify-content: center !important;
}}

/* Selectbox */
div[data-testid="stSelectbox"] {{
    max-width: 280px;
    margin: 0 auto;
}}

/* Radio */
div[data-testid="stRadio"] > div {{
    justify-content: center;
    gap: 8px;
}}

/* Dataframes */
div[data-testid="stDataFrame"] {{
    font-size: 0.76rem !important;
}}

</style>
""", unsafe_allow_html=True)


# =============================================
# DATA
# =============================================
@st.cache_resource
def init_models():
    return load_models()

@st.cache_resource
def load_test_dataset():
    from medmnist import PathMNIST as PathMNISTDS
    return PathMNISTDS(split='test', download=True, root=str(DATA_DIR))

@st.cache_data
def get_class_indices(_test_ds):
    """Index all test images by class for fast random pick."""
    indices = {c: [] for c in range(N_CLASSES)}
    for idx in range(len(_test_ds)):
        _, label = _test_ds[idx]
        lbl = int(np.array(label).flatten()[0])
        indices[lbl].append(idx)
    return indices

@st.cache_data
def get_class_reps(_test_ds):
    reps = {}
    for idx in range(len(_test_ds)):
        img, label = _test_ds[idx]
        lbl = int(np.array(label).flatten()[0])
        if lbl not in reps:
            reps[lbl] = np.array(img)
        if len(reps) == N_CLASSES:
            break
    return reps

@st.cache_data
def get_metrics():
    import pickle
    from sklearn.metrics import classification_report
    out = {}
    for name, fname in [('CNN v1', 'NB3_cnn_preds.pkl'), ('ResNet FT', 'resnet_preds.pkl')]:
        fpath = Path(DATA_DIR) / 'models' / fname
        if fpath.exists():
            with open(fpath, 'rb') as f:
                d = pickle.load(f)
            labels = np.array(d['all_labels']).flatten()
            preds = np.array(d['all_preds']).flatten()
            out[name] = classification_report(labels, preds, target_names=CLASSES, digits=4, output_dict=True)
    return out

def pil_to_b64(pil_img, size=80):
    pil_img = pil_img.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# =============================================
# HEADER — logo + titre
# =============================================
logo_path = Path(__file__).parent / 'assets' / 'IA.png'
logo_html = ""
if logo_path.exists():
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:80px; border-radius:6px;">'

st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:center; gap:18px; margin:10px 0 6px 0;">
    {logo_html}
    <span style="font-size:3rem; font-weight:700; color:{P['text']}; letter-spacing:-0.03em;">Cell.IA</span>
</div>
<div style="text-align:center; margin-bottom:16px;">
    <div style="font-size:0.95rem; color:{P['dim']}; letter-spacing:0.03em;">Classification Explicable de Lames par Learning — Intelligence Artificielle</div>
    <div style="font-size:0.95rem; color:{P['dim']}; letter-spacing:0.03em; margin-top:4px;">Projet universitaire — DU Sorbonne Data Analytics 2026</div>
    <div style="font-size:1.05rem; color:{P['text']}; letter-spacing:0.08em; font-weight:600; margin-top:6px;">DATA STREAM SCIENCES</div>
</div>
""", unsafe_allow_html=True)


# =============================================
# INIT DATA
# =============================================
models = init_models()
test_ds = load_test_dataset()
class_reps = get_class_reps(test_ds)

# =============================================
# TABS
# =============================================
tab1, tab2, tab5, tab4 = st.tabs(["Classification", "Grad-CAM", "Embeddings", "Resultats"])


# ═══════════════════════════════════════════
# TAB 1 — CLASSIFICATION
# ═══════════════════════════════════════════
with tab1:

    # Choose image
    st.markdown(f'<div style="text-align:center; font-size:0.72rem; color:{P["dim"]}; margin:4px 0 10px 0;">Clique sur une image pour l\'analyser</div>', unsafe_allow_html=True)

    # Microbe icons mapping (9 classes)
    MICROBE_FILES = [
        'assets/adipose.png',
        'assets/background.png',
        'assets/debris.png',
        'assets/lymphocytes.png',
        'assets/mucus.png',
        'assets/smooth_muscle.png',
        'assets/normal_mucosa.png',
        'assets/stroma.png',
        'assets/cancer.png',
    ]

    # Load microbe images and compose with label text below
    from PIL import ImageDraw, ImageFont

    FULL_NAMES = ['adipose', 'background', 'debris', 'lymphocytes', 'mucus',
                  'smooth muscle', 'normal colon\nmucosa', 'cancer-associated\nstroma', 'colorectal\nadenocarcinoma\nepithelium']

    microbe_b64 = []
    base_dir = Path(__file__).parent
    for i, fname in enumerate(MICROBE_FILES):
        fpath = base_dir / fname
        if fpath.exists():
            mic_img = Image.open(fpath).convert('RGBA')
            # Create composite: microbe + label below
            w, h = 135, 135
            composite = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            # Center microbe
            mic_resized = mic_img.resize((85, 85), Image.LANCZOS)
            composite.paste(mic_resized, ((w - 85) // 2, 3), mic_resized)
            # Draw label
            draw = ImageDraw.Draw(composite)
            label = FULL_NAMES[i]
            try:
                font = ImageFont.truetype("arial.ttf", 9)
            except:
                font = ImageFont.load_default()
            bbox = draw.multiline_textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.multiline_text(((w - tw) // 2, 92), label, fill=(100, 110, 140, 200), font=font, align='center')
            # To base64
            buf = io.BytesIO()
            composite.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()
            microbe_b64.append(f"data:image/png;base64,{b64}")
        else:
            img_arr = class_reps.get(i)
            if img_arr is not None:
                b64 = pil_to_b64(Image.fromarray(img_arr.astype(np.uint8)), size=80)
                microbe_b64.append(f"data:image/png;base64,{b64}")

    # Clickable microbe grid
    clicked = clickable_images(
        microbe_b64,
        titles=CLASSES,
        div_style={"display": "flex", "flex-wrap": "wrap", "justify-content": "center", "gap": "6px", "max-width": "500px", "margin": "0 auto"},
        img_style={"width": "135px", "height": "135px", "border-radius": "6px", "cursor": "pointer", "border": f"2px solid {P['border']}", "background": P['card'], "padding": "4px", "transition": "border-color 0.2s"},
    )

    # Track if new image selected
    new_image = False

    if clicked > -1:
        # Pick random image from this class
        cls_indices = get_class_indices(test_ds)
        rand_idx = cls_indices[clicked][np.random.randint(0, len(cls_indices[clicked]))]
        img, lbl = test_ds[rand_idx]
        st.session_state['selected_image'] = np.array(img)
        st.session_state['true_label'] = clicked
        new_image = True

    # Or: random / upload
    st.markdown(f'<div style="text-align:center; font-size:0.68rem; color:{P["dim"]}; margin:10px 0 4px 0;">ou</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
    with c2:
        random_clicked = st.button("Analyse aleatoire", use_container_width=True)
    with c3:
        uploaded = st.file_uploader("Glisse ton image ici", type=['png','jpg','jpeg'])
    if uploaded:
        upload_id = uploaded.name + str(uploaded.size)
        if upload_id != st.session_state.get('_last_upload_id'):
            st.session_state['_last_upload_id'] = upload_id
            img = Image.open(uploaded).convert('RGB').resize((28, 28))
            st.session_state['selected_image'] = np.array(img)
            st.session_state['true_label'] = -1
            new_image = True
    if random_clicked:
        idx = np.random.randint(0, len(test_ds))
        img, lbl = test_ds[idx]
        st.session_state['selected_image'] = np.array(img)
        st.session_state['true_label'] = int(np.array(lbl).flatten()[0])
        new_image = True

    # Auto-analyse on new image selection
    if new_image and 'selected_image' in st.session_state:
        img_arr = st.session_state['selected_image']
        st.session_state.pop('last_explanation', None)
        with st.spinner("Cell.IA analyse..."):
            result = run_agent(img_arr, mode='V1')
        st.session_state['last_result'] = result
        st.session_state['last_image'] = img_arr

    # Show image + results
    if 'selected_image' in st.session_state:
        img_arr = st.session_state['selected_image']
        b64_sel = pil_to_b64(Image.fromarray(img_arr), size=240)
        st.markdown(f'<div style="text-align:center; margin:10px 0;"><img src="data:image/png;base64,{b64_sel}" style="width:240px; height:240px; image-rendering:pixelated; border-radius:4px; border:1.5px solid {P["border"]};"></div>', unsafe_allow_html=True)

        # Display results if available
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            pred_cls = result['pred_idx']
            pred_name = CLASSES[pred_cls]
            conf = result['confidence']
            probas = result.get('probas', [])
            color = CLS_COLOR[pred_cls]

            # Badge
            st.markdown(f"""
            <div style="text-align:center; margin:14px 0 8px 0;">
                <span style="display:inline-block; padding:6px 18px; border-radius:4px;
                    background:{color}18; border:1.5px solid {color};
                    font-size:0.95rem; font-weight:700; color:{color};
                    letter-spacing:0.02em;">{pred_name.upper()}</span>
            </div>
            """, unsafe_allow_html=True)

            # Cancer alert signal
            CANCER_CLASSES = ['cancer epithelium', 'colorectal adenocarcinoma epithelium']
            RISK_CLASSES = ['stroma', 'cancer-associated stroma', 'debris']
            if any(c in pred_name.lower() for c in ['cancer', 'adenocarcinoma']):
                st.markdown(f"""
                <div style="text-align:center; padding:10px 16px; margin:8px auto; max-width:500px;
                    background:#c0405018; border:2px solid #c04050; border-radius:6px;">
                    <div style="font-size:1rem; font-weight:700; color:#c04050;">SIGNAL POSITIF — TISSU CANCEREUX DETECTE</div>
                    <div style="font-size:0.72rem; color:#e8a0a0; margin-top:4px;">Verification par un pathologiste recommandee. Recall {pred_name} = 0.9570 (CNN v1).</div>
                </div>
                """, unsafe_allow_html=True)
            elif any(c in pred_name.lower() for c in ['stroma', 'debris']):
                st.markdown(f"""
                <div style="text-align:center; padding:8px 16px; margin:8px auto; max-width:500px;
                    background:{P['jaune']}18; border:1.5px solid {P['jaune']}; border-radius:6px;">
                    <div style="font-size:0.85rem; font-weight:600; color:{P['jaune']};">SIGNAL INTERMEDIAIRE — Classe a faible recall</div>
                    <div style="font-size:0.68rem; color:{P['dim']}; margin-top:3px;">Le modele manque plus de 40% des {pred_name}. Supervision humaine necessaire.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align:center; padding:8px 16px; margin:8px auto; max-width:500px;
                    background:{P['vert']}18; border:1.5px solid {P['vert']}; border-radius:6px;">
                    <div style="font-size:0.85rem; font-weight:600; color:{P['vert']};">SIGNAL NEGATIF — Tissu non cancereux</div>
                </div>
                """, unsafe_allow_html=True)

            # True label comparison
            true_lbl = st.session_state.get('true_label', -1)
            if true_lbl >= 0:
                match = "Correct" if true_lbl == pred_cls else f"Erreur (vrai : {CLASSES[true_lbl]})"
                match_color = P['vert'] if true_lbl == pred_cls else '#e8a0a0'
                st.markdown(f'<div style="text-align:center; font-size:0.72rem; color:{match_color}; margin-bottom:6px;">{match}</div>', unsafe_allow_html=True)

            # Confidence
            bar_color = P['vert'] if conf > 0.8 else (P['jaune'] if conf > 0.5 else P['rose'])
            st.markdown(f"""
            <div style="max-width:320px; margin:0 auto 10px auto;">
                <div style="display:flex; justify-content:space-between; font-size:0.68rem; color:{P['dim']};">
                    <span>Confiance</span><span>{conf:.1%}</span>
                </div>
                <div style="background:{P['card']}; border-radius:3px; height:6px; border:1px solid {P['border']};">
                    <div style="width:{conf*100:.0f}%; height:100%; border-radius:3px; background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Monte Carlo Dropout — uncertainty as text badge only (no misleading bar)
            uncertainty = result.get('uncertainty', None)
            if uncertainty is not None:
                unc_pct = uncertainty * 100
                if unc_pct < 3:
                    unc_color = P['vert']
                    unc_label = "Prediction stable"
                elif unc_pct < 8:
                    unc_color = P['jaune']
                    unc_label = "Legere hesitation"
                else:
                    unc_color = P['rose']
                    unc_label = "Forte incertitude — verification recommandee"
                st.markdown(f"""
                <div style="text-align:center; margin:6px auto; max-width:400px;
                    padding:6px 14px; border-radius:4px; border:1.5px solid {unc_color};
                    background:{unc_color}12;">
                    <span style="font-size:0.72rem; color:{unc_color}; font-weight:600;">MC Dropout (20 passes) : {unc_label}</span>
                    <span style="font-size:0.68rem; color:{P['dim']}; margin-left:8px;">ecart-type = {unc_pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            # Metrics for predicted class
            metrics = get_metrics()
            mk = 'CNN v1'
            if mk in metrics and pred_name in metrics[mk]:
                m = metrics[mk][pred_name]
                rec, pre, f1 = m['recall'], m['precision'], m['f1-score']

                if rec < 0.7:
                    st.markdown(f"""
                    <div style="text-align:center; padding:6px 12px; margin:6px auto; max-width:420px;
                        background:#e8a0a012; border:1px solid #e8a0a0; border-radius:4px;
                        font-size:0.72rem; color:#e8a0a0;">
                        Recall {pred_name} = {rec:.4f} — le modele manque {(1-rec)*100:.0f}% de cette classe. Verification humaine recommandee.
                    </div>
                    """, unsafe_allow_html=True)

                mc = st.columns(3)
                for j, (label, val) in enumerate([("Precision", pre), ("Recall", rec), ("F1-score", f1)]):
                    with mc[j]:
                        st.markdown(f"""
                        <div style="background:{P['card']}; border:1px solid {P['border']}; border-radius:4px;
                            padding:10px 8px; text-align:center;">
                            <div style="font-size:0.62rem; color:{P['dim']}; text-transform:uppercase; letter-spacing:0.05em;">{label}</div>
                            <div style="font-size:1.2rem; font-weight:700; color:{P['text']}; margin-top:2px;">{val:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Macro / Weighted averages
            if mk in metrics:
                ma = metrics[mk].get('macro avg', {})
                wa = metrics[mk].get('weighted avg', {})
                acc = metrics[mk].get('accuracy', 0)
                st.markdown(f'<div style="margin:10px 0 4px 0; font-size:0.65rem; color:{P["dim"]}; text-transform:uppercase; letter-spacing:0.05em; text-align:center;">Moyennes globales ({mk})</div>', unsafe_allow_html=True)
                mc2 = st.columns(4)
                for j, (label, val) in enumerate([
                    ("Accuracy", acc),
                    ("Macro F1", ma.get('f1-score', 0)),
                    ("Weighted F1", wa.get('f1-score', 0)),
                    ("Macro Recall", ma.get('recall', 0)),
                ]):
                    with mc2[j]:
                        st.markdown(f"""
                        <div style="background:{P['card']}; border:1px solid {P['border']}; border-radius:4px;
                            padding:8px 6px; text-align:center;">
                            <div style="font-size:0.58rem; color:{P['dim']}; text-transform:uppercase; letter-spacing:0.04em;">{label}</div>
                            <div style="font-size:1rem; font-weight:700; color:{P['text']}; margin-top:2px;">{val:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Probabilities
            if len(probas) == N_CLASSES:
                st.markdown(f'<div style="height:10px;"></div>', unsafe_allow_html=True)
                prob_df = pd.DataFrame({'Classe': CLASSES, 'Proba': probas}).sort_values('Proba', ascending=True)
                st.bar_chart(prob_df.set_index('Classe'), height=220, use_container_width=True)

            # Agent IA — explain with Claude (adult + kids)
            st.markdown(f'<div style="height:8px;"></div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
            with c2:
                if st.button("Cell.IA explique", use_container_width=True):
                    with st.spinner("Cell.IA analyse..."):
                        try:
                            explanation = explain_with_claude(result=result, mode='V1', image=img_arr)
                            st.session_state['last_explanation'] = explanation
                        except Exception as e:
                            st.session_state['last_explanation'] = f"Erreur : {e}"
            with c3:
                if st.button("Cell.IA version junior", use_container_width=True):
                    with st.spinner("Cell.IA explique simplement..."):
                        try:
                            kid_prompt = (
                                f"Tu es Cell.IA, un assistant qui explique les resultats d'un microscope intelligent a un ado de 10-14 ans. "
                                f"Le modele a analyse une image de tissu colorectal et a identifie : '{CLASSES[pred_cls]}' avec {conf:.0%} de confiance. "
                                f"Explique en 3-4 phrases claires ce que c'est, ce que l'IA a regarde dans l'image, et pourquoi c'est utile pour les medecins. "
                                f"Utilise un vocabulaire accessible mais pas bebe. Tu peux utiliser des analogies simples. "
                                f"Ne minimise pas le sujet, sois honnete mais pas alarmiste."
                            )
                            explanation = explain_with_claude(result=result, mode='V1', image=img_arr, prompt_override=kid_prompt)
                            st.session_state['last_explanation'] = explanation
                        except Exception as e:
                            st.session_state['last_explanation'] = f"Erreur : {e}"

            if 'last_explanation' in st.session_state:
                st.markdown(f"""<div style="background:{P['card']}; border:1px solid {P['border']}; border-radius:6px;
                    padding:16px; margin:10px 0; font-size:0.78rem; color:{P['text']}; line-height:1.6;">
                    <div style="font-size:0.82rem; font-weight:700; color:{P['rose']}; margin-bottom:8px;">Cell.IA :</div>""", unsafe_allow_html=True)
                st.markdown(st.session_state['last_explanation'])
                st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TAB 2 — GRAD-CAM
# ═══════════════════════════════════════════
with tab2:
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:12px;">
        <div style="font-size:0.85rem; font-weight:600; color:{P['rose']};">De la Black-Box a la Glass-Box</div>
        <div style="font-size:0.7rem; color:{P['dim']}; margin-top:2px;">Le Grad-CAM montre ou le modele regarde. Exigence ethique pour le deploiement clinique.</div>
    </div>
    """, unsafe_allow_html=True)

    if 'last_image' in st.session_state:
        img_arr = st.session_state['last_image']
        result = st.session_state.get('last_result', {})
        pred_cls = result.get('pred_idx', 0)
        pred_name = CLASSES[pred_cls] if result else '?'
        color = CLS_COLOR[pred_cls]

        # Prediction reminder
        if result:
            conf = result.get('confidence', 0)
            st.markdown(f'<div style="text-align:center; margin:8px 0;"><span style="padding:4px 14px; border-radius:4px; background:{color}18; border:1.5px solid {color}; font-size:0.85rem; font-weight:700; color:{color};">{pred_name.upper()} — {conf:.0%}</span></div>', unsafe_allow_html=True)

        # ── Section 1 : Grad-CAM comparatif CNN vs ResNet ──
        st.markdown(f'<div style="text-align:center; font-size:0.72rem; color:{P["rose"]}; font-weight:600; margin:12px 0 6px 0;">Comparaison CNN v1 vs ResNet FT</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center; font-size:0.65rem; color:{P["dim"]}; margin-bottom:8px;">Zones chaudes = ou le modele regarde pour decider</div>', unsafe_allow_html=True)

        gc_cnn_result = None
        gc_resnet_result = None

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(Image.fromarray(img_arr), width=200, caption="Image originale")
        with c2:
            try:
                gc_cnn_result = grad_cam(img_arr, model='cnn')
                st.image(gc_cnn_result['overlay_28'], width=200, caption="CNN v1")
            except Exception:
                st.caption("CNN non disponible")
        with c3:
            try:
                gc_resnet_result = grad_cam(img_arr, model='resnet')
                st.image(gc_resnet_result['overlay_28'], width=200, caption="ResNet FT")
            except Exception:
                st.caption("ResNet non disponible")

        # Concordance score CNN vs ResNet
        if gc_cnn_result and gc_resnet_result:
            try:
                h1 = np.array(gc_cnn_result['heatmap_resized']).flatten().astype(float)
                h2 = np.array(gc_resnet_result['heatmap_resized']).flatten().astype(float)
                norm1 = np.linalg.norm(h1)
                norm2 = np.linalg.norm(h2)
                if norm1 > 0 and norm2 > 0:
                    concordance = float(np.dot(h1, h2) / (norm1 * norm2))
                else:
                    concordance = 0.0
                conc_color = P['vert'] if concordance > 0.7 else (P['jaune'] if concordance > 0.4 else P['rose'])
                conc_label = "forte concordance" if concordance > 0.7 else ("concordance moderee" if concordance > 0.4 else "faible concordance — verification recommandee")
                st.markdown(f"""
                <div style="max-width:400px; margin:8px auto; text-align:center;">
                    <div style="font-size:0.65rem; color:{P['dim']};">Concordance des zones d'attention</div>
                    <div style="background:{P['card']}; border-radius:3px; height:8px; border:1px solid {P['border']}; margin:4px 0;">
                        <div style="width:{concordance*100:.0f}%; height:100%; border-radius:3px; background:linear-gradient(90deg, {P['rose']}, {P['jaune']}, {P['vert']});"></div>
                    </div>
                    <div style="font-size:0.72rem; color:{conc_color}; font-weight:600;">{concordance:.0%} — {conc_label}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass

        st.markdown("---")

        # ── Section 2 : Grad-CAM sur recommandation ──
        st.markdown(f'<div style="text-align:center; font-size:0.72rem; color:{P["rose"]}; font-weight:600; margin:8px 0 6px 0;">Comparaison avec l\'image la plus similaire</div>', unsafe_allow_html=True)

        try:
            recs = get_recommendations(img_arr, n=1)
            if recs:
                rec = recs[0]
                rec_img = rec['image']
                rec_name = CLASSES[rec['label']]
                rec_sim = rec['similarity']

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.image(Image.fromarray(img_arr), width=150, caption=f"Analysee ({pred_name})")
                with c2:
                    if gc_cnn_result:
                        st.image(gc_cnn_result['overlay_28'], width=150, caption="Grad-CAM analysee")
                    else:
                        st.caption("—")
                with c3:
                    st.image(Image.fromarray(rec_img), width=150, caption=f"Similaire ({rec_name})")
                with c4:
                    try:
                        gc_rec = grad_cam(rec_img, model='cnn')
                        st.image(gc_rec['overlay_28'], width=150, caption="Grad-CAM similaire")
                    except Exception:
                        st.caption("—")

                st.markdown(f'<div style="text-align:center; font-size:0.68rem; color:{P["dim"]}; margin:4px 0;">Similarite cosine : {rec_sim:.1f}% — si les heatmaps se ressemblent, la prediction est coherente</div>', unsafe_allow_html=True)
        except Exception:
            st.caption("Recommandations non disponibles")

        st.markdown("---")

        # ── Section 3 : Cell.IA explique le Grad-CAM ──
        _, col_e, _ = st.columns([1, 2, 1])
        with col_e:
            if st.button("Cell.IA explique le Grad-CAM", use_container_width=True):
                with st.spinner("Cell.IA analyse les zones d'attention..."):
                    try:
                        prompt = f"Analyse cette image d'histologie colorectale et sa heatmap Grad-CAM. Le modele CNN a classifie ce tissu comme '{pred_name}' avec {conf:.0%} de confiance. En 3-4 phrases : 1) Decris ce que tu vois sur l'image originale 2) Explique quelles zones la heatmap met en evidence et pourquoi 3) Ce que cela signifie cliniquement pour un pathologiste. Sois precis sur l'histologie H&E."
                        gc_img = gc_cnn_result['overlay_28'] if gc_cnn_result else None
                        explanation = explain_with_claude(result=result, mode='V1', prompt_override=prompt, image=img_arr, gradcam_image=gc_img)
                        st.session_state['gc_explanation'] = explanation
                    except Exception as e:
                        st.session_state['gc_explanation'] = f"Erreur : {e}"

        if 'gc_explanation' in st.session_state:
            st.markdown(f"""<div style="background:{P['card']}; border:1px solid {P['border']}; border-radius:6px;
                padding:16px; margin:10px 0; font-size:0.78rem; color:{P['text']}; line-height:1.6;">
                <div style="font-size:0.82rem; font-weight:700; color:{P['rose']}; margin-bottom:8px;">Cell.IA :</div>""", unsafe_allow_html=True)
            st.markdown(st.session_state['gc_explanation'])
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Glass-box footer ──
        st.markdown(f"""
        <div style="max-width:500px; margin:14px auto; text-align:center; font-size:0.68rem; color:{P['dim']}; line-height:1.6;">
            <span style="color:{P['rose']};">Trust</span> — le medecin comprend &nbsp;|&nbsp;
            <span style="color:{P['rose']};">Safety</span> — bon recall + bon endroit &nbsp;|&nbsp;
            <span style="color:{P['rose']};">Liability</span> — raisonnement explicable
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Analyse d'abord une image dans l'onglet Classification.")



# ═══════════════════════════════════════════
# TAB 4 — RESULTATS DU PROJET
# ═══════════════════════════════════════════
with tab4:
    st.markdown(f'<div style="text-align:center; font-size:0.85rem; font-weight:600; color:{P["rose"]}; margin-bottom:10px;">Comparaison des modeles</div>', unsafe_allow_html=True)

    accs = [68.02, 88.86, 91.78, 87.14, 91.77, 81.98, 83.40]
    best_acc = max(accs)
    df_summary = pd.DataFrame({
        'Modele': ['MLP', 'CNN sans aug', 'CNN v1', 'ResNet frozen', 'ResNet FT', 'ViT', 'ViT sans pos'],
        'Test acc': ['68.02%', '88.86%', '91.78%', '87.14%', '91.77%', '81.98%', '83.40%'],
        'Precision cancer': ['0.5995', 'N/A', '0.9262', 'N/A', '0.9605', '0.8122', 'N/A'],
        'Recall cancer': ['0.7745', 'N/A', '0.9570', 'N/A', '0.9659', '0.8873', 'N/A'],
        'F1 cancer': ['0.6759', 'N/A', '0.9414', 'N/A', '0.9632', '0.8481', 'N/A'],
        'Params': ['1.37M', '436K', '436K', '4.6K', '11.1M', '816K', '816K'],
        'vs best': [f"{a - best_acc:+.2f}" for a in accs],
    })
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    metrics = get_metrics()

    def build_report_table(report, other_report):
        rows = []
        for cls in CLASSES:
            if cls in report and cls in other_report:
                m = report[cls]
                o = other_report[cls]
                delta_f1 = m['f1-score'] - o['f1-score']
                sign = '+' if delta_f1 >= 0 else ''
                rows.append({'Classe': cls, 'Precision': f"{m['precision']:.4f}", 'Recall': f"{m['recall']:.4f}", 'F1-score': f"{m['f1-score']:.4f}", 'Support': int(m['support']), 'vs': f"{sign}{delta_f1:.4f}"})
        return pd.DataFrame(rows)

    if metrics and 'CNN v1' in metrics and 'ResNet FT' in metrics:
        acc_cnn = metrics['CNN v1'].get('accuracy', 0)
        st.markdown(f'<div style="text-align:center; font-size:0.78rem; font-weight:600; color:{P["bleu"]}; margin:10px 0 2px 0;">CNN v1 (436K params)</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center; font-size:0.68rem; color:{P["dim"]}; margin-bottom:6px;">Accuracy : {acc_cnn:.4f} (91.78%)</div>', unsafe_allow_html=True)
        st.dataframe(build_report_table(metrics['CNN v1'], metrics['ResNet FT']), use_container_width=True, hide_index=True)
        st.markdown(f'<div style="font-size:0.6rem; color:{P["dim"]}; text-align:center; margin-bottom:16px;">vs = ecart F1 vs ResNet FT</div>', unsafe_allow_html=True)

        acc_resnet = metrics['ResNet FT'].get('accuracy', 0)
        st.markdown(f'<div style="text-align:center; font-size:0.78rem; font-weight:600; color:{P["rose"]}; margin:10px 0 2px 0;">ResNet FT (11.1M params)</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center; font-size:0.68rem; color:{P["dim"]}; margin-bottom:6px;">Accuracy : {acc_resnet:.4f} (91.77%)</div>', unsafe_allow_html=True)
        st.dataframe(build_report_table(metrics['ResNet FT'], metrics['CNN v1']), use_container_width=True, hide_index=True)
        st.markdown(f'<div style="font-size:0.6rem; color:{P["dim"]}; text-align:center;">vs = ecart F1 vs CNN v1</div>', unsafe_allow_html=True)

        # Macro / Weighted / Accuracy comparison table
        st.markdown(f'<div style="text-align:center; font-size:0.78rem; font-weight:600; color:{P["rose"]}; margin:18px 0 6px 0;">Moyennes globales — CNN v1 vs ResNet FT</div>', unsafe_allow_html=True)
        avg_rows = []
        for avg_name in ['macro avg', 'weighted avg']:
            a1 = metrics['CNN v1'].get(avg_name, {})
            a2 = metrics['ResNet FT'].get(avg_name, {})
            if a1 and a2:
                avg_rows.append({
                    'Metrique': avg_name.upper(),
                    'Prec CNN': f"{a1['precision']:.4f}",
                    'Prec ResNet': f"{a2['precision']:.4f}",
                    'Rec CNN': f"{a1['recall']:.4f}",
                    'Rec ResNet': f"{a2['recall']:.4f}",
                    'F1 CNN': f"{a1['f1-score']:.4f}",
                    'F1 ResNet': f"{a2['f1-score']:.4f}",
                })
        acc1 = metrics['CNN v1'].get('accuracy', 0)
        acc2 = metrics['ResNet FT'].get('accuracy', 0)
        avg_rows.append({
            'Metrique': 'ACCURACY',
            'Prec CNN': '', 'Prec ResNet': '',
            'Rec CNN': '', 'Rec ResNet': '',
            'F1 CNN': f"{acc1:.4f}", 'F1 ResNet': f"{acc2:.4f}",
        })
        st.dataframe(pd.DataFrame(avg_rows), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div style="max-width:540px; margin:14px auto; font-size:0.74rem; color:{P['text']}; line-height:1.7;">
        <span style="color:{P['vert']};">CNN v1 = ResNet FT</span> en accuracy (91.78% vs 91.77%) avec 25x moins de parametres<br>
        <span style="color:#e8a0a0;">Stroma</span> : recall ~0.53, limite physique de la resolution 28x28<br>
        <span style="color:{P['vert']};">Recall cancer > 95%</span> pour CNN v1 et ResNet FT — seuil clinique minimum<br>
        Macro F1 (0.89) vs Weighted F1 (0.92) : l'ecart revele le desequilibre des classes
    </div>
    """, unsafe_allow_html=True)



# ═══════════════════════════════════════════
# TAB 5 — CLUSTERS
# ═══════════════════════════════════════════
with tab5:
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:12px;">
        <div style="font-size:0.85rem; font-weight:600; color:{P['rose']};">Carte t-SNE des embeddings CNN v1</div>
        <div style="font-size:0.7rem; color:{P['dim']}; margin-top:2px;">7180 images du test set projetees de 128D vers 2D — chaque point est une image</div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def compute_tsne():
        from sklearn.manifold import TSNE
        emb_path = Path(DATA_DIR) / 'embeddings' / 'cnn_test_embeddings.npy'
        if not emb_path.exists():
            return None, None, None
        embeddings = np.load(emb_path)
        labels = []
        for idx in range(len(test_ds)):
            _, lbl = test_ds[idx]
            labels.append(int(np.array(lbl).flatten()[0]))
        labels = np.array(labels)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        coords = tsne.fit_transform(embeddings)
        return coords, labels, embeddings

    coords, labels_tsne, embeddings = compute_tsne()

    if coords is not None:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        colors_map = ['#c0a850', '#5a9ec0', '#8a70b8', '#5aaa78', '#c89060', '#d4688a', '#4a90b8', '#9070c8', '#c04050']

        # Main t-SNE plot
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8f9fc')
        for c in range(N_CLASSES):
            mask = labels_tsne == c
            ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.5,
                      c=colors_map[c], label=CLASSES[c])
        ax.legend(fontsize=8, loc='best', framealpha=0.8, markerscale=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('t-SNE CNN v1 embeddings (128D → 2D)', fontsize=11, color='#1a1a2e')
        for spine in ax.spines.values():
            spine.set_color('#e0e4ec')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""
        <div style="max-width:600px; margin:10px auto; font-size:0.74rem; color:{P['text']}; line-height:1.6; background:{P['card']}; border:1px solid {P['border']}; border-radius:6px; padding:14px;">
            <b>Lecture de la carte :</b><br>
            Les clusters bien separes (adipose, background, lymphocytes) correspondent aux classes a fort recall.<br>
            Les zones de chevauchement (stroma/smooth muscle) expliquent les confusions et le faible F1.<br>
            Le cancer epithelium forme un cluster distinct mais avec une frontiere floue vers la mucosa normale.
        </div>
        """, unsafe_allow_html=True)

        # UMAP
        st.markdown("---")
        st.markdown(f'<div style="text-align:center; font-size:0.78rem; font-weight:600; color:{P["rose"]}; margin:10px 0 6px 0;">Carte UMAP</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center; font-size:0.65rem; color:{P["dim"]}; margin-bottom:8px;">UMAP preserve mieux la structure globale que t-SNE</div>', unsafe_allow_html=True)

        @st.cache_data
        def compute_umap():
            try:
                from umap import UMAP
                emb_path = Path(DATA_DIR) / 'embeddings' / 'cnn_test_embeddings.npy'
                if not emb_path.exists():
                    return None
                embeddings_u = np.load(emb_path)
                reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
                return reducer.fit_transform(embeddings_u)
            except ImportError:
                return None

        umap_coords = compute_umap()
        if umap_coords is not None:
            fig_u, ax_u = plt.subplots(figsize=(10, 8))
            fig_u.patch.set_facecolor('#ffffff')
            ax_u.set_facecolor('#f8f9fc')
            for c in range(N_CLASSES):
                mask = labels_tsne == c
                ax_u.scatter(umap_coords[mask, 0], umap_coords[mask, 1], s=6, alpha=0.5,
                          c=colors_map[c], label=CLASSES[c])
            ax_u.legend(fontsize=8, loc='best', framealpha=0.8, markerscale=3)
            ax_u.set_xticks([]); ax_u.set_yticks([])
            ax_u.set_title('UMAP CNN v1 embeddings (128D → 2D)', fontsize=11, color='#1a1a2e')
            for spine in ax_u.spines.values():
                spine.set_color('#e0e4ec')
            plt.tight_layout()
            st.pyplot(fig_u)
            plt.close()
        else:
            st.caption("UMAP non disponible (pip install umap-learn)")

        st.markdown("---")

        # Highlight last analyzed image on t-SNE
        if 'last_image' in st.session_state and 'last_result' in st.session_state:
            result = st.session_state['last_result']
            pred_cls = result['pred_idx']
            pred_name = CLASSES[pred_cls]

            st.markdown(f'<div style="text-align:center; font-size:0.78rem; font-weight:600; color:{P["rose"]}; margin:10px 0 6px 0;">Position de l\'image analysee dans l\'espace des embeddings</div>', unsafe_allow_html=True)

            # Get embedding of analyzed image
            from agent_pathmnist import load_models
            import torch
            models_loaded = load_models()
            cnn = models_loaded.get('cnn')
            if cnn:
                img_arr = st.session_state['last_image']
                img_t = torch.tensor(img_arr.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
                norm_mean = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
                norm_std = torch.tensor(NORM_STD).view(1, 3, 1, 1)
                img_t = (img_t - norm_mean) / norm_std
                device = next(cnn.parameters()).device
                cnn.eval()
                with torch.no_grad():
                    # Get embedding from penultimate layer
                    x = img_t.to(device)
                    for layer in list(cnn.children())[:-1]:
                        x = layer(x)
                    if x.dim() > 2:
                        x = x.view(x.size(0), -1)
                    emb_query = x.cpu().numpy().flatten()

                # Project onto existing t-SNE using nearest neighbor position
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=5)
                nn.fit(embeddings)
                dists, idxs = nn.kneighbors(emb_query.reshape(1, -1))
                # Average position of 5 nearest neighbors
                pos_x = coords[idxs[0], 0].mean()
                pos_y = coords[idxs[0], 1].mean()

                fig2, ax2 = plt.subplots(figsize=(10, 8))
                fig2.patch.set_facecolor('#ffffff')
                ax2.set_facecolor('#f8f9fc')
                for c in range(N_CLASSES):
                    mask = labels_tsne == c
                    ax2.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.3,
                              c=colors_map[c], label=CLASSES[c])
                # Highlight analyzed image
                ax2.scatter([pos_x], [pos_y], s=200, c='red', marker='*', zorder=10, edgecolors='black', linewidths=1)
                ax2.annotate(f'{pred_name}', (pos_x, pos_y), textcoords="offset points",
                           xytext=(12, 12), fontsize=9, color='red', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red'))
                ax2.legend(fontsize=7, loc='best', framealpha=0.8, markerscale=3)
                ax2.set_xticks([]); ax2.set_yticks([])
                ax2.set_title('Position de l\'image analysee (etoile rouge)', fontsize=10, color='#1a1a2e')
                for spine in ax2.spines.values():
                    spine.set_color('#e0e4ec')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

                # Show 5 nearest neighbors
                st.markdown(f'<div style="text-align:center; font-size:0.72rem; color:{P["dim"]}; margin:8px 0;">5 plus proches voisins dans l\'espace des embeddings</div>', unsafe_allow_html=True)
                nn_html = '<div style="display:flex; justify-content:center; gap:10px;">'
                for ni in idxs[0]:
                    nn_img, nn_lbl = test_ds[ni]
                    nn_arr = np.array(nn_img)
                    nn_b64 = pil_to_b64(Image.fromarray(nn_arr), size=100)
                    nn_cls = int(np.array(nn_lbl).flatten()[0])
                    nn_html += f'<div style="text-align:center;"><img src="data:image/png;base64,{nn_b64}" style="width:100px; height:100px; image-rendering:pixelated; border-radius:4px; border:1.5px solid {P["border"]};"><div style="font-size:0.62rem; color:{colors_map[nn_cls]}; font-weight:600; margin-top:3px;">{CLASSES[nn_cls]}</div></div>'
                nn_html += '</div>'
                st.markdown(nn_html, unsafe_allow_html=True)

        st.markdown("---")

        # Recommendations (cosine similarity)
        if 'last_image' in st.session_state:
            st.markdown(f'<div style="text-align:center; font-size:0.78rem; font-weight:600; color:{P["rose"]}; margin:10px 0 6px 0;">Images les plus similaires (cosine similarity)</div>', unsafe_allow_html=True)
            try:
                recs = get_recommendations(st.session_state['last_image'], n=5)
                if recs:
                    recs_html = '<div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">'
                    for rec in recs:
                        rec_b64 = pil_to_b64(Image.fromarray(rec['image']), size=140)
                        c = CLS_COLOR[rec['label']]
                        cls_name = CLASSES[rec['label']]
                        sim = rec['similarity']
                        recs_html += f'''
                        <div style="width:150px; text-align:center;">
                            <img src="data:image/png;base64,{rec_b64}" style="width:140px; height:140px; image-rendering:pixelated; border-radius:4px; border:1.5px solid {P['border']}; display:block; margin:0 auto;">
                            <div style="font-size:0.7rem; color:{c}; font-weight:600; margin-top:4px;">{cls_name}</div>
                            <div style="font-size:0.62rem; color:{P['dim']};">sim: {sim:.1f}%</div>
                        </div>'''
                    recs_html += '</div>'
                    st.markdown(recs_html, unsafe_allow_html=True)
                else:
                    st.caption("Embeddings non disponibles.")
            except Exception as e:
                st.caption(f"Recommandations non disponibles : {e}")
    else:
        st.caption("Embeddings non disponibles. Lancez NB8 pour les generer.")


# =============================================
# FOOTER
# =============================================
st.markdown(f"""
<div style="text-align:center; margin-top:2rem; padding:1.5rem 0; border-top:1px solid {P['border']};">
    <div style="font-size:1.8rem; font-weight:700; color:{P['text']}; letter-spacing:-0.02em;">Cell.IA</div>
    <div style="font-size:0.82rem; color:{P['dim']}; margin-top:4px;">Classification Explicable de Lames par Learning — Intelligence Artificielle</div>
    <div style="font-size:0.72rem; color:{P['dim']}; margin-top:4px;">Projet universitaire — DU Sorbonne Data Analytics 2026</div>
    <div style="font-size:0.68rem; color:{P['dim']}; margin-top:2px;">DU Sorbonne Data Analytics 2026</div>
</div>
""", unsafe_allow_html=True)
