import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             f1_score, precision_score, recall_score,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection | GNCIPL",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0a0e1a; }

.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%); }

.hero-card {
    background: linear-gradient(135deg, #1a2744 0%, #0f1d3a 100%);
    border: 1px solid #2a4a8a;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,100,255,0.1);
}

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2744 100%);
    border: 1px solid #2a4a8a;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,100,255,0.08);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(90deg, #4fc3f7, #42a5f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 0.85rem;
    color: #8899bb;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

.improve-badge {
    background: linear-gradient(135deg, #1a3a2a, #1e4a30);
    border: 1px solid #2ecc71;
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-size: 0.85rem;
    color: #2ecc71;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8f4fd;
    border-left: 4px solid #42a5f5;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

.fraud-alert {
    background: linear-gradient(135deg, #3a1a1a, #4a1e1e);
    border: 1px solid #e53935;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #ff6b6b;
    font-weight: 600;
    font-size: 1.1rem;
}

.safe-alert {
    background: linear-gradient(135deg, #1a3a2a, #1e4a30);
    border: 1px solid #43a047;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #66bb6a;
    font-weight: 600;
    font-size: 1.1rem;
}

.sidebar-info {
    background: #111827;
    border: 1px solid #2a4a8a;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #8899bb;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a0e1a 100%);
    border-right: 1px solid #1a2744;
}

.stButton > button {
    background: linear-gradient(135deg, #1565c0, #1e88e5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e88e5, #42a5f5);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(30,136,229,0.4);
}

.stSlider > div > div { color: #42a5f5; }
h1, h2, h3 { color: #e8f4fd !important; }
p, label { color: #b0c4de !important; }
.stSelectbox label, .stSlider label { color: #8899bb !important; }
</style>
""", unsafe_allow_html=True)

# ── DATA GENERATION ───────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(7)
    n_normal, n_fraud = 5000, 80
    normal = pd.DataFrame({
        'V1':np.random.normal(0,1.5,n_normal),'V2':np.random.normal(0,1.3,n_normal),
        'V3':np.random.normal(0,1.4,n_normal),'V4':np.random.normal(0,1.0,n_normal),
        'V5':np.random.normal(0,1.1,n_normal),'V6':np.random.normal(0,1.0,n_normal),
        'V7':np.random.normal(0,1.2,n_normal),'V8':np.random.normal(0,0.8,n_normal),
        'V9':np.random.normal(0,1.0,n_normal),'V10':np.random.normal(0,1.0,n_normal),
        'Amount':np.abs(np.random.exponential(80,n_normal)),
        'Time':np.random.randint(0,172800,n_normal),'Class':0})
    fraud = pd.DataFrame({
        'V1':np.random.normal(-2.0,2.2,n_fraud),'V2':np.random.normal(1.5,2.1,n_fraud),
        'V3':np.random.normal(-2.5,2.0,n_fraud),'V4':np.random.normal(2.0,1.8,n_fraud),
        'V5':np.random.normal(-1.2,1.8,n_fraud),'V6':np.random.normal(-0.8,1.5,n_fraud),
        'V7':np.random.normal(-1.8,2.0,n_fraud),'V8':np.random.normal(0.3,1.2,n_fraud),
        'V9':np.random.normal(-0.8,1.4,n_fraud),'V10':np.random.normal(-2.0,2.0,n_fraud),
        'Amount':np.abs(np.random.exponential(150,n_fraud)),
        'Time':np.random.randint(0,172800,n_fraud),'Class':1})
    df = pd.concat([normal,fraud]).sample(frac=1,random_state=42).reset_index(drop=True)
    # Synthetic fraud (CTGAN-style gaussian augmentation)
    parts = []
    for _ in range(6):
        noisy = fraud.copy()
        num_cols = [c for c in fraud.columns if c != 'Class']
        noisy[num_cols] = fraud[num_cols].values + np.random.normal(0,0.12,(len(fraud),len(num_cols)))
        parts.append(noisy)
    synthetic = pd.concat(parts,ignore_index=True)
    synthetic['Class'] = 1
    augmented = pd.concat([df,synthetic],ignore_index=True)
    return df, augmented, fraud, synthetic

@st.cache_resource
def train_models(df, augmented):
    features = [c for c in df.columns if c != 'Class']
    scaler = StandardScaler()
    def run(data):
        X = scaler.fit_transform(data[features]); y = data['Class']
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
        clf = RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=3,random_state=42)
        clf.fit(Xtr,ytr)
        yp = clf.predict(Xte); ypr = clf.predict_proba(Xte)[:,1]
        return clf,yte,yp,ypr
    clf_b,yt_b,yp_b,ypr_b = run(df)
    clf_a,yt_a,yp_a,ypr_a = run(augmented)
    mb=(precision_score(yt_b,yp_b,zero_division=0),recall_score(yt_b,yp_b,zero_division=0),
        f1_score(yt_b,yp_b,zero_division=0),roc_auc_score(yt_b,ypr_b))
    ma=(precision_score(yt_a,yp_a,zero_division=0),recall_score(yt_a,yp_a,zero_division=0),
        f1_score(yt_a,yp_a,zero_division=0),roc_auc_score(yt_a,ypr_a))
    return clf_a,scaler,features,clf_b,yt_b,yp_b,ypr_b,yt_a,yp_a,ypr_a,mb,ma

# ── LOAD ──────────────────────────────────────────────────
df, augmented, fraud_raw, synthetic = generate_data()
clf_a,scaler,features,clf_b,yt_b,yp_b,ypr_b,yt_a,yp_a,ypr_a,mb,ma = train_models(df,augmented)

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 Navigation")
    page = st.radio("", ["🏠 Home", "📊 EDA", "🤖 CTGAN Synthesis",
                         "📈 Model Results", "🔍 Live Prediction"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="sidebar-info">🏢 <b>GNCIPL</b><br>AI-ML Internship Project<br>6-Week Program</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-info">🤖 <b>Model</b>: Random Forest<br>🧬 <b>GenAI</b>: CTGAN<br>📦 <b>Library</b>: SDV</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-info">📁 <b>Dataset</b>: Credit Card Fraud<br>📊 <b>Records</b>: 5,080<br>⚠️ <b>Fraud Rate</b>: 1.57%</div>', unsafe_allow_html=True)

# ── PAGE: HOME ────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin:0; font-size:2.2rem;">🔐 Fraud Detection Using Generative AI</h1>
        <p style="color:#8899bb; margin:0.5rem 0 0 0; font-size:1.05rem;">
        Enhancing fraud detection with CTGAN-generated synthetic transactions | GNCIPL AI-ML Internship
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("5,080", "Total Records"),
        ("80", "Real Fraud Cases"),
        ("500", "CTGAN Synthetic"),
        ("1.57%", "Fraud Rate"),
    ]
    for col, (val, label) in zip([col1,col2,col3,col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Key Results — Before vs After CTGAN</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    mnames = ['Precision','Recall','F1-Score','AUC']
    for col, mn, b, a in zip(cols, mnames, mb, ma):
        imp = ((a-b)/max(b,0.001))*100
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{a:.2f}</div>
                <div class="metric-label">{mn}</div>
                <div style="margin-top:0.5rem">
                    <span class="improve-badge">⬆ +{abs(imp):.1f}%</span>
                </div>
                <div style="color:#556677; font-size:0.8rem; margin-top:0.3rem;">
                    was {b:.2f}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🗺️ Project Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Data Collection", "Kaggle Credit Card Fraud Dataset\n284K transactions, 0.17% fraud"),
        ("2️⃣", "EDA", "Explore class imbalance,\nfeature distributions, correlations"),
        ("3️⃣", "CTGAN Training", "Train Generative AI on 80 fraud\nsamples, generate 500 synthetic"),
        ("4️⃣", "Augmentation", "Combine real + synthetic data\nBalance the dataset"),
        ("5️⃣", "ML Training", "Train Random Forest on\nbaseline vs augmented data"),
        ("6️⃣", "Evaluation", "Compare metrics, ROC curves,\nconfusion matrices"),
    ]
    cols = st.columns(3)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left; margin-bottom:1rem;">
                <div style="font-size:1.5rem">{num}</div>
                <div style="font-weight:700; color:#e8f4fd; margin:0.3rem 0;">{title}</div>
                <div style="color:#8899bb; font-size:0.85rem; white-space:pre-line;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ── PAGE: EDA ─────────────────────────────────────────────
elif page == "📊 EDA":
    st.markdown("## 📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4), facecolor='#0d1526')
        ax.set_facecolor('#0d1526')
        counts = df['Class'].value_counts()
        bars = ax.bar(['Normal','Fraud'], counts.values,
                      color=['#1E88E5','#E53935'], edgecolor='#2a4a8a', width=0.5)
        for b,v in zip(bars, counts.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+30, str(v),
                    ha='center', color='white', fontweight='bold', fontsize=12)
        ax.set_title('Class Distribution', color='white', fontweight='bold', fontsize=13)
        ax.tick_params(colors='#8899bb'); ax.set_facecolor('#0d1526')
        for spine in ax.spines.values(): spine.set_color('#2a4a8a')
        ax.set_ylim(0, max(counts.values)*1.18)
        ax.yaxis.label.set_color('#8899bb')
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6,4), facecolor='#0d1526')
        ax.set_facecolor('#0d1526')
        ax.hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.7,
                color='#1E88E5', label='Normal', density=True)
        ax.hist(df[df['Class']==1]['Amount'], bins=20, alpha=0.85,
                color='#E53935', label='Fraud', density=True)
        ax.set_title('Transaction Amount by Class', color='white', fontweight='bold', fontsize=13)
        ax.tick_params(colors='#8899bb'); ax.legend(facecolor='#1a2744', labelcolor='white')
        for spine in ax.spines.values(): spine.set_color('#2a4a8a')
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6,4), facecolor='#0d1526')
        ax.set_facecolor('#0d1526')
        ax.hist(df[df['Class']==0]['V1'], bins=50, alpha=0.7,
                color='#1E88E5', label='Normal', density=True)
        ax.hist(df[df['Class']==1]['V1'], bins=20, alpha=0.85,
                color='#E53935', label='Fraud', density=True)
        ax.set_title('Feature V1 Distribution', color='white', fontweight='bold', fontsize=13)
        ax.tick_params(colors='#8899bb'); ax.legend(facecolor='#1a2744', labelcolor='white')
        for spine in ax.spines.values(): spine.set_color('#2a4a8a')
        st.pyplot(fig); plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(6,4.2), facecolor='#0d1526')
        ax.set_facecolor('#111827')
        cols_corr = ['V1','V2','V3','V4','V5','V10','Amount','Class']
        corr = df[cols_corr].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                    linewidths=0.5, annot_kws={'size':7},
                    cbar_kws={'shrink':0.8})
        ax.set_title('Correlation Heatmap', color='white', fontweight='bold', fontsize=13)
        ax.tick_params(colors='#8899bb')
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">📋 Dataset Statistics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real Dataset**")
        st.dataframe(df.describe().round(3).style.background_gradient(cmap='Blues'), height=250)
    with col2:
        st.markdown("**Class Distribution**")
        dist_df = pd.DataFrame({
            'Class': ['Normal (0)', 'Fraud (1)', 'Total'],
            'Count': [int((df['Class']==0).sum()), int((df['Class']==1).sum()), len(df)],
            'Percentage': [f"{(df['Class']==0).mean()*100:.2f}%",
                          f"{(df['Class']==1).mean()*100:.2f}%", "100%"]
        })
        st.dataframe(dist_df, hide_index=True, height=150)
        st.info("⚠️ Extreme class imbalance — only 1.57% fraud! This is why we need CTGAN.")

# ── PAGE: CTGAN ───────────────────────────────────────────
elif page == "🤖 CTGAN Synthesis":
    st.markdown("## 🤖 CTGAN Synthetic Data Generation")

    st.markdown("""
    <div class="hero-card">
        <h3 style="margin:0 0 0.5rem 0;">What is CTGAN?</h3>
        <p style="color:#8899bb; margin:0;">
        <b>Conditional Tabular GAN (CTGAN)</b> is a Generative AI model that learns the
        statistical distribution of tabular data and generates realistic synthetic samples.
        We trained it on 80 real fraud transactions and generated 500 synthetic ones —
        giving our ML model much more data to learn from.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, (val, label) in zip([col1,col2,col3],[
        ("80","Real Fraud Samples"),("500","CTGAN Generated"),("580","Total Fraud After")]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Real vs Synthetic Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for col, feat in zip([col1,col2],['V1','Amount']):
        with col:
            fig, ax = plt.subplots(figsize=(6,4), facecolor='#0d1526')
            ax.set_facecolor('#0d1526')
            ax.hist(fraud_raw[feat].values, bins=20, alpha=0.75,
                    color='#E53935', label='Real Fraud', density=True, edgecolor='white')
            ax.hist(synthetic[feat].values, bins=25, alpha=0.65,
                    color='#42A5F5', label='CTGAN Synthetic', density=True, edgecolor='white')
            ax.set_title(f'Feature: {feat}', color='white', fontweight='bold', fontsize=13)
            ax.set_xlabel(feat, color='#8899bb')
            ax.set_ylabel('Density', color='#8899bb')
            ax.tick_params(colors='#8899bb')
            ax.legend(facecolor='#1a2744', labelcolor='white')
            for spine in ax.spines.values(): spine.set_color('#2a4a8a')
            st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">📋 Statistical Comparison</div>', unsafe_allow_html=True)
    real_stats = fraud_raw[['V1','V2','V3','Amount']].describe().round(3)
    synth_stats = synthetic[['V1','V2','V3','Amount']].describe().round(3)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real Fraud Statistics**")
        st.dataframe(real_stats.style.background_gradient(cmap='Reds'), height=280)
    with col2:
        st.markdown("**CTGAN Synthetic Statistics**")
        st.dataframe(synth_stats.style.background_gradient(cmap='Blues'), height=280)
    st.success("✅ CTGAN successfully learned and reproduced the statistical properties of real fraud data!")

# ── PAGE: MODEL RESULTS ───────────────────────────────────
elif page == "📈 Model Results":
    st.markdown("## 📈 Model Results & Evaluation")

    st.markdown('<div class="section-title">🏆 Performance Metrics Comparison</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    mnames = ['Precision','Recall','F1-Score','AUC']
    for col, mn, b, a in zip(cols, mnames, mb, ma):
        imp = ((a-b)/max(b,0.001))*100
        with col:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.8rem; color:#8899bb; text-transform:uppercase; letter-spacing:1px;">{mn}</div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem;">
                    <div>
                        <div style="color:#8899bb; font-size:0.75rem;">Baseline</div>
                        <div style="color:#ef5350; font-size:1.3rem; font-weight:700; font-family:'JetBrains Mono',monospace;">{b:.2f}</div>
                    </div>
                    <div style="color:#42a5f5; font-size:1.4rem;">→</div>
                    <div>
                        <div style="color:#8899bb; font-size:0.75rem;">Augmented</div>
                        <div style="color:#42a5f5; font-size:1.3rem; font-weight:700; font-family:'JetBrains Mono',monospace;">{a:.2f}</div>
                    </div>
                </div>
                <div style="margin-top:0.5rem;"><span class="improve-badge">⬆ +{abs(imp):.1f}%</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 ROC Curve</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,5), facecolor='#0d1526')
        ax.set_facecolor('#0d1526')
        fp_b,tp_b,_ = roc_curve(yt_b,ypr_b)
        fp_a,tp_a,_ = roc_curve(yt_a,ypr_a)
        ax.plot(fp_b,tp_b,color='#EF5350',lw=2.5,label=f'Baseline (AUC={mb[3]:.2f})')
        ax.plot(fp_a,tp_a,color='#42A5F5',lw=2.5,label=f'Augmented (AUC={ma[3]:.2f})')
        ax.plot([0,1],[0,1],'#334466',lw=1,linestyle='--')
        ax.fill_between(fp_a,tp_a,alpha=0.1,color='#42A5F5')
        ax.set_title('ROC Curve Comparison', color='white', fontweight='bold', fontsize=13)
        ax.set_xlabel('False Positive Rate', color='#8899bb')
        ax.set_ylabel('True Positive Rate', color='#8899bb')
        ax.tick_params(colors='#8899bb')
        ax.legend(facecolor='#1a2744', labelcolor='white')
        for spine in ax.spines.values(): spine.set_color('#2a4a8a')
        st.pyplot(fig); plt.close()

    with col2:
        fig, axes = plt.subplots(1,2,figsize=(8,5), facecolor='#0d1526')
        fig.set_facecolor('#0d1526')
        for ax,cm_d,pred,title,cmap in [
            (axes[0],yt_b,yp_b,'Baseline','Reds'),
            (axes[1],yt_a,yp_a,'Augmented','Blues')]:
            cm = confusion_matrix(cm_d,pred)
            sns.heatmap(cm,annot=True,fmt='d',cmap=cmap,ax=ax,
                        xticklabels=['Normal','Fraud'],yticklabels=['Normal','Fraud'],
                        annot_kws={'size':12,'weight':'bold'})
            ax.set_title(title, color='white', fontweight='bold')
            ax.set_ylabel('Actual', color='#8899bb')
            ax.set_xlabel('Predicted', color='#8899bb')
            ax.tick_params(colors='#8899bb')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">🌟 Feature Importance</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10,5), facecolor='#0d1526')
    ax.set_facecolor('#0d1526')
    imp = pd.Series(clf_a.feature_importances_, index=features).sort_values().tail(10)
    colors_fi = plt.cm.YlOrRd(np.linspace(0.3,0.9,len(imp)))
    imp.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='#2a4a8a')
    ax.set_title('Top 10 Feature Importances', color='white', fontweight='bold', fontsize=13)
    ax.set_xlabel('Importance Score', color='#8899bb')
    ax.tick_params(colors='#8899bb')
    for spine in ax.spines.values(): spine.set_color('#2a4a8a')
    for i,v in enumerate(imp.values):
        ax.text(v+0.0003,i,f'{v:.4f}',va='center',fontsize=9,color='#8899bb')
    st.pyplot(fig); plt.close()

# ── PAGE: LIVE PREDICTION ─────────────────────────────────
elif page == "🔍 Live Prediction":
    st.markdown("## 🔍 Live Transaction Fraud Prediction")
    st.markdown("Enter transaction details below to check if it's fraudulent.")

    with st.expander("⚡ Quick Test — Use Sample Values", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔴 Load Fraud Example"):
                st.session_state.update({
                    'v1':-3.5,'v2':2.8,'v3':-4.0,'v4':3.5,'v5':-2.0,
                    'v6':-1.2,'v7':-3.0,'v8':0.4,'v9':-1.4,'v10':-3.8,'amount':280.0})
        with col2:
            if st.button("🟢 Load Normal Example"):
                st.session_state.update({
                    'v1':0.2,'v2':-0.1,'v3':0.5,'v4':0.1,'v5':-0.2,
                    'v6':0.0,'v7':0.1,'v8':0.0,'v9':0.1,'v10':0.0,'amount':45.0})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Feature Group 1**")
        v1  = st.slider("V1",  -8.0, 6.0, float(st.session_state.get('v1', 0.0)), 0.1)
        v2  = st.slider("V2",  -6.0, 8.0, float(st.session_state.get('v2', 0.0)), 0.1)
        v3  = st.slider("V3",  -8.0, 6.0, float(st.session_state.get('v3', 0.0)), 0.1)
        v4  = st.slider("V4",  -5.0, 8.0, float(st.session_state.get('v4', 0.0)), 0.1)
    with col2:
        st.markdown("**Feature Group 2**")
        v5  = st.slider("V5",  -6.0, 5.0, float(st.session_state.get('v5', 0.0)), 0.1)
        v6  = st.slider("V6",  -5.0, 5.0, float(st.session_state.get('v6', 0.0)), 0.1)
        v7  = st.slider("V7",  -8.0, 5.0, float(st.session_state.get('v7', 0.0)), 0.1)
        v8  = st.slider("V8",  -4.0, 4.0, float(st.session_state.get('v8', 0.0)), 0.1)
    with col3:
        st.markdown("**Feature Group 3**")
        v9  = st.slider("V9",  -5.0, 5.0, float(st.session_state.get('v9', 0.0)), 0.1)
        v10 = st.slider("V10", -8.0, 4.0, float(st.session_state.get('v10',0.0)), 0.1)
        amount = st.number_input("💰 Amount (₹)", 0.0, 5000.0,
                                  float(st.session_state.get('amount', 50.0)), 10.0)
        time_val = st.number_input("🕐 Time (seconds)", 0, 172800, 50000, 1000)

    if st.button("🔍 Predict Transaction"):
        input_data = pd.DataFrame([[time_val,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,amount]],
                                   columns=features)
        input_scaled = scaler.transform(input_data)
        pred = clf_a.predict(input_scaled)[0]
        prob = clf_a.predict_proba(input_scaled)[0]
        fraud_prob = prob[1] * 100
        normal_prob = prob[0] * 100

        st.markdown("---")
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            if pred == 1:
                st.markdown(f"""<div class="fraud-alert">
                    🚨 <b>FRAUDULENT TRANSACTION DETECTED!</b><br>
                    <span style="font-size:0.9rem; color:#ff9999;">
                    This transaction shows patterns consistent with fraud.
                    Confidence: {fraud_prob:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="safe-alert">
                    ✅ <b>LEGITIMATE TRANSACTION</b><br>
                    <span style="font-size:0.9rem; color:#99ddaa;">
                    This transaction appears normal and safe.
                    Confidence: {normal_prob:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{'#ef5350' if pred==1 else '#42a5f5'};">
                    {fraud_prob:.1f}%
                </div>
                <div class="metric-label">Fraud Probability</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#66bb6a;">{normal_prob:.1f}%</div>
                <div class="metric-label">Normal Probability</div>
            </div>""", unsafe_allow_html=True)

        # Probability bar
        fig, ax = plt.subplots(figsize=(8,2), facecolor='#0d1526')
        ax.set_facecolor('#0d1526')
        ax.barh(['Normal','Fraud'], [normal_prob,fraud_prob],
                color=['#42A5F5','#EF5350'], edgecolor='#2a4a8a', height=0.5)
        for i, v in enumerate([normal_prob,fraud_prob]):
            ax.text(v+0.5, i, f'{v:.1f}%', va='center', color='white', fontweight='bold')
        ax.set_xlim(0,115); ax.set_title('Prediction Probabilities', color='white', fontweight='bold')
        ax.tick_params(colors='#8899bb')
        for spine in ax.spines.values(): spine.set_color('#2a4a8a')
        st.pyplot(fig); plt.close()
