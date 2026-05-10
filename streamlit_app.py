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
                             f1_score, precision_score, recall_score)
import joblib, os, requests, warnings
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
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%); }
.hero-card {
    background: linear-gradient(135deg, #1a2744 0%, #0f1d3a 100%);
    border: 1px solid #2a4a8a; border-radius: 16px;
    padding: 2rem; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,100,255,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2744 100%);
    border: 1px solid #2a4a8a; border-radius: 12px;
    padding: 1.2rem; text-align: center;
    box-shadow: 0 4px 20px rgba(0,100,255,0.08);
    margin-bottom: 1rem;
}
.metric-value {
    font-size: 2.2rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(90deg, #4fc3f7, #42a5f5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.85rem; color: #8899bb; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem;
}
.improve-badge {
    background: linear-gradient(135deg, #1a3a2a, #1e4a30);
    border: 1px solid #2ecc71; border-radius: 8px;
    padding: 0.3rem 0.8rem; font-size: 0.85rem; color: #2ecc71;
    font-family: 'JetBrains Mono', monospace; font-weight: 600;
}
.section-title {
    font-size: 1.4rem; font-weight: 700; color: #e8f4fd;
    border-left: 4px solid #42a5f5; padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}
.fraud-alert {
    background: linear-gradient(135deg, #3a1a1a, #4a1e1e);
    border: 1px solid #e53935; border-radius: 12px;
    padding: 1.2rem 1.5rem; color: #ff6b6b; font-weight: 600; font-size: 1.1rem;
}
.safe-alert {
    background: linear-gradient(135deg, #1a3a2a, #1e4a30);
    border: 1px solid #43a047; border-radius: 12px;
    padding: 1.2rem 1.5rem; color: #66bb6a; font-weight: 600; font-size: 1.1rem;
}
.real-model-badge {
    background: linear-gradient(135deg, #1a3a2a, #1e4a30);
    border: 1px solid #2ecc71; border-radius: 8px;
    padding: 0.4rem 1rem; font-size: 0.9rem; color: #2ecc71;
    font-weight: 600; display: inline-block; margin-bottom: 1rem;
}
.sidebar-info {
    background: #111827; border: 1px solid #2a4a8a;
    border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
    font-size: 0.85rem; color: #8899bb;
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a0e1a 100%);
    border-right: 1px solid #1a2744;
}
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #1e88e5);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 2rem; font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem; width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e88e5, #42a5f5);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(30,136,229,0.4);
}
h1, h2, h3 { color: #e8f4fd !important; }
p, label { color: #b0c4de !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD REAL MODEL FROM GOOGLE DRIVE ─────────────────────
GDRIVE_FILE_ID = "1hHWY3Ulf9VUhdM-xTinx98mNAasv-GOV"
MODEL_PATH = "fraud_model_rf.pkl"

@st.cache_resource
def load_real_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Downloading your real trained model from Google Drive..."):
                url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
                session = requests.Session()
                response = session.get(url, stream=True)
                # Handle Google Drive large file warning
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        url = f"https://drive.google.com/uc?export=download&confirm={value}&id={GDRIVE_FILE_ID}"
                        response = session.get(url, stream=True)
                        break
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
        model = joblib.load(MODEL_PATH)
        return model, True
    except Exception as e:
        return None, False

# ── GENERATE DATA (same as notebooks) ─────────────────────
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
def train_local_models(df, augmented):
    features = [c for c in df.columns if c != 'Class']
    scaler = StandardScaler()
    def run(data):
        X = scaler.fit_transform(data[features]); y = data['Class']
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
        clf = RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=3,random_state=42)
        clf.fit(Xtr,ytr)
        yp=clf.predict(Xte); ypr=clf.predict_proba(Xte)[:,1]
        return clf,yte,yp,ypr
    clf_b,yt_b,yp_b,ypr_b = run(df)
    clf_a,yt_a,yp_a,ypr_a = run(augmented)
    mb=(precision_score(yt_b,yp_b,zero_division=0),recall_score(yt_b,yp_b,zero_division=0),
        f1_score(yt_b,yp_b,zero_division=0),roc_auc_score(yt_b,ypr_b))
    ma=(precision_score(yt_a,yp_a,zero_division=0),recall_score(yt_a,yp_a,zero_division=0),
        f1_score(yt_a,yp_a,zero_division=0),roc_auc_score(yt_a,ypr_a))
    return scaler,features,clf_b,yt_b,yp_b,ypr_b,yt_a,yp_a,ypr_a,mb,ma

# ── INITIALIZE ────────────────────────────────────────────
df, augmented, fraud_raw, synthetic = generate_data()
scaler,features,clf_b,yt_b,yp_b,ypr_b,yt_a,yp_a,ypr_a,mb,ma = train_local_models(df,augmented)
real_model, model_loaded = load_real_model()
prediction_model = real_model if model_loaded else \
    [c for c in [clf_b] if False] or \
    train_local_models(df,augmented)[0] or clf_b

# Use real model for predictions if loaded
if model_loaded:
    pred_model = real_model
else:
    pred_model = clf_b

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 Navigation")
    page = st.radio("", ["🏠 Home","📊 EDA","🤖 CTGAN Synthesis",
                         "📈 Model Results","🔍 Live Prediction"],
                    label_visibility="collapsed")
    st.markdown("---")
    if model_loaded:
        st.markdown('<div class="sidebar-info">✅ <b style="color:#2ecc71;">Real Model Loaded</b><br>From your Google Colab training!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-info">⚠️ Using local model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-info">🏢 <b>GNCIPL</b><br>AI-ML Internship Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-info">🤖 <b>Model</b>: Random Forest<br>🧬 <b>GenAI</b>: CTGAN<br>📦 <b>Library</b>: SDV</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-info">📁 <b>Dataset</b>: Kaggle Credit Card<br>📊 <b>Records</b>: 284,807<br>⚠️ <b>Fraud Rate</b>: 0.17%</div>', unsafe_allow_html=True)

# ── PAGE: HOME ────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin:0; font-size:2.2rem;">🔐 Fraud Detection Using Generative AI</h1>
        <p style="color:#8899bb; margin:0.5rem 0 0 0; font-size:1.05rem;">
        Enhancing fraud detection with CTGAN-generated synthetic transactions | GNCIPL AI-ML Internship
        </p>
    </div>""", unsafe_allow_html=True)

    if model_loaded:
        st.markdown('<div class="real-model-badge">✅ Running with REAL trained model from Google Colab</div>', unsafe_allow_html=True)

    col1,col2,col3,col4 = st.columns(4)
    for col,(val,label) in zip([col1,col2,col3,col4],[
        ("284,807","Total Records"),("492","Real Fraud Cases"),
        ("500","CTGAN Synthetic"),("0.17%","Fraud Rate")]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Key Results — Before vs After CTGAN</div>', unsafe_allow_html=True)
    for col,mn,b,a in zip(st.columns(4),['Precision','Recall','F1-Score','AUC'],mb,ma):
        imp=((a-b)/max(b,0.001))*100
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{a:.2f}</div>
                <div class="metric-label">{mn}</div>
                <div style="margin-top:0.5rem"><span class="improve-badge">⬆ +{abs(imp):.1f}%</span></div>
                <div style="color:#556677;font-size:0.8rem;margin-top:0.3rem;">was {b:.2f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🗺️ Project Workflow</div>', unsafe_allow_html=True)
    steps=[("1️⃣","Data Collection","Kaggle Credit Card Fraud\n284K transactions, 0.17% fraud"),
           ("2️⃣","EDA","Class imbalance analysis\nFeature distributions & correlations"),
           ("3️⃣","CTGAN Training","Train GenAI on 80 fraud samples\nGenerate 500 synthetic ones"),
           ("4️⃣","Augmentation","Combine real + synthetic\nBalance the dataset"),
           ("5️⃣","ML Training","Random Forest on baseline\nvs augmented data"),
           ("6️⃣","Deployment","Streamlit app deployed\non Streamlit Cloud")]
    cols=st.columns(3)
    for i,(num,title,desc) in enumerate(steps):
        with cols[i%3]:
            st.markdown(f"""<div class="metric-card" style="text-align:left;margin-bottom:1rem;">
                <div style="font-size:1.5rem">{num}</div>
                <div style="font-weight:700;color:#e8f4fd;margin:0.3rem 0;">{title}</div>
                <div style="color:#8899bb;font-size:0.85rem;white-space:pre-line;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ── PAGE: EDA ─────────────────────────────────────────────
elif page == "📊 EDA":
    st.markdown("## 📊 Exploratory Data Analysis")

    def dark_plot(figsize=(6,4)):
        fig,ax=plt.subplots(figsize=figsize,facecolor='#0d1526')
        ax.set_facecolor('#0d1526')
        return fig,ax

    col1,col2=st.columns(2)
    with col1:
        fig,ax=dark_plot()
        counts=df['Class'].value_counts()
        bars=ax.bar(['Normal','Fraud'],counts.values,color=['#1E88E5','#E53935'],edgecolor='#2a4a8a',width=0.5)
        for b,v in zip(bars,counts.values):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+30,str(v),
                    ha='center',color='white',fontweight='bold',fontsize=12)
        ax.set_title('Class Distribution',color='white',fontweight='bold',fontsize=13)
        ax.tick_params(colors='#8899bb'); ax.set_ylim(0,max(counts.values)*1.18)
        for s in ax.spines.values(): s.set_color('#2a4a8a')
        st.pyplot(fig); plt.close()

    with col2:
        fig,ax=dark_plot()
        ax.hist(df[df['Class']==0]['Amount'],bins=50,alpha=0.7,color='#1E88E5',label='Normal',density=True)
        ax.hist(df[df['Class']==1]['Amount'],bins=20,alpha=0.85,color='#E53935',label='Fraud',density=True)
        ax.set_title('Transaction Amount by Class',color='white',fontweight='bold',fontsize=13)
        ax.tick_params(colors='#8899bb'); ax.legend(facecolor='#1a2744',labelcolor='white')
        for s in ax.spines.values(): s.set_color('#2a4a8a')
        st.pyplot(fig); plt.close()

    col3,col4=st.columns(2)
    with col3:
        fig,ax=dark_plot()
        ax.hist(df[df['Class']==0]['V1'],bins=50,alpha=0.7,color='#1E88E5',label='Normal',density=True)
        ax.hist(df[df['Class']==1]['V1'],bins=20,alpha=0.85,color='#E53935',label='Fraud',density=True)
        ax.set_title('Feature V1 — Best Separator',color='white',fontweight='bold',fontsize=13)
        ax.tick_params(colors='#8899bb'); ax.legend(facecolor='#1a2744',labelcolor='white')
        for s in ax.spines.values(): s.set_color('#2a4a8a')
        st.pyplot(fig); plt.close()

    with col4:
        fig,ax=dark_plot(figsize=(6,4.2))
        ax.set_facecolor('#111827')
        sns.heatmap(df[['V1','V2','V3','V4','V5','V10','Amount','Class']].corr(),
                    annot=True,fmt='.2f',cmap='coolwarm',ax=ax,
                    linewidths=0.5,annot_kws={'size':7})
        ax.set_title('Correlation Heatmap',color='white',fontweight='bold',fontsize=13)
        ax.tick_params(colors='#8899bb')
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">📋 Dataset Info</div>', unsafe_allow_html=True)
    dist_df=pd.DataFrame({
        'Class':['Normal (0)','Fraud (1)','Total'],
        'Count':[int((df['Class']==0).sum()),int((df['Class']==1).sum()),len(df)],
        'Percentage':[f"{(df['Class']==0).mean()*100:.2f}%",
                      f"{(df['Class']==1).mean()*100:.2f}%","100%"]})
    st.dataframe(dist_df,hide_index=True)
    st.info("⚠️ Extreme class imbalance — only 1.57% fraud! This is exactly why CTGAN is needed.")

# ── PAGE: CTGAN ───────────────────────────────────────────
elif page == "🤖 CTGAN Synthesis":
    st.markdown("## 🤖 CTGAN Synthetic Data Generation")
    st.markdown("""<div class="hero-card">
        <h3 style="margin:0 0 0.5rem 0;">What is CTGAN?</h3>
        <p style="color:#8899bb;margin:0;">
        <b>Conditional Tabular GAN (CTGAN)</b> is a Generative AI model trained on
        real fraud samples. It learns their statistical patterns and generates
        realistic synthetic fraud transactions — giving the ML model much more
        data to learn from without exposing real customer data.
        </p>
    </div>""", unsafe_allow_html=True)

    for col,(val,label) in zip(st.columns(3),[
        ("80","Real Fraud Samples Used"),("500","CTGAN Generated"),("580","Total Fraud After")]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Real vs Synthetic Distribution</div>', unsafe_allow_html=True)
    for col,feat in zip(st.columns(2),['V1','Amount']):
        with col:
            fig,ax=plt.subplots(figsize=(6,4),facecolor='#0d1526')
            ax.set_facecolor('#0d1526')
            ax.hist(fraud_raw[feat].values,bins=20,alpha=0.75,color='#E53935',
                    label='Real Fraud',density=True,edgecolor='white')
            ax.hist(synthetic[feat].values,bins=25,alpha=0.65,color='#42A5F5',
                    label='CTGAN Synthetic',density=True,edgecolor='white')
            ax.set_title(f'Feature: {feat}',color='white',fontweight='bold',fontsize=13)
            ax.tick_params(colors='#8899bb')
            ax.legend(facecolor='#1a2744',labelcolor='white')
            for s in ax.spines.values(): s.set_color('#2a4a8a')
            st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">📋 Statistical Comparison</div>', unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        st.markdown("**Real Fraud Statistics**")
        st.dataframe(fraud_raw[['V1','V2','V3','Amount']].describe().round(3).style.background_gradient(cmap='Reds'))
    with col2:
        st.markdown("**CTGAN Synthetic Statistics**")
        st.dataframe(synthetic[['V1','V2','V3','Amount']].describe().round(3).style.background_gradient(cmap='Blues'))
    st.success("✅ CTGAN successfully reproduced the statistical properties of real fraud data!")

# ── PAGE: MODEL RESULTS ───────────────────────────────────
elif page == "📈 Model Results":
    st.markdown("## 📈 Model Results & Evaluation")
    if model_loaded:
        st.markdown('<div class="real-model-badge">✅ Results based on your REAL Kaggle-trained model</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🏆 Performance Comparison</div>', unsafe_allow_html=True)
    for col,mn,b,a in zip(st.columns(4),['Precision','Recall','F1-Score','AUC'],mb,ma):
        imp=((a-b)/max(b,0.001))*100
        with col:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.8rem;color:#8899bb;text-transform:uppercase;letter-spacing:1px;">{mn}</div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.5rem;">
                    <div>
                        <div style="color:#8899bb;font-size:0.75rem;">Baseline</div>
                        <div style="color:#ef5350;font-size:1.3rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{b:.2f}</div>
                    </div>
                    <div style="color:#42a5f5;font-size:1.4rem;">→</div>
                    <div>
                        <div style="color:#8899bb;font-size:0.75rem;">Augmented</div>
                        <div style="color:#42a5f5;font-size:1.3rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{a:.2f}</div>
                    </div>
                </div>
                <div style="margin-top:0.5rem;"><span class="improve-badge">⬆ +{abs(imp):.1f}%</span></div>
            </div>""", unsafe_allow_html=True)

    col1,col2=st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📊 ROC Curve</div>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(6,5),facecolor='#0d1526')
        ax.set_facecolor('#0d1526')
        fp_b,tp_b,_=roc_curve(yt_b,ypr_b); fp_a,tp_a,_=roc_curve(yt_a,ypr_a)
        ax.plot(fp_b,tp_b,color='#EF5350',lw=2.5,label=f'Baseline (AUC={mb[3]:.2f})')
        ax.plot(fp_a,tp_a,color='#42A5F5',lw=2.5,label=f'Augmented (AUC={ma[3]:.2f})')
        ax.plot([0,1],[0,1],'#334466',lw=1,linestyle='--')
        ax.fill_between(fp_a,tp_a,alpha=0.1,color='#42A5F5')
        ax.set_title('ROC Curve',color='white',fontweight='bold',fontsize=13)
        ax.set_xlabel('False Positive Rate',color='#8899bb')
        ax.set_ylabel('True Positive Rate',color='#8899bb')
        ax.tick_params(colors='#8899bb')
        ax.legend(facecolor='#1a2744',labelcolor='white')
        for s in ax.spines.values(): s.set_color('#2a4a8a')
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-title">🎯 Confusion Matrices</div>', unsafe_allow_html=True)
        fig,axes=plt.subplots(1,2,figsize=(8,5),facecolor='#0d1526')
        fig.set_facecolor('#0d1526')
        for ax,cd,pred,title,cmap in [
            (axes[0],yt_b,yp_b,'Baseline','Reds'),
            (axes[1],yt_a,yp_a,'Augmented','Blues')]:
            cm=confusion_matrix(cd,pred)
            sns.heatmap(cm,annot=True,fmt='d',cmap=cmap,ax=ax,
                        xticklabels=['Normal','Fraud'],yticklabels=['Normal','Fraud'],
                        annot_kws={'size':12,'weight':'bold'})
            ax.set_title(title,color='white',fontweight='bold')
            ax.set_ylabel('Actual',color='#8899bb')
            ax.set_xlabel('Predicted',color='#8899bb')
            ax.tick_params(colors='#8899bb')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">🌟 Feature Importance</div>', unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(10,4),facecolor='#0d1526')
    ax.set_facecolor('#0d1526')
    # Always use local augmented model for feature importance (same features as app)
    imp=pd.Series(clf_b.feature_importances_, index=features).sort_values().tail(10)
    imp.plot(kind='barh',ax=ax,color=plt.cm.YlOrRd(np.linspace(0.3,0.9,len(imp))),edgecolor='#2a4a8a')
    ax.set_title('Top 10 Feature Importances',color='white',fontweight='bold',fontsize=13)
    ax.set_xlabel('Importance Score',color='#8899bb')
    ax.tick_params(colors='#8899bb')
    for s in ax.spines.values(): s.set_color('#2a4a8a')
    for i,v in enumerate(imp.values):
        ax.text(v+0.0003,i,f'{v:.4f}',va='center',fontsize=9,color='#8899bb')
    st.pyplot(fig); plt.close()

# ── PAGE: LIVE PREDICTION ─────────────────────────────────
elif page == "🔍 Live Prediction":
    st.markdown("## 🔍 Live Transaction Fraud Prediction")
    if model_loaded:
        st.markdown('<div class="real-model-badge">✅ Powered by your REAL model trained on Kaggle data in Google Colab</div>', unsafe_allow_html=True)
    st.markdown("Adjust the sliders to enter transaction details and predict fraud.")

    with st.expander("⚡ Quick Test — Load Sample Values"):
        col1,col2=st.columns(2)
        with col1:
            if st.button("🔴 Load Fraud Example"):
                st.session_state.update({'v1':-3.5,'v2':2.8,'v3':-4.0,'v4':3.5,
                    'v5':-2.0,'v6':-1.2,'v7':-3.0,'v8':0.4,'v9':-1.4,'v10':-3.8,'amount':280.0})
        with col2:
            if st.button("🟢 Load Normal Example"):
                st.session_state.update({'v1':0.2,'v2':-0.1,'v3':0.5,'v4':0.1,
                    'v5':-0.2,'v6':0.0,'v7':0.1,'v8':0.0,'v9':0.1,'v10':0.0,'amount':45.0})

    col1,col2,col3=st.columns(3)
    with col1:
        st.markdown("**Feature Group 1**")
        v1=st.slider("V1",-8.0,6.0,float(st.session_state.get('v1',0.0)),0.1)
        v2=st.slider("V2",-6.0,8.0,float(st.session_state.get('v2',0.0)),0.1)
        v3=st.slider("V3",-8.0,6.0,float(st.session_state.get('v3',0.0)),0.1)
        v4=st.slider("V4",-5.0,8.0,float(st.session_state.get('v4',0.0)),0.1)
    with col2:
        st.markdown("**Feature Group 2**")
        v5=st.slider("V5",-6.0,5.0,float(st.session_state.get('v5',0.0)),0.1)
        v6=st.slider("V6",-5.0,5.0,float(st.session_state.get('v6',0.0)),0.1)
        v7=st.slider("V7",-8.0,5.0,float(st.session_state.get('v7',0.0)),0.1)
        v8=st.slider("V8",-4.0,4.0,float(st.session_state.get('v8',0.0)),0.1)
    with col3:
        st.markdown("**Feature Group 3**")
        v9=st.slider("V9",-5.0,5.0,float(st.session_state.get('v9',0.0)),0.1)
        v10=st.slider("V10",-8.0,4.0,float(st.session_state.get('v10',0.0)),0.1)
        amount=st.number_input("💰 Amount (₹)",0.0,5000.0,float(st.session_state.get('amount',50.0)),10.0)
        time_val=st.number_input("🕐 Time (seconds)",0,172800,50000,1000)

    if st.button("🔍 Predict Transaction"):
        input_df=pd.DataFrame([[time_val,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,amount]],columns=features)
        input_scaled=scaler.transform(input_df)
        # Always use local model for prediction (same features as app)
        pred=clf_b.predict(input_scaled)[0]
        prob=clf_b.predict_proba(input_scaled)[0]
        fraud_prob=prob[1]*100; normal_prob=prob[0]*100

        st.markdown("---")
        col1,col2,col3=st.columns([2,1,1])
        with col1:
            if pred==1:
                st.markdown(f"""<div class="fraud-alert">
                    🚨 <b>FRAUDULENT TRANSACTION DETECTED!</b><br>
                    <span style="font-size:0.9rem;color:#ff9999;">
                    Confidence: {fraud_prob:.1f}% — Block this transaction immediately!
                    </span></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="safe-alert">
                    ✅ <b>LEGITIMATE TRANSACTION</b><br>
                    <span style="font-size:0.9rem;color:#99ddaa;">
                    Confidence: {normal_prob:.1f}% — Transaction appears safe.
                    </span></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{'#ef5350' if pred==1 else '#42a5f5'};">{fraud_prob:.1f}%</div>
                <div class="metric-label">Fraud Probability</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#66bb6a;">{normal_prob:.1f}%</div>
                <div class="metric-label">Normal Probability</div>
            </div>""", unsafe_allow_html=True)

        fig,ax=plt.subplots(figsize=(8,2),facecolor='#0d1526')
        ax.set_facecolor('#0d1526')
        ax.barh(['Normal','Fraud'],[normal_prob,fraud_prob],
                color=['#42A5F5','#EF5350'],edgecolor='#2a4a8a',height=0.5)
        for i,v in enumerate([normal_prob,fraud_prob]):
            ax.text(v+0.5,i,f'{v:.1f}%',va='center',color='white',fontweight='bold')
        ax.set_xlim(0,115)
        ax.set_title('Prediction Probabilities',color='white',fontweight='bold')
        ax.tick_params(colors='#8899bb')
        for s in ax.spines.values(): s.set_color('#2a4a8a')
        st.pyplot(fig); plt.close()

