import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Novel Rock Physics Gas Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .highlight-box {
        background-color: #E6F3FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #DEE2E6;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🛢️ Novel Rock Physics Gas Detection Framework</h1>", unsafe_allow_html=True)
st.markdown("### Adaptive DEM Modeling with Multi-Model Uncertainty Quantification")

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Data Input")
    
    uploaded_file = st.file_uploader(
        "Upload Well Log CSV File", 
        type=['csv'],
        help="Upload CSV file with columns: Vp, Vs, RHO, PHI, GR, RT, VCLAY, SW, DEPTH"
    )
    
    st.markdown("---")
    st.markdown("## ⚙️ Analysis Parameters")
    
    analysis_tab1, analysis_tab2 = st.tabs(["Model Settings", "Detection Parameters"])
    
    with analysis_tab1:
        model_type = st.selectbox(
            "Rock Physics Model",
            ["Adaptive DEM", "Multi-Model Ensemble", "Soft Sand", "Stiff Sand", "Constant Cement"],
            index=0,
            help="Select the rock physics modeling approach"
        )
        
        use_dispersion = st.checkbox(
            "Include Dispersion Analysis", 
            value=True,
            help="Include frequency-dependent dispersion effects"
        )
        
        use_ml = st.checkbox(
            "Include Machine Learning", 
            value=True,
            help="Include ML classifier in ensemble"
        )
        
        uncertainty_weighting = st.select_slider(
            "Uncertainty Weighting",
            options=["Low", "Medium", "High"],
            value="Medium",
            help="Weighting of model uncertainties in ensemble"
        )
    
    with analysis_tab2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Minimum confidence for gas zone identification"
        )
        
        n_clusters = st.slider(
            "Number of Rock Fabric Clusters",
            min_value=2,
            max_value=5,
            value=3,
            help="For adaptive DEM pore geometry classification"
        )
        
        generate_rpt = st.checkbox(
            "Generate Rock Physics Templates",
            value=True,
            help="Generate RPTs for visualization"
        )
    
    st.markdown("---")
    
    if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
    else:
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False

# Main content
class NovelRockPhysicsGasDetector:
    """Advanced gas detection framework with adaptive DEM modeling."""
    
    def __init__(self):
        # Physical constants
        self.physical_constants = {
            'K_quartz': 37.0, 'G_quartz': 44.0,
            'K_clay': 21.0, 'G_clay': 7.0,
            'K_water': 2.25, 'K_gas': 0.05,
            'RHO_quartz': 2.65, 'RHO_clay': 2.70,
            'RHO_water': 1.00, 'RHO_gas': 0.25,
            'critical_porosity': 0.40
        }
        
    def load_data(self, data):
        """Load and prepare well log data."""
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()
        
        # Standardize column names
        self._standardize_columns()
        
        # Validate and calculate derived properties
        self._process_data()
        
        return self.data
    
    def _standardize_columns(self):
        """Standardize column names."""
        column_mapping = {
            'VP': 'Vp', 'V_P': 'Vp', 'P_VEL': 'Vp', 'DT': 'Vp',
            'VS': 'Vs', 'V_S': 'Vs', 'S_VEL': 'Vs', 'DTS': 'Vs',
            'RHOB': 'RHO', 'DEN': 'RHO', 'DENSITY': 'RHO',
            'PHIT': 'PHI', 'PHIE': 'PHI', 'POR': 'PHI',
            'GR': 'GR', 'GAMMA': 'GR',
            'RT': 'RT', 'RES': 'RT',
            'VSH': 'VCLAY', 'SH': 'VCLAY',
            'SW': 'SW', 'SWT': 'SW',
            'DEPTH': 'DEPTH', 'DEPT': 'DEPTH'
        }
        
        self.data.columns = [column_mapping.get(col.upper(), col) 
                           for col in self.data.columns]
    
    def _process_data(self):
        """Process and validate data."""
        # Ensure required columns
        required = ['Vp', 'Vs', 'RHO', 'PHI']
        for col in required:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert units if needed
        if self.data['Vp'].mean() < 1000:
            self.data['Vp'] *= 1000
        if self.data['Vs'].mean() < 500:
            self.data['Vs'] *= 1000
        if self.data['RHO'].mean() > 3000:
            self.data['RHO'] /= 1000
        
        # Calculate derived properties
        self.data['Ip'] = self.data['Vp'] * self.data['RHO']
        self.data['Is'] = self.data['Vs'] * self.data['RHO']
        self.data['Vp_Vs'] = self.data['Vp'] / self.data['Vs']
        self.data['PR'] = (0.5 * (self.data['Vp_Vs']**2 - 2)) / (self.data['Vp_Vs']**2 - 1)
        
        # Lame parameters
        self.data['MU'] = self.data['RHO'] * self.data['Vs']**2 / 1e6
        self.data['LAMBDA'] = self.data['RHO'] * self.data['Vp']**2 / 1e6 - 2 * self.data['MU']
        self.data['MU_RHO'] = self.data['MU'] * self.data['RHO']
        self.data['LAMBDA_RHO'] = self.data['LAMBDA'] * self.data['RHO']
        
        # Clean data
        self.data = self.data.dropna(subset=required)
        mask = (
            (self.data['Vp'] > 1000) & (self.data['Vp'] < 8000) &
            (self.data['Vs'] > 500) & (self.data['Vs'] < 5000) &
            (self.data['RHO'] > 1.8) & (self.data['RHO'] < 3.0) &
            (self.data['PHI'] >= 0) & (self.data['PHI'] <= 0.5)
        )
        self.data = self.data[mask].reset_index(drop=True)
    
    def differential_effective_medium(self, K0, G0, phi, aspect_ratios, Sw=1.0):
        """DEM theory implementation."""
        # Fluid properties
        Kf = 1/(Sw/self.physical_constants['K_water'] + 
               (1-Sw)/self.physical_constants['K_gas'])
        rhof = Sw * self.physical_constants['RHO_water'] + \
               (1-Sw) * self.physical_constants['RHO_gas']
        
        # Initialize
        K_eff, G_eff = K0, G0
        steps = 100
        dphi = phi / steps
        phi_current = 0.0
        
        for _ in range(steps):
            K_sum, G_sum = 0, 0
            for alpha in aspect_ratios:
                # Simplified Eshelby tensor
                if alpha < 1.0:
                    A = alpha**2 / (1 - alpha**2)**1.5
                    A *= (np.arccos(alpha) - alpha * np.sqrt(1 - alpha**2))
                    F1 = 1 + A * (3/(1-alpha**2) - 1/(1-alpha**2)**0.5)
                    F2 = 2 + A * (1.5/(1-alpha**2) - (1+2*alpha**2)/(1-alpha**2)**1.5)
                else:
                    alpha_inv = 1/alpha
                    A = alpha_inv**2 / (alpha_inv**2 - 1)**1.5
                    A *= (alpha_inv * np.sqrt(alpha_inv**2-1) - np.arccosh(alpha_inv))
                    F1 = 1 + A * (3/(alpha_inv**2-1) - 1/(alpha_inv**2-1)**0.5)
                    F2 = 2 + A * (1.5/(alpha_inv**2-1) - 
                                 (1+2*alpha_inv**2)/(alpha_inv**2-1)**1.5)
                
                P = (Kf - K_eff) / (Kf + 4/3 * G_eff) * F1
                Q = (Kf - G_eff) / (Kf + G_eff * (9*K_eff + 8*G_eff) / 
                                   (6*(K_eff + 2*G_eff))) * F2
                
                K_sum += -K_eff * P / len(aspect_ratios)
                G_sum += -G_eff * Q / len(aspect_ratios)
            
            K_eff += K_sum * dphi / (1 - phi_current)
            G_eff += G_sum * dphi / (1 - phi_current)
            phi_current += dphi
        
        # Calculate velocities
        rho_eff = (1 - phi) * 2.65 + phi * rhof  # Simplified density
        Vp = np.sqrt((K_eff + 4/3 * G_eff) / rho_eff) * 1000
        Vs = np.sqrt(G_eff / rho_eff) * 1000
        
        return Vp, Vs, rho_eff
    
    def classify_rock_fabric(self, n_clusters=3):
        """Classify rock fabric using clustering."""
        features = np.column_stack([
            self.data['PHI'].values,
            self.data['Vp'].values / 1000,
            self.data['Vp_Vs'].values
        ])
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        clusters = gmm.fit_predict(features_scaled)
        
        # Assign aspect ratios based on clusters
        aspect_ratio_map = []
        for i in range(n_clusters):
            cluster_data = self.data[clusters == i]
            if len(cluster_data) > 0:
                avg_phi = cluster_data['PHI'].mean()
                avg_vp_vs = cluster_data['Vp_Vs'].mean()
                
                if avg_phi > 0.25:
                    aspect_ratio_map.append(np.random.uniform(0.05, 0.15, 3))  # Soft
                elif avg_vp_vs > 1.8:
                    aspect_ratio_map.append(np.random.uniform(0.1, 0.3, 3))   # Intermediate
                else:
                    aspect_ratio_map.append(np.random.uniform(0.5, 0.8, 3))   # Stiff
        
        self.data['FABRIC_CLUSTER'] = clusters
        self.fabric_aspect_ratios = aspect_ratio_map
        
        return clusters, aspect_ratio_map
    
    def multi_model_gas_detection(self, use_dispersion=True, use_ml=True, 
                                 confidence_threshold=0.7, n_clusters=3):
        """Main gas detection algorithm."""
        n_samples = len(self.data)
        
        # Step 1: Rock fabric classification
        clusters, aspect_ratios = self.classify_rock_fabric(n_clusters)
        
        # Step 2: Multiple detection methods
        method_probs = []
        method_names = []
        uncertainties = []
        
        # Method 1: Vp/Vs cutoff
        vp_vs_ratio = self.data['Vp'] / self.data['Vs']
        prob1 = np.clip(1.8 - vp_vs_ratio, 0, 1)
        method_probs.append(prob1)
        method_names.append("Vp/Vs Cutoff")
        uncertainties.append(0.2)
        
        # Method 2: Lambda-Rho
        LMR = self.data['LAMBDA_RHO']
        prob2 = 1 - (LMR - LMR.min()) / (LMR.max() - LMR.min())
        method_probs.append(prob2)
        method_names.append("Lambda-Rho")
        uncertainties.append(0.15)
        
        # Method 3: Adaptive DEM
        prob3 = np.zeros(n_samples)
        for i in range(n_samples):
            cluster = clusters[i]
            if cluster < len(aspect_ratios):
                ar_set = aspect_ratios[cluster]
            else:
                ar_set = [0.1, 0.3, 0.6]
            
            # Simplified DEM calculation
            phi = self.data.iloc[i]['PHI']
            Sw = self.data.iloc[i]['SW'] if 'SW' in self.data.columns else 1.0
            
            # Water-saturated baseline
            vp_water, _, _ = self.differential_effective_medium(37, 44, phi, ar_set, 1.0)
            vp_current, _, _ = self.differential_effective_medium(37, 44, phi, ar_set, Sw)
            
            if vp_water > 0:
                velocity_drop = (vp_water - vp_current) / vp_water
                prob3[i] = np.clip(velocity_drop * 3, 0, 1)
        
        method_probs.append(prob3)
        method_names.append("Adaptive DEM")
        uncertainties.append(0.12)
        
        # Method 4: RPT distance
        prob4 = self._calculate_rpt_probability()
        method_probs.append(prob4)
        method_names.append("RPT Distance")
        uncertainties.append(0.1)
        
        # Method 5: ML classifier
        if use_ml and n_samples > 50:
            prob5 = self._ml_gas_classification()
            method_probs.append(prob5)
            method_names.append("ML Classifier")
            uncertainties.append(0.08)
        
        # Weighted combination
        uncertainties = np.array(uncertainties)
        weights = 1 / (uncertainties + 1e-6)
        weights = weights / weights.sum()
        
        combined_prob = np.average(method_probs, axis=0, weights=weights)
        
        # Apply dispersion if requested
        if use_dispersion:
            dispersion = self._calculate_dispersion_index()
            combined_prob = combined_prob * (1 + 0.3 * dispersion)
            combined_prob = np.clip(combined_prob, 0, 1)
        
        # Calculate confidence
        method_std = np.std(method_probs, axis=0)
        confidence = 1 - np.clip(method_std / 0.5, 0, 1)
        
        # Store results
        self.results = {
            'gas_probability': combined_prob,
            'gas_confidence': confidence,
            'method_probabilities': method_probs,
            'method_names': method_names,
            'method_weights': weights,
            'fabric_clusters': clusters,
            'high_confidence_gas': (combined_prob > confidence_threshold) & (confidence > 0.6)
        }
        
        # Calculate statistics
        self._calculate_statistics(confidence_threshold)
        
        return combined_prob, confidence
    
    def _calculate_rpt_probability(self):
        """Calculate RPT-based probability."""
        n_samples = len(self.data)
        prob = np.zeros(n_samples)
        
        # Simplified RPT distance calculation
        Ip_mean = self.data['Ip'].mean()
        Vp_Vs_mean = self.data['Vp_Vs'].mean()
        
        for i in range(n_samples):
            Ip = self.data.iloc[i]['Ip']
            Vp_Vs = self.data.iloc[i]['Vp_Vs']
            
            # Distance to "gas trend" (simplified)
            gas_trend_vp_vs = 1.6  # Typical for gas sands
            distance = abs(Vp_Vs - gas_trend_vp_vs) / 0.5
            
            prob[i] = np.exp(-distance)
        
        return prob
    
    def _ml_gas_classification(self):
        """ML-based gas classification."""
        n_samples = len(self.data)
        
        # Feature engineering
        features = []
        feature_names = ['Vp', 'Vs', 'RHO', 'Vp_Vs', 'Ip', 'LAMBDA_RHO', 'MU_RHO']
        
        for feat in feature_names:
            if feat in self.data.columns:
                features.append(self.data[feat].values)
        
        if 'PHI' in self.data.columns:
            features.append(self.data['PHI'].values)
            feature_names.append('PHI')
        
        X = np.column_stack(features)
        
        # Create synthetic labels for demonstration
        y = np.zeros(n_samples)
        vp_vs_ratio = self.data['Vp'] / self.data['Vs']
        lambda_rho = self.data['LAMBDA_RHO']
        
        gas_candidates = (vp_vs_ratio < 1.8) & (lambda_rho < lambda_rho.quantile(0.3))
        brine_candidates = (vp_vs_ratio > 2.0) & (lambda_rho > lambda_rho.quantile(0.7))
        
        y[gas_candidates] = 1
        y[brine_candidates] = 0
        
        train_mask = gas_candidates | brine_candidates
        
        if np.sum(train_mask) < 20:
            return np.zeros(n_samples)
        
        # Train classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X[train_mask], y[train_mask])
        
        # Predict
        prob = rf.predict_proba(X)[:, 1]
        
        return prob
    
    def _calculate_dispersion_index(self):
        """Calculate dispersion index."""
        lambda_mu_ratio = self.data['LAMBDA'] / (self.data['MU'] + 1e-6)
        dispersion = 1 - (lambda_mu_ratio - lambda_mu_ratio.min()) / \
                     (lambda_mu_ratio.max() - lambda_mu_ratio.min())
        return dispersion
    
    def _calculate_statistics(self, threshold):
        """Calculate detection statistics."""
        gas_prob = self.results['gas_probability']
        
        stats = {
            'mean_probability': float(np.mean(gas_prob)),
            'max_probability': float(np.max(gas_prob)),
            'std_probability': float(np.std(gas_prob)),
            'samples_above_threshold': int(np.sum(gas_prob > threshold)),
            'percentage_above_threshold': float(np.mean(gas_prob > threshold) * 100),
            'high_confidence_samples': int(np.sum(self.results['high_confidence_gas'])),
            'high_confidence_percentage': float(np.mean(self.results['high_confidence_gas']) * 100),
        }
        
        if np.any(self.results['high_confidence_gas']):
            high_prob_data = self.data[self.results['high_confidence_gas']]
            stats.update({
                'hc_mean_phi': float(high_prob_data['PHI'].mean()),
                'hc_mean_vp_vs': float(high_prob_data['Vp_Vs'].mean()),
                'hc_mean_ip': float(high_prob_data['Ip'].mean()),
            })
        
        self.results['statistics'] = stats
        return stats

def create_dashboard(detector):
    """Create interactive dashboard."""
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🛢️ Gas Detection", 
        "📊 Crossplots", 
        "🔬 Rock Physics", 
        "📋 Statistics"
    ])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Samples",
                f"{len(detector.data):,}",
                help="Number of valid well log samples"
            )
        
        with col2:
            if 'results' in detector.__dict__:
                stats = detector.results['statistics']
                st.metric(
                    "High-Confidence Gas Zones",
                    f"{stats['high_confidence_samples']:,}",
                    f"{stats['high_confidence_percentage']:.1f}%",
                    help="Samples with gas probability > threshold and confidence > 0.6"
                )
        
        with col3:
            if 'results' in detector.__dict__:
                st.metric(
                    "Mean Gas Probability",
                    f"{detector.results['statistics']['mean_probability']:.3f}",
                    help="Average gas probability across all samples"
                )
        
        # Data preview
        st.markdown("### 📋 Data Preview")
        st.dataframe(detector.data.head(10), use_container_width=True)
        
        # Log curves
        st.markdown("### 📈 Well Log Curves")
        
        if 'DEPTH' in detector.data.columns:
            depth = detector.data['DEPTH']
        else:
            depth = np.arange(len(detector.data))
        
        fig = make_subplots(
            rows=1, cols=4,
            subplot_titles=('Gamma Ray', 'Porosity', 'Vp/Vs Ratio', 'Density'),
            horizontal_spacing=0.05
        )
        
        if 'GR' in detector.data.columns:
            fig.add_trace(
                go.Scatter(x=detector.data['GR'], y=depth, mode='lines', 
                          name='GR', line=dict(color='green', width=1)),
                row=1, col=1
            )
        
        fig.add_trace(
            go.Scatter(x=detector.data['PHI'], y=depth, mode='lines',
                      name='PHI', line=dict(color='blue', width=1)),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=detector.data['Vp_Vs'], y=depth, mode='lines',
                      name='Vp/Vs', line=dict(color='purple', width=1)),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Scatter(x=detector.data['RHO'], y=depth, mode='lines',
                      name='RHO', line=dict(color='black', width=1)),
            row=1, col=4
        )
        
        # Update layout
        for i in range(1, 5):
            fig.update_yaxes(title_text="Depth", row=1, col=i, autorange="reversed")
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'results' not in detector.__dict__:
            st.warning("Please run analysis first")
            return
        
        st.markdown("### 🛢️ Gas Probability Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gas probability curve
            fig = go.Figure()
            
            if 'DEPTH' in detector.data.columns:
                depth = detector.data['DEPTH']
            else:
                depth = np.arange(len(detector.data))
            
            # Gas probability
            fig.add_trace(go.Scatter(
                x=detector.results['gas_probability'],
                y=depth,
                mode='lines',
                name='Gas Probability',
                line=dict(color='red', width=2),
                fill='tozerox',
                fillcolor='rgba(255, 0, 0, 0.2)'
            ))
            
            # Confidence
            fig.add_trace(go.Scatter(
                x=detector.results['gas_confidence'],
                y=depth,
                mode='lines',
                name='Confidence',
                line=dict(color='green', width=1, dash='dash'),
                opacity=0.7
            ))
            
            # Threshold line
            fig.add_vline(x=confidence_threshold, line_dash="dot", 
                         line_color="black", opacity=0.5)
            
            fig.update_layout(
                title="Gas Probability with Confidence",
                xaxis_title="Probability / Confidence",
                yaxis_title="Depth",
                yaxis=dict(autorange="reversed"),
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, 
                           xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Method weights
            st.markdown("#### Method Weights")
            weights_df = pd.DataFrame({
                'Method': detector.results['method_names'],
                'Weight': detector.results['method_weights']
            })
            
            fig = px.bar(weights_df, x='Weight', y='Method', orientation='h',
                        color='Weight', color_continuous_scale='viridis')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # High-confidence zones
            st.markdown("#### High-Confidence Zones")
            hc_zones = detector.results['high_confidence_gas']
            if np.any(hc_zones):
                hc_data = detector.data[hc_zones]
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Count", f"{np.sum(hc_zones):,}")
                    st.metric("Avg Porosity", f"{hc_data['PHI'].mean():.3f}")
                
                with metrics_col2:
                    st.metric("Percentage", f"{np.mean(hc_zones)*100:.1f}%")
                    st.metric("Avg Vp/Vs", f"{hc_data['Vp_Vs'].mean():.2f}")
            else:
                st.info("No high-confidence gas zones detected")
        
        # Method probabilities
        st.markdown("#### Individual Method Probabilities")
        method_fig = go.Figure()
        
        for i, (name, probs) in enumerate(zip(detector.results['method_names'], 
                                            detector.results['method_probabilities'])):
            method_fig.add_trace(go.Scatter(
                x=probs,
                y=depth,
                mode='lines',
                name=name,
                line=dict(width=1),
                opacity=0.7
            ))
        
        method_fig.update_layout(
            title="Method Probabilities Comparison",
            xaxis_title="Probability",
            yaxis_title="Depth",
            yaxis=dict(autorange="reversed"),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(method_fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 📊 Crossplot Analysis")
        
        plot_type = st.selectbox(
            "Select Crossplot Type",
            ["Vp/Vs vs Ip", "Lambda-Rho vs Mu-Rho", "Vp vs Porosity", 
             "Rock Fabric Clusters", "Gas Probability Map"]
        )
        
        if plot_type == "Vp/Vs vs Ip":
            fig = px.scatter(
                detector.data,
                x='Ip',
                y='Vp_Vs',
                color=detector.results['gas_probability'] if 'results' in detector.__dict__ else None,
                color_continuous_scale='RdBu_r',
                title="Vp/Vs vs Acoustic Impedance",
                labels={'Ip': 'Acoustic Impedance', 'Vp_Vs': 'Vp/Vs Ratio'}
            )
            
            # Add trend lines
            if 'results' in detector.__dict__:
                fig.add_hline(y=1.8, line_dash="dash", line_color="red", 
                             annotation_text="Gas Sand Threshold")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Lambda-Rho vs Mu-Rho":
            fig = px.scatter(
                detector.data,
                x='LAMBDA_RHO',
                y='MU_RHO',
                color=detector.results['gas_probability'] if 'results' in detector.__dict__ else None,
                color_continuous_scale='RdBu_r',
                title="Lambda-Rho vs Mu-Rho Crossplot",
                labels={'LAMBDA_RHO': 'λρ', 'MU_RHO': 'μρ'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Vp vs Porosity":
            fig = px.scatter(
                detector.data,
                x='PHI',
                y='Vp',
                color=detector.results['gas_probability'] if 'results' in detector.__dict__ else None,
                color_continuous_scale='RdBu_r',
                title="P-wave Velocity vs Porosity",
                labels={'PHI': 'Porosity', 'Vp': 'P-wave Velocity (m/s)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Rock Fabric Clusters":
            if 'fabric_clusters' in detector.results:
                fig = px.scatter(
                    detector.data,
                    x='Ip',
                    y='Vp_Vs',
                    color=detector.results['fabric_clusters'].astype(str),
                    title="Rock Fabric Clusters",
                    labels={'Ip': 'Acoustic Impedance', 'Vp_Vs': 'Vp/Vs Ratio'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Gas Probability Map":
            if 'results' in detector.__dict__:
                fig = px.density_contour(
                    detector.data,
                    x='Ip',
                    y='Vp_Vs',
                    z=detector.results['gas_probability'],
                    title="Gas Probability Density Map",
                    labels={'Ip': 'Acoustic Impedance', 'Vp_Vs': 'Vp/Vs Ratio'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 🔬 Rock Physics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Rock Fabric Classification")
            if 'fabric_clusters' in detector.results:
                cluster_counts = pd.Series(detector.results['fabric_clusters']).value_counts().sort_index()
                
                fig = px.pie(
                    values=cluster_counts.values,
                    names=[f'Cluster {i}' for i in cluster_counts.index],
                    title="Rock Fabric Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster characteristics
                st.markdown("**Cluster Characteristics:**")
                for cluster_id in np.unique(detector.results['fabric_clusters']):
                    cluster_data = detector.data[detector.results['fabric_clusters'] == cluster_id]
                    with st.expander(f"Cluster {cluster_id} - {len(cluster_data)} samples"):
                        st.write(f"Average Porosity: {cluster_data['PHI'].mean():.3f}")
                        st.write(f"Average Vp/Vs: {cluster_data['Vp_Vs'].mean():.2f}")
                        st.write(f"Average Ip: {cluster_data['Ip'].mean():.0f}")
        
        with col2:
            st.markdown("#### Aspect Ratio Distribution")
            if hasattr(detector, 'fabric_aspect_ratios'):
                aspect_data = []
                for i, ar_set in enumerate(detector.fabric_aspect_ratios):
                    for ar in ar_set:
                        aspect_data.append({'Cluster': f'Cluster {i}', 'Aspect Ratio': ar})
                
                if aspect_data:
                    aspect_df = pd.DataFrame(aspect_data)
                    fig = px.box(aspect_df, x='Cluster', y='Aspect Ratio',
                                title="Aspect Ratio Distribution by Cluster")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Elastic moduli
        st.markdown("#### Elastic Moduli Analysis")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Bulk Modulus (K)', 'Shear Modulus (μ)'),
            horizontal_spacing=0.1
        )
        
        fig.add_trace(
            go.Histogram(x=detector.data['LAMBDA'] + 2/3 * detector.data['MU'], 
                        name='K', nbinsx=30),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=detector.data['MU'], name='μ', nbinsx=30),
            row=1, col=2
        )
        
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### 📋 Detailed Statistics")
        
        if 'results' in detector.__dict__:
            stats = detector.results['statistics']
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Gas Probability", f"{stats['mean_probability']:.3f}")
            
            with col2:
                st.metric("Max Probability", f"{stats['max_probability']:.3f}")
            
            with col3:
                st.metric("Samples > Threshold", 
                         f"{stats['samples_above_threshold']:,}",
                         f"{stats['percentage_above_threshold']:.1f}%")
            
            with col4:
                st.metric("High Confidence", 
                         f"{stats['high_confidence_samples']:,}",
                         f"{stats['high_confidence_percentage']:.1f}%")
            
            # Detailed statistics table
            st.markdown("#### Complete Statistics")
            stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
            st.dataframe(stats_df, use_container_width=True)
            
            # Method statistics
            st.markdown("#### Method Performance")
            if 'method_probabilities' in detector.results:
                method_stats = []
                for name, probs in zip(detector.results['method_names'], 
                                     detector.results['method_probabilities']):
                    method_stats.append({
                        'Method': name,
                        'Mean': np.mean(probs),
                        'Std': np.std(probs),
                        'Correlation with Ensemble': np.corrcoef(
                            probs, detector.results['gas_probability'])[0, 1]
                    })
                
                method_df = pd.DataFrame(method_stats)
                st.dataframe(method_df, use_container_width=True)
            
            # Export options
            st.markdown("#### Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download Results CSV", use_container_width=True):
                    results_df = detector.data.copy()
                    results_df['GAS_PROBABILITY'] = detector.results['gas_probability']
                    results_df['GAS_CONFIDENCE'] = detector.results['gas_confidence']
                    results_df['HIGH_CONFIDENCE_GAS'] = detector.results['high_confidence_gas']
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="gas_detection_results.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Download Plots", use_container_width=True):
                    st.info("Plot download feature coming soon!")

def main():
    """Main Streamlit app."""
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
    
    # Main content based on file upload
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Loading and processing data..."):
            if st.session_state.detector is None:
                st.session_state.detector = NovelRockPhysicsGasDetector()
                df = pd.read_csv(uploaded_file)
                st.session_state.detector.load_data(df)
        
        # Show data info
        st.success(f"✅ Data loaded successfully: {len(st.session_state.detector.data):,} samples")
        
        # Display data summary
        with st.expander("📊 Data Summary", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", len(st.session_state.detector.data))
            with col2:
                st.metric("Columns", len(st.session_state.detector.data.columns))
            with col3:
                st.metric("Memory Usage", 
                         f"{st.session_state.detector.data.memory_usage().sum() / 1024:.1f} KB")
            
            # Show column info
            st.write("**Available Columns:**")
            cols = st.session_state.detector.data.columns.tolist()
            for i in range(0, len(cols), 4):
                st.write(" | ".join(cols[i:i+4]))
        
        # Run analysis button
        if st.session_state.run_analysis or st.button("▶️ Run Analysis", type="primary"):
            with st.spinner("Running advanced gas detection analysis..."):
                progress_bar = st.progress(0)
                
                # Step 1: Rock fabric classification
                progress_bar.progress(20)
                st.info("Step 1/5: Classifying rock fabric...")
                
                # Step 2: Run gas detection
                progress_bar.progress(40)
                st.info("Step 2/5: Running multi-model gas detection...")
                
                gas_prob, confidence = st.session_state.detector.multi_model_gas_detection(
                    use_dispersion=use_dispersion,
                    use_ml=use_ml,
                    confidence_threshold=confidence_threshold,
                    n_clusters=n_clusters
                )
                
                progress_bar.progress(80)
                st.info("Step 3/5: Calculating statistics...")
                
                progress_bar.progress(100)
                st.success("✅ Analysis completed successfully!")
                
                st.session_state.analysis_run = True
        
        # Show dashboard if analysis was run
        if st.session_state.analysis_run and st.session_state.detector is not None:
            create_dashboard(st.session_state.detector)
        
        # Show raw data preview if no analysis yet
        elif not st.session_state.analysis_run:
            st.markdown("### 📋 Data Preview")
            st.dataframe(st.session_state.detector.data.head(), use_container_width=True)
            
            # Quick stats
            st.markdown("### 📊 Quick Statistics")
            numeric_cols = st.session_state.detector.data.select_dtypes(include=[np.number]).columns
            stats_df = st.session_state.detector.data[numeric_cols].describe().T
            st.dataframe(stats_df, use_container_width=True)
    
    else:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h3>🛢️ Welcome to Novel Rock Physics Gas Detection</h3>
                <p>Upload your well log data to begin analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
            **Supported Data Format:**
            - CSV file with well log data
            - Required columns: Vp, Vs, RHO, PHI
            - Optional columns: GR, RT, VCLAY, SW, DEPTH
            """)
            
            # Example data download
            st.markdown("---")
            st.markdown("**Need example data?**")
            
            # Create example data
            example_data = pd.DataFrame({
                'DEPTH': np.arange(0, 1000, 2),
                'Vp': np.random.uniform(2500, 4000, 500),
                'Vs': np.random.uniform(1200, 2500, 500),
                'RHO': np.random.uniform(2.0, 2.6, 500),
                'PHI': np.random.uniform(0.1, 0.35, 500),
                'GR': np.random.uniform(20, 120, 500),
                'VCLAY': np.random.uniform(0.0, 0.4, 500),
                'SW': np.random.uniform(0.2, 1.0, 500)
            })
            
            csv = example_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Example Data",
                data=csv,
                file_name="example_well_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
