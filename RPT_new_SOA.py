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

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'example_data' not in st.session_state:
    st.session_state.example_data = None
if 'use_example_data' not in st.session_state:
    st.session_state.use_example_data = False
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Data Input")
    
    # File uploader - store in session state
    uploaded_file = st.file_uploader(
        "Upload Well Log CSV File", 
        type=['csv'],
        help="Upload CSV file with columns: Vp, Vs, RHO, PHI, GR, RT, VCLAY, SW, DEPTH"
    )
    
    # Store uploaded file in session state
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.use_example_data = False
    
    # Example data button
    if st.button("📊 Load Example Data", use_container_width=True):
        # Create synthetic example data
        np.random.seed(42)
        n_samples = 500
        
        example_data = pd.DataFrame({
            'DEPTH': np.arange(0, n_samples * 2, 2),
            'Vp': np.random.uniform(2500, 4000, n_samples),
            'Vs': np.random.uniform(1200, 2500, n_samples),
            'RHO': np.random.uniform(2.0, 2.6, n_samples),
            'PHI': np.random.uniform(0.1, 0.35, n_samples),
            'GR': np.random.uniform(20, 120, n_samples),
            'VCLAY': np.random.uniform(0.0, 0.4, n_samples),
            'SW': np.random.uniform(0.2, 1.0, n_samples)
        })
        
        # Add some gas zones
        gas_zones = (example_data['DEPTH'] > 300) & (example_data['DEPTH'] < 400)
        example_data.loc[gas_zones, 'Vp'] *= 0.85  # Velocity drop for gas
        example_data.loc[gas_zones, 'Vs'] *= 0.95
        example_data.loc[gas_zones, 'SW'] = np.random.uniform(0.1, 0.4, np.sum(gas_zones))
        
        # Save to session state
        st.session_state.example_data = example_data
        st.session_state.use_example_data = True
        st.session_state.uploaded_file = None
    
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
        st.rerun()

# Main content - Novel Rock Physics Class
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
        
        self.data = None
        self.results = {}
        
    def load_data(self, data):
        """Load and prepare well log data."""
        if isinstance(data, (pd.DataFrame, str)):
            if isinstance(data, str):
                self.data = pd.read_csv(data)
            else:
                self.data = data.copy()
        else:
            raise ValueError("Data must be a DataFrame or file path")
        
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
            'PHIT': 'PHI', 'PHIE': 'PHI', 'POR': 'PHI', 'POROSITY': 'PHI',
            'GR': 'GR', 'GAMMA': 'GR', 'GAMMA_RAY': 'GR',
            'RT': 'RT', 'RES': 'RT', 'RESISTIVITY': 'RT',
            'VSH': 'VCLAY', 'SH': 'VCLAY', 'CLAY': 'VCLAY',
            'SW': 'SW', 'SWT': 'SW', 'WATER_SAT': 'SW',
            'DEPTH': 'DEPTH', 'DEPT': 'DEPTH', 'MD': 'DEPTH'
        }
        
        new_columns = []
        for col in self.data.columns:
            col_upper = col.upper()
            if col_upper in column_mapping:
                new_columns.append(column_mapping[col_upper])
            else:
                new_columns.append(col)
        
        self.data.columns = new_columns
    
    def _process_data(self):
        """Process and validate data."""
        # Ensure required columns
        required = ['Vp', 'Vs', 'RHO', 'PHI']
        
        # Check if we have the basics
        missing_required = [col for col in required if col not in self.data.columns]
        if missing_required:
            st.warning(f"Missing columns: {missing_required}. Attempting to estimate...")
            
            # Try to estimate missing columns
            if 'Vp' in missing_required and 'DT' in self.data.columns:
                self.data['Vp'] = 304.8 / self.data['DT']  # Convert from DT to Vp in m/s
            
            if 'Vs' in missing_required and 'DTS' in self.data.columns:
                self.data['Vs'] = 304.8 / self.data['DTS']  # Convert from DTS to Vs in m/s
            elif 'Vs' in missing_required and 'Vp' in self.data.columns:
                # Estimate Vs from Vp using empirical relation
                self.data['Vs'] = 0.804 * self.data['Vp'] - 0.856
            
            if 'RHO' in missing_required and 'Vp' in self.data.columns:
                # Gardner's relation
                self.data['RHO'] = 310 * (self.data['Vp']/1000) ** 0.25
        
        # Now check again
        missing_required = [col for col in required if col not in self.data.columns]
        if missing_required:
            raise ValueError(f"Cannot proceed without: {missing_required}")
        
        # Ensure numeric data
        for col in required:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Remove rows with NaN in required columns
        self.data = self.data.dropna(subset=required)
        
        # Convert units if needed
        if self.data['Vp'].mean() < 1000:
            self.data['Vp'] = self.data['Vp'] * 1000
        if self.data['Vs'].mean() < 500:
            self.data['Vs'] = self.data['Vs'] * 1000
        if self.data['RHO'].mean() > 3000:
            self.data['RHO'] = self.data['RHO'] / 1000
        
        # Calculate derived properties
        self.data['Ip'] = self.data['Vp'] * self.data['RHO']
        self.data['Is'] = self.data['Vs'] * self.data['RHO']
        self.data['Vp_Vs'] = self.data['Vp'] / self.data['Vs']
        
        # Avoid division by zero
        self.data['Vp_Vs'] = self.data['Vp_Vs'].replace([np.inf, -np.inf], np.nan)
        self.data = self.data.dropna(subset=['Vp_Vs'])
        
        # Poisson's ratio
        mask = (self.data['Vp_Vs']**2 - 1) != 0
        self.data.loc[mask, 'PR'] = (0.5 * (self.data.loc[mask, 'Vp_Vs']**2 - 2)) / \
                                   (self.data.loc[mask, 'Vp_Vs']**2 - 1)
        
        # Lame parameters (in GPa)
        self.data['MU'] = self.data['RHO'] * self.data['Vs']**2 / 1e9
        self.data['LAMBDA'] = self.data['RHO'] * self.data['Vp']**2 / 1e9 - 2 * self.data['MU']
        self.data['MU_RHO'] = self.data['MU'] * self.data['RHO']
        self.data['LAMBDA_RHO'] = self.data['LAMBDA'] * self.data['RHO']
        
        # Clean data - remove physically impossible values
        mask = (
            (self.data['Vp'] > 1000) & (self.data['Vp'] < 8000) &
            (self.data['Vs'] > 500) & (self.data['Vs'] < 5000) &
            (self.data['RHO'] > 1.8) & (self.data['RHO'] < 3.0) &
            (self.data['PHI'] >= 0) & (self.data['PHI'] <= 0.5)
        )
        
        self.data = self.data[mask].reset_index(drop=True)
        
        if len(self.data) == 0:
            raise ValueError("No valid data after cleaning. Check input values.")
    
    def differential_effective_medium(self, K0, G0, phi, aspect_ratios, Sw=1.0):
        """DEM theory implementation with robust calculations."""
        try:
            # Fluid properties
            Kf = 1/(Sw/self.physical_constants['K_water'] + 
                   (1-Sw)/self.physical_constants['K_gas'])
            rhof = Sw * self.physical_constants['RHO_water'] + \
                   (1-Sw) * self.physical_constants['RHO_gas']
            
            # Initialize
            K_eff, G_eff = K0, G0
            steps = min(100, int(phi * 1000) + 10)  # Adaptive steps
            dphi = phi / steps
            phi_current = 0.0
            
            for _ in range(steps):
                K_sum, G_sum = 0, 0
                n_ratios = len(aspect_ratios)
                
                for alpha in aspect_ratios:
                    # Protect against invalid alpha
                    alpha = max(0.001, min(0.999, alpha))
                    
                    # Simplified Eshelby tensor for spherical inclusions
                    # For now, use simplified approach
                    P = (Kf - K_eff) / (Kf + 4/3 * G_eff)
                    Q = (Kf - G_eff) / (Kf + G_eff)
                    
                    K_sum += -K_eff * P / n_ratios
                    G_sum += -G_eff * Q / n_ratios
                
                # Update
                K_eff += K_sum * dphi / max(0.001, 1 - phi_current)
                G_eff += G_sum * dphi / max(0.001, 1 - phi_current)
                phi_current += dphi
            
            # Calculate velocities
            rho_eff = (1 - phi) * 2.65 + phi * rhof  # Simplified density
            Vp = np.sqrt(max(0.1, (K_eff + 4/3 * G_eff) / rho_eff)) * 1000
            Vs = np.sqrt(max(0.1, G_eff / rho_eff)) * 1000
            
            return Vp, Vs, rho_eff
            
        except Exception as e:
            st.warning(f"DEM calculation error: {e}")
            # Return safe defaults
            return 3000, 1500, 2.3
    
    def classify_rock_fabric(self, n_clusters=3):
        """Classify rock fabric using clustering with error handling."""
        try:
            # Prepare features
            features_list = []
            feature_names = []
            
            # Always include these if available
            for feat in ['PHI', 'Vp', 'Vp_Vs']:
                if feat in self.data.columns:
                    features_list.append(self.data[feat].values)
                    feature_names.append(feat)
            
            if len(features_list) < 2:
                # If not enough features, use simple approach
                st.warning("Not enough features for clustering. Using default clusters.")
                clusters = np.zeros(len(self.data), dtype=int)
                aspect_ratios = [np.array([0.1, 0.3, 0.6])]  # Default aspect ratios
                return clusters, aspect_ratios
            
            # Combine features
            X = np.column_stack(features_list)
            
            # Handle NaN values
            X = np.nan_to_num(X)
            
            # Normalize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters if not specified
            if n_clusters > len(X_scaled):
                n_clusters = min(3, len(X_scaled))
            
            # Use KMeans (more stable than GMM for this)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Assign aspect ratios based on clusters
            aspect_ratio_map = []
            for i in range(n_clusters):
                cluster_mask = clusters == i
                if np.sum(cluster_mask) > 0:
                    cluster_data = self.data[cluster_mask]
                    
                    # Calculate cluster statistics
                    avg_phi = cluster_data['PHI'].mean() if 'PHI' in cluster_data.columns else 0.2
                    avg_vp_vs = cluster_data['Vp_Vs'].mean() if 'Vp_Vs' in cluster_data.columns else 1.8
                    
                    # Assign aspect ratios based on characteristics
                    if avg_phi > 0.25:
                        # High porosity -> soft sand
                        ar_set = np.random.uniform(0.05, 0.15, 3)
                    elif avg_vp_vs > 1.8:
                        # High Vp/Vs -> intermediate
                        ar_set = np.random.uniform(0.1, 0.3, 3)
                    else:
                        # Low Vp/Vs -> stiff
                        ar_set = np.random.uniform(0.5, 0.8, 3)
                    
                    aspect_ratio_map.append(ar_set)
                else:
                    # Default aspect ratios for empty clusters
                    aspect_ratio_map.append(np.array([0.1, 0.3, 0.6]))
            
            self.data['FABRIC_CLUSTER'] = clusters
            self.fabric_aspect_ratios = aspect_ratio_map
            
            return clusters, aspect_ratio_map
            
        except Exception as e:
            st.warning(f"Clustering error: {e}. Using single cluster.")
            clusters = np.zeros(len(self.data), dtype=int)
            aspect_ratios = [np.array([0.1, 0.3, 0.6])]
            self.data['FABRIC_CLUSTER'] = clusters
            self.fabric_aspect_ratios = aspect_ratios
            return clusters, aspect_ratios
    
    def _calculate_rpt_probability(self):
        """Calculate RPT-based probability safely."""
        n_samples = len(self.data)
        prob = np.zeros(n_samples)
        
        if 'Ip' not in self.data.columns or 'Vp_Vs' not in self.data.columns:
            return prob
        
        # Calculate distance to gas trend
        gas_trend_vp_vs = 1.6  # Typical for gas sands
        
        for i in range(n_samples):
            try:
                Vp_Vs = self.data.iloc[i]['Vp_Vs']
                # Simple probability based on Vp/Vs ratio
                if Vp_Vs < gas_trend_vp_vs:
                    prob[i] = 1 - (Vp_Vs / gas_trend_vp_vs)
                else:
                    prob[i] = 0
            except:
                prob[i] = 0
        
        return np.clip(prob, 0, 1)
    
    def _ml_gas_classification(self):
        """ML-based gas classification with robust error handling."""
        n_samples = len(self.data)
        
        if n_samples < 50:
            st.warning("Not enough samples for ML classification. Returning zeros.")
            return np.zeros(n_samples)
        
        try:
            # Feature engineering
            features_list = []
            feature_names = []
            
            # Basic features
            basic_features = ['Vp', 'Vs', 'RHO', 'Vp_Vs', 'Ip', 'LAMBDA_RHO', 'MU_RHO']
            for feat in basic_features:
                if feat in self.data.columns:
                    features_list.append(self.data[feat].values)
                    feature_names.append(feat)
            
            # Additional features if available
            for feat in ['PHI', 'VCLAY', 'GR']:
                if feat in self.data.columns:
                    features_list.append(self.data[feat].values)
                    feature_names.append(feat)
            
            if len(features_list) < 3:
                st.warning("Not enough features for ML. Returning zeros.")
                return np.zeros(n_samples)
            
            # Create feature matrix
            X = np.column_stack(features_list)
            
            # Handle NaN values
            X = np.nan_to_num(X)
            
            # Create synthetic labels for training
            y = np.zeros(n_samples)
            
            # Simple heuristic: low Vp/Vs and low Lambda-Rho suggest gas
            if 'Vp_Vs' in self.data.columns and 'LAMBDA_RHO' in self.data.columns:
                vp_vs_ratio = self.data['Vp_Vs'].values
                lambda_rho = self.data['LAMBDA_RHO'].values
                
                # Gas candidates
                gas_mask = (vp_vs_ratio < 1.8) & (lambda_rho < np.percentile(lambda_rho, 30))
                # Brine candidates
                brine_mask = (vp_vs_ratio > 2.0) & (lambda_rho > np.percentile(lambda_rho, 70))
                
                y[gas_mask] = 1
                y[brine_mask] = 0
                
                train_mask = gas_mask | brine_mask
                
                if np.sum(train_mask) < 20:
                    st.warning("Not enough labeled samples for ML training. Using simple heuristic.")
                    # Use simple Vp/Vs based probability
                    prob = np.clip(1.8 - vp_vs_ratio, 0, 1)
                    return prob
            
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=50, 
                random_state=42, 
                class_weight='balanced',
                max_depth=5
            )
            
            # Only train if we have enough labeled samples
            if 'train_mask' in locals() and np.sum(train_mask) >= 20:
                rf.fit(X[train_mask], y[train_mask])
                
                # Predict probabilities
                prob = rf.predict_proba(X)[:, 1]
            else:
                # Fallback to simple method
                if 'vp_vs_ratio' in locals():
                    prob = np.clip(1.8 - vp_vs_ratio, 0, 1)
                else:
                    prob = np.zeros(n_samples)
            
            return np.clip(prob, 0, 1)
            
        except Exception as e:
            st.warning(f"ML classification error: {e}. Using fallback method.")
            # Fallback to simple Vp/Vs method
            if 'Vp_Vs' in self.data.columns:
                vp_vs_ratio = self.data['Vp_Vs'].values
                return np.clip(1.8 - vp_vs_ratio, 0, 1)
            else:
                return np.zeros(n_samples)
    
    def _calculate_dispersion_index(self):
        """Calculate dispersion index safely."""
        if 'LAMBDA' not in self.data.columns or 'MU' not in self.data.columns:
            return np.zeros(len(self.data))
        
        try:
            lambda_mu_ratio = self.data['LAMBDA'] / (self.data['MU'] + 1e-6)
            dispersion = 1 - (lambda_mu_ratio - lambda_mu_ratio.min()) / \
                         max(1e-6, (lambda_mu_ratio.max() - lambda_mu_ratio.min()))
            return np.nan_to_num(dispersion)
        except:
            return np.zeros(len(self.data))
    
    def multi_model_gas_detection(self, use_dispersion=True, use_ml=True, 
                                 confidence_threshold=0.7, n_clusters=3):
        """Main gas detection algorithm with robust error handling."""
        try:
            n_samples = len(self.data)
            
            if n_samples == 0:
                raise ValueError("No data available for analysis")
            
            # Step 1: Rock fabric classification
            with st.spinner("Classifying rock fabric..."):
                clusters, aspect_ratios = self.classify_rock_fabric(n_clusters)
            
            # Step 2: Initialize results
            method_probs = []
            method_names = []
            uncertainties = []
            
            # Method 1: Vp/Vs cutoff
            with st.spinner("Applying Vp/Vs method..."):
                if 'Vp' in self.data.columns and 'Vs' in self.data.columns:
                    vp_vs_ratio = self.data['Vp'] / self.data['Vs']
                    prob1 = np.clip(1.8 - vp_vs_ratio, 0, 1)
                    method_probs.append(prob1)
                    method_names.append("Vp/Vs Cutoff")
                    uncertainties.append(0.2)
            
            # Method 2: Lambda-Rho
            with st.spinner("Applying Lambda-Rho method..."):
                if 'LAMBDA_RHO' in self.data.columns:
                    LMR = self.data['LAMBDA_RHO']
                    if LMR.max() > LMR.min():
                        prob2 = 1 - (LMR - LMR.min()) / (LMR.max() - LMR.min())
                    else:
                        prob2 = np.zeros(n_samples)
                    method_probs.append(prob2)
                    method_names.append("Lambda-Rho")
                    uncertainties.append(0.15)
            
            # Method 3: Adaptive DEM
            with st.spinner("Running Adaptive DEM..."):
                prob3 = np.zeros(n_samples)
                for i in range(min(n_samples, 1000)):  # Limit for performance
                    if i < len(clusters) and clusters[i] < len(aspect_ratios):
                        ar_set = aspect_ratios[clusters[i]]
                    else:
                        ar_set = [0.1, 0.3, 0.6]
                    
                    phi = self.data.iloc[i]['PHI'] if i < len(self.data) else 0.2
                    Sw = self.data.iloc[i]['SW'] if 'SW' in self.data.columns and i < len(self.data) else 1.0
                    
                    # Simplified DEM calculation
                    vp_water, _, _ = self.differential_effective_medium(37, 44, phi, ar_set, 1.0)
                    vp_current, _, _ = self.differential_effective_medium(37, 44, phi, ar_set, Sw)
                    
                    if vp_water > 0:
                        velocity_drop = (vp_water - vp_current) / vp_water
                        prob3[i] = np.clip(velocity_drop * 3, 0, 1)
                
                method_probs.append(prob3)
                method_names.append("Adaptive DEM")
                uncertainties.append(0.12)
            
            # Method 4: RPT distance
            with st.spinner("Calculating RPT distances..."):
                prob4 = self._calculate_rpt_probability()
                method_probs.append(prob4)
                method_names.append("RPT Distance")
                uncertainties.append(0.1)
            
            # Method 5: ML classifier (optional)
            if use_ml:
                with st.spinner("Running ML classification..."):
                    prob5 = self._ml_gas_classification()
                    method_probs.append(prob5)
                    method_names.append("ML Classifier")
                    uncertainties.append(0.08)
            
            # Ensure all probabilities have same length
            for i in range(len(method_probs)):
                if len(method_probs[i]) != n_samples:
                    method_probs[i] = np.zeros(n_samples)
            
            # Weighted combination
            uncertainties = np.array(uncertainties)
            weights = 1 / (uncertainties + 1e-6)
            weights = weights / weights.sum()
            
            # Combine probabilities
            combined_prob = np.zeros(n_samples)
            for i, prob in enumerate(method_probs):
                combined_prob += weights[i] * prob
            
            # Apply dispersion if requested
            if use_dispersion:
                dispersion = self._calculate_dispersion_index()
                combined_prob = combined_prob * (1 + 0.3 * dispersion)
                combined_prob = np.clip(combined_prob, 0, 1)
            
            # Calculate confidence based on method agreement
            if len(method_probs) > 1:
                method_std = np.std(method_probs, axis=0)
                confidence = 1 - np.clip(method_std / 0.5, 0, 1)
            else:
                confidence = np.ones(n_samples) * 0.5
            
            # Apply physical constraints
            for i in range(n_samples):
                if i < len(self.data):
                    phi = self.data.iloc[i]['PHI']
                    max_effect = min(phi * 2, 0.9)
                    combined_prob[i] = min(combined_prob[i], max_effect)
            
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
            
        except Exception as e:
            st.error(f"Error in gas detection: {str(e)}")
            # Return safe defaults
            n_samples = len(self.data)
            return np.zeros(n_samples), np.ones(n_samples) * 0.5
    
    def _calculate_statistics(self, threshold):
        """Calculate detection statistics."""
        if 'gas_probability' not in self.results:
            return
        
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
                'hc_mean_phi': float(high_prob_data['PHI'].mean()) if 'PHI' in high_prob_data.columns else 0,
                'hc_mean_vp_vs': float(high_prob_data['Vp_Vs'].mean()) if 'Vp_Vs' in high_prob_data.columns else 0,
                'hc_mean_ip': float(high_prob_data['Ip'].mean()) if 'Ip' in high_prob_data.columns else 0,
            })
        
        self.results['statistics'] = stats
        return stats

def create_dashboard(detector, confidence_threshold):
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
            if 'results' in detector.__dict__ and 'statistics' in detector.results:
                stats = detector.results['statistics']
                st.metric(
                    "High-Confidence Gas Zones",
                    f"{stats['high_confidence_samples']:,}",
                    f"{stats['high_confidence_percentage']:.1f}%",
                    help="Samples with gas probability > threshold and confidence > 0.6"
                )
        
        with col3:
            if 'results' in detector.__dict__ and 'statistics' in detector.results:
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
        
        if 'PHI' in detector.data.columns:
            fig.add_trace(
                go.Scatter(x=detector.data['PHI'], y=depth, mode='lines',
                          name='PHI', line=dict(color='blue', width=1)),
                row=1, col=2
            )
        
        if 'Vp_Vs' in detector.data.columns:
            fig.add_trace(
                go.Scatter(x=detector.data['Vp_Vs'], y=depth, mode='lines',
                          name='Vp/Vs', line=dict(color='purple', width=1)),
                row=1, col=3
            )
        
        if 'RHO' in detector.data.columns:
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
        if 'results' not in detector.__dict__ or 'gas_probability' not in detector.results:
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
            if 'gas_confidence' in detector.results:
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
            if 'method_names' in detector.results and 'method_weights' in detector.results:
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
            if 'high_confidence_gas' in detector.results:
                hc_zones = detector.results['high_confidence_gas']
                if np.any(hc_zones):
                    hc_data = detector.data[hc_zones]
                    
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("Count", f"{np.sum(hc_zones):,}")
                        if 'PHI' in hc_data.columns:
                            st.metric("Avg Porosity", f"{hc_data['PHI'].mean():.3f}")
                    
                    with metrics_col2:
                        st.metric("Percentage", f"{np.mean(hc_zones)*100:.1f}%")
                        if 'Vp_Vs' in hc_data.columns:
                            st.metric("Avg Vp/Vs", f"{hc_data['Vp_Vs'].mean():.2f}")
                else:
                    st.info("No high-confidence gas zones detected")
        
        # Method probabilities
        st.markdown("#### Individual Method Probabilities")
        if 'method_probabilities' in detector.results and 'method_names' in detector.results:
            method_fig = go.Figure()
            
            for i, (name, probs) in enumerate(zip(detector.results['method_names'], 
                                                detector.results['method_probabilities'])):
                if i < 5:  # Limit to first 5 methods for clarity
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
        
        if plot_type == "Vp/Vs vs Ip" and 'Ip' in detector.data.columns and 'Vp_Vs' in detector.data.columns:
            fig = px.scatter(
                detector.data,
                x='Ip',
                y='Vp_Vs',
                color=detector.results.get('gas_probability', None),
                color_continuous_scale='RdBu_r',
                title="Vp/Vs vs Acoustic Impedance",
                labels={'Ip': 'Acoustic Impedance', 'Vp_Vs': 'Vp/Vs Ratio'}
            )
            
            if 'results' in detector.__dict__:
                fig.add_hline(y=1.8, line_dash="dash", line_color="red", 
                             annotation_text="Gas Sand Threshold")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Lambda-Rho vs Mu-Rho" and 'LAMBDA_RHO' in detector.data.columns and 'MU_RHO' in detector.data.columns:
            fig = px.scatter(
                detector.data,
                x='LAMBDA_RHO',
                y='MU_RHO',
                color=detector.results.get('gas_probability', None),
                color_continuous_scale='RdBu_r',
                title="Lambda-Rho vs Mu-Rho Crossplot",
                labels={'LAMBDA_RHO': 'λρ', 'MU_RHO': 'μρ'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Vp vs Porosity" and 'PHI' in detector.data.columns and 'Vp' in detector.data.columns:
            fig = px.scatter(
                detector.data,
                x='PHI',
                y='Vp',
                color=detector.results.get('gas_probability', None),
                color_continuous_scale='RdBu_r',
                title="P-wave Velocity vs Porosity",
                labels={'PHI': 'Porosity', 'Vp': 'P-wave Velocity (m/s)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Rock Fabric Clusters" and 'FABRIC_CLUSTER' in detector.data.columns:
            fig = px.scatter(
                detector.data,
                x='Ip' if 'Ip' in detector.data.columns else 'Vp',
                y='Vp_Vs' if 'Vp_Vs' in detector.data.columns else 'Vs',
                color=detector.data['FABRIC_CLUSTER'].astype(str),
                title="Rock Fabric Clusters",
                labels={'FABRIC_CLUSTER': 'Cluster'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Gas Probability Map" and 'results' in detector.__dict__:
            fig = px.density_contour(
                detector.data,
                x='Ip' if 'Ip' in detector.data.columns else 'Vp',
                y='Vp_Vs' if 'Vp_Vs' in detector.data.columns else 'Vs',
                z=detector.results['gas_probability'],
                title="Gas Probability Density Map"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Required data not available for this plot type")
    
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
                        if 'PHI' in cluster_data.columns:
                            st.write(f"Average Porosity: {cluster_data['PHI'].mean():.3f}")
                        if 'Vp_Vs' in cluster_data.columns:
                            st.write(f"Average Vp/Vs: {cluster_data['Vp_Vs'].mean():.2f}")
                        if 'Ip' in cluster_data.columns:
                            st.write(f"Average Ip: {cluster_data['Ip'].mean():.0f}")
        
        with col2:
            st.markdown("#### Elastic Moduli Analysis")
            
            if 'LAMBDA' in detector.data.columns and 'MU' in detector.data.columns:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Bulk Modulus (K)', 'Shear Modulus (μ)'),
                    horizontal_spacing=0.1
                )
                
                # Bulk modulus K = lambda + 2/3 * mu
                K = detector.data['LAMBDA'] + 2/3 * detector.data['MU']
                fig.add_trace(
                    go.Histogram(x=K, name='K', nbinsx=30),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(x=detector.data['MU'], name='μ', nbinsx=30),
                    row=1, col=2
                )
                
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Elastic moduli not available")
    
    with tab5:
        st.markdown("### 📋 Detailed Statistics")
        
        if 'results' in detector.__dict__ and 'statistics' in detector.results:
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
            
            # Export options
            st.markdown("#### Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Create results dataframe
                results_df = detector.data.copy()
                if 'gas_probability' in detector.results:
                    results_df['GAS_PROBABILITY'] = detector.results['gas_probability']
                if 'gas_confidence' in detector.results:
                    results_df['GAS_CONFIDENCE'] = detector.results['gas_confidence']
                if 'high_confidence_gas' in detector.results:
                    results_df['HIGH_CONFIDENCE_GAS'] = detector.results['high_confidence_gas']
                if 'fabric_clusters' in detector.results:
                    results_df['FABRIC_CLUSTER'] = detector.results['fabric_clusters']
                
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="gas_detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                if st.button("📊 Generate Report PDF", use_container_width=True):
                    st.info("PDF report generation coming soon!")
        
        else:
            st.info("Run analysis to see statistics")

def main():
    """Main Streamlit app."""
    
    # Check if we have data to load
    data_available = False
    
    # Check for uploaded file
    if st.session_state.uploaded_file is not None:
        data_available = True
        data_source = "uploaded"
        data_to_load = st.session_state.uploaded_file
    
    # Check for example data
    elif st.session_state.use_example_data and st.session_state.example_data is not None:
        data_available = True
        data_source = "example"
        data_to_load = st.session_state.example_data
    
    # Main content based on data availability
    if data_available:
        # Load and process data
        with st.spinner("Loading and processing data..."):
            try:
                if st.session_state.detector is None:
                    st.session_state.detector = NovelRockPhysicsGasDetector()
                
                if data_source == "uploaded":
                    df = pd.read_csv(data_to_load)
                    st.session_state.detector.load_data(df)
                    st.success(f"✅ Data loaded successfully: {len(st.session_state.detector.data):,} samples")
                elif data_source == "example":
                    st.session_state.detector.load_data(data_to_load)
                    st.success(f"✅ Example data loaded: {len(st.session_state.detector.data):,} samples")
                
                st.session_state.data_loaded = True
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.session_state.data_loaded = False
                return
        
        # Show data info if loaded
        if st.session_state.data_loaded and st.session_state.detector is not None:
            # Display data summary
            with st.expander("📊 Data Summary", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Samples", len(st.session_state.detector.data))
                with col2:
                    st.metric("Columns", len(st.session_state.detector.data.columns))
                with col3:
                    numeric_cols = st.session_state.detector.data.select_dtypes(include=[np.number]).columns
                    st.metric("Numeric Columns", len(numeric_cols))
                
                # Show column info
                st.write("**Available Columns:**")
                cols = st.session_state.detector.data.columns.tolist()
                cols_per_row = 4
                for i in range(0, len(cols), cols_per_row):
                    st.code(" | ".join(cols[i:i+cols_per_row]))
            
            # Run analysis if requested
            if st.session_state.run_analysis:
                with st.spinner("Running advanced gas detection analysis..."):
                    # Create progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("Step 1/5: Initializing analysis...")
                        progress_bar.progress(20)
                        
                        status_text.text("Step 2/5: Classifying rock fabric...")
                        progress_bar.progress(40)
                        
                        status_text.text("Step 3/5: Running multi-model detection...")
                        progress_bar.progress(60)
                        
                        # Get parameters from sidebar (they should be in scope)
                        # We'll use default values if not available
                        use_dispersion_val = True
                        use_ml_val = True
                        confidence_threshold_val = 0.7
                        n_clusters_val = 3
                        
                        # Try to get from sidebar widgets (if they exist)
                        try:
                            use_dispersion_val = use_dispersion
                        except:
                            pass
                        
                        try:
                            use_ml_val = use_ml
                        except:
                            pass
                        
                        try:
                            confidence_threshold_val = confidence_threshold
                        except:
                            pass
                        
                        try:
                            n_clusters_val = n_clusters
                        except:
                            pass
                        
                        gas_prob, confidence = st.session_state.detector.multi_model_gas_detection(
                            use_dispersion=use_dispersion_val,
                            use_ml=use_ml_val,
                            confidence_threshold=confidence_threshold_val,
                            n_clusters=n_clusters_val
                        )
                        
                        status_text.text("Step 4/5: Calculating statistics...")
                        progress_bar.progress(80)
                        
                        status_text.text("Step 5/5: Finalizing results...")
                        progress_bar.progress(100)
                        
                        st.success("✅ Analysis completed successfully!")
                        st.session_state.analysis_run = True
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Show dashboard if analysis was run
            if st.session_state.analysis_run and st.session_state.detector is not None:
                # Get confidence threshold
                try:
                    confidence_threshold_val = confidence_threshold
                except:
                    confidence_threshold_val = 0.7
                
                create_dashboard(st.session_state.detector, confidence_threshold_val)
            
            # Show raw data preview if no analysis yet
            elif not st.session_state.analysis_run:
                st.markdown("### 📋 Data Preview")
                st.dataframe(st.session_state.detector.data.head(), use_container_width=True)
                
                # Quick stats
                st.markdown("### 📊 Quick Statistics")
                numeric_cols = st.session_state.detector.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_df = st.session_state.detector.data[numeric_cols].describe().T
                    st.dataframe(stats_df, use_container_width=True)
                
                # Show analysis button
                if st.button("▶️ Run Analysis Now", type="primary", use_container_width=True):
                    st.session_state.run_analysis = True
                    st.rerun()
    
    else:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h3>🛢️ Welcome to Novel Rock Physics Gas Detection</h3>
                <p>Upload your well log data or use example data to begin analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
            **Supported Data Format:**
            - CSV file with well log data
            - Required columns: Vp, Vs, RHO, PHI
            - Optional columns: GR, RT, VCLAY, SW, DEPTH
            
            **Key Features:**
            1. Adaptive DEM modeling with pore geometry classification
            2. Multi-model ensemble with uncertainty quantification
            3. Machine learning integration
            4. Interactive visualizations
            5. Export capabilities
            """)
            
            # Features in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Methods", "5")
            with col2:
                st.metric("Models", "Adaptive")
            with col3:
                st.metric("Visualizations", "Interactive")
            with col4:
                st.metric("Output", "CSV + Plots")

if __name__ == "__main__":
    main()
