import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from scipy import interpolate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
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
if 'depth_range' not in st.session_state:
    st.session_state.depth_range = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None

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
        st.rerun()
    
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

# Main content - Enhanced Novel Rock Physics Class
class EnhancedRockPhysicsGasDetector:
    """Advanced gas detection framework with adaptive DEM modeling and Avseth-Oddgard RPTs."""
    
    def __init__(self):
        # Physical constants
        self.physical_constants = {
            'K_quartz': 37.0, 'G_quartz': 44.0,
            'K_clay': 21.0, 'G_clay': 7.0,
            'K_calcite': 76.8, 'G_calcite': 32.0,
            'K_water': 2.25, 'K_gas': 0.05,
            'RHO_quartz': 2.65, 'RHO_clay': 2.70,
            'RHO_calcite': 2.71, 'RHO_water': 1.00, 
            'RHO_gas': 0.25,
            'critical_porosity': 0.40
        }
        
        self.data = None
        self.original_data = None
        self.results = {}
        self.rpt_templates = {}
        
    def load_data(self, data, depth_range=None):
        """Load and prepare well log data with optional depth range."""
        if isinstance(data, (pd.DataFrame, str)):
            if isinstance(data, str):
                self.original_data = pd.read_csv(data)
            else:
                self.original_data = data.copy()
        else:
            raise ValueError("Data must be a DataFrame or file path")
        
        # Store original data
        self.original_data = self.original_data.copy()
        
        # Apply depth range if specified
        if depth_range is not None and 'DEPTH' in self.original_data.columns:
            min_depth, max_depth = depth_range
            mask = (self.original_data['DEPTH'] >= min_depth) & (self.original_data['DEPTH'] <= max_depth)
            self.data = self.original_data[mask].copy()
        else:
            self.data = self.original_data.copy()
        
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
        
        # Elastic moduli
        self.data['K'] = self.data['LAMBDA'] + (2/3) * self.data['MU']
        
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
    
    def set_depth_range(self, min_depth, max_depth):
        """Set depth range for analysis."""
        if self.original_data is None:
            st.warning("No original data loaded. Cannot apply depth range.")
            return self.data
        
        if 'DEPTH' not in self.original_data.columns:
            st.warning("No DEPTH column in data. Cannot apply depth range.")
            return self.data
        
        mask = (self.original_data['DEPTH'] >= min_depth) & (self.original_data['DEPTH'] <= max_depth)
        self.data = self.original_data[mask].copy()
        
        # Re-process the filtered data
        self._process_data()
        
        return self.data
    
    def hertz_mindlin(self, phi_c=0.4, sigma=10.0, Cn=8.5):
        """Hertz-Mindlin contact theory."""
        nu = 0.12  # Poisson's ratio for quartz pack
        
        # Hertz-Mindlin moduli
        K_hm = (Cn**2 * (1 - phi_c)**2 * sigma**2 / 
                (18 * np.pi**2 * (1 - nu)**2))**(1/3)
        G_hm = (3 * Cn**2 * (1 - phi_c)**2 * sigma**2 * (5 - 4*nu) / 
                (2 * np.pi**2 * (1 - nu)**2 * (5 * (2 - nu))))**(1/3)
        
        return K_hm, G_hm
    
    def hashin_shtrikman_bounds(self, phi, K1, G1, K2, G2, rho1, rho2):
        """Hashin-Shtrikman bounds for two-phase composite."""
        # Upper bound (stiff inclusions)
        K_upper = K1 + phi / ((1/(K2 - K1)) + ((1 - phi)/(K1 + 4/3 * G1)))
        G_upper = G1 + phi / ((1/(G2 - G1)) + (2*(1-phi)*(K1+2*G1)/(5*G1*(K1+4/3*G1))))
        
        # Lower bound (soft inclusions)
        K_lower = K2 + (1-phi) / ((1/(K1 - K2)) + (phi/(K2 + 4/3 * G2)))
        G_lower = G2 + (1-phi) / ((1/(G1 - G2)) + (2*phi*(K2+2*G2)/(5*G2*(K2+4/3*G2))))
        
        # Density
        rho = (1 - phi) * rho1 + phi * rho2
        
        return {
            'K_upper': K_upper, 'G_upper': G_upper,
            'K_lower': K_lower, 'G_lower': G_lower,
            'rho': rho
        }
    
    def gassmann_fluid_substitution(self, K_dry, G_dry, phi, K_min, Kf, rho_min, rhof):
        """Gassmann fluid substitution."""
        # Gassmann equation
        K_sat = K_dry + (1 - K_dry/K_min)**2 / (
            phi/Kf + (1-phi)/K_min - K_dry/K_min**2
        )
        
        G_sat = G_dry  # Shear modulus unchanged
        
        # Density
        rho_sat = rho_min * (1 - phi) + rhof * phi
        
        return K_sat, G_sat, rho_sat
    
    def generate_avseth_oddgard_rpts(self, phi_range=(0.05, 0.35), Sw_range=(0.0, 1.0),
                                    vclay=0.2, cement=0.0, n_points=30):
        """
        Generate Avseth-Oddgard style Rock Physics Templates.
        Based on the "Quantitative Seismic Interpretation" approach.
        """
        print("Generating Avseth-Oddgard RPTs...")
        
        # Create meshgrid
        phi_values = np.linspace(phi_range[0], phi_range[1], n_points)
        Sw_values = np.linspace(Sw_range[0], Sw_range[1], n_points)
        
        # Initialize templates
        templates = {
            'soft_sand': {'Ip': np.zeros((n_points, n_points)),
                         'Vp_Vs': np.zeros((n_points, n_points)),
                         'Phi': np.zeros((n_points, n_points)),
                         'Sw': np.zeros((n_points, n_points))},
            'stiff_sand': {'Ip': np.zeros((n_points, n_points)),
                          'Vp_Vs': np.zeros((n_points, n_points)),
                          'Phi': np.zeros((n_points, n_points)),
                          'Sw': np.zeros((n_points, n_points))},
            'constant_cement': {'Ip': np.zeros((n_points, n_points)),
                               'Vp_Vs': np.zeros((n_points, n_points)),
                               'Phi': np.zeros((n_points, n_points)),
                               'Sw': np.zeros((n_points, n_points))}
        }
        
        # Mineral mixture
        K_min = (1 - vclay) * self.physical_constants['K_quartz'] + \
                vclay * self.physical_constants['K_clay']
        G_min = (1 - vclay) * self.physical_constants['G_quartz'] + \
                vclay * self.physical_constants['G_clay']
        rho_min = (1 - vclay) * self.physical_constants['RHO_quartz'] + \
                  vclay * self.physical_constants['RHO_clay']
        
        # Generate templates for each model
        for i, phi in enumerate(phi_values):
            for j, Sw in enumerate(Sw_values):
                # Fluid properties
                Kf = 1/(Sw/self.physical_constants['K_water'] + 
                       (1-Sw)/self.physical_constants['K_gas'])
                rhof = Sw * self.physical_constants['RHO_water'] + \
                       (1-Sw) * self.physical_constants['RHO_gas']
                
                # 1. Soft Sand (uncemented) model
                # Hertz-Mindlin for dry frame
                K_hm, G_hm = self.hertz_mindlin(phi_c=0.4, sigma=10)
                
                # Apply Gassmann for fluid substitution
                K_sat_soft, G_sat_soft, rho_sat_soft = self.gassmann_fluid_substitution(
                    K_hm, G_hm, phi, K_min, Kf, rho_min, rhof
                )
                
                # Calculate velocities
                Vp_soft = np.sqrt((K_sat_soft + 4/3 * G_sat_soft) / rho_sat_soft) * 1000
                Vs_soft = np.sqrt(G_sat_soft / rho_sat_soft) * 1000
                
                templates['soft_sand']['Ip'][i, j] = Vp_soft * rho_sat_soft
                templates['soft_sand']['Vp_Vs'][i, j] = Vp_soft / Vs_soft
                templates['soft_sand']['Phi'][i, j] = phi
                templates['soft_sand']['Sw'][i, j] = Sw
                
                # 2. Stiff Sand (well-cemented) model
                # Use modified lower Hashin-Shtrikman bound
                hs_stiff = self.hashin_shtrikman_bounds(
                    phi, K_min, G_min, Kf, 0.1, rho_min, rhof
                )
                
                Vp_stiff = np.sqrt((hs_stiff['K_upper'] + 4/3 * hs_stiff['G_upper']) / hs_stiff['rho']) * 1000
                Vs_stiff = np.sqrt(hs_stiff['G_upper'] / hs_stiff['rho']) * 1000
                
                templates['stiff_sand']['Ip'][i, j] = Vp_stiff * hs_stiff['rho']
                templates['stiff_sand']['Vp_Vs'][i, j] = Vp_stiff / Vs_stiff
                templates['stiff_sand']['Phi'][i, j] = phi
                templates['stiff_sand']['Sw'][i, j] = Sw
                
                # 3. Constant Cement model
                # Simplified approach - intermediate between soft and stiff
                K_cc = (K_hm + hs_stiff['K_upper']) / 2
                G_cc = (G_hm + hs_stiff['G_upper']) / 2
                
                # Apply Gassmann
                K_sat_cc, G_sat_cc, rho_sat_cc = self.gassmann_fluid_substitution(
                    K_cc, G_cc, phi, K_min, Kf, rho_min, rhof
                )
                
                Vp_cc = np.sqrt((K_sat_cc + 4/3 * G_sat_cc) / rho_sat_cc) * 1000
                Vs_cc = np.sqrt(G_sat_cc / rho_sat_cc) * 1000
                
                templates['constant_cement']['Ip'][i, j] = Vp_cc * rho_sat_cc
                templates['constant_cement']['Vp_Vs'][i, j] = Vp_cc / Vs_cc
                templates['constant_cement']['Phi'][i, j] = phi
                templates['constant_cement']['Sw'][i, j] = Sw
        
        self.rpt_templates = templates
        self.rpt_params = {
            'phi_range': phi_range,
            'Sw_range': Sw_range,
            'vclay': vclay,
            'cement': cement
        }
        
        print("Avseth-Oddgard RPTs generated successfully.")
        return templates
    
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
        
        if 'Vp_Vs' not in self.data.columns:
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
            
            # Basic features - check each one exists
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
            else:
                # If we don't have the columns for heuristic, skip ML
                st.warning("Missing columns for ML training. Using Vp/Vs method.")
                if 'Vp_Vs' in self.data.columns:
                    vp_vs_ratio = self.data['Vp_Vs'].values
                    return np.clip(1.8 - vp_vs_ratio, 0, 1)
                else:
                    return np.zeros(n_samples)
            
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=50, 
                random_state=42, 
                class_weight='balanced',
                max_depth=5
            )
            
            # Only train if we have enough labeled samples
            if np.sum(train_mask) >= 20:
                rf.fit(X[train_mask], y[train_mask])
                
                # Predict probabilities - handle both binary and multi-class
                if hasattr(rf, 'predict_proba'):
                    proba = rf.predict_proba(X)
                    # Check shape of proba
                    if len(proba.shape) == 2 and proba.shape[1] > 1:
                        # Multi-class case
                        prob = proba[:, 1]  # Probability of class 1 (gas)
                    else:
                        # Binary case or single class
                        prob = proba.flatten()
                else:
                    # If predict_proba not available, use decision function
                    prob = rf.predict(X)
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
                else:
                    # Add placeholder if column doesn't exist
                    prob2 = np.zeros(n_samples)
                    method_probs.append(prob2)
                    method_names.append("Lambda-Rho")
                    uncertainties.append(0.15)
            
            # Method 3: Adaptive DEM
            with st.spinner("Running Adaptive DEM..."):
                prob3 = np.zeros(n_samples)
                # Limit calculations for performance
                calc_samples = min(n_samples, 1000)
                step = max(1, n_samples // 1000)
                
                indices = range(0, n_samples, step)
                for idx in indices:
                    if idx < len(clusters) and clusters[idx] < len(aspect_ratios):
                        ar_set = aspect_ratios[clusters[idx]]
                    else:
                        ar_set = [0.1, 0.3, 0.6]
                    
                    phi = self.data.iloc[idx]['PHI'] if idx < len(self.data) else 0.2
                    Sw = self.data.iloc[idx]['SW'] if 'SW' in self.data.columns and idx < len(self.data) else 1.0
                    
                    # Simplified DEM calculation
                    # Use Hertz-Mindlin as baseline
                    K_hm, G_hm = self.hertz_mindlin(phi_c=phi, sigma=10)
                    
                    # Apply DEM effect based on aspect ratios
                    K_factor = 1.0
                    G_factor = 1.0
                    for ar in ar_set:
                        if ar < 0.2:  # Cracks
                            K_factor *= 0.8
                            G_factor *= 0.9
                        elif ar > 0.7:  # Stiff pores
                            K_factor *= 1.2
                            G_factor *= 1.1
                    
                    K_dry = K_hm * K_factor
                    G_dry = G_hm * G_factor
                    
                    # Fluid substitution
                    Kf = 1/(Sw/self.physical_constants['K_water'] + 
                           (1-Sw)/self.physical_constants['K_gas'])
                    
                    K_min = 37.0  # Quartz
                    K_sat, G_sat, rho_sat = self.gassmann_fluid_substitution(
                        K_dry, G_dry, phi, K_min, Kf, 2.65, 1.0
                    )
                    
                    vp_water = np.sqrt((K_sat + 4/3 * G_sat) / rho_sat) * 1000
                    
                    # For gas case
                    Kf_gas = self.physical_constants['K_gas']
                    K_sat_gas, G_sat_gas, rho_sat_gas = self.gassmann_fluid_substitution(
                        K_dry, G_dry, phi, K_min, Kf_gas, 2.65, 0.25
                    )
                    
                    vp_gas = np.sqrt((K_sat_gas + 4/3 * G_sat_gas) / rho_sat_gas) * 1000
                    
                    if vp_water > 0:
                        velocity_drop = (vp_water - vp_gas) / vp_water
                        # Adjust probability based on Sw
                        prob3[idx] = np.clip(velocity_drop * 3 * (1 - Sw), 0, 1)
                
                # Fill in missing values
                if step > 1:
                    # Interpolate for skipped samples
                    x = np.array(list(indices))
                    y = prob3[x]
                    f = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)
                    prob3 = f(np.arange(n_samples))
                
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
                'hc_mean_sw': float(high_prob_data['SW'].mean()) if 'SW' in high_prob_data.columns else 0,
                'hc_mean_vclay': float(high_prob_data['VCLAY'].mean()) if 'VCLAY' in high_prob_data.columns else 0,
            })
        
        self.results['statistics'] = stats
        return stats

def create_enhanced_dashboard(detector, confidence_threshold):
    """Create enhanced interactive dashboard with depth range selection and Avseth-Oddgard RPTs."""
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🛢️ Gas Detection", 
        "📊 Crossplots & RPTs", 
        "🔬 Rock Physics", 
        "📋 Statistics & Export"
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
        
        # Depth Range Selection
        st.markdown("### 🌡️ Depth Range Selection")
        if 'DEPTH' in detector.data.columns and detector.original_data is not None:
            min_depth = float(detector.original_data['DEPTH'].min())
            max_depth = float(detector.original_data['DEPTH'].max())
            current_min = float(detector.data['DEPTH'].min()) if len(detector.data) > 0 else min_depth
            current_max = float(detector.data['DEPTH'].max()) if len(detector.data) > 0 else max_depth
            
            col1, col2 = st.columns(2)
            with col1:
                new_min = st.number_input(
                    "Minimum Depth",
                    min_value=min_depth,
                    max_value=max_depth,
                    value=current_min,
                    step=10.0
                )
            with col2:
                new_max = st.number_input(
                    "Maximum Depth",
                    min_value=min_depth,
                    max_value=max_depth,
                    value=current_max,
                    step=10.0
                )
            
            if st.button("Apply Depth Range", use_container_width=True):
                if new_min < new_max:
                    detector.set_depth_range(new_min, new_max)
                    st.success(f"Applied depth range: {new_min} - {new_max} m")
                    st.rerun()
                else:
                    st.error("Minimum depth must be less than maximum depth")
        
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
        st.markdown("### 📊 Crossplot Analysis with Avseth-Oddgard RPTs")
        
        # Generate RPTs if not already done
        if not hasattr(detector, 'rpt_templates') or len(detector.rpt_templates) == 0:
            with st.spinner("Generating Avseth-Oddgard RPTs..."):
                detector.generate_avseth_oddgard_rpts()
        
        plot_type = st.selectbox(
            "Select Plot Type",
            ["Vp/Vs vs Ip with RPTs", "Lambda-Rho vs Mu-Rho", "Vp vs Porosity", 
             "Gas Probability Map", "Rock Fabric Clusters", "Fluid Substitution"]
        )
        
        if plot_type == "Vp/Vs vs Ip with RPTs":
            # Create figure with RPT background
            fig = go.Figure()
            
            # Add RPT contours for soft sand model
            if 'soft_sand' in detector.rpt_templates:
                rpt = detector.rpt_templates['soft_sand']
                
                # Add porosity contours
                n_contours = 5
                phi_vals = np.linspace(detector.rpt_params['phi_range'][0], 
                                      detector.rpt_params['phi_range'][1], n_contours)
                
                for phi in phi_vals:
                    phi_idx = np.argmin(np.abs(rpt['Phi'][:, 0] - phi))
                    fig.add_trace(go.Scatter(
                        x=rpt['Ip'][phi_idx, :],
                        y=rpt['Vp_Vs'][phi_idx, :],
                        mode='lines',
                        name=f'φ={phi:.2f}',
                        line=dict(color='gray', width=1, dash='dot'),
                        showlegend=False
                    ))
                
                # Add water saturation contours
                Sw_vals = [0.2, 0.5, 0.8]
                for Sw in Sw_vals:
                    Sw_idx = np.argmin(np.abs(rpt['Sw'][0, :] - Sw))
                    fig.add_trace(go.Scatter(
                        x=rpt['Ip'][:, Sw_idx],
                        y=rpt['Vp_Vs'][:, Sw_idx],
                        mode='lines',
                        name=f'Sw={Sw:.1f}',
                        line=dict(color='blue', width=1, dash='dash'),
                        showlegend=True
                    ))
            
            # Add data points with gas probability coloring
            if 'results' in detector.__dict__:
                fig.add_trace(go.Scatter(
                    x=detector.data['Ip'],
                    y=detector.data['Vp_Vs'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=detector.results['gas_probability'],
                        colorscale='RdBu_r',
                        showscale=True,
                        colorbar=dict(title="Gas Probability"),
                        line=dict(width=0.5, color='black')
                    ),
                    text=[f"Depth: {d:.1f}<br>Gas Prob: {p:.2f}" 
                          for d, p in zip(detector.data['DEPTH'] if 'DEPTH' in detector.data.columns else range(len(detector.data)), 
                                         detector.results['gas_probability'])],
                    hoverinfo='text',
                    name='Well Data'
                ))
            
            # Add gas sand threshold line
            fig.add_hline(y=1.8, line_dash="dash", line_color="red", 
                         annotation_text="Gas Sand Threshold", 
                         annotation_position="bottom right")
            
            # Add shale baseline
            fig.add_hline(y=2.0, line_dash="dash", line_color="brown", 
                         annotation_text="Shale Baseline",
                         annotation_position="bottom right")
            
            fig.update_layout(
                title="Vp/Vs vs Acoustic Impedance with Avseth-Oddgard RPTs",
                xaxis_title="Acoustic Impedance (g/cc * m/s)",
                yaxis_title="Vp/Vs Ratio",
                height=600,
                showlegend=True,
                legend=dict(x=1.02, y=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Lambda-Rho vs Mu-Rho":
            fig = px.scatter(
                detector.data,
                x='LAMBDA_RHO' if 'LAMBDA_RHO' in detector.data.columns else 'LAMBDA',
                y='MU_RHO' if 'MU_RHO' in detector.data.columns else 'MU',
                color=detector.results.get('gas_probability', None) if 'results' in detector.__dict__ else None,
                color_continuous_scale='RdBu_r',
                title="Lambda-Rho vs Mu-Rho Crossplot",
                labels={'LAMBDA_RHO': 'λρ (GPa*g/cc)', 'MU_RHO': 'μρ (GPa*g/cc)'}
            )
            
            # Add fluid discrimination lines
            fig.add_shape(type="line",
                         x0=detector.data['LAMBDA_RHO'].min() if 'LAMBDA_RHO' in detector.data.columns else 0,
                         x1=detector.data['LAMBDA_RHO'].max() if 'LAMBDA_RHO' in detector.data.columns else 50,
                         y0=detector.data['MU_RHO'].min() if 'MU_RHO' in detector.data.columns else 0,
                         y1=detector.data['MU_RHO'].max() if 'MU_RHO' in detector.data.columns else 50,
                         line=dict(color="gray", dash="dash"))
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Vp vs Porosity":
            fig = px.scatter(
                detector.data,
                x='PHI',
                y='Vp',
                color=detector.results.get('gas_probability', None) if 'results' in detector.__dict__ else None,
                color_continuous_scale='RdBu_r',
                title="P-wave Velocity vs Porosity",
                labels={'PHI': 'Porosity', 'Vp': 'P-wave Velocity (m/s)'}
            )
            
            # Add trend lines for different lithologies
            phi_range = np.linspace(0.05, 0.35, 10)
            # Sand trend
            vp_sand = 5500 - 8000 * phi_range
            fig.add_trace(go.Scatter(x=phi_range, y=vp_sand, mode='lines', 
                                    name='Sand Trend', line=dict(color='yellow', dash='dash')))
            # Shale trend
            vp_shale = 3000 - 4000 * phi_range
            fig.add_trace(go.Scatter(x=phi_range, y=vp_shale, mode='lines', 
                                    name='Shale Trend', line=dict(color='brown', dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Gas Probability Map":
            # Create 2D histogram with probability coloring
            fig = go.Figure()
            
            if 'Ip' in detector.data.columns and 'Vp_Vs' in detector.data.columns and 'results' in detector.__dict__:
                # Create 2D grid
                x = detector.data['Ip']
                y = detector.data['Vp_Vs']
                z = detector.results['gas_probability']
                
                # Create heatmap
                fig.add_trace(go.Histogram2dContour(
                    x=x, y=y, z=z,
                    colorscale='RdBu_r',
                    contours=dict(
                        coloring='heatmap',
                        showlabels=True,
                        labelfont=dict(size=12, color='white')
                    ),
                    colorbar=dict(title="Gas Probability"),
                    name='Probability Density'
                ))
                
                # Add scatter points for reference
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='black',
                        opacity=0.3
                    ),
                    name='Data Points'
                ))
                
                fig.update_layout(
                    title="Gas Probability Density Map (Colored by Probability)",
                    xaxis_title="Acoustic Impedance (g/cc * m/s)",
                    yaxis_title="Vp/Vs Ratio",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Rock Fabric Clusters" and 'FABRIC_CLUSTER' in detector.data.columns:
            fig = px.scatter(
                detector.data,
                x='Ip' if 'Ip' in detector.data.columns else 'Vp',
                y='Vp_Vs' if 'Vp_Vs' in detector.data.columns else 'Vs',
                color=detector.data['FABRIC_CLUSTER'].astype(str),
                title="Rock Fabric Clusters",
                labels={'FABRIC_CLUSTER': 'Cluster'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            # Add cluster statistics
            cluster_stats = detector.data.groupby('FABRIC_CLUSTER').agg({
                'PHI': 'mean',
                'Vp_Vs': 'mean',
                'Ip': 'mean'
            }).reset_index()
            
            st.dataframe(cluster_stats, use_container_width=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Fluid Substitution":
            # Simple fluid substitution visualization
            st.info("Fluid substitution analysis coming soon...")
    
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
                    title="Rock Fabric Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster characteristics
                st.markdown("**Cluster Characteristics:**")
                for cluster_id in np.unique(detector.results['fabric_clusters']):
                    cluster_data = detector.data[detector.results['fabric_clusters'] == cluster_id]
                    with st.expander(f"Cluster {cluster_id} - {len(cluster_data)} samples"):
                        cols = st.columns(2)
                        with cols[0]:
                            if 'PHI' in cluster_data.columns:
                                st.metric("Avg Porosity", f"{cluster_data['PHI'].mean():.3f}")
                            if 'Vp_Vs' in cluster_data.columns:
                                st.metric("Avg Vp/Vs", f"{cluster_data['Vp_Vs'].mean():.2f}")
                        with cols[1]:
                            if 'Ip' in cluster_data.columns:
                                st.metric("Avg Ip", f"{cluster_data['Ip'].mean():.0f}")
                            if 'SW' in cluster_data.columns:
                                st.metric("Avg Sw", f"{cluster_data['SW'].mean():.3f}")
        
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
                    go.Histogram(x=K, name='K', nbinsx=30,
                                marker_color='lightblue'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(x=detector.data['MU'], name='μ', nbinsx=30,
                                marker_color='lightgreen'),
                    row=1, col=2
                )
                
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Moduli statistics
                moduli_stats = pd.DataFrame({
                    'Parameter': ['K (GPa)', 'μ (GPa)', 'K/μ ratio'],
                    'Mean': [float(K.mean()), float(detector.data['MU'].mean()), 
                            float(K.mean()/detector.data['MU'].mean())],
                    'Std': [float(K.std()), float(detector.data['MU'].std()), 
                           float((K/detector.data['MU']).std())]
                })
                st.dataframe(moduli_stats, use_container_width=True)
            else:
                st.info("Elastic moduli not available")
    
    with tab5:
        st.markdown("### 📋 Detailed Statistics & Export")
        
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
            
            # High-confidence zones details
            if 'high_confidence_samples' in stats and stats['high_confidence_samples'] > 0:
                st.markdown("#### High-Confidence Zone Characteristics")
                hc_cols = ['hc_mean_phi', 'hc_mean_vp_vs', 'hc_mean_ip', 
                          'hc_mean_sw', 'hc_mean_vclay']
                
                hc_stats = {}
                for col in hc_cols:
                    if col in stats:
                        hc_stats[col.replace('hc_', '').replace('_', ' ').title()] = stats[col]
                
                if hc_stats:
                    hc_df = pd.DataFrame.from_dict(hc_stats, orient='index', columns=['Value'])
                    st.dataframe(hc_df, use_container_width=True)
            
            # Export options
            st.markdown("#### Export Results")
            col1, col2, col3 = st.columns(3)
            
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
                
                # Add method probabilities
                if 'method_probabilities' in detector.results and 'method_names' in detector.results:
                    for i, (name, probs) in enumerate(zip(detector.results['method_names'], 
                                                        detector.results['method_probabilities'])):
                        results_df[f'PROB_{name.replace(" ", "_").upper()}'] = probs
                
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="gas_detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export statistics
                stats_csv = pd.DataFrame([stats]).to_csv(index=False)
                st.download_button(
                    label="📊 Download Statistics",
                    data=stats_csv,
                    file_name="gas_detection_statistics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                if st.button("📈 Download All Plots", use_container_width=True):
                    st.info("Plot download feature coming soon!")
        
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
                    st.session_state.detector = EnhancedRockPhysicsGasDetector()
                
                if data_source == "uploaded":
                    df = pd.read_csv(data_to_load)
                    # Store original data
                    st.session_state.original_data = df.copy()
                    st.session_state.detector.load_data(df)
                    st.success(f"✅ Data loaded successfully: {len(st.session_state.detector.data):,} samples")
                elif data_source == "example":
                    # Store original data
                    st.session_state.original_data = data_to_load.copy()
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
                
                # Show depth range if available
                if 'DEPTH' in st.session_state.detector.data.columns:
                    min_depth = st.session_state.detector.data['DEPTH'].min()
                    max_depth = st.session_state.detector.data['DEPTH'].max()
                    st.metric("Depth Range", f"{min_depth:.1f} - {max_depth:.1f} m")
                
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
                        
                        # Get parameters from sidebar widgets
                        # Use try-except to handle any missing variables
                        try:
                            use_dispersion_val = use_dispersion
                        except NameError:
                            use_dispersion_val = True
                        
                        try:
                            use_ml_val = use_ml
                        except NameError:
                            use_ml_val = True
                        
                        try:
                            confidence_threshold_val = confidence_threshold
                        except NameError:
                            confidence_threshold_val = 0.7
                        
                        try:
                            n_clusters_val = n_clusters
                        except NameError:
                            n_clusters_val = 3
                        
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
                except NameError:
                    confidence_threshold_val = 0.7
                
                create_enhanced_dashboard(st.session_state.detector, confidence_threshold_val)
            
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
            
            **Novel Features:**
            1. **Adaptive DEM modeling** with pore geometry classification
            2. **Avseth-Oddgard RPTs** for quantitative interpretation
            3. **Multi-model ensemble** with uncertainty quantification
            4. **Depth range selection** for targeted analysis
            5. **Enhanced visualizations** with probability coloring
            6. **Rock fabric analysis** using machine learning
            7. **Export capabilities** for further analysis
            """)
            
            # Features in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Methods", "5")
            with col2:
                st.metric("RPT Models", "3")
            with col3:
                st.metric("Visualizations", "Enhanced")
            with col4:
                st.metric("Output", "CSV + Plots")

if __name__ == "__main__":
    main()
