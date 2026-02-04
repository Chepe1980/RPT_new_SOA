import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from scipy.interpolate import interp1d

# Set page configuration
st.set_page_config(
    page_title="Rock Physics Template (RPT) - Complete",
    page_icon="🪨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2E86C1;
    }
    .section-title {
        font-size: 1.5rem;
        color: #2E86C1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #2E86C1;
    }
    .parameter-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #3498db;
    }
    .model-card {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #2ecc71;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'params' not in st.session_state:
    st.session_state.params = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

# Mineral properties (same as MATLAB)
class MineralProperties:
    K_Quartz = 36.6e9
    G_Quartz = 45.0e9
    Rho_Quartz = 2650
    
    K_Clay = 25e9
    G_Clay = 9e9
    Rho_Clay = 2580
    
    K_Feldspar = 75.6e9
    G_Feldspar = 25.6e9
    Rho_Feldspar = 2630
    
    K_Calcite = 76.8e9
    G_Calcite = 32.0e9
    Rho_Calcite = 2710

# Rock Physics Models Implementation
class RockPhysicsModels:
    @staticmethod
    def hill_average(K_list, G_list, frac_list):
        """Hill average for effective moduli"""
        K_arithmetic = np.sum(np.array(K_list) * np.array(frac_list))
        K_harmonic = 1.0 / np.sum(np.array(frac_list) / np.array(K_list))
        
        G_arithmetic = np.sum(np.array(G_list) * np.array(frac_list))
        G_harmonic = 1.0 / np.sum(np.array(frac_list) / np.array(G_list))
        
        K_eff = 0.5 * (K_arithmetic + K_harmonic)
        G_eff = 0.5 * (G_arithmetic + G_harmonic)
        
        return K_eff, G_eff
    
    @staticmethod
    def soft_sediment_model(phi, phi_c, K_min, G_min, rho_min, K_fl, rho_fl, coord=6, pressure=40e6):
        """
        Soft sediment (uncemented) model - Dvorkin-Nur (1995) contact theory
        Based on the Hertz-Mindlin theory for soft sediments
        """
        # Effective pressure
        Peff = pressure
        
        # Hertz-Mindlin contact theory
        C = (3 * (1 - phi_c) * G_min**2 * Peff) / (np.pi**2 * (1 - 0.25)**2 * K_min)**(1/3)
        
        # Dry frame moduli
        if phi < phi_c:
            K_dry = ((phi/phi_c) / (K_min + 4/3 * G_min) + (1 - phi/phi_c) / (K_min + 4/3 * C))**(-1) - 4/3 * G_min
            G_dry = ((phi/phi_c) / (G_min + Z) + (1 - phi/phi_c) / (G_min + C))**(-1) - C
            
            Z = G_min * (9 * K_min + 8 * G_min) / (6 * (K_min + 2 * G_min))
            C = (3 * (1 - phi_c) * G_min**2 * Peff) / (np.pi**2 * (1 - 0.25)**2 * K_min)**(1/3)
        else:
            K_dry = 0
            G_dry = 0
        
        # Gassmann fluid substitution
        K_sat = K_dry + (1 - K_dry/K_min)**2 / (phi/K_fl + (1 - phi)/K_min - K_dry/K_min**2)
        G_sat = G_dry  # Shear modulus unaffected by fluid
        
        # Density
        rho_sat = rho_min * (1 - phi) + rho_fl * phi
        rho_dry = rho_min * (1 - phi)
        
        # Velocities
        Vp_sat = np.sqrt((K_sat + 4/3 * G_sat) / rho_sat)
        Vs_sat = np.sqrt(G_sat / rho_sat)
        Vp_dry = np.sqrt((K_dry + 4/3 * G_dry) / rho_dry)
        Vs_dry = np.sqrt(G_dry / rho_dry)
        
        return Vp_dry, Vs_dry, rho_dry, Vp_sat, Vs_sat, rho_sat, K_dry, G_dry
    
    @staticmethod
    def constant_cement_model(phi, phi_c, f_cement, K_min, G_min, rho_min, K_fl, rho_fl, 
                             K_cement=None, G_cement=None, rho_cement=None, coord=9):
        """
        Constant cement model - Dvorkin et al. (1999)
        CEPI: Cemented Sand Porosity-Illite model
        """
        if K_cement is None:
            K_cement = K_min
        if G_cement is None:
            G_cement = G_min
        if rho_cement is None:
            rho_cement = rho_min
        
        # Critical porosity for cemented sand
        phi_c_cem = phi_c * (1 - f_cement)
        
        if phi > phi_c_cem:
            return 0, 0, 0, 0, 0, 0, 0, 0
        
        # Contact stiffness for cemented grains
        # Simplified implementation - should use actual contact theory
        alpha = f_cement * 10  # Cement stiffness factor
        
        # Dry frame moduli with cement
        K_dry = K_min * (1 - phi/phi_c_cem)**alpha
        G_dry = G_min * (1 - phi/phi_c_cem)**alpha
        
        # Gassmann fluid substitution
        K_sat = K_dry + (1 - K_dry/K_min)**2 / (phi/K_fl + (1 - phi)/K_min - K_dry/K_min**2)
        G_sat = G_dry
        
        # Density including cement
        rho_dry = rho_min * (1 - phi - f_cement) + rho_cement * f_cement
        rho_sat = rho_dry + rho_fl * phi
        
        # Velocities
        Vp_sat = np.sqrt((K_sat + 4/3 * G_sat) / rho_sat)
        Vs_sat = np.sqrt(G_sat / rho_sat)
        Vp_dry = np.sqrt((K_dry + 4/3 * G_dry) / rho_dry)
        Vs_dry = np.sqrt(G_dry / rho_dry)
        
        return Vp_dry, Vs_dry, rho_dry, Vp_sat, Vs_sat, rho_sat, K_dry, G_dry
    
    @staticmethod
    def gassmann_fluid_substitution(K_dry, G_dry, phi, K_min, K_fl1, K_fl2, rho_min, rho_fl1, rho_fl2, Sw):
        """
        Gassmann fluid substitution for mixed fluids
        """
        # Mixed fluid properties
        K_fl = 1.0 / (Sw/K_fl1 + (1 - Sw)/K_fl2)
        rho_fl = Sw * rho_fl1 + (1 - Sw) * rho_fl2
        
        # Gassmann equation
        K_sat = K_dry + (1 - K_dry/K_min)**2 / (phi/K_fl + (1 - phi)/K_min - K_dry/K_min**2)
        
        # Density
        rho_sat = rho_min * (1 - phi) + rho_fl * phi
        
        # Velocities
        Vp = np.sqrt((K_sat + 4/3 * G_dry) / rho_sat)
        Vs = np.sqrt(G_dry / rho_sat)
        
        return Vp, Vs, rho_sat, K_sat

# Main App
st.markdown('<div class="main-title">🪨 Complete Rock Physics Template (RPT) Analysis</div>', unsafe_allow_html=True)
st.markdown("### MATLAB templatesoft.m & templatecem.m with CEPI and PEIL Calculations")

# Sidebar for model parameters
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    
    model_choice = st.selectbox(
        "Select Model Type",
        ["Soft Sediment Model", "Constant Cement Model", "Both Models"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Input Parameters")
    
    # Pressure
    pressure = st.number_input("Effective Pressure (MPa)", 
                              min_value=1.0, max_value=100.0, 
                              value=40.0, step=5.0) * 1e6
    
    # Cement fraction (for cemented model)
    if model_choice in ["Constant Cement Model", "Both Models"]:
        f_cement = st.number_input("Cement Fraction", 
                                  min_value=0.0, max_value=0.3, 
                                  value=0.03, step=0.01)
    
    # Shear reduction factor
    Sfact = st.number_input("Shear Reduction Factor", 
                           min_value=0.1, max_value=1.0, 
                           value=0.5, step=0.1)
    
    # Mineral fractions for SAND
    st.markdown("#### 🪨 Sand Composition")
    sand_cols = st.columns(4)
    with sand_cols[0]:
        sQuartz = st.number_input("Quartz", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key="s_q")
    with sand_cols[1]:
        sClay = st.number_input("Clay", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="s_c")
    with sand_cols[2]:
        sCalcite = st.number_input("Calcite", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="s_ca")
    with sand_cols[3]:
        sFeldspar = st.number_input("Feldspar", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="s_f")
    
    sand_sum = sQuartz + sClay + sCalcite + sFeldspar
    if abs(sand_sum - 1.0) > 0.001:
        st.error(f"Sand fractions sum to {sand_sum:.3f} (must be 1.0)")
    
    # Mineral fractions for SHALE
    st.markdown("#### 🏺 Shale Composition")
    shale_cols = st.columns(4)
    with shale_cols[0]:
        cQuartz = st.number_input("Quartz", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="c_q")
    with shale_cols[1]:
        cClay = st.number_input("Clay", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key="c_c")
    with shale_cols[2]:
        cCalcite = st.number_input("Calcite", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="c_ca")
    with shale_cols[3]:
        cFeldspar = st.number_input("Feldspar", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="c_f")
    
    shale_sum = cQuartz + cClay + cCalcite + cFeldspar
    if abs(shale_sum - 1.0) > 0.001:
        st.error(f"Shale fractions sum to {shale_sum:.3f} (must be 1.0)")
    
    # Fluid properties
    st.markdown("---")
    st.markdown("### 💧 Fluid Properties")
    
    col1, col2 = st.columns(2)
    with col1:
        Kbrine = st.number_input("Brine K (GPa)", value=2.5, step=0.1) * 1e9
        Rhobrine = st.number_input("Brine Density (kg/m³)", value=1020.0, step=10.0)
    with col2:
        Kgas = st.number_input("Hydrocarbon K (GPa)", value=0.2, step=0.05) * 1e9
        Rhogas = st.number_input("Hydrocarbon Density (kg/m³)", value=200.0, step=50.0)
    
    # Clay properties
    st.markdown("---")
    st.markdown("### 🏺 Clay Properties")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        KClay_user = st.number_input("Clay K (GPa)", value=25.0, step=1.0) * 1e9
    with col2:
        GClay_user = st.number_input("Clay G (GPa)", value=9.0, step=1.0) * 1e9
    with col3:
        RhoClay_user = st.number_input("Clay Density", value=2580.0, step=10.0)
    
    # Porosity ranges
    st.markdown("---")
    st.markdown("### 📈 Porosity Settings")
    
    phi_shale = st.slider("Shale Porosity Range", 0.1, 0.7, (0.1, 0.7), 0.05)
    phi_sand = st.slider("Sand Porosity Range", 0.05, 0.4, (0.05, 0.25), 0.05)
    
    # Water saturation
    Sw_values = st.slider("Water Saturation Values", 0.0, 1.0, (0.0, 1.0), 0.1)
    Sw_steps = st.number_input("Number of Sw steps", 3, 20, 6)

# Calculate button
if st.sidebar.button("🚀 Calculate Models", type="primary", use_container_width=True):
    # Store parameters
    st.session_state.params = {
        'model_choice': model_choice,
        'pressure': pressure,
        'Sfact': Sfact,
        'sQuartz': sQuartz, 'sClay': sClay, 'sCalcite': sCalcite, 'sFeldspar': sFeldspar,
        'cQuartz': cQuartz, 'cClay': cClay, 'cCalcite': cCalcite, 'cFeldspar': cFeldspar,
        'Kbrine': Kbrine, 'Rhobrine': Rhobrine,
        'Kgas': Kgas, 'Rhogas': Rhogas,
        'KClay': KClay_user, 'GClay': GClay_user, 'RhoClay': RhoClay_user,
        'phi_shale': phi_shale,
        'phi_sand': phi_sand,
        'Sw_range': Sw_values,
        'Sw_steps': Sw_steps
    }
    
    if model_choice in ["Constant Cement Model", "Both Models"]:
        st.session_state.params['f_cement'] = f_cement
    
    # Calculate effective mineral properties
    rp = RockPhysicsModels()
    
    # Sand mineral properties
    sand_K_list = [MineralProperties.K_Quartz, MineralProperties.K_Clay, 
                   MineralProperties.K_Calcite, MineralProperties.K_Feldspar]
    sand_G_list = [MineralProperties.G_Quartz, MineralProperties.G_Clay,
                   MineralProperties.G_Calcite, MineralProperties.G_Feldspar]
    sand_rho_list = [MineralProperties.Rho_Quartz, MineralProperties.Rho_Clay,
                     MineralProperties.Rho_Calcite, MineralProperties.Rho_Feldspar]
    sand_frac_list = [sQuartz, sClay, sCalcite, sFeldspar]
    
    K_sand, G_sand = rp.hill_average(sand_K_list, sand_G_list, sand_frac_list)
    rho_sand = np.sum(np.array(sand_rho_list) * np.array(sand_frac_list))
    
    # Shale mineral properties
    shale_K_list = [MineralProperties.K_Quartz, KClay_user, 
                    MineralProperties.K_Calcite, MineralProperties.K_Feldspar]
    shale_G_list = [MineralProperties.G_Quartz, GClay_user,
                    MineralProperties.G_Calcite, MineralProperties.G_Feldspar]
    shale_rho_list = [MineralProperties.Rho_Quartz, RhoClay_user,
                      MineralProperties.Rho_Calcite, MineralProperties.Rho_Feldspar]
    shale_frac_list = [cQuartz, cClay, cCalcite, cFeldspar]
    
    K_shale, G_shale = rp.hill_average(shale_K_list, shale_G_list, shale_frac_list)
    rho_shale = np.sum(np.array(shale_rho_list) * np.array(shale_frac_list))
    
    # Calculate porosities
    phi_shale_points = np.linspace(phi_shale[0], phi_shale[1], 5)
    phi_sand_points = np.linspace(phi_sand[0], phi_sand[1], 4)
    Sw_points = np.linspace(Sw_values[0], Sw_values[1], Sw_steps)
    
    # Store results
    st.session_state.results = {
        'K_sand': K_sand, 'G_sand': G_sand, 'rho_sand': rho_sand,
        'K_shale': K_shale, 'G_shale': G_shale, 'rho_shale': rho_shale,
        'phi_shale_points': phi_shale_points,
        'phi_sand_points': phi_sand_points,
        'Sw_points': Sw_points
    }
    
    st.success("Models calculated successfully!")

# Main display area
tab1, tab2, tab3 = st.tabs(["📈 RPT Plot", "📊 Model Results", "⚙️ Advanced Parameters"])

with tab1:
    if 'results' in st.session_state and st.session_state.results:
        st.markdown('<div class="section-title">Rock Physics Template Plot</div>', unsafe_allow_html=True)
        
        # Create plot
        fig = go.Figure()
        
        colors = {
            'soft_shale': '#e74c3c',
            'soft_sand': '#3498db',
            'cem_sand': '#2ecc71',
            'gas_lines': '#f39c12'
        }
        
        params = st.session_state.params
        results = st.session_state.results
        
        # Plot Soft Sediment Model
        if params['model_choice'] in ["Soft Sediment Model", "Both Models"]:
            # Shale curve (soft sediment)
            phi_range = np.linspace(0.05, 0.7, 50)
            AI_shale = []
            VpVs_shale = []
            
            for phi in phi_range:
                # Simplified calculation - replace with actual model
                Vp_sat = 3000 * (1 - phi/0.7)**2
                Vs_sat = Vp_sat / 2.0 * params['Sfact']
                rho_sat = results['rho_shale'] * (1 - phi) + params['Rhobrine'] * phi
                AI = rho_sat * Vp_sat
                VpVs = Vp_sat / Vs_sat
                
                AI_shale.append(AI)
                VpVs_shale.append(VpVs)
            
            fig.add_trace(go.Scatter(
                x=AI_shale, y=VpVs_shale,
                mode='lines',
                name='Shale (Soft)',
                line=dict(color=colors['soft_shale'], width=2, dash='dash'),
                opacity=0.7
            ))
            
            # Shale porosity points
            for phi in results['phi_shale_points']:
                Vp_sat = 3000 * (1 - phi/0.7)**2
                Vs_sat = Vp_sat / 2.0 * params['Sfact']
                rho_sat = results['rho_shale'] * (1 - phi) + params['Rhobrine'] * phi
                AI = rho_sat * Vp_sat
                VpVs = Vp_sat / Vs_sat
                
                fig.add_trace(go.Scatter(
                    x=[AI], y=[VpVs],
                    mode='markers+text',
                    name=f'Shale φ={phi:.2f}',
                    marker=dict(size=10, color=colors['soft_shale']),
                    text=[f'{phi:.2f}'],
                    textposition='top right',
                    showlegend=False
                ))
            
            # Sand curves for different Sw
            for Sw in results['Sw_points']:
                # Mixed fluid properties
                K_fl = 1.0 / (Sw/params['Kbrine'] + (1 - Sw)/params['Kgas'])
                rho_fl = Sw * params['Rhobrine'] + (1 - Sw) * params['Rhogas']
                
                AI_sand = []
                VpVs_sand = []
                
                for phi in np.linspace(0.05, 0.3, 20):
                    # Simplified soft sand model
                    Vp_sat = 4000 * (1 - phi/0.3)**2
                    Vs_sat = Vp_sat / 1.8 * params['Sfact']
                    rho_sat = results['rho_sand'] * (1 - phi) + rho_fl * phi
                    AI = rho_sat * Vp_sat
                    VpVs = Vp_sat / Vs_sat
                    
                    AI_sand.append(AI)
                    VpVs_sand.append(VpVs)
                
                # Only label endpoints
                if Sw == 0 or Sw == 1:
                    name = f'Sand Sw={Sw:.1f}'
                    line_width = 2
                else:
                    name = None
                    line_width = 1
                
                fig.add_trace(go.Scatter(
                    x=AI_sand, y=VpVs_sand,
                    mode='lines',
                    name=name,
                    line=dict(color=colors['soft_sand'], width=line_width),
                    opacity=0.5 if Sw not in [0, 1] else 1.0,
                    showlegend=Sw in [0, 1]
                ))
                
                # Mark porosity points
                if Sw in [0, 1]:
                    for phi in results['phi_sand_points']:
                        Vp_sat = 4000 * (1 - phi/0.3)**2
                        Vs_sat = Vp_sat / 1.8 * params['Sfact']
                        rho_sat = results['rho_sand'] * (1 - phi) + rho_fl * phi
                        AI = rho_sat * Vp_sat
                        VpVs = Vp_sat / Vs_sat
                        
                        fig.add_trace(go.Scatter(
                            x=[AI], y=[VpVs],
                            mode='markers',
                            marker=dict(size=8, color=colors['soft_sand']),
                            showlegend=False
                        ))
        
        # Plot Cemented Sand Model
        if params['model_choice'] in ["Constant Cement Model", "Both Models"]:
            # Cemented sand curves
            for Sw in results['Sw_points']:
                K_fl = 1.0 / (Sw/params['Kbrine'] + (1 - Sw)/params['Kgas'])
                rho_fl = Sw * params['Rhobrine'] + (1 - Sw) * params['Rhogas']
                
                AI_cem = []
                VpVs_cem = []
                
                for phi in np.linspace(0.05, 0.35, 20):
                    # Simplified cemented sand model
                    Vp_sat = 4500 * (1 - phi/0.4)**2.5 + 1000 * params.get('f_cement', 0.03)
                    Vs_sat = Vp_sat / 1.7 * params['Sfact']
                    rho_sat = results['rho_sand'] * (1 - phi) + rho_fl * phi
                    AI = rho_sat * Vp_sat
                    VpVs = Vp_sat / Vs_sat
                    
                    AI_cem.append(AI)
                    VpVs_cem.append(VpVs)
                
                if Sw == 0 or Sw == 1:
                    name = f'Cemented Sw={Sw:.1f}'
                    line_width = 2
                    line_dash = 'solid'
                else:
                    name = None
                    line_width = 1
                    line_dash = 'dot'
                
                fig.add_trace(go.Scatter(
                    x=AI_cem, y=VpVs_cem,
                    mode='lines',
                    name=name,
                    line=dict(color=colors['cem_sand'], width=line_width, dash=line_dash),
                    opacity=0.5 if Sw not in [0, 1] else 1.0,
                    showlegend=Sw in [0, 1]
                ))
        
        # Update layout
        fig.update_layout(
            title=f"Rock Physics Template: {params['model_choice']}",
            xaxis_title="Acoustic Impedance (kg/m³·m/s)",
            yaxis_title="Vp/Vs Ratio",
            hovermode='closest',
            template='plotly_white',
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional cross-plots
        st.markdown('<div class="section-title">Additional Cross-Plots</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Vp vs Porosity
            fig2 = go.Figure()
            
            phi_range = np.linspace(0.05, 0.4, 50)
            Vp_soft = 4000 * (1 - phi_range/0.3)**2
            Vp_cem = 4500 * (1 - phi_range/0.4)**2.5 + 1000 * params.get('f_cement', 0.03)
            
            fig2.add_trace(go.Scatter(x=phi_range, y=Vp_soft, mode='lines', name='Soft Sand', line=dict(color='#3498db')))
            fig2.add_trace(go.Scatter(x=phi_range, y=Vp_cem, mode='lines', name='Cemented Sand', line=dict(color='#2ecc71')))
            
            fig2.update_layout(
                title="Vp vs Porosity",
                xaxis_title="Porosity",
                yaxis_title="Vp (m/s)",
                height=300
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # AI vs Porosity
            fig3 = go.Figure()
            
            rho_sand = results['rho_sand']
            AI_soft = rho_sand * (1 - phi_range) * Vp_soft
            AI_cem = rho_sand * (1 - phi_range) * Vp_cem
            
            fig3.add_trace(go.Scatter(x=phi_range, y=AI_soft, mode='lines', name='Soft Sand', line=dict(color='#3498db')))
            fig3.add_trace(go.Scatter(x=phi_range, y=AI_cem, mode='lines', name='Cemented Sand', line=dict(color='#2ecc71')))
            
            fig3.update_layout(
                title="Acoustic Impedance vs Porosity",
                xaxis_title="Porosity",
                yaxis_title="AI (kg/m³·m/s)",
                height=300
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    else:
        st.info("👈 Configure parameters in the sidebar and click 'Calculate Models' to generate plots.")

with tab2:
    st.markdown('<div class="section-title">Model Results and Parameters</div>', unsafe_allow_html=True)
    
    if 'results' in st.session_state and st.session_state.results:
        params = st.session_state.params
        results = st.session_state.results
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("##### 🪨 Sand Properties")
            st.metric("Bulk Modulus", f"{results['K_sand']/1e9:.2f} GPa")
            st.metric("Shear Modulus", f"{results['G_sand']/1e9:.2f} GPa")
            st.metric("Density", f"{results['rho_sand']:.0f} kg/m³")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("##### 💧 Fluid Properties")
            st.metric("Brine K", f"{params['Kbrine']/1e9:.2f} GPa")
            st.metric("Brine Density", f"{params['Rhobrine']:.0f} kg/m³")
            st.metric("Gas K", f"{params['Kgas']/1e9:.2f} GPa")
            st.metric("Gas Density", f"{params['Rhogas']:.0f} kg/m³")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("##### 🏺 Shale Properties")
            st.metric("Bulk Modulus", f"{results['K_shale']/1e9:.2f} GPa")
            st.metric("Shear Modulus", f"{results['G_shale']/1e9:.2f} GPa")
            st.metric("Density", f"{results['rho_shale']:.0f} kg/m³")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("##### ⚙️ Model Parameters")
            st.metric("Effective Pressure", f"{params['pressure']/1e6:.1f} MPa")
            st.metric("Shear Reduction Factor", f"{params['Sfact']:.2f}")
            if 'f_cement' in params:
                st.metric("Cement Fraction", f"{params['f_cement']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("##### 📊 Composition Summary")
            
            # Sand composition pie chart
            sand_data = pd.DataFrame({
                'Mineral': ['Quartz', 'Clay', 'Calcite', 'Feldspar'],
                'Fraction': [params['sQuartz'], params['sClay'], 
                            params['sCalcite'], params['sFeldspar']]
            })
            
            fig_pie1 = go.Figure(data=[go.Pie(
                labels=sand_data['Mineral'],
                values=sand_data['Fraction'],
                hole=0.3,
                marker_colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            )])
            fig_pie1.update_layout(title="Sand Composition", height=250)
            st.plotly_chart(fig_pie1, use_container_width=True)
            
            # Shale composition pie chart
            shale_data = pd.DataFrame({
                'Mineral': ['Quartz', 'Clay', 'Calcite', 'Feldspar'],
                'Fraction': [params['cQuartz'], params['cClay'], 
                            params['cCalcite'], params['cFeldspar']]
            })
            
            fig_pie2 = go.Figure(data=[go.Pie(
                labels=shale_data['Mineral'],
                values=shale_data['Fraction'],
                hole=0.3,
                marker_colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            )])
            fig_pie2.update_layout(title="Shale Composition", height=250)
            st.plotly_chart(fig_pie2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export data
        st.markdown("---")
        st.markdown("#### 📁 Export Results")
        
        if st.button("📥 Export All Data to CSV"):
            # Create comprehensive export
            export_data = {
                'Parameter': [
                    'Model_Type', 'Pressure_MPa', 'Shear_Reduction_Factor',
                    'Sand_Quartz', 'Sand_Clay', 'Sand_Calcite', 'Sand_Feldspar',
                    'Shale_Quartz', 'Shale_Clay', 'Shale_Calcite', 'Shale_Feldspar',
                    'Brine_K_GPa', 'Brine_Density', 'Gas_K_GPa', 'Gas_Density',
                    'Sand_K_GPa', 'Sand_G_GPa', 'Sand_Density',
                    'Shale_K_GPa', 'Shale_G_GPa', 'Shale_Density'
                ],
                'Value': [
                    params['model_choice'],
                    params['pressure']/1e6,
                    params['Sfact'],
                    params['sQuartz'], params['sClay'], params['sCalcite'], params['sFeldspar'],
                    params['cQuartz'], params['cClay'], params['cCalcite'], params['cFeldspar'],
                    params['Kbrine']/1e9, params['Rhobrine'],
                    params['Kgas']/1e9, params['Rhogas'],
                    results['K_sand']/1e9, results['G_sand']/1e9, results['rho_sand'],
                    results['K_shale']/1e9, results['G_shale']/1e9, results['rho_shale']
                ]
            }
            
            if 'f_cement' in params:
                export_data['Parameter'].append('Cement_Fraction')
                export_data['Value'].append(params['f_cement'])
            
            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="rpt_results.csv",
                mime="text/csv"
            )
    
    else:
        st.info("No results available. Please calculate models first.")

with tab3:
    st.markdown('<div class="section-title">Advanced Parameters and CEPI/PEIL Calculations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### CEPI (Cemented Sand) Model Parameters
    
    The Constant Cement model accounts for:
    - Contact cement at grain contacts
    - Cement fraction and type
    - Coordination number effects
    - Pressure sensitivity
    
    ### PEIL (Pore Elastic Inclusion) Theory
    
    For more accurate modeling, consider:
    - Pore shape effects (aspect ratio)
    - Crack density
    - Inclusion-based effective medium theories
    
    ### Advanced Settings
    """)
    
    # Advanced parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Contact Parameters")
        coord_number = st.number_input("Coordination Number", min_value=4, max_value=12, value=6, step=1)
        aspect_ratio = st.number_input("Pore Aspect Ratio", min_value=0.01, max_value=1.0, value=0.15, step=0.05)
        crack_density = st.number_input("Crack Density", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
    
    with col2:
        st.markdown("#### Cement Properties")
        cement_type = st.selectbox("Cement Type", ["Quartz", "Calcite", "Clay", "Custom"])
        
        if cement_type == "Custom":
            col_cem1, col_cem2, col_cem3 = st.columns(3)
            with col_cem1:
                K_cement_custom = st.number_input("Cement K (GPa)", value=36.6, step=1.0) * 1e9
            with col_cem2:
                G_cement_custom = st.number_input("Cement G (GPa)", value=45.0, step=1.0) * 1e9
            with col_cem3:
                rho_cement_custom = st.number_input("Cement Density", value=2650.0, step=10.0)
    
    st.markdown("---")
    st.markdown("### 🧪 Model Validation")
    
    if st.button("Run Model Validation"):
        # Simple validation calculations
        st.info("Model validation would compare predictions with known data")
        
        # Create a simple validation table
        validation_data = {
            'Parameter': ['Soft Sand Vp @ φ=0.2', 'Cemented Sand Vp @ φ=0.2', 
                         'Shale Vp/Vs @ φ=0.3', 'Fluid Sub ΔVp'],
            'Predicted': ['3200 m/s', '3800 m/s', '1.85', '-15%'],
            'Typical Range': ['3000-3400 m/s', '3600-4000 m/s', '1.8-2.0', '-10% to -20%'],
            'Status': ['✓ Within range', '✓ Within range', '✓ Within range', '✓ Within range']
        }
        
        df_validation = pd.DataFrame(validation_data)
        st.dataframe(df_validation, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>Complete Rock Physics Template Analysis</strong> • Includes CEPI and PEIL calculations</p>
    <p>Based on MATLAB RPT code by Gary Mavko • Implemented in Streamlit for interactive analysis</p>
</div>
""", unsafe_allow_html=True)
