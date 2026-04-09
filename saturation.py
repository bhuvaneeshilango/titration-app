import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Saturated Isotherm & Titration Modeler")
st.write("Overlay theoretical saturation plateaus directly onto your raw physical data.")

# ==========================================
# 1. FILE UPLOAD & SETUP
# ==========================================
uploaded_file = st.file_uploader("Upload Titration Data (.txt or .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep='\s+')
        wavelengths = df.iloc[:, 0].values
        intensities_orig = df.iloc[:, 1:].values
        num_orig = intensities_orig.shape[1]

        # ==========================================
        # 2. SIDEBAR: EXPERIMENTAL SETUP
        # ==========================================
        st.sidebar.header("1. Experimental Setup")
        V_0 = st.sidebar.number_input("Initial Probe Volume (µL)", value=2000.0, step=100.0)
        C_stock = st.sidebar.number_input("Analyte Stock Conc. (µM)", value=250.0, step=10.0)
        V_add = st.sidebar.number_input("Volume per addition (µL)", value=3.0, step=0.5)

        # ==========================================
        # 3. SIDEBAR: PROBE METRICS
        # ==========================================
        st.sidebar.header("2. Probe Metrics")
        probe_type = st.sidebar.selectbox("Probe Behavior", ["Ratiometric (I1 / I2)", "Turn-On (I - I0)", "Quenching (I0 - I)"])
        wl_1 = st.sidebar.number_input("Monitor Wavelength 1 (nm)", int(wavelengths.min()), int(wavelengths.max()), 466)
        
        if "Ratiometric" in probe_type:
            wl_2 = st.sidebar.number_input("Monitor Wavelength 2 (nm)", int(wavelengths.min()), int(wavelengths.max()), 390)

        # ==========================================
        # 4. SIDEBAR: SATURATION SLIDERS
        # ==========================================
        st.sidebar.header("3. Saturation Control")
        st.sidebar.write("*Your raw data will always be plotted. Use these sliders to generate a theoretical curve to compare it against.*")
        
        # Select the pivot (where the simulation branches off)
        pivot_idx = st.sidebar.slider("Pivot Point (Onset of Saturation)", min_value=1, max_value=num_orig-1, value=min(10, num_orig-1))
        
        # How many steps to simulate past the pivot
        sim_steps = st.sidebar.slider("Simulated steps past pivot", min_value=5, max_value=30, value=10)
        
        sat_limit = st.sidebar.slider("Saturation Limit Factor", 1.0, 2.0, 1.15, 0.01)
        sharpness = st.sidebar.slider("Curve Sharpness", 0.1, 1.0, 0.3)

        # ==========================================
        # 5. MATH: EXACT CONCENTRATIONS
        # ==========================================
        # Calculate concentrations for ORIGINAL data
        vols_orig = np.arange(num_orig) * V_add
        concs_orig = (vols_orig * C_stock) / (V_0 + vols_orig)

        # Calculate concentrations for SIMULATED data
        # Simulation starts exactly at the pivot and moves forward
        vols_sim = np.arange(pivot_idx, pivot_idx + sim_steps + 1) * V_add
        concs_sim = (vols_sim * C_stock) / (V_0 + vols_sim)

        # ==========================================
        # 6. MATH: ASYMPTOTIC EXTRAPOLATION
        # ==========================================
        intensities_sim = np.zeros((len(wavelengths), sim_steps + 1))
        
        S_base = intensities_orig[:, 0]
        S_pivot = intensities_orig[:, pivot_idx]

        # Generate the simulated curves
        for i in range(sim_steps + 1):
            alpha = sat_limit - (sat_limit - 1.0) * np.exp(-sharpness * i)
            intensities_sim[:, i] = np.clip(S_base + alpha * (S_pivot - S_base), a_min=0, a_max=None)

        # ==========================================
        # 7. METRIC EXTRACTION
        # ==========================================
        def extract_y(intensities_matrix):
            I_vals_1 = intensities_matrix[(np.abs(wavelengths - wl_1)).argmin(), :]
            I_0_1 = intensities_orig[(np.abs(wavelengths - wl_1)).argmin(), 0] # Base always from original
            
            if "Turn-On" in probe_type:
                return I_vals_1 - I_0_1, f"I - I0 (at {wl_1} nm)"
            elif "Quenching" in probe_type:
                return I_0_1 - I_vals_1, f"I0 - I (at {wl_1} nm)"
            else:
                I_vals_2 = intensities_matrix[(np.abs(wavelengths - wl_2)).argmin(), :]
                return I_vals_1 / I_vals_2, f"Ratio I({wl_1}) / I({wl_2})"

        y_orig, ylabel = extract_y(intensities_orig)
        y_sim, _ = extract_y(intensities_sim)

        # ==========================================
        # 8. PLOTTING
        # ==========================================
        st.subheader("1. Binding Isotherm Overlay")
        fig2, ax2 = plt.subplots(figsize=(7, 5))

        # Plot 100% of Real Data
        ax2.plot(concs_orig, y_orig, marker='o', color='black', linestyle='-', linewidth=2, label='Real Physical Data')
        
        # Highlight the Pivot Point
        ax2.plot(concs_orig[pivot_idx], y_orig[pivot_idx], marker='o', color='green', markersize=10, label='Pivot Point')

        # Plot the Simulated Data branching off
        ax2.plot(concs_sim, y_sim, marker='x', color='red', linestyle='--', linewidth=2, label='Theoretical Saturation')

        ax2.set_xlabel("Analyte Concentration (µM)")
        ax2.set_ylabel(ylabel)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)

        st.subheader("2. Spectra Preview")
        st.write("*Note: For visual clarity, only the real baseline, real pivot, and simulated curves are shown.*")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        
        # Plot Base and Pivot
        ax1.plot(wavelengths, S_base, color='black', label="Real Baseline (0 µM)")
        ax1.plot(wavelengths, S_pivot, color='green', label=f"Real Pivot ({round(concs_orig[pivot_idx],2)} µM)")
        
        # Plot Simulated Additions
        for i in range(1, sim_steps + 1): # Skip 0 as it's the pivot
            alpha_val = np.clip(0.2 + (i/sim_steps)*0.8, 0, 1)
            ax1.plot(wavelengths, intensities_sim[:, i], color='red', linestyle='--', alpha=alpha_val)

        ax1.axvline(x=wl_1, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Intensity")
        ax1.legend()
        st.pyplot(fig1)
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
