import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Probe Saturation & Linearity Modeler")
st.write("Simulate probe saturation for additional additions and visualize binding curves.")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload Titration Data (.txt or .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep='\s+')
        wavelengths = df.iloc[:, 0].values
        intensities = df.iloc[:, 1:].values
        num_original = intensities.shape[1]
        
        st.success(f"Loaded {num_original} original curves.")
        
        # -----------------------------------------
        # UI: PROBE AND CONCENTRATION SETTINGS
        # -----------------------------------------
        st.sidebar.header("1. Experimental Setup")
        probe_type = st.sidebar.selectbox("Probe Type", ["Turn-On", "Quenching", "Ratiometric"])
        conc_step = st.sidebar.number_input("Analyte added per step (e.g., µM)", value=5.0)
        
        st.sidebar.header("2. Saturation Extrapolation")
        extra_curves = st.sidebar.number_input("How many additional curves to simulate?", 1, 20, 5)
        
        st.sidebar.write("*Saturation Limit: 1.0 means it stops exactly at your last curve. 1.2 means it can grow/quench 20% more before fully plateauing.*")
        sat_limit = st.sidebar.slider("Saturation Limit Factor", 1.01, 2.0, 1.15)
        curve_sharpness = st.sidebar.slider("How fast does it saturate?", 0.1, 1.0, 0.4)

        # -----------------------------------------
        # UI: PLOT SETTINGS
        # -----------------------------------------
        st.sidebar.header("3. Monitor Wavelengths")
        wl_1 = st.sidebar.number_input("Main Peak (nm)", int(wavelengths.min()), int(wavelengths.max()), 466)
        if probe_type == "Ratiometric":
            wl_2 = st.sidebar.number_input("Reference Peak (nm)", int(wavelengths.min()), int(wavelengths.max()), 390)

        # -----------------------------------------
        # MATH: ASYMPTOTIC SATURATION
        # -----------------------------------------
        total_curves = num_original + extra_curves
        concentrations = np.arange(total_curves) * conc_step
        
        # Extract base (first) and final (last physical) spectra
        S_base = intensities[:, 0]
        S_last = intensities[:, -1]
        
        new_intensities = np.zeros((len(wavelengths), total_curves))
        
        # Fill in original data
        for i in range(num_original):
            new_intensities[:, i] = intensities[:, i]
            
        # Simulate saturated additions using an asymptotic decay approach
        # α represents the "reaction progress" or binding fraction
        for i in range(num_original, total_curves):
            # Asymptotic formula: approaches sat_limit but never exceeds it
            alpha = sat_limit - (sat_limit - 1.0) * np.exp(-curve_sharpness * (i - num_original + 1))
            new_intensities[:, i] = np.clip(S_base + alpha * (S_last - S_base), a_min=0, a_max=None)

        # -----------------------------------------
        # PLOTTING: SPECTRA
        # -----------------------------------------
        st.subheader("1. Full Spectra (Original + Saturated Additions)")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for i in range(total_curves):
            color = 'blue' if i < num_original else 'red'
            linestyle = '-' if i < num_original else '--'
            alpha_val = np.clip(0.3 + (i/total_curves)*0.7, 0, 1)
            label = f"{concentrations[i]} µM" if i == 0 or i == num_original-1 or i == total_curves-1 else ""
            ax1.plot(wavelengths, new_intensities[:, i], color=color, linestyle=linestyle, alpha=alpha_val, label=label)
            
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Intensity")
        ax1.legend()
        st.pyplot(fig1)

        # -----------------------------------------
        # PLOTTING: BINDING / LINEARITY
        # -----------------------------------------
        st.subheader(f"2. {probe_type} Binding Curve")
        
        idx_1 = (np.abs(wavelengths - wl_1)).argmin()
        I_vals = new_intensities[idx_1, :]
        I_0 = I_vals[0]
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        
        if probe_type == "Turn-On":
            y_vals = I_vals - I_0
            ylabel = f"I - I0 (at {wl_1} nm)"
        elif probe_type == "Quenching":
            y_vals = I_0 - I_vals
            ylabel = f"I0 - I (at {wl_1} nm)"
        elif probe_type == "Ratiometric":
            idx_2 = (np.abs(wavelengths - wl_2)).argmin()
            I_vals_2 = new_intensities[idx_2, :]
            y_vals = I_vals / I_vals_2
            ylabel = f"Ratio I({wl_1}) / I({wl_2})"
            
        # Plot physical vs simulated points differently
        ax2.plot(concentrations[:num_original], y_vals[:num_original], marker='o', color='black', linestyle='-', label='Original Data')
        ax2.plot(concentrations[num_original-1:], y_vals[num_original-1:], marker='x', color='red', linestyle='--', label='Simulated Saturation')
        
        ax2.set_xlabel("Analyte Concentration (µM)")
        ax2.set_ylabel(ylabel)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)

        # -----------------------------------------
        # DATA EXPORT
        # -----------------------------------------
        st.subheader("Export Data for Origin")
        new_df = pd.DataFrame(new_intensities)
        new_df.insert(0, 'Wavelength', wavelengths)
        new_df.columns = ['Wavelength'] + [f'{c}_uM' for c in concentrations]
        
        csv = new_df.to_csv(sep='\t', index=False).encode('utf-8')
        st.download_button(
            label="Download Complete Dataset",
            data=csv,
            file_name="Saturated_Titration.txt",
            mime="text/plain",
        )
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
