import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Saturated Isotherm & Titration Modeler")
st.write("Project saturation plateaus with exact volume-corrected concentration math.")

# ==========================================
# 1. FILE UPLOAD & SETUP
# ==========================================
uploaded_file = st.file_uploader("Upload Titration Data (.txt or .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep='\s+')
        wavelengths = df.iloc[:, 0].values
        intensities = df.iloc[:, 1:].values
        num_original = intensities.shape[1]

        # ==========================================
        # 2. SIDEBAR: EXPERIMENTAL SETUP
        # ==========================================
        st.sidebar.header("1. Experimental Setup")
        V_0 = st.sidebar.number_input("Initial Probe Volume (µL)", value=2000.0, step=100.0)
        C_stock = st.sidebar.number_input("Analyte Stock Conc. (µM)", value=250.0, step=10.0)
        V_add_per_step = st.sidebar.number_input("Volume per addition (µL)", value=3.0, step=0.5)

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
        
        # Slider to change the total number of points (e.g., 15 up to 30)
        total_points = st.sidebar.slider("Total Additions (Real + Simulated)", min_value=num_original, max_value=40, value=max(20, num_original))
        
        # Slider to define where the linear trend ends and saturation begins
        pivot_idx = st.sidebar.slider("Pivot Point (Onset of Saturation)", min_value=1, max_value=num_original-1, value=min(10, num_original-1))
        
        sat_limit = st.sidebar.slider("Saturation Limit Factor", 1.0, 2.0, 1.15, 0.01)
        sharpness = st.sidebar.slider("Curve Sharpness", 0.1, 1.0, 0.3)

        # ==========================================
        # 5. MATH: EXACT CONCENTRATIONS
        # ==========================================
        volumes_added = np.arange(total_points) * V_add_per_step
        total_volumes = V_0 + volumes_added
        # C_final = (V_added * C_stock) / (V_initial + V_added)
        concs = (volumes_added * C_stock) / total_volumes

        # ==========================================
        # 6. MATH: ASYMPTOTIC EXTRAPOLATION
        # ==========================================
        new_intensities = np.zeros((len(wavelengths), total_points))

        # Fill up to the pivot point with your real experimental data
        for i in range(pivot_idx + 1):
            new_intensities[:, i] = intensities[:, i]

        # Extract the start and the pivot spectra
        S_base = intensities[:, 0]
        S_pivot = intensities[:, pivot_idx]

        # Simulate the remaining points using a Langmuir-style asymptote
        for i in range(pivot_idx + 1, total_points):
            step_past_pivot = i - pivot_idx
            alpha = sat_limit - (sat_limit - 1.0) * np.exp(-sharpness * step_past_pivot)
            new_intensities[:, i] = np.clip(S_base + alpha * (S_pivot - S_base), a_min=0, a_max=None)

        # ==========================================
        # 7. METRIC EXTRACTION & DATA TABLE
        # ==========================================
        idx_1 = (np.abs(wavelengths - wl_1)).argmin()
        I_vals_1 = new_intensities[idx_1, :]
        I_0_1 = I_vals_1[0]

        if "Turn-On" in probe_type:
            y_vals = I_vals_1 - I_0_1
            ylabel = f"I - I0 (at {wl_1} nm)"
        elif "Quenching" in probe_type:
            y_vals = I_0_1 - I_vals_1
            ylabel = f"I0 - I (at {wl_1} nm)"
        else:
            idx_2 = (np.abs(wavelengths - wl_2)).argmin()
            I_vals_2 = new_intensities[idx_2, :]
            y_vals = I_vals_1 / I_vals_2
            ylabel = f"Ratio I({wl_1}) / I({wl_2})"

        st.subheader("Titration Parameters & Extracted Values")
        
        # Build the detailed dataframe
        table_data = {
            "Addition #": np.arange(total_points),
            "Vol Added (µL)": volumes_added,
            "Total Vol (µL)": total_volumes,
            "[Analyte] (µM)": np.round(concs, 3),
            f"I at {wl_1}nm": np.round(I_vals_1, 2),
        }
        if "Ratiometric" in probe_type:
            table_data[f"I at {wl_2}nm"] = np.round(I_vals_2, 2)
            
        table_data["Y-Axis Metric"] = np.round(y_vals, 3)
        table_data["Data Source"] = ["Real" if i <= pivot_idx else "Simulated" for i in range(total_points)]
        
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df, use_container_width=True)

        # ==========================================
        # 8. PLOTTING
        # ==========================================
        st.subheader("1. Full Spectra (Real + Simulated)")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for i in range(total_points):
            color = 'blue' if i <= pivot_idx else 'red'
            linestyle = '-' if i <= pivot_idx else '--'
            alpha_val = np.clip(0.3 + (i/total_points)*0.7, 0, 1)
            
            # Only label a few key curves to keep the legend clean
            label = f"{np.round(concs[i], 2)} µM" if i == 0 or i == pivot_idx or i == total_points-1 else ""
            ax1.plot(wavelengths, new_intensities[:, i], color=color, linestyle=linestyle, alpha=alpha_val, label=label)

        ax1.axvline(x=wl_1, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Intensity")
        ax1.legend()
        st.pyplot(fig1)

        st.subheader("2. Binding Isotherm")
        fig2, ax2 = plt.subplots(figsize=(7, 5))

        # Plot Real vs Simulated points
        ax2.plot(concs[:pivot_idx+1], y_vals[:pivot_idx+1], marker='o', color='black', linestyle='-', label='Real Data (Pre-Pivot)')
        ax2.plot(concs[pivot_idx:], y_vals[pivot_idx:], marker='x', color='red', linestyle='--', label='Simulated Saturation')

        ax2.set_xlabel("Analyte Concentration (µM)")
        ax2.set_ylabel(ylabel)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)

        # ==========================================
        # 9. EXPORT FOR ORIGIN
        # ==========================================
        export_df = pd.DataFrame(new_intensities)
        export_df.insert(0, 'Wavelength', wavelengths)
        export_df.columns = ['Wavelength'] + [f"{round(c, 2)}_uM" for c in concs]
        
        csv = export_df.to_csv(sep='\t', index=False).encode('utf-8')
        st.download_button(
            label="Download Calibrated Saturation Data",
            data=csv,
            file_name="Calibrated_Saturation.txt",
            mime="text/plain",
        )
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
