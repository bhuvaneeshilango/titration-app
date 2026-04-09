import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Titration & Linearity Analyzer")
st.write("Calculate exact concentrations, select your linear regime, and extract proper $y=mx+c$ statistics.")

# ==========================================
# 1. FILE UPLOAD & SETUP
# ==========================================
uploaded_file = st.file_uploader("Upload Titration Data (.txt or .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        # Handle both spaces and tabs gracefully
        df = pd.read_csv(uploaded_file, sep='\s+')
        wavelengths = df.iloc[:, 0].values
        intensities = df.iloc[:, 1:].values
        num_curves = intensities.shape[1]
        
        st.success(f"Loaded baseline + {num_curves - 1} additions.")

        # ==========================================
        # 2. SIDEBAR: CONCENTRATION MATH
        # ==========================================
        st.sidebar.header("1. Experimental Setup")
        V_0 = st.sidebar.number_input("Initial Probe Volume (µL)", value=2000.0, step=100.0)
        C_stock = st.sidebar.number_input("Analyte Stock Conc. (µM)", value=1000.0, step=10.0)
        V_add = st.sidebar.number_input("Volume per addition (µL)", value=3.0, step=0.5)

        # Calculate exact concentrations accounting for volume change
        # Formula: C_final = (n * V_add * C_stock) / (V_0 + n * V_add)
        concs = [0.0] # Baseline is 0
        for n in range(1, num_curves):
            c_final = (n * V_add * C_stock) / (V_0 + n * V_add)
            concs.append(c_final)
        concs = np.array(concs)

        # ==========================================
        # 3. SIDEBAR: PROBE METRICS
        # ==========================================
        st.sidebar.header("2. Probe Metrics")
        probe_type = st.sidebar.selectbox("Probe Behavior", ["Turn-On (I - I0)", "Quenching (I0 - I)", "Ratiometric (I1 / I2)"])
        wl_1 = st.sidebar.number_input("Monitor Wavelength 1 (nm)", int(wavelengths.min()), int(wavelengths.max()), 466)
        
        if probe_type == "Ratiometric (I1 / I2)":
            wl_2 = st.sidebar.number_input("Monitor Wavelength 2 (nm)", int(wavelengths.min()), int(wavelengths.max()), 390)

        # ==========================================
        # 4. SIDEBAR: LINEAR RANGE SELECTOR
        # ==========================================
        st.sidebar.header("3. Linear Fit Range")
        st.sidebar.write("*Select which addition points to include in the statistical fit. All points will be plotted.*")
        
        # Dual slider to pick start and end points for the fit
        fit_range = st.sidebar.slider("Select Point Indices (0 = Baseline)", 
                                      min_value=0, 
                                      max_value=num_curves-1, 
                                      value=(0, min(10, num_curves-1)))
        
        start_idx, end_idx = fit_range[0], fit_range[1]

        # ==========================================
        # 5. DATA PROCESSING (Y-AXIS)
        # ==========================================
        idx_1 = (np.abs(wavelengths - wl_1)).argmin()
        I_vals_1 = intensities[idx_1, :]
        I_0_1 = I_vals_1[0]

        if probe_type == "Turn-On (I - I0)":
            y_vals = I_vals_1 - I_0_1
            ylabel = f"$I - I_0$ (at {wl_1} nm)"
        elif probe_type == "Quenching (I0 - I)":
            y_vals = I_0_1 - I_vals_1
            ylabel = f"$I_0 - I$ (at {wl_1} nm)"
        else: # Ratiometric
            idx_2 = (np.abs(wavelengths - wl_2)).argmin()
            I_vals_2 = intensities[idx_2, :]
            y_vals = I_vals_1 / I_vals_2
            ylabel = f"Ratio $I_{{{wl_1}}} / I_{{{wl_2}}}$"

        # ==========================================
        # 6. STATISTICAL LINEAR REGRESSION
        # ==========================================
        # Extract only the user-selected range for the math
        fit_x = concs[start_idx : end_idx + 1]
        fit_y = y_vals[start_idx : end_idx + 1]

        # Calculate slope (m) and intercept (c)
        coeffs = np.polyfit(fit_x, fit_y, 1)
        m, c = coeffs

        # Calculate R-squared
        y_fit_values = m * fit_x + c
        ss_res = np.sum((fit_y - y_fit_values)**2)
        ss_tot = np.sum((fit_y - np.mean(fit_y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Generate the fit line coordinates for plotting
        line_x = np.array([concs[start_idx], concs[end_idx]])
        line_y = m * line_x + c

        # ==========================================
        # 7. VISUALIZATION
        # ==========================================
        st.subheader("Linearity Plot & Statistics")
        
        # Display Stats clearly
        col1, col2, col3 = st.columns(3)
        col1.metric("Slope (m)", f"{m:.4f}")
        col2.metric("Intercept (c)", f"{c:.4f}")
        col3.metric("R² Value", f"{r_squared:.5f}")

        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Plot ALL points (Excluded points in gray, Included points in blue)
        ax.scatter(concs, y_vals, color='lightgray', label='Excluded Data', zorder=1)
        ax.scatter(fit_x, fit_y, color='blue', label='Fitted Data', zorder=2)
        
        # Plot the Line of Best Fit
        ax.plot(line_x, line_y, color='red', linestyle='--', label=f'Fit: y = {m:.3f}x + {c:.3f}', zorder=3)

        ax.set_xlabel("Analyte Concentration (µM)")
        ax.set_ylabel(ylabel)
        ax.set_title("Photophysical Linearity Assessment")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
        st.pyplot(fig)

        # ==========================================
        # 8. EXPORT DATA FOR ORIGIN
        # ==========================================
        st.subheader("Export Data")
        
        # Create a clean DataFrame with actual concentrations as headers
        export_df = pd.DataFrame(intensities)
        export_df.insert(0, 'Wavelength', wavelengths)
        
        # Name columns: Wavelength, 0.0_uM, 1.49_uM, 2.98_uM, etc.
        col_names = ['Wavelength'] + [f"{round(c, 2)}_uM" for c in concs]
        export_df.columns = col_names
        
        csv = export_df.to_csv(sep='\t', index=False).encode('utf-8')
        st.download_button(
            label="Download Concentration-Calibrated Data",
            data=csv,
            file_name="Calibrated_Titration.txt",
            mime="text/plain",
        )
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
