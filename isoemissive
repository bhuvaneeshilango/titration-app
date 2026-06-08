import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Isoemissive Extrapolator & Slope Controller")
st.write("Generate clean, wobble-free extrapolated spectra from partial data while locking the isoemissive point.")

# ==========================================
# 1. FILE UPLOAD & SETUP
# ==========================================
uploaded_file = st.file_uploader("Upload Partial Titration Data (e.g., first 3-5 curves)", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep='\s+')
        wavelengths = df.iloc[:, 0].values
        intensities_orig = df.iloc[:, 1:].values
        num_orig = intensities_orig.shape[1]
        
        st.success(f"Loaded baseline + {num_orig - 1} physical additions.")

        # ==========================================
        # 2. SIDEBAR: EXPERIMENTAL SETUP
        # ==========================================
        st.sidebar.header("1. Extrapolation Goals")
        total_curves = st.sidebar.number_input("Total curves to generate (e.g., 15)", min_value=num_orig, max_value=50, value=15)
        monitor_wl = st.sidebar.number_input("Monitor Wavelength (nm)", int(wavelengths.min()), int(wavelengths.max()), 466)
        
        st.sidebar.header("2. Slope & Rate Control")
        st.sidebar.write("*Control how aggressively the quenching/enhancement proceeds past your uploaded data.*")
        
        # 1.0 means it continues the exact trend of your uploaded data. 
        # 1.5 means it quenches 50% faster. 0.5 means it quenches 50% slower.
        slope_multiplier = st.sidebar.slider("Rate Multiplier (Slope Control)", 0.1, 3.0, 1.0, 0.05)
        
        # Determine if it stays linear or curves into saturation
        trend_type = st.sidebar.radio("Trend Geometry", ["Strictly Linear", "Asymptotic (Saturating)"])
        if trend_type == "Asymptotic (Saturating)":
            sharpness = st.sidebar.slider("Saturation Curvature", 0.1, 1.0, 0.3)

        # ==========================================
        # 3. MATH: THE STRUCTURAL VECTOR
        # ==========================================
        S_base = intensities_orig[:, 0]
        S_last = intensities_orig[:, -1]
        
        # This vector defines the pure chemical shift. At the isoemissive point, Delta_S = 0.
        Delta_S = S_last - S_base 
        
        new_intensities = np.zeros((len(wavelengths), total_curves))
        
        # Fill in the original data exactly as uploaded
        for i in range(num_orig):
            new_intensities[:, i] = intensities_orig[:, i]

        # Calculate the mathematical progress of the uploaded data
        # Let's define the base as progress=0, and the last uploaded curve as progress=1
        
        for i in range(num_orig, total_curves):
            # How many steps past the uploaded data are we?
            step_projection = (i - (num_orig - 1)) * slope_multiplier
            
            if trend_type == "Strictly Linear":
                # Continues linearly based on your slope multiplier
                progress_factor = 1.0 + step_projection
            else:
                # Curves smoothly based on the sharpness
                progress_factor = 1.0 + (3.0 * slope_multiplier) * (1 - np.exp(-sharpness * step_projection))
            
            # Apply the progress to the structural vector
            # Because Delta_S is 0 at the isoemissive point, that specific wavelength mathematically cannot move.
            generated_spectrum = S_base + (progress_factor * Delta_S)
            
            # Ensure intensity doesn't mathematically drop below zero
            new_intensities[:, i] = np.clip(generated_spectrum, a_min=0, a_max=None)

        # ==========================================
        # 4. PLOTTING
        # ==========================================
        # Extract monitor wavelength trend for the linearity plot
        wl_idx = (np.abs(wavelengths - monitor_wl)).argmin()
        trend_y = new_intensities[wl_idx, :]
        x_axis_steps = np.arange(total_curves)

        st.subheader("1. Isoemissive Spectral Expansion")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for i in range(total_curves):
            if i < num_orig:
                # Highlight the uploaded data you provided
                ax1.plot(wavelengths, new_intensities[:, i], color='black', linewidth=1.5, alpha=0.8)
            else:
                # Plot the extrapolated data with a smooth color gradient
                alpha_val = np.clip(0.3 + (i/total_curves)*0.7, 0, 1)
                ax1.plot(wavelengths, new_intensities[:, i], color='blue', linestyle='--', alpha=alpha_val)

        # Add a custom legend
        ax1.plot([], [], color='black', linewidth=1.5, label='Uploaded Physical Data')
        ax1.plot([], [], color='blue', linestyle='--', label='Extrapolated Trend')
        ax1.axvline(x=monitor_wl, color='red', linestyle=':', alpha=0.5, label=f'Monitor ({monitor_wl} nm)')
        
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Intensity")
        ax1.legend()
        st.pyplot(fig1)

        st.subheader("2. Slope & Quenching Rate Control")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        
        ax2.plot(x_axis_steps[:num_orig], trend_y[:num_orig], marker='o', color='black', linestyle='-', label='Uploaded Data')
        ax2.plot(x_axis_steps[num_orig-1:], trend_y[num_orig-1:], marker='x', color='red', linestyle='--', label='Generated Trajectory')
        
        ax2.set_xlabel("Addition Step")
        ax2.set_ylabel(f"Intensity at {monitor_wl} nm")
        ax2.set_title("Intensity Modification Tracker")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        st.pyplot(fig2)

        # ==========================================
        # 5. EXPORT FOR ORIGIN
        # ==========================================
        export_df = pd.DataFrame(new_intensities)
        export_df.insert(0, 'Wavelength', wavelengths)
        export_df.columns = ['Wavelength'] + [f"y{i+1}" for i in range(total_curves)]
        
        csv = export_df.to_csv(sep='\t', index=False).encode('utf-8')
        st.download_button(
            label="Download Complete Dataset",
            data=csv,
            file_name="Isoemissive_Extrapolation.txt",
            mime="text/plain",
        )
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
