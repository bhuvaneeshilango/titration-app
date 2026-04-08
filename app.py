import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Photophysical Titration Interpolator")
st.write("Upload your raw titration data (Wavelength + Intensity columns) to mathematically expand your curve steps.")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload Titration Data (.txt or .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    # 2. Read the Data
    try:
        df = pd.read_csv(uploaded_file, sep='\s+')
        wavelengths = df.iloc[:, 0].values
        intensities = df.iloc[:, 1:].values
        
        num_original_curves = intensities.shape[1]
        st.success(f"Successfully loaded {num_original_curves} titration curves!")
        
        # 3. User Controls
        st.subheader("Interpolation Settings")
        target_curves = st.slider("How many total curves do you want?", 
                                  min_value=num_original_curves, 
                                  max_value=50, 
                                  value=15)
        
        # 4. The Math: Map old columns to new steps
        old_steps = np.linspace(0, 1, num_original_curves)
        new_steps = np.linspace(0, 1, target_curves)
        
        new_intensities = np.zeros((len(wavelengths), target_curves))
        
        # Interpolate every single wavelength across the reaction progress
        for i in range(len(wavelengths)):
            new_intensities[i, :] = np.interp(new_steps, old_steps, intensities[i, :])
            
        # 5. Build the New DataFrame
        new_df = pd.DataFrame(new_intensities)
        new_df.insert(0, 'Wavelength', wavelengths)
        
        # Format column names for Origin (e.g., y1, y2, y3...)
        new_df.columns = ['Wavelength'] + [f'y{i+1}' for i in range(target_curves)]
        
        # 6. Visual Preview
        st.subheader("Preview")
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(target_curves):
            ax.plot(wavelengths, new_intensities[:, i], color='blue', alpha=(i+1)/target_curves)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        st.pyplot(fig)
        
        # 7. Download Button for Origin
        st.subheader("Export for Origin")
        csv = new_df.to_csv(sep='\t', index=False).encode('utf-8')
        st.download_button(
            label="Download Interpolated Data",
            data=csv,
            file_name="Interpolated_Titration.txt",
            mime="text/plain",
        )
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
