# Gaussian Deconvolution Tool - User Guide

## Overview
This tool performs Gaussian deconvolution of chromatogram data, allowing you to analyze and fit peaks to your data with various calibration and visualization options.

## Data Upload

### Chromatogram Data
- **File Format**: Upload tab-delimited `.txt` or `.csv` files, other `.csv` file formats which are not tab-delimited should also work 
- **Required Columns**:
  - **Column 1**: Retention Time
  - **Column 2**: Signal Intensity
- **Note**: The first 2 rows are automatically skipped as headers. Your actual data should start from the 3rd row

### Calibration Options
Choose one of the following calibration methods:

#### 📁 Upload Calibration File
- Upload calibration chromatogram data
- Same file format requirements as chromatogram data
- First 2 rows skipped as headers

#### 📝 Enter Calibration Equation
- Manually enter calibration curve equation
- Supports **linear** or **quadratic** equations

#### ⚡ No Calibration
- If graphing against retention time only, no calibration data is required

## Configuration Parameters

### Number of Peaks
- **Set between 1-10 peaks**

### Mode Selection
- **Change Between Retention Time and Molecular Weigth Graphing**
### Basic Settings
- **Mode Selection**:
  - **MW Mode**: Specify MW Lower Bound and MW Upper Bound
  - **RT Mode**: Specify RT Lower Bound and RT Upper Bound
- **Peak Width Search**: Adjust for narrower or broader peaks
- **Y-axis Bounds**: Set display range for Y-axis
- **Normalization**: All data normalized against the highest peak

### Baseline Correction
- **None**: No baseline correction (default)
- **Arpls**: **Recommended** automatic correction method
- **Flat**: Adds or subtracts constant value
- **Linear**: Corrects using linear equation
- **Quadratic**: Corrects using quadratic equation

*For manual methods, specify values/ranges where baseline should be applied*

## Visualization & Analysis

### Peak Appearance
- Rename **Peaks** and **Original Data**
- Customize colors for each element
- **Enable/Disable Peaks**: Uncheck boxes to hide specific peaks
- **Sum of Gaussians**: Plot combined sum of all selected peaks

### Residual Plots
- **Residual Calculation**: Sum of Gaussians subtracted from original data
- **Color Customization**: Change residual plot colors
- **Results Table**: View area comparison of fitted gaussians vs. original chromatogram

### Peak Integration
- **Enable Integration**: Calculate molecular weight properties
- **Integration Range**: 
  - Dotted red vertical lines indicate limits
  - Red shading shows integration area
  - Specify range for each peak (highest to lowest MW)
- **Integration Rules**: Only positive Y-values are integrated
- **Output Metrics**: 
  - **Mn**, **Mw**, and **Dispersity** in Molecular Weight Results table
  - **Integration Results**: Area breakdown by peak

### Figure Appearance
- Customize graph styling
- Rename axis labels
- Various visualization options

## Graph Controls

### Auto-Update
- **Enabled by default**: Graph updates automatically with changes
- **Recommendation**: Turn off when making multiple rapid changes, then re-enable

### Manual Update
- **Update Graph Button**: Manually refresh graph if changes aren't reflected
- Use if graph doesn't display properly after modifications

---

**Tip**: If experiencing peak fitting issues, try adjusting the **Peak Width Search** parameters in Basic Settings.