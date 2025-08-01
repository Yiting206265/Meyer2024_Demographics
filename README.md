# Occurrence Rate Estimator for Giant Planets and Brown Dwarfs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains a Streamlit application that calculates and visualizes the frequency of giant planets and brown dwarfs around different stellar types. The model is based on the research presented in Meyer et al. 2025, "Demographics of Planetary and Brown Dwarf Companions".

The application implements a companion population model using log-normal distributions for orbital separations and power-law distributions for mass ratios to estimate companion frequencies around different stellar types.

## Features

- Interactive selection of stellar type (M, FGK, or A stars)
- Adjustable model parameters for both brown dwarf and giant planet populations
- Visualization of mass ratio and orbital separation distributions
- Real-time calculation of companion frequencies
- Mathematical model explanation with LaTeX equations
- Customizable integration ranges for orbital separation and mass ratio

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Meyer2024_Demographics.git
cd Meyer2024_Demographics

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run frequency.py
```

The application will open in your default web browser.

## Mathematical Model

The model calculates companion frequencies using:

1. **Mass Ratio Distribution**: Power-law distributions with different exponents for giant planets and brown dwarfs
2. **Orbital Separation Distribution**: Log-normal distributions in log10 space
3. **Frequency Calculation**: Normalization constants multiplied by integrals over the mass ratio and orbital separation distributions

The mathematical details are explained within the application.

## Dependencies

- Python 3.7+
- Streamlit
- NumPy
- SciPy
- Matplotlib

## Citation

If you use this code in your research, please cite:

```
Meyer et al. (2025). Demographics of Planetary and Brown Dwarf Companions. [Journal details]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Yiting Li