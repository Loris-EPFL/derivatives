### **Question 6 — VIX Futures Pricing**

> *"Use your preferred language to write a code that computes the futures prices in a VIX future as a function of time to maturity, the current squared volatility \( V_t \), and the parameters \((\lambda, \theta, \xi)\) of the squared volatility process. Analyze numerically how the result depends on the parameters \((\lambda, \theta, \xi)\)."*

- **Code file:** `VIX.py` (contains the futures pricing formula)
- **Analysis notebook:** `vix_analysis.ipynb` (runs the pricing function and explores parameter sensitivity)

---

### **Question 7 — Variance Futures Pricing**

> *"Write a code that computes the futures prices in Variance futures as a function of time to maturity \( T - t \), the accrued variance, the current squared volatility \( V_t \), and the parameters \((\lambda, \theta, \xi)\). Analyze graphically how the result depends on the parameters."*

- **Code file:** `Variance.py` (contains the pricing formula)
- **Analysis notebook:** `variance_analysis.ipynb` (runs the model and visualizes parameter effects)

---

### **Question 8 — Calibration to Market Data**

> *"Use the code you developed in the previous two questions together with the market data in the Excel file `Fin404-2025-VIXnCo-Data.xlsx` to calibrate the current squared volatility \( V_t \) and the parameters \((\lambda, \theta, \xi)\)."*

- **Calibration notebook:** `calibration_final.ipynb`  
  Contains all required data, pricing functions, and calibration procedures.
