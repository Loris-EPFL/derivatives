import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def compute_vix_futures(T_t, Vt, lambda_, theta, xi, eta=30/365):
    """
    Compute VIX futures price
    
    Parameters:
    T_t : float - time to maturity (T-t)
    Vt : float - current squared volatility
    lambda_ : float - mean reversion speed
    theta : float - long-term mean level
    xi : float - volatility of volatility
    eta : float - VIX time window (default 30 days)
    
    Returns:
    float - VIX futures price
    """
    # Constants from derivation
    a_prime = theta * (eta - (1 - np.exp(-lambda_ * eta)) / lambda_)
    b_prime = (1 - np.exp(-lambda_ * eta)) / lambda_
    
    def d_function(tau, s):
        """Compute d(τ;s) function"""
        numerator = 2 * lambda_ * s
        denominator = np.exp(lambda_ * tau) * (2 * lambda_ + xi**2 * s) - xi**2 * s
        return numerator / denominator
    
    def c_function(tau, s):
        """Compute c(τ;s) function"""
        term1 = -2 * theta * lambda_ / xi**2
        term2 = lambda_ * tau + np.log(2 * lambda_)
        term3 = np.log(np.exp(lambda_ * tau) * (2 * lambda_ + xi**2 * s) - xi**2 * s)
        return term1 * (term2 - term3)
    
    def ell_function(s):
        """Compute ℓ(s,T-t,Vt) function"""
        return (s * a_prime + 
                c_function(T_t, s * b_prime) + 
                d_function(T_t, s * b_prime) * Vt)
    
    def integrand(s):
        """Compute the integrand for the VIX futures formula"""
        if s < 1e-10:  # Handle near-zero values
            return 0
        try:
            result = (1 - np.exp(-ell_function(s))) / (s**1.5)
            return result
        except (OverflowError, ZeroDivisionError, RuntimeWarning):
            return 0
    
    # Numerical integration with upper bound adjustment for practical computation
    integral, _ = integrate.quad(integrand, 0, 100)  # Upper bound of 100 is sufficient in most cases
    
    # Final VIX futures price formula from Derivatives_part3 file
    return 50 / np.sqrt(np.pi * eta) * integral

def analyze_single_parameter(base_params, maturity=1.0):
    """
    Analyze sensitivity to individual parameters
    """
    # Define parameter ranges
    params_to_analyze = {
        'lambda_': (np.linspace(0.5, 5.0, 50), 2.0, 'Mean Reversion Speed', r'\lambda'),
        'theta': (np.linspace(0.01, 0.09, 50), 0.04, 'Long-term Mean Level', r'\theta'),
        'xi': (np.linspace(0.1, 1.0, 50), 0.4, 'Volatility of Volatility', r'\xi'),
        'Vt': (np.linspace(0.01, 0.09, 50), 0.04, 'Current Squared Volatility', 'V_t')
    }
    
    # Analyze each parameter individually
    for param_name, (param_range, base_value, label, tex_symbol) in params_to_analyze.items():
        prices = []
        params = base_params.copy()
        
        for value in param_range:
            params[param_name] = value
            price = compute_vix_futures(maturity, **params)
            prices.append(price)
            
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, prices)
        plt.axvline(x=base_value, color='r', linestyle='--', label='Base value')
        plt.title(f'VIX Futures Price Sensitivity to {label} ({tex_symbol})')
        plt.xlabel(label)
        plt.ylabel('VIX Futures Price')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Calculate sensitivity metrics
        base_idx = np.abs(param_range - base_value).argmin()
        base_price = prices[base_idx]
        
        # Find low and high values for elasticity calculation
        low_idx = max(0, base_idx - 5)
        high_idx = min(len(param_range) - 1, base_idx + 5)
        
        low_price, high_price = prices[low_idx], prices[high_idx]
        low_value, high_value = param_range[low_idx], param_range[high_idx]
        
        # Approximate elasticity
        pct_price_change = (high_price - low_price) / base_price
        pct_param_change = (high_value - low_value) / base_value
        elasticity = pct_price_change / pct_param_change if pct_param_change != 0 else 0
        
        print(f"Parameter: {label} ({tex_symbol})")
        print(f"Base value: {base_value}, Base price: {base_price:.4f}")
        print(f"Elasticity around base value: {elasticity:.4f}")
        print("-" * 50)

def analyze_parameter_pairs(base_params, maturity=1.0):
    """
    Analyze pairwise parameter interactions
    """
    # Define parameter ranges (using fewer points for faster computation)
    lambda_range = np.linspace(0.5, 5.0, 15)
    theta_range = np.linspace(0.01, 0.09, 15)
    xi_range = np.linspace(0.2, 0.6, 15)
    vt_range = np.linspace(0.01, 0.09, 15)
    
    # Define parameter pairs to analyze
    param_pairs = [
        ('lambda_', 'theta', lambda_range, theta_range, 'Mean Reversion Speed', 'Long-term Mean Level'),
        ('lambda_', 'xi', lambda_range, xi_range, 'Mean Reversion Speed', 'Volatility of Volatility'),
        ('theta', 'xi', theta_range, xi_range, 'Long-term Mean Level', 'Volatility of Volatility'),
        ('lambda_', 'Vt', lambda_range, vt_range, 'Mean Reversion Speed', 'Current Squared Volatility'),
        ('theta', 'Vt', theta_range, vt_range, 'Long-term Mean Level', 'Current Squared Volatility'),
        ('xi', 'Vt', xi_range, vt_range, 'Volatility of Volatility', 'Current Squared Volatility')
    ]
    
    # Analyze each parameter pair
    for param1, param2, range1, range2, label1, label2 in param_pairs:
        # Create meshgrid
        X, Y = np.meshgrid(range1, range2)
        Z = np.zeros_like(X)
        
        # Compute prices for each parameter combination
        for i in range(len(range1)):
            for j in range(len(range2)):
                params = base_params.copy()
                params[param1] = X[j,i]
                params[param2] = Y[j,i]
                Z[j,i] = compute_vix_futures(maturity, **params)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        heatmap = plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
        plt.colorbar(heatmap, label='VIX Futures Price')
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title(f'VIX Futures Price: {label1}-{label2} Interaction')
        plt.grid(True)
        plt.show()
        
        # Create 3D surface plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_zlabel('VIX Futures Price')
        ax.set_title(f'3D Surface: {label1}-{label2} Interaction')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

def analyze_term_structure(base_params):
    """
    Analyze term structure effects
    """
    # Define maturities range
    maturities = np.linspace(0.1, 2.0, 30)
    
    # Base case
    base_prices = [compute_vix_futures(T, **base_params) for T in maturities]
    
    plt.figure(figsize=(12, 8))
    plt.plot(maturities, base_prices, 'k-', linewidth=2, label='Base Case')
    
    # Vary lambda
    for lambda_val in [1.0, 3.0, 5.0]:
        params = base_params.copy()
        params['lambda_'] = lambda_val
        prices = [compute_vix_futures(T, **params) for T in maturities]
        plt.plot(maturities, prices, '--', label=f'λ={lambda_val}')
    
    # Vary theta
    for theta_val in [0.02, 0.04, 0.08]:
        params = base_params.copy()
        params['theta'] = theta_val
        prices = [compute_vix_futures(T, **params) for T in maturities]
        plt.plot(maturities, prices, ':', label=f'θ={theta_val}')
    
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('VIX Futures Price')
    plt.title('Term Structure of VIX Futures Prices')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Analyze case where Vt > theta vs Vt < theta
    plt.figure(figsize=(12, 8))
    
    # Case 1: Vt > theta (high current volatility)
    params1 = base_params.copy()
    params1['Vt'] = 0.08
    params1['theta'] = 0.03
    prices1 = [compute_vix_futures(T, **params1) for T in maturities]
    
    # Case 2: Vt < theta (low current volatility)
    params2 = base_params.copy()
    params2['Vt'] = 0.03
    params2['theta'] = 0.08
    prices2 = [compute_vix_futures(T, **params2) for T in maturities]
    
    plt.plot(maturities, prices1, 'b-', linewidth=2, label='Case Vt > θ')
    plt.plot(maturities, prices2, 'r-', linewidth=2, label='Case Vt < θ')
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('VIX Futures Price')
    plt.title('Term Structure: Effect of Vt vs θ Relationship')
    plt.grid(True)
    plt.legend()
    plt.show()

def analyze_theta_vt_at_different_maturities(base_params):
    """
    Analyze Vt-theta interaction at different maturities
    """
    # Define parameter ranges
    theta_range = np.linspace(0.01, 0.09, 15)
    vt_range = np.linspace(0.01, 0.09, 15)
    
    # Define maturities to analyze
    maturities = [0.1, 0.5, 1.0, 2.0]
    
    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, maturity in enumerate(maturities):
        X, Y = np.meshgrid(vt_range, theta_range)
        Z = np.zeros_like(X)
        
        for j in range(len(vt_range)):
            for k in range(len(theta_range)):
                params = base_params.copy()
                params['Vt'] = X[k,j]
                params['theta'] = Y[k,j]
                Z[k,j] = compute_vix_futures(maturity, **params)
        
        im = axes[i].pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
        axes[i].set_title(f'Maturity T-t = {maturity} years')
        axes[i].set_xlabel('Current Squared Volatility (Vt)')
        axes[i].set_ylabel('Long-term Mean Level (θ)')
        fig.colorbar(im, ax=axes[i], label='VIX Futures Price')
        axes[i].grid(True)
    
    plt.suptitle('Effect of Maturity on Vt-θ Interaction', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def analyze_complete(base_params):
    """
    Run all analyses
    """
    print("1. Analyzing individual parameter sensitivity...")
    analyze_single_parameter(base_params)
    
    print("\n2. Analyzing parameter pair interactions...")
    analyze_parameter_pairs(base_params)
    
    print("\n3. Analyzing term structure effects...")
    analyze_term_structure(base_params)
    
    print("\n4. Analyzing theta-Vt interaction at different maturities...")
    analyze_theta_vt_at_different_maturities(base_params)

