import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



def compute_variance_futures(Vt, lambda_, theta, xi, T_t, T_t0, accrued_variance):
    """
    Compute variance futures price
    
    Parameters:
    Vt : float - current squared volatility
    lambda_ : float - mean reversion speed
    theta : float - long-term mean level
    xi : float - volatility of volatility
    T_t : float - time to maturity (in years)
    T_t0 : float - total time span (in years)
    accrued_variance : float - accrued variance
    
    Returns:
    float - Variance futures price (in variance points)
    """
    # print(lambda_)
    # Calculate a*(T-t) and b*(T-t)
    a_star = theta * (T_t - (1 - np.exp(-lambda_ * T_t)) / lambda_)
    b_star = (1 - np.exp(-lambda_ * T_t)) / lambda_
    
    # Formula from Derivatives_part3_bis without correction term
    return (1 / (T_t0)) * (accrued_variance + 10000*(a_star + b_star * Vt))


def analyze_individual_parameters(base_params):
    """
    Analyze how variance futures price depends on individual parameters
    """
    # Set up base parameters
    T_t = 1.0  # Time to maturity (T-t)
    accrued_variance = 0.02  # Example accrued variance
    T_t0 = 2.0  # Total time span (T-t0)
    
    # Define parameter ranges
    params_to_analyze = {
        'lambda_': (np.linspace(0.1, 5.0, 100), 2.0, 'Mean Reversion Speed', r'\lambda'),
        'theta': (np.linspace(0.01, 0.09, 100), 0.04, 'Long-term Mean Level', r'\theta'),
        'xi': (np.linspace(0.1, 1.0, 100), 0.4, 'Volatility of Volatility', r'\xi'),
        'Vt': (np.linspace(0.01, 0.09, 100), 0.04, 'Current Squared Volatility', 'V_t'),
        'accrued_variance': (np.linspace(0.0, 0.05, 100), 0.02, 'Accrued Variance', '\int_{t_0}^t V_u du')
    }
    
    # Analyze each parameter individually
    for param_name, (param_range, base_value, label, tex_symbol) in params_to_analyze.items():
        prices = []
        params = base_params.copy()
        
        for value in param_range:
            if param_name == 'accrued_variance':
                price = compute_variance_futures(
                    T_t=T_t,
                    accrued_variance=value,
                    Vt=params['Vt'],
                    lambda_=params['lambda_'],
                    theta=params['theta'],
                    xi=params['xi'],
                    T_t0=T_t0
                )
            else:
                params[param_name] = value
                price = compute_variance_futures(
                    T_t=T_t,
                    accrued_variance=accrued_variance,
                    Vt=params['Vt'],
                    lambda_=params['lambda_'],
                    theta=params['theta'],
                    xi=params['xi'],
                    T_t0=T_t0
                )
            prices.append(price)
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, prices)
        plt.axvline(x=base_value, color='r', linestyle='--', label='Base value')
        plt.title(f'Variance Futures Price Sensitivity to {label} ({tex_symbol})')
        plt.xlabel(label)
        plt.ylabel('Variance Futures Price')
        plt.grid(True)
        plt.legend()
        plt.show()

def analyze_parameter_pairs(base_params):
    """
    Analyze how variance futures price depends on parameter pairs
    """
    # Set up base parameters
    T_t = 1.0  # Time to maturity (T-t)
    accrued_variance = 0.02  # Example accrued variance
    T_t0 = 2.0  # Total time span (T-t0)
    
    # Define parameter ranges
    lambda_range = np.linspace(0.5, 5.0, 20)
    theta_range = np.linspace(0.01, 0.09, 20)
    xi_range = np.linspace(0.1, 0.8, 20)
    vt_range = np.linspace(0.01, 0.09, 20)
    accvar_range = np.linspace(0.0, 0.04, 20)
    
    # Define parameter pairs to analyze
    param_pairs = [
        ('lambda_', 'theta', lambda_range, theta_range, 'Mean Reversion Speed', 'Long-term Mean Level'),
        ('lambda_', 'Vt', lambda_range, vt_range, 'Mean Reversion Speed', 'Current Squared Volatility'),
        ('theta', 'Vt', theta_range, vt_range, 'Long-term Mean Level', 'Current Squared Volatility'),
        ('Vt', 'accrued_variance', vt_range, accvar_range, 'Current Squared Volatility', 'Accrued Variance')
    ]
    
    # Analyze each parameter pair
    for param1, param2, range1, range2, label1, label2 in param_pairs:
        # Create meshgrid
        X, Y = np.meshgrid(range1, range2)
        Z = np.zeros_like(X)
        
        # Compute prices for each parameter combination
        for i in range(len(range1)):
            for j in range(len(range2)):
                params_copy = base_params.copy()
                
                # Set first parameter
                params_copy[param1] = X[j,i]
                
                # Set second parameter or accrued variance
                if param2 == 'accrued_variance':
                    accrued_var = Y[j,i]
                    price = compute_variance_futures(
                        T_t=T_t,
                        accrued_variance=accrued_var,
                        Vt=params_copy['Vt'],
                        lambda_=params_copy['lambda_'],
                        theta=params_copy['theta'],
                        xi=params_copy['xi'],
                        T_t0=T_t0
                    )
                else:
                    params_copy[param2] = Y[j,i]
                    price = compute_variance_futures(
                        T_t=T_t,
                        accrued_variance=accrued_variance,
                        Vt=params_copy['Vt'],
                        lambda_=params_copy['lambda_'],
                        theta=params_copy['theta'],
                        xi=params_copy['xi'],
                        T_t0=T_t0
                    )
                
                Z[j,i] = price
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        heatmap = plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
        plt.colorbar(heatmap, label='Variance Futures Price')
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title(f'Variance Futures Price: {label1}-{label2} Interaction')
        plt.grid(True)
        plt.show()
        
        # Create 3D surface plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_zlabel('Variance Futures Price')
        ax.set_title(f'3D Surface: {label1}-{label2} Interaction')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

def analyze_term_structure(base_params):
    """
    Analyze term structure of variance futures prices with detailed parameter analysis
    """
    # Set up base parameters
    accrued_variance = 0  # Example accrued variance
    T_t0 = 2.0  # Assuming 2 years total time span
    
    # Define maturities range
    maturities = np.linspace(0.1, 1.9, 30)  # Maturity can't exceed T_t0
    
    # Base case for reference
    base_prices = []
    for T_t in maturities:
        price = compute_variance_futures(
            T_t=T_t,
            accrued_variance=accrued_variance,
            Vt=base_params['Vt'],
            lambda_=base_params['lambda_'],
            theta=base_params['theta'],
            xi=base_params['xi'],
            T_t0=T_t0
        )
        base_prices.append(price)
    
    # 1. Effect of Lambda (mean reversion speed)
    plt.figure(figsize=(12, 8))
    plt.plot(maturities, base_prices, 'k-', linewidth=2, label=f'Base Case (λ={base_params["lambda_"]})')
    
    for lambda_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
        if lambda_val == base_params['lambda_']:
            continue  # Skip base case as it's already plotted
        prices = []
        for T_t in maturities:
            price = compute_variance_futures(
                T_t=T_t,
                accrued_variance=accrued_variance,
                Vt=base_params['Vt'],
                lambda_=lambda_val,
                theta=base_params['theta'],
                xi=base_params['xi'],
                T_t0=T_t0
            )
            # print(
            #     "price found for lambda value : ",
            #     lambda_val,
            #     " and maturity : ",
            #     T_t,
            #     " is : ",
            #     price,
            # )
            prices.append(price)
        print(prices, "for lambda value : ", lambda_val)
        plt.plot(maturities, prices, '--', label=f'λ={lambda_val}')
    
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('Variance Futures Price')
    plt.title('Term Structure of Variance Futures: Effect of Mean Reversion Speed (λ)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 2. Effect of Theta (long-term mean)
    plt.figure(figsize=(12, 8))
    plt.plot(maturities, base_prices, 'k-', linewidth=2, label=f'Base Case (θ={base_params["theta"]})')
    
    for theta_val in [0.01, 0.03, 0.05, 0.07, 0.09]:
        if abs(theta_val - base_params['theta']) < 1e-6:
            continue  # Skip base case as it's already plotted
        prices = []
        for T_t in maturities:
            price = compute_variance_futures(
                T_t=T_t,
                accrued_variance=accrued_variance,
                Vt=base_params['Vt'],
                lambda_=base_params['lambda_'],
                theta=theta_val,
                xi=base_params['xi'],
                T_t0=T_t0
            )
            prices.append(price)
        plt.plot(maturities, prices, ':', label=f'θ={theta_val}')
    
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('Variance Futures Price')
    plt.title('Term Structure of Variance Futures: Effect of Long-term Mean (θ)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 3. Effect of Xi (volatility of volatility)
    plt.figure(figsize=(12, 8))
    plt.plot(maturities, base_prices, 'k-', linewidth=2, label=f'Base Case (ξ={base_params["xi"]})')
    
    for xi_val in [0.2, 0.3, 0.5, 0.7, 0.9]:
        if abs(xi_val - base_params['xi']) < 1e-6:
            continue  # Skip base case as it's already plotted
        prices = []
        for T_t in maturities:
            price = compute_variance_futures(
                T_t=T_t,
                accrued_variance=accrued_variance,
                Vt=base_params['Vt'],
                lambda_=base_params['lambda_'],
                theta=base_params['theta'],
                xi=xi_val,
                T_t0=T_t0
            )
            prices.append(price)
        plt.plot(maturities, prices, '-.', label=f'ξ={xi_val}')
    
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('Variance Futures Price')
    plt.title('Term Structure of Variance Futures: Effect of Volatility of Volatility (ξ)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 4. Effect of Vt vs Theta relationship
    plt.figure(figsize=(12, 8))
    
    # Various combinations of Vt and theta
    scenarios = [
        {'Vt': 0.08, 'theta': 0.03, 'label': 'Vt(0.08) > θ(0.03)', 'color': 'b'},
        {'Vt': 0.05, 'theta': 0.05, 'label': 'Vt(0.05) = θ(0.05)', 'color': 'g'},
        {'Vt': 0.03, 'theta': 0.08, 'label': 'Vt(0.03) < θ(0.08)', 'color': 'r'},
        {'Vt': 0.02, 'theta': 0.09, 'label': 'Vt(0.02) << θ(0.09)', 'color': 'm'},
        {'Vt': 0.09, 'theta': 0.02, 'label': 'Vt(0.09) >> θ(0.02)', 'color': 'c'}
    ]
    
    for scenario in scenarios:
        prices = []
        for T_t in maturities:
            price = compute_variance_futures(
                T_t=T_t,
                accrued_variance=accrued_variance,
                Vt=scenario['Vt'],
                lambda_=base_params['lambda_'],
                theta=scenario['theta'],
                xi=base_params['xi'],
                T_t0=T_t0
            )
            prices.append(price)
        plt.plot(maturities, prices, color=scenario['color'], linewidth=2, label=scenario['label'])
    
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('Variance Futures Price')
    plt.title('Term Structure of Variance Futures: Effect of Vt vs θ Relationship')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 5. Effect of accrued variance
    plt.figure(figsize=(12, 8))
    
    for acc_var in [0.01, 0.02, 0.05, 0.10, 0.15]:
        prices = []
        for T_t in maturities:
            price = compute_variance_futures(
                T_t=T_t,
                accrued_variance=acc_var,
                Vt=base_params['Vt'],
                lambda_=base_params['lambda_'],
                theta=base_params['theta'],
                xi=base_params['xi'],
                T_t0=T_t0
            )
            prices.append(price)
        plt.plot(maturities, prices, linewidth=2, label=f'Accrued Var={acc_var}')
    
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('Variance Futures Price')
    plt.title('Term Structure of Variance Futures: Effect of Accrued Variance')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 6. Comparison of different T_t0 values
    plt.figure(figsize=(12, 8))
    
    for t0_val in [1.0, 2.0, 3.0, 5.0]:
        # Adjust maturities to not exceed T_t0
        valid_mats = np.linspace(0.1, t0_val * 0.95, 30)
        prices = []
        for T_t in valid_mats:
            price = compute_variance_futures(
                T_t=T_t,
                accrued_variance=accrued_variance,
                Vt=base_params['Vt'],
                lambda_=base_params['lambda_'],
                theta=base_params['theta'],
                xi=base_params['xi'],
                T_t0=t0_val
            )
            prices.append(price)
        plt.plot(valid_mats, prices, linewidth=2, label=f'T_t0={t0_val}')
    
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('Variance Futures Price')
    plt.title('Term Structure of Variance Futures: Effect of Total Time Span (T_t0)')
    plt.grid(True)
    plt.legend()
    plt.show()

def analyze_accrued_variance_impact(base_params):
    """
    Analyze how accrued variance impacts variance futures prices at different maturities
    """
    # Set up different levels of accrued variance
    accrued_levels = [0.0, 0.01, 0.02, 0.04, 0.06]
    T_t0 = 2.0  # Total time span
    
    # Define maturities range
    maturities = np.linspace(0.1, 1.9, 30)
    
    plt.figure(figsize=(12, 8))
    
    # For each level of accrued variance
    for accrued_var in accrued_levels:
        prices = []
        for T_t in maturities:
            price = compute_variance_futures(
                T_t=T_t,
                accrued_variance=accrued_var,
                Vt=base_params['Vt'],
                lambda_=base_params['lambda_'],
                theta=base_params['theta'],
                xi=base_params['xi'],
                T_t0=T_t0
            )
            prices.append(price)
        
        plt.plot(maturities, prices, linewidth=2, label=f'Accrued Variance = {accrued_var}')
    
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('Variance Futures Price')
    plt.title('Impact of Accrued Variance on Term Structure')
    plt.grid(True)
    plt.legend()
    plt.show()

def analyze_vix_vs_variance_futures(base_params):
    """
    Compare VIX futures price with variance futures price
    """
    # This would require the VIX futures pricing function - we'll skip it for now
    # This would be an interesting comparison to see the difference between the two products
    pass

def run_comprehensive_analysis(base_params):
    """
    Run comprehensive analysis of variance futures
    """

    
    # print("1. Analyzing individual parameter sensitivity...")
    # analyze_individual_parameters(base_params)
    
    # print("\n2. Analyzing parameter pair interactions...")
    # analyze_parameter_pairs(base_params)
    
    print("\n3. Analyzing term structure effects...")
    analyze_term_structure(base_params)
    
    print("\n4. Analyzing accrued variance impact...")
    analyze_accrued_variance_impact(base_params)

def analyze_lambda_effect(base_params):
    """
    Analyze lambda effect specifically
    """
    # Set up base parameters
    T_t = 1.0  # Time to maturity (T-t)
    accrued_variance = 0.02  # Example accrued variance
    T_t0 = 2.0  # Total time span (T-t0)
    
    # Use wider range for lambda to see non-linear effects
    lambda_range = np.linspace(0.01, 10.0, 200)
    
    # Calculate a* and b* components separately
    a_star_values = []
    b_star_values = []
    total_prices = []
    
    for lambda_val in lambda_range:
        a_star = base_params['theta'] * ((T_t) - (1 - np.exp(-lambda_val * T_t)) / lambda_val)
        b_star = (1 - np.exp(-lambda_val * T_t)) / lambda_val
        
        # Store components
        a_star_values.append(a_star)
        b_star_values.append(b_star)
        
        # Calculate price
        price = (1 / T_t0) * (accrued_variance + 10000 * (a_star + b_star * base_params['Vt']))
        total_prices.append(price)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot a* component
    axs[0].plot(lambda_range, a_star_values)
    axs[0].set_title('a*(T-t) Component vs λ')
    axs[0].set_xlabel('Mean Reversion Speed (λ)')
    axs[0].set_ylabel('a*(T-t) Value')
    axs[0].grid(True)
    
    # Plot b* component
    axs[1].plot(lambda_range, b_star_values)
    axs[1].set_title('b*(T-t) Component vs λ')
    axs[1].set_xlabel('Mean Reversion Speed (λ)')
    axs[1].set_ylabel('b*(T-t) Value')
    axs[1].grid(True)
    
    # Plot total price
    axs[2].plot(lambda_range, total_prices)
    axs[2].set_title('Variance Futures Price vs λ')
    axs[2].set_xlabel('Mean Reversion Speed (λ)')
    axs[2].set_ylabel('Variance Futures Price')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Analysis of lambda effect:")
    print(f"For λ=0.01: a*={a_star_values[0]:.6f}, b*={b_star_values[0]:.6f}, price={total_prices[0]:.2f}")
    print(f"For λ=5.0: a*={a_star_values[100]:.6f}, b*={b_star_values[100]:.6f}, price={total_prices[100]:.2f}")
    print(f"For λ=10.0: a*={a_star_values[-1]:.6f}, b*={b_star_values[-1]:.6f}, price={total_prices[-1]:.2f}")

def analyze_lambda_effect_different_scenarios():
    """
    Analyze lambda effect with different Vt-theta relationships
    """
    # Set up base parameters
    T_t = 1.0  # Time to maturity (T-t)
    accrued_variance = 0.02  # Example accrued variance
    T_t0 = 2.0  # Total time span (T-t0)
    theta = 0.04  # Long-term mean
    
    # Use wider range for lambda
    lambda_range = np.linspace(0.01, 10.0, 200)
    
    # Define scenarios
    scenarios = [
        {'name': 'Vt = θ', 'Vt': 0.04, 'color': 'k'},
        {'name': 'Vt > θ', 'Vt': 0.08, 'color': 'r'},
        {'name': 'Vt < θ', 'Vt': 0.02, 'color': 'b'}
    ]
    
    plt.figure(figsize=(12, 8))
    
    for scenario in scenarios:
        prices = []
        for lambda_val in lambda_range:
            a_star = theta * ((T_t) - (1 - np.exp(-lambda_val * T_t)) / lambda_val)
            b_star = (1 - np.exp(-lambda_val * T_t)) / lambda_val
            
            # Calculate price
            price = (1 / T_t0) * (accrued_variance + 10000 * (a_star + b_star * scenario['Vt']))
            prices.append(price)
        
        plt.plot(lambda_range, prices, color=scenario['color'], linewidth=2, label=scenario['name'])
    
    plt.title('Variance Futures Price Sensitivity to λ Under Different Volatility Scenarios')
    plt.xlabel('Mean Reversion Speed (λ)')
    plt.ylabel('Variance Futures Price')
    plt.legend()
    plt.grid(True)
    plt.show()
