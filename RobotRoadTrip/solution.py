from scipy.integrate import dblquad
from scipy.optimize import minimize_scalar

def objective_function_limits(a):
    """
    Calculate the objective function in the double limit:
    lim[N→∞] lim[z→0+] f(z,N)
    """
    def integrand(v1, v2):
        v_slow = min(v1, v2)
        v_fast = max(v1, v2)
        
        if abs(v_fast - v_slow) < 1e-10:
            return 0.0
        
        # Interaction probability weight from geometric analysis
        weight = 2 * abs(v_fast - v_slow) / (v1 * v2)
        
        cost = 0.0
        
        # Both in slow lane: slower car decelerates to 0
        if v_slow <= a and v_fast <= a:
            cost += v_slow**2
        
        # Both in fast lane: slower car decelerates to 'a'
        elif v_slow >= a and v_fast >= a:
            cost += (v_slow - a)**2
        
        return weight * cost
    
    result, _ = dblquad(integrand, 1, 2, lambda v1: 1, lambda v1: 2)
    return result

def solve_robot_road_trip():
    result = minimize_scalar(
        objective_function_limits, 
        bounds=(1, 2), 
        method='bounded',
        options={'xatol': 1e-15}
    )
    
    return result.x

if __name__ == "__main__":
    optimal_a = solve_robot_road_trip()
    print(f"Optimal a: {optimal_a:.10f}")
