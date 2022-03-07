import numpy as np
import sympy as sym

class gamma():
    def __init__(self, var_a, var_b = 1):
        self.var_a = var_a
        self.var_b = var_b
    
    def function(self, print_f_flag = False, print_F_flag = False):
        # variable t
        var_t = sym.symbols('t')
        
        # function and integral
        f_t = var_t ** (self.var_a - 1) * sym.exp(-var_t)
        gamma = sym.Integral(f_t, (var_t, 0, sym.oo))

        F_t = sym.integrate(f_t, var_t)

        F_b = sym.limit(F_t, var_t, sym.oo)
        F_a = sym.limit(F_t, var_t, 0)
        # F_a = F_t.subs(var_t, 0).evalf()

        if print_f_flag:
            print(sym.pretty(gamma))

        if print_F_flag:
            print(F_t)
        
        return sym.N(F_b - F_a)

    def pdf(self, var_x):
        y = (1 / self.function()) * (self.var_b ** self.var_a) * (var_x ** (self.var_a - 1)) * np.exp(-self.var_b * var_x)
        
        return y.astype(float)

class beta():
    # mean, mode, var
    def __init__(self, var_a, var_b):
        self.var_a = var_a
        self.var_b = var_b
    
    def function(self):
        numerator = gamma(self.var_a).function()*gamma(self.var_b).function()
        denominator = gamma(self.var_a + self.var_b).function()

        return numerator/denominator
    
    def pdf(self, var_x):
        y = (1 / self.function()) * var_x ** (self.var_a - 1) * (1 - var_x) ** (self.var_b - 1)
        
        return y.astype(float)


# 디리클레 분포