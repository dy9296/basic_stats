import numpy as np
import sympy as sym

# mean, mode, var, cdf

class normal():
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def pdf(self, var_x):
        norm_x = (var_x - self.mean) / np.sqrt(2 * self.variance)
        y = (1 / np.sqrt(2 * np.pi * self.variance)) * np.exp(- norm_x ** 2)

        return y.astype(float)

    def cdf(self, var_x):
        # variable t
        var_t = sym.symbols('t')
        
        # function and integral
        erf_f_t = (2 / sym.sqrt(sym.pi)) * sym.exp(- var_t ** 2)
        erf_F_t = sym.integrate(erf_f_t, var_t) # erf

        norm_x = (var_x - self.mean)/np.sqrt(2 * self.variance)

        F_b = np.array([(1 + sym.limit(erf_F_t, var_t, x))/2 for x in norm_x])
        F_a = (1 + sym.limit(erf_F_t, var_t, -sym.oo))/2

        return (F_b - F_a).astype(float)

class bernoulli():
    def __init__(self, mu):
        self.mu = mu

    def pdf(self, var_x):
        y = self.mu ** var_x * (1 - self.mu) ** (1 - var_x)

        return y.astype(float)

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
    
    def cdf(self, var_x):
        # variable t
        var_t = sym.symbols('t')
        
        ## function and integral
        # pdf function f(t)
        yt = (1 / self.function()) * (self.var_b ** self.var_a) * (var_t ** (self.var_a - 1)) * sym.exp(-self.var_b * var_t)
        
        # cdf function F(t)
        cumm_yt = sym.integrate(yt, var_t)
        
        F_b = np.array([sym.limit(cumm_yt, var_t, x) for x in var_x])
        F_a = sym.limit(cumm_yt, var_t, 0)

        return (F_b - F_a).astype(float)

class beta():
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