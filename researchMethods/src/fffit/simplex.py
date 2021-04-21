#!/usr/bin/env python
"""Simplex Algorithm


"""
    
import numpy as np
from scipy.optimize import minimize

class Simplex(object):
    
    def __init__(self, func, x0, xatol=1e-8, disp=True, maxiter=None, maxfev=None, fatol=None, adaptive=True):
      self.func = func
      self.x0 = x0
      self.pos_best_glob = None
      self.fitness = None
      self.step_number = None
      if fatol is None:
        self.options = {'xatol': xatol, 'disp': disp, 'maxiter': maxiter, 'maxfev': maxfev,'adaptive': adaptive }
      else:
        self.options = {'xatol': xatol, 'disp': disp, 'maxiter': maxiter, 'maxfev': maxfev, 'fatol': fatol, 'adaptive': adaptive }
        


    def do_full_step(self, **kwargs):
        simplex = minimize(self.func, self.x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True, 'maxiter': 1, 'maxfev': None})

        result = {'x':simplex.x ,'fun':simplex.fun, 'success': simplex.success}
        return result
    def run(self, func, enum_particles=False, add_step_num=False, DEBUG=None, **kwargs):
        """Execute uma etapa completa de Simplex.
        
        Este método passa por todos os outros métodos para realizar uma completa
        Etapa Simplex.
        """
        if add_step_num:
            kwargs['step_num'] = 3
        if enum_particles:
            kwargs['part_idx'] = enumerate(range(0))
    
        simplex = minimize(func, self.x0, method='nelder-mead',
               options=self.options)
        result = {'x':simplex.x ,'fun':simplex.fun, 'success': simplex.success, 'step_number':simplex.nit}
        self.step_number = simplex.nit
        self.fitness = simplex.fun
        self.pos_best_glob = simplex.x
        return result