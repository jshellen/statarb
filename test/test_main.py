# -*- coding: utf-8 -*-


import os
dirname = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

import sys
sys.path.append(dirname)



from unittest import TestCase, main

class Test_OU_Spread_Model(TestCase):
    
    def test_parameter_fit(self):
        
        from src.optimal_controls.estimate_ou_params import estimate_ou_parameters
        import numpy as np
        
        
        data = np.array([ 3.0000,1.7600, 1.2693, 1.1960, 0.9468, 0.9532, 0.6252, 0.8604, 1.0984,
                          1.4310, 1.3019, 1.4005, 1.2686, 0.7147, 0.9237, 0.7297, 0.7105, 
                          0.8683, 0.7406, 0.7314, 0.6232])
                        
        res = estimate_ou_parameters(data,0.25)
        
        kappa = res[0]
        theta = res[1]
        sigma = res[2]
        
        self.assertAlmostEqual(kappa,3.1288,places=2)
        self.assertAlmostEqual(theta,0.9075,places=2)
        self.assertAlmostEqual(sigma,0.5531,places=2)
    
if __name__ == '__main__':
    main()
    
