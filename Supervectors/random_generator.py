# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:30:03 2015

@author: buckler
"""
import numpy as np
def randomint_right_seed(list_len=10):
    """Return a random vector with range 1-3 withe the same nember of occurance 1, 2, 3
        If the list_len is not divisible by 3, the "3" occurance in number free.
        -----------
        
        example: randomint_right_seed(list_len=10)
        return 3 times 1, 3 times 2 and 4 times 3
        
        randomint_right_seed(list_len=12)
        return 4 times 1, 4 times 2 and 4 times 3
    """
    h=0;
    k=0;
    l=0;
    seed=10
    part= list_len/3;
    while h!=part or k!=part:
        prng = np.random.RandomState(seed);
        random_vector=prng.randint(1, 4, size=list_len );
        h=0;
        k=0;
        l=0;
        for i in random_vector:
            if(i==1): h=h+1
            if(i==2): k=k+1
            if(i==3): l=l+1
        seed=seed+1;
    
    return random_vector
