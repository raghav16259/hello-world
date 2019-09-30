import numpy as np
import math
def bh_bound():
    global m_w1,m_w2,v_w1,v_w2,p_w1,p_w2
    temp=((m_w2-m_w1)**2)/(v_w1+v_w2)
    temp=temp/8
    t1=(v_w1+v_w2)/2
    t1=t1/math.sqrt(v_w1*v_w2)
    t1=t1/2
    temp=np.exp((-1)*(temp+t1))
    temp=temp*math.sqrt(p_w1*p_w2)
    return temp
m_w1=-0.5
v_w1=1
m_w2=0.5
v_w2=1
p_w1=0.5
p_w2=0.5
print(bh_bound())
