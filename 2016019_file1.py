import numpy as np
import math
def get_mean(x1):
    return np.sum(x1)/len(x1)
def get_mean_2(x):
    s1=0.0
    s2=0.0
    for i in range(len(x)):
        s1+=x[i][0]
        s2+=x[i][1]
    return [s1/len(x),s2/len(x)]
def get_mean_3(x):
    s1=0.0
    s2=0.0
    s3=0.0
    for i in range(len(x)):
        s1+=x[i][0]
        s2+=x[i][1]
        s3+=x[i][2]
    return [s1/len(x),s2/len(x),s3/len(x)]
def get_covar(x1,x2):
    prod=1.0
    m1=get_mean(x1)
    m2=get_mean(x2)
    for i in range(len(x1)):
        prod+=(x1[i]-m1)*(x2[i]-m2)
    return prod/len(x1)
def get_covar_matrix(x1,x2):
    mat=[[0,0],[0,0]]
    mat[0][0]=get_covar(x1,x1)
    mat[0][1]=get_covar(x1,x2)
    mat[1][0]=mat[0][1]
    mat[1][1]=get_covar(x2,x2)
    return mat
def get_covar_mat3(x1,x2,x3):
    mat=[[0,0,0],[0,0,0],[0,0,0]]
    mat[0][0]=get_covar(x1,x1)
    mat[1][1]=get_covar(x2,x2)
    mat[2][2]=get_covar(x3,x3)
    mat[0][1]=get_covar(x1,x2)
    mat[1][0]=mat[0][1]
    mat[0][2]=get_covar(x1,x3)
    mat[2][0]=mat[0][2]
    mat[1][2]=get_covar(x2,x3)
    mat[2][1]=mat[1][2]
    return mat
def find_gi(x,w,dim):
    if dim==1:
        m=get_mean(w)
        v=get_covar(w,w)
        temp=(x-m)**2
        temp=temp/v
        temp=temp/(-2)
        temp+=np.log(v)/(-2)
        return temp
    if dim==2:
        m=get_mean_2(w)
        v=get_covar_matrix(w[:][0],w[:][1])
        v_inv=np.linalg.inv(v)
        #print(v_inv)
        k=[x[0]-m[0],x[1]-m[1]]
        temp=np.matmul(k,v_inv)
        temp=np.matmul(temp,np.transpose(k))
        temp=temp/(-2)
        t1=np.log(np.absolute(v[0][0]*v[1][1]-v[1][0]**2))
        t1=t1/(-2)
        return temp+t1
    if dim==3:
        m=get_mean_3(w)
        v=get_covar_mat3(w[:][0],w[:][1],w[:][2])
        v_inv=np.linalg.inv(v)
        k=[x[0]-m[0],x[1]-m[1],x[2]-m[2]]
        temp=np.matmul(k,v_inv)
        temp=np.matmul(temp,np.transpose(k))
        temp=temp/(-2)
        t1=np.log(np.linalg.det(v))
        t1=t1/(-2)
        return temp+t1
def decide_class(x,dim):
    global w1
    global w2
    g1=find_gi(x,w1,dim)
    g2=find_gi(x,w2,dim)
    g=g1-g2
    if g>0:
        return 1
    else:
        return 2
def bh_bound(dim):
    global w1,w2,p_w1,p_w2
    if dim==1:
        m1=get_mean(w1)
        m2=get_mean(w2)
        v1=get_covar(w1,w1)
        v2=get_covar(w2,w2)
        temp=((m2-m1)**2)/(4*(v1+v2))
        t1=((v1+v2)/2)/math.sqrt(v1*v2)
        t1=(np.log(t1))/2
        temp+=t1
        temp=temp*(-1)
        val=math.sqrt(p_w1*p_w2)
        val=val*np.exp(temp)
        return val
    if dim==2:
        m1=get_mean_2(w1)
        m2=get_mean_2(w2)
        v1=get_covar_matrix(w1[:][0],w1[:][1])
        v2=get_covar_matrix(w2[:][0],w2[:][1])
        k=[m2[0]-m1[0],m2[1]-m1[1]]
        temp=np.transpose(k)
        temp_v=[[0,0],[0,0]]
        for i in range(2):
            for j in range(2):
                temp_v[i][j]=(v1[i][j]+v2[i][j])/2
        v_s=np.linalg.inv(temp_v)
        temp=np.matmul(temp,v_s)
        temp=np.matmul(temp,k)
        temp=temp/8
        mod1=temp_v[0][0]+temp_v[1][1]-temp_v[0][1]**2
        mod2=v1[0][0]+v1[1][1]-v1[0][1]**2
        mod3=v2[0][0]+v2[1][1]-v2[0][1]**2
        temp_2=np.log(mod1/(math.sqrt(mod2*mod3)))
        temp_2=temp_2*(-0.5)
        val=math.sqrt(p_w1*p_w2)
        val=val*np.exp(temp+temp_2)
        return val
    if dim==3:
        m1=get_mean_3(w1)
        m2=get_mean_3(w2)
        v1=get_covar_mat3(w1[:][0],w1[:][1],w1[:][2])
        v2=get_covar_mat3(w2[:][0],w2[:][1],w2[:][2])
        k=[m2[0]-m1[0],m2[1]-m1[1],m2[2]-m1[2]]
        temp=np.transpose(k)
        temp_v=[[0,0,0],[0,0,0],[0,0,0]]
        for i in range(3):
            for j in range(3):
                temp_v[i][j]=(v1[i][j]+v2[i][j])/2
        v_s=np.linalg.inv(temp_v)
        temp=np.matmul(temp,v_s)
        temp=np.matmul(temp,k)
        temp=temp/8
        mod1=np.linalg.det(temp_v)
        mod2=np.linalg.det(v1)
        mod3=np.linalg.det(v2)
        temp_2=np.log(mod1/math.sqrt(mod2*mod3))
        temp_2=temp_2*(-0.5)
        val=math.sqrt(p_w1*p_w2)
        val=val*np.exp(temp+temp_2)
        return val
w1=[-5.01,-5.43,1.08,0.86,-2.67,4.94,-2.52,-2.25,5.56,1.03]
w2=[-0.91,1.30,-7.75,-5.47,6.14,3.60,5.37,7.18,-7.39,-7.50]
p_w1=0.5
p_w2=0.5
err=0
for i in range(len(w1)):
    val=decide_class(w1[i],1)
    if val!=1:
        err+=1
for i in range(len(w2)):
    val=decide_class(w2[i],1)
    if val!=2:
        err+=1
print("Emperical error:",err/(2*len(w1)))
print("Bhattacharya Bound:",bh_bound(1))
w1=[(-5.01,-8.12),(-5.43,-3.48),(1.08,-5.52),(0.86,-3.78),(-2.67,0.63),(4.94,3.29),(-2.52,2.09),(-2.25,-2.13),(5.56,2.86),(1.03,-3.33)]
w2=[(-0.91,-0.18),(1.30,-2.06),(-7.75,-4.54),(-5.47,0.50),(6.14,5.72),(3.60,1.26),(5.37,-4.63),(7.18,1.46),(-7.39,1.17),(-7.50,-6.32)]
err=0
for i in range(len(w1)):
    val=decide_class(list(w1[i]),2)
    if val!=1:
        err+=1
for i in range(len(w2)):
    val=decide_class(list(w2[i]),2)
    if val!=2:
        err+=1
print("Emperical error:",err/(2*len(w1)))
print("Bhattacharya bound:",bh_bound(2))
w1=[(-5.01,-8.12,-3.68),(-5.43,-3.48,-3.54),(1.08,-5.52,1.66),(0.86,-3.78,-4.11),(-2.67,0.63,7.39),(4.94,3.29,2.08),(-2.51,2.09,-2.59),(-2.25,-2.13,-6.94),(5.56,2.86,-2.26),(1.03,-3.33,4.33)]
w2=[(-0.91,-0.18,-0.05),(1.30,-2.06,-3.53),(-7.75,-4.54,-0.95),(-5.47,0.50,3.92),(6.14,5.72,-4.85),(3.60,1.26,4.36),(5.37,-4.63,-3.65),(7.18,1.46,-6.66),(-7.39,1.17,6.30),(-7.50,-6.32,-0.31)]
err=0
for i in range(len(w1)):
    val=decide_class(list(w1[i]),3)
    if val!=1:
        err+=1
for i in range(len(w2)):
    val=decide_class(list(w2[i]),3)
    if val!=2:
        err+=1
print("Emperical error:",err/(2*len(w1)))
print("Bhattacharya Bound:",bh_bound(3))
