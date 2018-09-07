
# coding: utf-8

# In[5]:


get_ipython().system('pip install numpy')
import numpy as np


# Creating an Array in Numpy

# In[15]:


my_list=[1,2,3,4]
print("Original List is "+str(my_list)+" and Type is "+str(type(my_list)))
arr=np.array(my_list)
print("New Numpy Array is "+str(arr)+" and Type is "+str(type(arr)))


# Creating 2d array in numpy

# In[22]:


my_list1=[1,2,3,4]
my_list2=[5,6,7,8]
#method 1
arr2d_m1=np.array([my_list1,my_list2])
print("2d Arrays is \n"+str(arr2d_m1))
#method 2
arr2d_m2=np.array([[11,22,33,44],[55,66,77,88]])
print("2d Arrays is \n"+str(arr2d_m2))


# Use of shape,dtype(datatype of the members of the array)

# In[32]:


print(arr2d_m1)
print("Shape of above matrix is "+str(arr2d_m1.shape))
print("Datatype of members of array is of type "+str(arr.dtype))


# #zeros, ones, empty, eye, arange

# In[47]:


array1 = np.zeros(4)
print("Use of Zeros Function of Numpy "+str(array1))
array1 = np.ones([2,2]) 
print("Use of ones Function of Numpy \n"+str(array1))
array1 = np.eye(2)
print("Use of eye Function of Numpy \n"+str(array1))
array1 = np.arange(4,24,4)
print("Use of arange Function of Numpy "+str(array1))


# Scalar Matrix

# In[54]:


array1 = np.array([[1,2,3,4],[5,6,7,8]])
array2 = np.array([[5,6,7,8],[1,2,3,4]])
print("Array 1 \n"+str(array1))
print("Array 2 \n"+str(array2))

#multiplication
array3 = array1*array2
print("Mutiplication of above arrays \n"+str(array3))

#exp multiplication
array3 = array1 ** 3
print("Expo Mutiplication of above arrays \n"+str(array3))

#subtraction
array3 = array2 - array1;
print("Subtraction of above arrays \n"+str(array3))

#reciprocal
array3 = 1/array1;
print("Reciprocal of above arrays \n"+str(array3))


# Indexing of Arrays

# In[66]:


arr = np.arange(0,12)
print("Defined array as "+str(arr))

print("Slicing array(0,5) as "+str(arr[0:5]))
print("Slicing array(2,6) as "+str(arr[2:6]))

arr[0:5] = 20
print("Slicing and assinging value is "+str(arr))

arr2 = arr[0:6]
arr2[:] = 29 
print("Array after slicing value is "+str(arr))

# creating new array copy
arrcopy = arr.copy()


# More about Indexing

# In[77]:


arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Array is \n"+str(arr2d))
print("Value at Row 2 Column 3 is "+str(arr2d[1][2]))

#slices of 2d array
slice1 = arr2d[0:2,0:2]
print("Slice 1 is as \n"+str(slice1))
arr2d[:2,1:] = 15
print("Array is as \n"+str(arr2d))

#using loops to index
arr_len = arr2d.shape[0]

for i in range(arr_len):
    arr2d[i] = i;
print("Array after indexing is \n"+str(arr2d))

#one more way of accessing the rows
print("Slice with [[0,1]] is as \n"+str(arr2d[[0,1]]))
print("Slice with [[1,0]] is as \n"+str(arr2d[[1,0]]))


# Universal Array Functions

# #arange
# #sqrt
# #exp
# #random
# #addition
# #maximum

# In[82]:


A = np.arange(15)
print("A is "+str(A))
A = np.arange(1,15,2)
print("A is "+str(A))

#sqrt
B = np.sqrt(A)
print("B is \n"+str(B))

#exp
C = np.exp(A)
print("C is \n"+str(C))

#add
D = np.add(A,B)
print("D is \n"+str(D))

#maximum
E = np.maximum(A,B)
print("E is \n"+str(E))


# Saving array

# In[84]:


# Saving single arrays
arr = np.arange(10)
print(arr)

np.save('saved_array',arr)
#New file is created - saved_array.npy

new_array = np.load('saved_array.npy')
print(new_array)


# In[86]:


# Save Multiple arrays
array_1 = np.arange(25)
array_2 = np.arange(30)

np.savez('saved_archive.npz',x = array_1,y = array_2)

load_archive = np.load('saved_archive.npz')

print('load_archive[x] is '+str(load_archive['x']))

print('load_archive[y] is '+str(load_archive['y']))


# In[88]:


#save to txtfile
np.savetxt('notepadfile.txt',array_1,delimiter=',')
#loading of txt files
load_txt_file = np.loadtxt('notepadfile.txt',delimiter=',')
print("load_txt_file is"+str(load_txt_file))


# Statistics

# In[92]:


import matplotlib.pyplot as plt
axes_values = np.arange(-100,100,10)
dx, dy = np.meshgrid(axes_values,axes_values)

#print("dx:"+str(dx))
#print("dy"+str(dy))

function1 = 2*dx+3*dy
function2 = np.cos(dx)+np.cos(dy)

#print(function1)
#print(function2)

#replace function2 by function1 to get graph of function1
#plotting the function on graph
plt.imshow(function2)
plt.title("function cos plot")
plt.colorbar()
plt.savefig('myfig2.png')


# Clausing 

# In[12]:


x = np.array([100,400,500,600]) #each member 'a'
y = np.array([10,15,20,25]) #each member 'b'
condition = np.array([True,True,False,False]) #each menber cond

#use loops indirectly to perform this
z = [a if cond else b for a,cond,b in zip(x,condition,y)]
print("Condition using loops \n"+str(z))

#np.where(#condition,#value for yes, #value for No)
z2 = np.where(condition,x,y)
print("Condition using where \n"+str(z2))

z3 = np.where(x>0,0,1)
print("Using where clause where if x>0 then 0 else 1 \n"+str(z3))


# Standard functions of numpy

# In[27]:


n = np.array([[1,2],[3,4]])
print(n)

print("Sum of \n"+str(n.sum()))
#column sum
print("Sum of Columns \n"+str(n.sum(0)))

print("Sum of Mean : "+str(x.mean()))
print("Sum of Standard Devation : "+str(x.std()))
print("Sum of varainace : "+str(x.var()))

#logical operations - and / or operations

condition2 = np.array([True,False,True])
print("OR Operation : "+str(condition2.any())) #or operator
print("AND Operation : "+str(condition2.all())) #and operator

#sorting in numpy arrays

unsorted_array = np.array([1,2,8,10,7,3])
print("Array before sorting : "+str(unsorted_array))
unsorted_array.sort()
print("Array after sorting : "+str(unsorted_array))

arr2 = np.array(['solid','solid','solid','liquid','liquid','gas','gas'])
print("Original Array "+str(arr2))
print("Unique elements in Array"+str(np.unique(arr2)))

print("Check if array contains ['solid','gas','plasma'] : "+str(np.in1d(['solid','gas','plasma'],arr2)))


# NOW PANDAS
