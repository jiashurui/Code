import numpy as np

print(np.array([1, 2, 3, 4, 5]))

#不加上dtype就是float64
print(np.zeros([3,2],dtype=int))
print(np.zeros([3,2],dtype=np.int8))
zero_array = np.zeros([3,2],dtype=np.int8)
zero_array.astype(np.int16)

print(np.ones((5,5,)))

#左闭右开的递增的数字
print(np.arange(3,7))

#等间距的数 从3到7,分成5份
print(np.linspace(3,7,5))

#随机生成5x5矩阵
print(np.random.rand(5,5))

print("------------------------------------")
#相同长度的矩阵可直接加法 (不同的话直接报错)
add1 = np.array([[1,2]]) #1x2
add2 = np.array([[1],[2]]) #2x1
print(add1)
print(add2)
print("------------------------------------")

print(add1+add2)
print(np.dot(add1,add2)) #add1.dot(add2)
print("==================================-")

#向量乘法
print(add1@add2)
print(np.matmul(add1,add2))
print(100*add1) #标量乘法
print("------------------------------------")

#min max
print(np.min([1,2,3,4,5]))
print(np.max([1,2,3,4,5]))
#最小或者最大数字的索引
print("-----------------index-------------------")

np.argmin(np.min([1,2,3,4,5,1]))
np.argmax(np.max([1,2,3,99,5,1]))

#求平方根
sqrt_number = 4
print(np.sqrt(sqrt_number,dtype=np.float64)) #int8 error  why?
print(np.sqrt([4,16,64]))

print(np.log(10))
print("=================三角函数=================-")

#sin函数
print(np.sin(0.5))
#cos函数
print(np.cos(30))
#角度 to rad
print(np.sin(np.deg2rad(90)))

#开方
print(np.power(2,3))
#e的指数
print(np.exp(sqrt_number))

#和
print(np.sum([1,2,3,4,5]))

#平均值
print(np.average([1,2,3,4,5,99]))
print(np.mean([1,2,3,4,5,99]))

#中位数
print(np.median([1,2,3,4,5]))

#常数
print(np.pi)
print(np.nan)
print(np.e)

#方差 标准差
print(np.var([1,2,3,4,5]))
print(np.std([1,2,3,4,5]))

#axis 是代表维度


#获取元素
a = np.array([[1,2,3,4,5],
              [2,3,4,5,6]])
#直接用下标索引
print(a[1,1])
#直接用条件式索引
print(a[a<4])
print(a[(a<4) & (a>1)])
#切片语法
print(a[0,0:4])
print(a[0,0:4:2]) #切片语法 stride
print(a[0,::-1]) # stride为负数,从右往左遍历数组,这里的就是reverse操作

#形状
print(a.shape)

print()