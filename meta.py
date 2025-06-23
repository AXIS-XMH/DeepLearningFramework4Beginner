import numpy as np

# 1. 实现变量类
class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None
        self.creator = None
    def set_creator(self,func):
        self.creator = func


# 2. 实现function类
class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x) # 具体计算在forward方法中进行
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
        
    def forward(self,x):
        raise NotImplementedError()
    def backward(self,gy):
        raise NotImplementedError()

    # 2.1 平方
class Square(Function):
    def forward(self,x):
        y = x ** 2
        return y
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
    # 3.exp实现
class Exp(Function):
    def forward(self,x):
        y = np.exp(x)
        return y
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
    #数值微分
def numerical_diff(f,x,eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
