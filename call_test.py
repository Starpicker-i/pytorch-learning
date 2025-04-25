'''
介绍类中的__call__内置函数和自定义函数的区别
'''
class Person:
    def __call__(slef, name):
        print('__call__' + 'Hello' + name)

    def hello(self, name):
        print('hello' + name)

person = Person()
person('zhangsan')
person.hello('lisi')

'''
输出结果；
__call__Hellozhangsan
hellolisi

内置函数直接传入参数即可
自定义函数需要采样点调用的方式
'''