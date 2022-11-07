from nnn import print, Model, Variable

print('-- Regression --')
func = lambda x: 2*x*x - 4*x + 3
X = [x/100. for x in range(-10000, 10000)]
Y = [func(i) for i in X]
print('[+] Find a, b, and c')
print('    X = [{}, {}, ..., {}]'.format(X[0], X[1], X[-1]))
print('    Y = [{}, {}, ..., {}]'.format(Y[0], Y[1], Y[-1]))
print('    hypothesis = a*x*x + b*x + c')

x = Model()
a, b, c = Variable('a'), Variable('b'), Variable('c')
y = a*x*x + b*x + c

def print_values(epoch):
    print('    (a, b, c) = ({}, {}, {})'.format(Variable(a), Variable(b), Variable(c)))

y.train((X, Y), loss='mse', epochs=10, validset=(1, func(1)), callback_epoch=print_values)

