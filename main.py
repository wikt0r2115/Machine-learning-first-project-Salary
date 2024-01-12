import functions as fn

x_train = list()
y_train = list()

x_train , y_train = fn.load_data('Salary_dataset.txt')
w = 0
b = 0

while True:
    print('1.Visualize data with linear function')
    print('2.Run gradient descent for x iterations')
    print('3.Predict value for x yers of experience')
    print('4.End program')
    print(f'Current value for b and w {w}, {b}')
    inp = input('')
    if inp == '1':
        fn.training_data_visualization(x_train,y_train,w,b)
    elif inp == '2':
        print('How many iterations do u want to do?')
        iterations = int(input())
        print('What learning rate do u want to do?')
        alpha = float(input())
        new_w , new_b = fn.gradient_descent(x_train,y_train,w,b,alpha,iterations)
        print(f'New values for w anb b are: {new_w}, {new_b} do u want to change them(Y/N)')
        yn = input('')
        if yn == 'y' or yn == 'Y':
            w = new_w
            b = new_b
    elif inp == '3':
        print('How many years of experience?')
        years = float(input(''))
        fn.prediction(years,w,b)
    elif inp == '4': 
        break
    else:
        print('Wrong input')
