from math import exp
import numpy as np
import matplotlib.pyplot as plt
from correlation import *
from moving_average import *
from autoregression import *
from arma import *

data = np.array(list(map(float, open('./tsp.txt'))))

if __name__ == '__main__':
    maxiter = 15000
    np.random.seed(0)

    mean = data.mean()
    std = data.std()

    print('Mean: {}'.format(mean))
    print('Disp.: {}'.format(std ** 2))
    print('STD: {}'.format(std))

    x = range(0, 121)
    y = data[0:121]

    fig = plt.figure()
    ax = plt.subplot(111)

    plt.rc('lines', linewidth=1)

    source, = ax.plot(x, y, 'black')
    avg, = ax.plot([0, 120], [mean] * 2, 'red')
    mn, = ax.plot([0, 120], [mean + std] * 2, 'blue')
    mn, = ax.plot([0, 120], [mean - std] * 2, 'blue')

    plt.xlabel('Index number')
    plt.ylabel('Random sequence values')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend([source, avg, mn], ['Source process', 'Average', 'Standart deviation'],
              loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fancybox=True)

    plt.grid(True)

    plt.savefig('plot1.png')
    plt.show()

    cnst = exp(-1)
    ncf = normalized_corr_func(data)
    cf = corr_func(data)

    limit = 300

    for n in range(limit):
        ncf(n)

    interval = limit

    while abs(ncf(interval)) < exp(-1):
        interval -= 1

    print('Correlation interval: {}'.format(interval))

    print('Corr.f., ncf - first 10 values')
    for n in range(11):
        print(n, ' & ', round(cf(n), 5), '& ', round(ncf(n), 5), '     \\\\ \\hline')

    print('Corr.f, Norm.c.f in interval')
    for n in range(interval + 1):
        print(round(cf(n), 5), round(ncf(n), 5))

    fig = plt.figure()
    ax = plt.subplot(111)

    plt.rc('text', usetex=True)

    x = range(interval + 20)
    y = list(map(ncf, x))

    source, = ax.plot(x, y, 'black')
    cnst_plt, = ax.plot([min(x), max(x)], [cnst] * 2, 'blue')
    cnst_plt, = ax.plot([min(x), max(x)], [-cnst] * 2, 'blue')
    interval, = ax.plot([interval, interval], [
                        min(y), max(y)], 'red', linestyle='--')

    plt.xlabel('Index number')
    plt.ylabel('Normalized correlation function')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend([source,
               cnst_plt,
               interval],
              ['Source',
               r'$\pm e^{-1}$',
               'Correlation interval'],
              loc='upper center',
              bbox_to_anchor=(0.5,
                              -0.13),
              ncol=3,
              fancybox=True)

    plt.grid(True)

    plt.savefig('plot2.png')
    plt.show()

    # Autoregression
    print('~' * 25)
    print('Autoregression')
    best_ar = 0
    best_ar_err = 1e6
    for m in range(4):
        try:
            print('Autoregression {}:'.format(m))
            print(' ' * 4 + str(calc_model_ar(m, data)))

            ncf_th = theory_ncf_ar(m, data)

            for x in range(1, 11):
                print(' ' * 4, x, round(ncf_th(x), 5))

            err = corr_error(ncf, ncf_th)
            print(' ' * 4 + str(err))

            if best_ar_err > err:
                best_ar = m
                best_ar_err = err
        except Exception as e:
            print(e)
    print('Best ar err: {}, model ar({})'.format(best_ar_err, best_ar))

    print('~' * 25)
    print('Moving average')
    best_ma = 0
    best_ma_err = 1e6
    for n in range(4):
        try:
            print('Moving average {}'.format(n))
            print(' ' * 4 + str(calc_model_ma(n, data, maxiter=maxiter)))

            ncf_th = theory_ncf_ma(n, data)

            for x in range(1, 11):
                print(' '*4, x, ncf_th(x))

            err = corr_error(ncf, ncf_th)
            print(' ' * 4 + str(err))

            if best_ma_err > err:
                best_ma = n
                best_ma_err = err
        except Exception as e:
            print(e)
    print('Best ma err: {}, model ma({})'.format(best_ma_err, best_ma))

    print('~' * 25)
    print('ARMA')
    best_arma_ar = 0
    best_arma_ma = 0
    best_arma_err = 1e6
    for m in range(1, 4):
        for n in range(1, 4):
            try:
                print('ARMA({}, {})'.format(m, n))
                print(' ' * 4 + str(calc_model_arma(m, n, data, maxiter=maxiter)))

                ncf_th = theory_ncf_arma(m, n, data)

                for x in range(1, 11):
                    print(' '*4, x, ncf_th(x))

                err = corr_error(ncf, ncf_th)
                print(' ' * 4 + str(err))

                if best_arma_err > err:
                    best_arma_ar = m
                    best_arma_ma = n
                    best_arma_err = err
            except Exception as e:
                print(e)
    print('Best arma err: {}, model arma({}, {})'.format(
        best_arma_err, best_arma_ar, best_arma_ma))

    print('Data')
    print('Mean: {}'.format(data.mean()))
    print('Std: {}'.format(data.std()))

    ar = model_ar(best_ar, data)([1] * best_ar, 1)
    data_ar = np.array([next(ar) for _ in range(6000)][1000:]) + data.mean()

    print('Modeled ar')
    print('Mean: {}'.format(data_ar.mean()))
    print('Mean err: {}'.format(abs(data_ar.mean() - data.mean())))
    print('Std: {}'.format(data_ar.std()))
    print('Std err: {}'.format(abs(data_ar.std() - data.std())))
    ncf_ar = normalized_corr_func(data_ar)
    for i in range(1, 11):
        print(i, round(ncf_ar(i), 5))
    print('Error: {}'.format(round(corr_error(ncf, ncf_ar), 5)))

    ma = model_ma(best_ma, data, maxiter=maxiter)(1)
    data_ma = np.array([next(ma) for _ in range(6000)][1000:]) + data.mean()

    print('Modeled ma')
    print('Mean: {}'.format(data_ma.mean()))
    print('Mean err: {}'.format(abs(data_ma.mean() - data.mean())))
    print('Std: {}'.format(data_ma.std()))
    print('Std err: {}'.format(abs(data_ma.std() - data.std())))
    ncf_ma = normalized_corr_func(data_ma)
    for i in range(1, 11):
        print(i, round(ncf_ma(i), 5))
    print('Error: {}'.format(round(corr_error(ncf, ncf_ma), 5)))

    arma = model_arma(
        best_arma_ar,
        best_arma_ma,
        data,
        maxiter=maxiter)(
        [1] *
        best_arma_ar,
        1)
    data_arma = np.array([next(arma)
                          for _ in range(6000)][1000:]) + data.mean()

    print('Modeled arma')
    print('Mean: {}'.format(data_arma.mean()))
    print('Mean err: {}'.format(abs(data_arma.mean() - data.mean())))
    print('Std: {}'.format(data_arma.std()))
    print('Std err: {}'.format(abs(data_arma.std() - data.std())))
    ncf_arma = normalized_corr_func(data_arma)
    for i in range(1, 11):
        print(i, round(ncf_arma(i), 5))
    print('Error: {}'.format(round(corr_error(ncf, ncf_arma), 5)))

    # AR plot

    x = range(10)
    y1 = list(map(ncf, x))
    y2 = list(map(theory_ncf_ar(best_ar, data), x))
    y3 = list(map(normalized_corr_func(data_ar), x))

    fig = plt.figure()
    ax = plt.subplot(111)

    ncf_plot, = ax.plot(x, y1, 'red')
    th_ncf_plot, = ax.plot(x, y2, 'green')
    mdl_ncf_plot, = ax.plot(x, y3, 'blue')

    plt.xlabel('Index number')
    plt.ylabel('Normalized correlation function AR')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend([ncf_plot, th_ncf_plot, mdl_ncf_plot], ['Source', 'Theoretical AR({})'.format(
        best_ar), 'Modelling'], loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fancybox=True)

    plt.grid(True)

    plt.savefig('plot_ar_ncf.png')
    plt.show()

    # MA plot

    x = range(10)
    y1 = list(map(ncf, x))
    y2 = list(map(theory_ncf_ma(best_ma, data), x))
    y3 = list(map(normalized_corr_func(data_ma), x))

    fig = plt.figure()
    ax = plt.subplot(111)

    ncf_plot, = ax.plot(x, y1, 'red')
    th_ncf_plot, = ax.plot(x, y2, 'green')
    mdl_ncf_plot, = ax.plot(x, y3, 'blue')

    plt.xlabel('Index number')
    plt.ylabel('Normalized correlation function MA')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend([ncf_plot, th_ncf_plot, mdl_ncf_plot], ['Source', 'Theoretical MA({})'.format(
        best_ma), 'Modelling'], loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fancybox=True)

    plt.grid(True)

    plt.savefig('plot_ma_ncf.png')
    plt.show()

    # ARMA plot

    x = range(10)
    y1 = list(map(ncf, x))
    y2 = list(map(theory_ncf_arma(best_arma_ar, best_arma_ma, data), x))
    y3 = list(map(normalized_corr_func(data_arma), x))

    fig = plt.figure()
    ax = plt.subplot(111)

    ncf_plot, = ax.plot(x, y1, 'red')
    th_ncf_plot, = ax.plot(x, y2, 'green')
    mdl_ncf_plot, = ax.plot(x, y3, 'blue')

    plt.xlabel('Index number')
    plt.ylabel('Normalized correlation function ARMA')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(
        [
            ncf_plot, th_ncf_plot, mdl_ncf_plot], [
            'Source', 'Theoretical ARMA({}, {})'.format(
                best_arma_ar, best_arma_ma), 'Modelling'], loc='upper center', bbox_to_anchor=(
                    0.5, -0.13), ncol=3, fancybox=True)

    plt.grid(True)

    plt.savefig('plot_arma_ncf.png')
    plt.show()

    # best would be arma

    x = range(0, 121)
    y = data_arma[0:121]

    mean = data_arma.mean()
    std = data.std()

    fig = plt.figure()
    ax = plt.subplot(111)

    data_plot, = ax.plot(x, y, 'black')
    mean_plot, = ax.plot([0, 120], [mean, mean], 'blue')
    std_plot, = ax.plot([0, 120], [mean - std, mean - std], 'red')
    _ = ax.plot([0, 120], [mean + std, mean + std], 'red')

    plt.xlabel('Index number')
    plt.ylabel('Random sequence values')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(
        [
            data_plot, mean_plot, std_plot], [
            'ARMA({},{}) process'.format(
                best_arma_ar, best_arma_ma), 'Average', 'Standart deviation'], loc='upper center', bbox_to_anchor=(
                    0.5, -0.13), ncol=3, fancybox=True)

    plt.grid(True)

    plt.savefig('plot_arma_modeled.png')
    plt.show()
