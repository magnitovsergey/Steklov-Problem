import math
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import os
from joblib import Parallel, delayed

class InverseIterationMethod:

    def __init__(self, file, h, Number_Of_Vectors, Accuracy):
        self.coordinates = []
        length = 0
        with open(file, 'r') as file:
            k = 0
            for line in file:
                current_line = line.strip()
                length = len(current_line)
                for i in range(length):
                    if current_line[i] == '1':
                        self.coordinates.append([i, k])
                k += 1

        self.accuracy = Accuracy
        self.h = h
        self.Number_Of_Points_X = length
        self.Number_Of_Points_Y = k
        self.Number_Of_Points = len(self.coordinates)
        self.Number_Of_Vectors = Number_Of_Vectors
        self.Approximate_Vectors = [self.createApproximateVector() for i in range(self.Number_Of_Vectors)]
        self.Matrix_M = self.createMatrixM(self.Number_Of_Points)
        self.Matrix_A = self.createMatrixA(self.Number_Of_Points) + self.Matrix_M
        self.setAxis()

        self.result_x = []
        self.result_y = []
        self.draw_coordinates = []

        for i in range(len(self.coordinates)):
            self.result_x.append(self.h * self.coordinates[i][0])
            self.result_y.append(self.h * self.coordinates[i][1])
            self.draw_coordinates.append([self.h * self.coordinates[i][0], self.h * self.coordinates[i][1]])

    def setAxis(self):
        self.x = np.linspace(0, 0, self.Number_Of_Points_X)
        self.y = np.linspace(0, 0, self.Number_Of_Points_Y)

        for i in range(1, self.Number_Of_Points_X):
            self.x[i] = self.x[i - 1] + self.h
        for i in range(1, self.Number_Of_Points_Y):
            self.y[i] = self.y[i - 1] + self.h

        self.x2 = np.linspace(0, 0, max(self.Number_Of_Points_X, self.Number_Of_Points_Y))
        self.y2 = np.linspace(0, 0, max(self.Number_Of_Points_X, self.Number_Of_Points_Y))

        for i in range(1, max(self.Number_Of_Points_X, self.Number_Of_Points_Y)):
            self.x2[i] = self.x2[i - 1] + self.h
            self.y2[i] = self.y2[i - 1] + self.h

    def createApproximateVector(self):
        App_Vec = np.zeros(self.Number_Of_Points)
        for i in range(0, len(App_Vec)):
            App_Vec[i] = random.randint(0, 10)/10
        return App_Vec

    def check_inside(self, x, y):
        if [x - 1, y] in self.coordinates and [x + 1, y] in self.coordinates and [x, y - 1] in self.coordinates and [x, y + 1] in self.coordinates:
            return True
        return False

    def check_first_case_angle(self, x, y):
        if [x - 1, y] not in self.coordinates and [x + 1, y] in self.coordinates and [x, y - 1] not in self.coordinates and [x, y + 1] in self.coordinates:
            return True
        if [x - 1, y] in self.coordinates and [x + 1, y] not in self.coordinates and [x, y - 1] in self.coordinates and [x, y + 1] not in self.coordinates:
            return True
        return False

    def check_second_case_angle(self, x, y):
        if [x - 1, y] not in self.coordinates and [x + 1, y] in self.coordinates and [x, y - 1] in self.coordinates and [x, y + 1] not in self.coordinates:
            return True
        if [x - 1, y] in self.coordinates and [x + 1, y] not in self.coordinates and [x, y - 1] not in self.coordinates and [x, y + 1] in self.coordinates:
            return True
        return False

    def check_fixed_y(self, x, y):
        if [x - 1, y] in self.coordinates and [x + 1, y] in self.coordinates and [x, y - 1] in self.coordinates and [x, y + 1] not in self.coordinates:
            return True
        if [x - 1, y] in self.coordinates and [x + 1, y] in self.coordinates and [x, y - 1] not in self.coordinates and [x, y + 1] in self.coordinates:
            return True
        return False

    def check_fixed_x(self, x, y):
        if [x - 1, y] not in self.coordinates and [x + 1, y] in self.coordinates and [x, y - 1] in self.coordinates and [x, y + 1] in self.coordinates:
            return True
        if [x - 1, y] in self.coordinates and [x + 1, y] not in self.coordinates and [x, y - 1] in self.coordinates and [x, y + 1] in self.coordinates:
            return True
        return False

    def createMatrixM(self, Size):
        cpus = os.cpu_count()
        cSize = Size // cpus
        remaining = Size % cpus
        mxs = list(Parallel(n_jobs=-1)(
            delayed(self.fillMChuck)(i * cSize, (i + 1) * cSize + (remaining if i == cpus - 1 else 0), Size) for i in
            range(cpus)))
        return np.vstack(mxs)

    def fillMChuck(self, startR, endR, colls):
        chunck = np.zeros((endR - startR, colls))
        for i in range(startR, endR):
            for j in range(colls):
                chunck[i - startR, j] = self.fillMatrixM(i, j)
        return chunck

    def fillMatrixM(self, n, k):
        Matr_M = 0
        n_i = self.coordinates[n][0]
        n_j = self.coordinates[n][1]
        k_i = self.coordinates[k][0]
        k_j = self.coordinates[k][1]

        if self.check_first_case_angle(n_i, n_j):
            if n == k:
                Matr_M = 2 * self.h / 3
            elif (abs(n_i - k_i) == 1 and n_j == k_j) or (abs(n_j - k_j) == 1 and n_i == k_i):
                Matr_M = self.h / 6

        elif self.check_second_case_angle(n_i, n_j):
            if n == k:
                Matr_M = 2 * self.h / 3
            elif (abs(n_i - k_i) == 1 and n_j == k_j) or (abs(n_j - k_j) == 1 and n_i == k_i):
                Matr_M = self.h / 6

        elif self.check_fixed_y(n_i, n_j):
            if n == k:
                Matr_M = 2 * self.h / 3
            elif (abs(n_i - k_i) == 1 and n_j == k_j):
                Matr_M = self.h / 6

        elif self.check_fixed_x(n_i, n_j):
            if n == k:
                Matr_M = 2 * self.h / 3
            elif (abs(n_j - k_j) == 1 and n_i == k_i):
                Matr_M = self.h / 6

        return Matr_M

    def createMatrixA(self, Size):
        cpus = os.cpu_count()
        cSize = Size // cpus
        remaining = Size % cpus
        mxs = list(Parallel(n_jobs=-1)(
            delayed(self.fillAChuck)(i * cSize, (i + 1) * cSize + (remaining if i == cpus - 1 else 0), Size) for i in
            range(cpus)))
        return np.vstack(mxs)

    def fillAChuck(self, startR, endR, colls):
        chunck = np.zeros((endR - startR, colls))
        for i in range(startR, endR):
            for j in range(colls):
                chunck[i - startR, j] = self.fillMatrixA(i, j)
        return chunck

    def fillMatrixA(self, n, k):
        Matr_A = 0
        n_i = self.coordinates[n][0]
        n_j = self.coordinates[n][1]
        k_i = self.coordinates[k][0]
        k_j = self.coordinates[k][1]
        if self.check_inside(n_i, n_j):
            if n == k:
                Matr_A = 4
            elif (abs(n_i - k_i) == 1 and n_j == k_j) or (abs(n_j - k_j) == 1 and n_i == k_i):
                Matr_A = -1
        elif self.check_first_case_angle(n_i, n_j):
            if n == k:
                Matr_A = 1
            elif (abs(n_i - k_i) == 1 and n_j == k_j) or (abs(n_j - k_j) == 1 and n_i == k_i):
                Matr_A = -1 / 2
        elif self.check_second_case_angle(n_i, n_j):
            if n == k:
                Matr_A = 1
            elif (abs(n_i - k_i) == 1 and n_j == k_j) or (abs(n_j - k_j) == 1 and n_i == k_i):
                Matr_A = -1 / 2
        elif self.check_fixed_y(n_i, n_j):
            if n == k:
                Matr_A = 2
            elif abs(n_i - k_i) == 1 and n_j == k_j:
                Matr_A = -1 / 2
            elif abs(n_j - k_j) == 1 and n_i == k_i:
                Matr_A = -1
        elif self.check_fixed_x(n_i, n_j):
            if n == k:
                Matr_A = 2
            elif abs(n_i - k_i) == 1 and n_j == k_j:
                Matr_A = -1
            elif abs(n_j - k_j) == 1 and n_i == k_i:
                Matr_A = -1 / 2
        return Matr_A

    def DotProduct(self, Vector_1, Vector_2):
        return np.dot(np.transpose(Vector_2), np.dot(self.Matrix_M, Vector_1))

    def Norma(self, Vector):
        return np.sqrt(self.DotProduct(Vector, Vector))

    def Ortogonalization(self, System_Of_Vectors):
        Result_System = []
        for i in range(self.Number_Of_Vectors):
            New_Vector = System_Of_Vectors[i]
            for j in range(len(Result_System)):
                top = self.DotProduct(System_Of_Vectors[i], Result_System[j])
                bottom = self.DotProduct(Result_System[j], Result_System[j])
                New_Vector -= np.dot(top/bottom, Result_System[j])
            Result_System.append(New_Vector)
        return Result_System

    def calculateEigenvalue(self, Eigenvector):
        top = np.dot(np.transpose(Eigenvector), np.dot(self.Matrix_A, Eigenvector))
        bottom = np.dot(np.transpose(Eigenvector), np.dot(self.Matrix_M, Eigenvector))
        return top/bottom

    def Iteration(self, Vector):
        New_Vector = scipy.sparse.linalg.cg(self.Matrix_A, np.dot(self.Matrix_M, Vector))
        New_Vector = New_Vector[0]/self.Norma(New_Vector[0])
        return New_Vector

    def checkAllVectors(self, Prev, Cur):
        for i in range(len(Prev)):
            if not np.all(np.isclose(Prev[i], Cur[i], atol=0.0001)):
                return True
        return False

    def Run(self, Vector):
        Prev_Vector = Vector
        Current_Vector = [self.Iteration(i) for i in Vector]
        Current_Vector = self.Ortogonalization(Current_Vector)
        while self.checkAllVectors(Prev_Vector, Current_Vector):
            Prev_Vector = Current_Vector
            Current_Vector = [self.Iteration(i) for i in Current_Vector]
            Current_Vector = self.Ortogonalization(Current_Vector)
        Current_Eigenvalue = [self.calculateEigenvalue(i) for i in Current_Vector]
        return Current_Vector, Current_Eigenvalue

    def Save(self, figure, figure_heat, figure_counter):
        results_path = os.path.join(os.path.expanduser('~'), 'Results')
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        pictures_path = os.path.join(results_path, 'Pictures')
        if not os.path.exists(pictures_path):
            os.makedirs(pictures_path)

        file_path = os.path.join(results_path, 'Eigenvalues.txt')
        with open(file_path, 'w') as file:
            for i in range(len(self.Eigenvalue)):
                file.write(f'Собственное число №{i + 1}: {round(self.Eigenvalue[i] - 1, 7)} \n')

        file_path = os.path.join(results_path, 'Eigenfunctions.txt')
        with open(file_path, 'w') as file:
            for i in range(len(self.Eigenfunctions)):
                file.write(f'Собственная функция №{i + 1}:\n')
                for j in range(len(self.Eigenfunctions[i])):
                    for k in range(len(self.Eigenfunctions[i][j])):
                        if self.Eigenfunctions[i][j][k]:
                            file.write(f'{round(self.Eigenfunctions[i][j][k], self.accuracy)} ')
                        else:
                            file.write(f'Nan')
                    file.write(f'\n')

        figure.savefig(os.path.join(pictures_path, 'Eigenfuctions.png'), bbox_inches='tight')
        figure_heat.savefig(os.path.join(pictures_path, 'Heatmap.png'), bbox_inches='tight')
        figure_counter.savefig(os.path.join(pictures_path, 'Counter.png'), bbox_inches='tight')

    def Print(self):
        First_Vector = self.Ortogonalization(self.Approximate_Vectors)
        Eigenvector, self.Eigenvalue = self.Run(First_Vector)
        self.Eigenfunctions = []


        columns = int(math.ceil(np.sqrt(self.Number_Of_Vectors)))
        if self.Number_Of_Vectors % columns == 0:
            rows = int(self.Number_Of_Vectors // columns)
        else:
            rows = int(self.Number_Of_Vectors // columns) + 1

        fig = plt.figure(figsize=(12 * columns/3, 6.3 * rows/2))
        plt.tight_layout()

        for i in range(self.Number_Of_Vectors):
            ax = fig.add_subplot(rows, columns, i + 1, projection='3d')
            plt.gca().invert_xaxis()

            Z = np.zeros((self.Number_Of_Points_Y, self.Number_Of_Points_X))
            X, Y = np.meshgrid(self.x, self.y)

            if i == 0:
                for j in range(len(self.coordinates)):
                    Z[self.coordinates[j][1], self.coordinates[j][0]] = round(Eigenvector[i][j], 1)
            else:
                for j in range(len(self.coordinates)):
                    Z[self.coordinates[j][1], self.coordinates[j][0]] = Eigenvector[i][j]

            for j in range(self.Number_Of_Points_Y):
                for k in range(self.Number_Of_Points_X):
                    if Z[j][k] == 0 and [k, j] not in self.coordinates:
                        Z[j][k] = None

            surf = ax.plot_surface(X, Y, Z, cmap="viridis")

            color_bar = plt.colorbar(surf, shrink=0.5, aspect=5, pad=0.2)
            plt.grid()
            plt.xlabel('Ось X', fontsize=12)
            plt.ylabel('Ось Y', fontsize=12)
            plt.title("График функции №"+str(i+1), fontsize=15)


        fig_heat = plt.figure(figsize=(18.3 * columns/3, 10 * rows/2))
        plt.tight_layout()
        for i in range(self.Number_Of_Vectors):
            ax = fig_heat.add_subplot(rows, columns, i + 1)
            plt.gca().invert_yaxis()

            Z = np.zeros((self.Number_Of_Points_Y, self.Number_Of_Points_X))

            if i == 0:
                for j in range(len(self.coordinates)):
                    Z[self.coordinates[j][1], self.coordinates[j][0]] = round(Eigenvector[i][j], self.accuracy)
            else:
                for j in range(len(self.coordinates)):
                    Z[self.coordinates[j][1], self.coordinates[j][0]] = Eigenvector[i][j]
            for j in range(self.Number_Of_Points_Y):
                for k in range(self.Number_Of_Points_X):
                    if Z[j][k] == 0 and [k, j] not in self.coordinates:
                        Z[j][k] = None
            plt.imshow(Z, cmap='viridis', interpolation='nearest')
            self.Eigenfunctions.append(Z)
            plt.gca().invert_yaxis()

            if len(self.y) <= 9:
                ax.set_yticks([i for i in range(0, len(self.y), len(self.y) // 5 + 1)])
                ax.set_yticklabels([f'{self.h * i:.1f}' for i in range(0, len(self.y), len(self.y) // 5 + 1)])
            else:
                ax.set_yticks([i for i in range(0, len(self.y), len(self.y) // 5 - 1)])
                ax.set_yticklabels([f'{self.h * i:.1f}' for i in range(0, len(self.y), len(self.y) // 5 - 1)])

            if len(self.x) <= 9:
                ax.set_xticks([i for i in range(0, len(self.x), len(self.x) // 5 + 1)])
                ax.set_xticklabels([f'{self.h * i:.1f}' for i in range(0, len(self.x), len(self.x) // 5 + 1)])
            else:
                ax.set_xticks([i for i in range(0, len(self.x), len(self.x) // 5 - 1)])
                ax.set_xticklabels([f'{self.h*i:.1f}' for i in range(0, len(self.x), len(self.x) // 5 - 1)])

            color_bar = plt.colorbar(shrink=0.7, aspect=10, pad=0.05)
            plt.xlabel('Ось X', fontsize=12)
            plt.ylabel('Ось Y', fontsize=12)
            plt.title("Тепловой график функции №"+str(i+1), fontsize=15)

        fig_counter = plt.figure(figsize=(18.3 * columns / 3, 10 * rows / 2))
        plt.tight_layout()
        X, Y = np.meshgrid(self.x, self.y)
        for i in range(self.Number_Of_Vectors):
            ax = fig_counter.add_subplot(rows, columns, i + 1)

            plt.contourf(X, Y, self.Eigenfunctions[i], cmap='viridis')

            color_bar = plt.colorbar(shrink=1, aspect=15, pad=0.05)
            formatter = ticker.ScalarFormatter(useOffset=False)
            color_bar.formatter = formatter
            color_bar.formatter = ticker.FuncFormatter(self.custom_round)
            color_bar.update_ticks()
            plt.xlabel('Ось X', fontsize=12)
            plt.ylabel('Ось Y', fontsize=12)
            plt.title("Контурный график функции №" + str(i + 1), fontsize=15)


        self.Save(fig, fig_heat, fig_counter)

    def printOneFunction(self, num):
        rows = 2
        columns = 2
        X, Y = np.meshgrid(self.x, self.y)

        fig = plt.figure(figsize=(5.5, 4.5))
        plt.tight_layout()

        for i in range(4):
            ax = fig.add_subplot(rows, columns, i + 1, projection='3d')
            surf = ax.plot_surface(X, Y, self.Eigenfunctions[num], cmap="viridis")
            color_bar = plt.colorbar(surf, shrink=0.5, aspect=5)
            ax.set_zticks([])

            if i == 0:
                ax.view_init(elev=0, azim=0)
                plt.ylabel('Ось Y', fontsize=8)
                ax.set_xticks([])
            if i == 1:
                ax.view_init(elev=0, azim=90)
                plt.xlabel('Ось X', fontsize=8)
                ax.set_yticks([])
            if i == 2:
                ax.view_init(elev=45, azim=45)
                plt.xlabel('Ось X', fontsize=8)
                plt.ylabel('Ось Y', fontsize=8)
            if i == 3:
                ax.view_init(elev=45, azim=-45)
                plt.xlabel('Ось X', fontsize=8)
                plt.ylabel('Ось Y', fontsize=8)

            plt.gca().invert_xaxis()
            plt.grid()
        fig.savefig('Pictures/Results/Eigenfunction.png', bbox_inches='tight')
        plt.close(fig)

    def printHeatMap(self, num):
        rows = 1
        columns = 1
        Y, X = np.meshgrid(self.y, self.x)

        fig = plt.figure(figsize=(4.75, 5))
        ax = fig.add_subplot(rows, columns, 1)
        plt.tight_layout()
        plt.gca().invert_yaxis()

        plt.imshow(self.Eigenfunctions[num], cmap='viridis', interpolation='nearest')

        if len(self.y) <= 9:
            ax.set_yticks([i for i in range(0, len(self.y), len(self.y) // 5 + 1)])
            ax.set_yticklabels([f'{self.h * i:.1f}' for i in range(0, len(self.y), len(self.y) // 5 + 1)])
        else:
            ax.set_yticks([i for i in range(0, len(self.y), len(self.y) // 5 - 1)])
            ax.set_yticklabels([f'{self.h * i:.1f}' for i in range(0, len(self.y), len(self.y) // 5 - 1)])

        if len(self.x) <= 9:
            ax.set_xticks([i for i in range(0, len(self.x), len(self.x) // 5 + 1)])
            ax.set_xticklabels([f'{self.h * i:.1f}' for i in range(0, len(self.x), len(self.x) // 5 + 1)])
        else:
            ax.set_xticks([i for i in range(0, len(self.x), len(self.x) // 5 - 1)])
            ax.set_xticklabels([f'{self.h * i:.1f}' for i in range(0, len(self.x), len(self.x) // 5 - 1)])

        color_bar = plt.colorbar(shrink=0.7, aspect=10)

        plt.gca().invert_yaxis()
        plt.xlabel('Ось X', fontsize=12)
        plt.ylabel('Ось Y', fontsize=12)
        fig.savefig('Pictures/Results/Eigenfunction.png', bbox_inches='tight')
        plt.close(fig)

    def printCounterMap(self, num):
        rows = 1
        columns = 1
        X, Y = np.meshgrid(self.x, self.y)

        fig = plt.figure(figsize=(4.75, 5))
        ax = fig.add_subplot(rows, columns, 1)
        plt.tight_layout()

        plt.contourf(X, Y, self.Eigenfunctions[num], cmap='viridis')
        cbar = plt.colorbar(shrink=0.7, aspect=10)
        formatter = ticker.ScalarFormatter(useOffset=False)
        cbar.formatter = formatter
        cbar.formatter = ticker.FuncFormatter(self.custom_round)
        cbar.update_ticks()
        plt.xlabel('Ось X', fontsize=12)
        plt.ylabel('Ось Y', fontsize=12)
        fig.savefig('Pictures/Results/Eigenfunction.png', bbox_inches='tight')
        plt.close(fig)

    def custom_round(self, x, pos):
        return f'{x:.3f}'