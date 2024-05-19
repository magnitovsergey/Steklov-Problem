class Processor:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.lines = [list(line.strip()) for line in file.readlines()]
        self.fill_ones_between()
        self.remove_sparse_lines()
        self.process_matrix()
        self.final_process_matrix()
        self.replace_zeros_with_ones()
        self.remove_empty_columns()
        self.result_writing()

    def replace_zeros_with_ones(self):
        rows = len(self.lines)
        cols = len(self.lines[0]) if rows > 0 else 0

        for row in range(rows):
            for col in range(cols):
                if self.lines[row][col] == '0':
                    if row > 0 and row < rows - 1 and self.lines[row - 1][col] == '1' and self.lines[row + 1][col] == '1':
                        self.lines[row][col] = '1'
                    elif col > 0 and col < cols - 1 and self.lines[row][col - 1] == '1' and self.lines[row][col + 1] == '1':
                        self.lines[row][col] = '1'
    def fill_ones_between(self):
        for i in range(len(self.lines)):
            line = self.lines[i]
            if '1' in line:
                left_index = line.index('1')
                right_index = len(line) - 1 - line[::-1].index('1')
                for j in range(left_index, right_index + 1):
                    self.lines[i][j] = '1'

    def remove_sparse_lines(self):
        self.lines = [line for line in self.lines if line.count('1') > 1]

    def process_matrix(self):
        rows = len(self.lines)
        cols = len(self.lines[0]) if rows > 0 else 0
        for row in range(rows):
            for col in range(cols):
                if self.lines[row][col] == '1' and self.six_or_more_zeros(row, col):
                    self.lines[row][col] = '0'

    def six_or_more_zeros(self, row, col):
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if (dr != 0 or dc != 0) and self.is_forbidden_neighbor(row + dr, col + dc, '0'):
                    count += 1
        return count >= 6

    def is_forbidden_neighbor(self, row, col, neighbor):
        if 0 <= row < len(self.lines) and 0 <= col < len(self.lines[0]):
            return self.lines[row][col] == neighbor
        if neighbor == '0':
            return True
        return False

    def final_process_matrix(self):
        rows = len(self.lines)
        cols = len(self.lines[0]) if rows > 0 else 0
        for row in range(rows):
            for col in range(cols):
                if self.lines[row][col] == '1' and self.four_points(row, col):
                    self.lines[row][col] = '0'

    def four_points(self, row, col):
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if (dr != 0 or dc != 0) and self.is_forbidden_neighbor(row + dr, col + dc, '1'):
                    if dr*dc == 0:
                        count += 2
                    else:
                        count += 1
        return count == 4

    def remove_empty_columns(self):
        if not self.lines:
            return
        rows = len(self.lines)
        cols = len(self.lines[0])
        columns_to_remove = set()

        for col in range(cols):
            if all(self.lines[row][col] == '0' for row in range(rows)):
                columns_to_remove.add(col)

       
        for i in range(rows):
            self.lines[i] = [
                self.lines[i][col] for col in range(cols) if col not in columns_to_remove
            ]

    def result_writing(self):
        with open('Data/area.txt', 'w') as file:
            for line in self.lines:
                file.write(''.join(line) + '\n')
