class Vector:
    def __init__(self, vector):
        self.vector = list(vector)

    def set_element(self, index, value): #eстановка элемента по индексу
        self.vector[index] = value

    def get_element(self, index): #доступ к элементу по индексу
        return self.vector[index]

    def length(self): # Длина
        return len(self.vector)


class matrix():
    def __init__(self, data):
        if not (isinstance(data, list) and all(isinstance(row, list) for row in data)):
            raise ValueError("!= list[list[]]")

        row_len = len(data[0])

        if not all(len(row) == row_len for row in data):
            raise ValueError("len(list[1]) != len(list[2])")

        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __str__()

    def __repr__(self):
        return f"Matrix({self.data})"

    def transpose(self):
        transposed = []
        for j in range(self.cols):
            new_row = []
            for i in range(self.rows):
                new_row.append(self.data[i][j])
            transposed.append(new_row)

        return matrix(transposed)


m = matrix([[1, 2], [2, 1]])
print(m)


class Vector():
    vector = []
    def __init__(self, data):
        self.vector = list(data)