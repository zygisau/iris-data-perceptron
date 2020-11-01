import matplotlib.pyplot as plt


class PaintingService:
    @staticmethod
    def paint(arr):
        for i in range(0, len(arr), 2):
            plt.plot(arr[i], i, 'ro-')

        plt.show()
