import matplotlib.pyplot as plt


class ChartsViewer:
    """
    ChartsViewer
    =======
    A class used to display line charts.
    """

    @staticmethod
    def get_line_chart(values, title="Line Chart", x_label="X", y_label="Y"):
        """
        Displays a line chart with the specified values, title, and labels.

        Args:
            values (list): The values to be plotted.
            title (str): The title of the chart. Defaults to "Line Chart".
            x_label (str): The label for the x-axis. Defaults to "X".
            y_label (str): The label for the y-axis. Defaults to "Y".
        """
        plt.plot(range(1, len(values) + 1), values, marker='o')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid()
        plt.show()
