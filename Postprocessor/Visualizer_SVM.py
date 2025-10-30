"""
AIStructDynSolve
A framework focused on solving structural dynamics problems using artificial intelligence (AI) methods
solve the following ODE of MDOF
M*U_dotdot+C*U_dot+K*U=Pt
initial condition: U(t=0)=InitialU, and U_dot(t=0)=InitialU_dot
Author: 杜轲 duke@iem.ac.cn
Date: 2023/12/26
"""
import matplotlib.pyplot as plt
import numpy as np

class Visualizer_SVM:
    def __init__(self, result_SVM):
        """
        Initialize the Visualizer class.

        Parameters:
        - result_SVM: A dictionary containing displacement, velocity, and acceleration data,
                          with keys 'U', 'U_dt', and 'U_dt2', respectively.
        - eq_time: The time series.
        - dof: Number of degrees of freedom (DOF).
        """
        self.U = result_SVM['U']
        self.U_dt = result_SVM['U_dt']
        self.U_dt2 = result_SVM['U_dt2']
        self.time = result_SVM['time']
        self.dof = result_SVM['DOF']

    def displacement(self):
        """
        Plot the displacement graph and save the displacement data.
        The default filename is displacement_results.txt.
        """
        self.plot_displacement()
        self.save_displacement('displacement_SVM_results.txt')

    def velocity(self):
        """
        Plot the velocity graph and save the velocity data.
        The default filename is velocity_results.txt.
        """
        self.plot_velocity()
        self.save_velocity('velocity_SVM_results.txt')

    def acceleration(self):
        """
        Plot the acceleration graph and save the acceleration data.
        The default filename is acceleration_results.txt.
        """
        self.plot_acceleration()
        self.save_acceleration('acceleration_SVM_results.txt')

    def plot_displacement(self):
        """Plot the displacement graph."""
        self._plot(self.U, 'Displacement_SVM_results', 'Displacement (U)')

    def plot_velocity(self):
        """Plot the velocity graph."""
        self._plot(self.U_dt, 'Velocity_SVM_results', 'Velocity (U_dt)')

    def plot_acceleration(self):
        """Plot the acceleration graph."""
        self._plot(self.U_dt2, 'Acceleration_SVM_results', 'Acceleration (U_dt2)')

    def save_displacement(self, file_name):
        """Save the displacement data to a text file."""
        self._save_to_txt(file_name, self.U, 'Displacement')

    def save_velocity(self, file_name):
        """Save the velocity data to a text file."""
        self._save_to_txt(file_name, self.U_dt, 'Velocity')

    def save_acceleration(self, file_name):
        """Save the acceleration data to a text file."""
        self._save_to_txt(file_name, self.U_dt2, 'Acceleration')

    def _plot(self, data, title, ylabel):
        """
        Private method to plot a graph.

        Parameters:
        - data: The data array to be plotted.
        - title: The graph title.
        - ylabel: The label for the y-axis.
        """
        plt.figure(figsize=(8, 6))
        for i in range(self.dof):
            plt.plot(self.time, data[:, i], label=f'DOF {i + 1}', linewidth=2)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.grid(True, linestyle=':', alpha=1)
        plt.title(title)
        plt.show()

    def _save_to_txt(self, file_name, data, data_type):
        """
        Private method to save data to a text file.

        Parameters:
        - file_name: The path of the file to save.
        - data: The data to be saved.
        - data_type: The type of data (Displacement, Velocity, or Acceleration).
        """
        header = ['Time'] + [f'{data_type}_DOF{i + 1}' for i in range(self.dof)]
        time_array = np.array(self.time)  # Convert to NumPy array
        combined_data = np.hstack([time_array.reshape(-1, 1), data])
        np.savetxt(file_name, combined_data, header='\t'.join(header), delimiter='\t')
        print(f"{data_type} results saved to {file_name}")
