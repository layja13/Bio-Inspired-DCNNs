import pandas as pd
import matplotlib.pyplot as plt
from functions import plot_perturbed, plot_cifar10g_results


# Load the CSV files
cifar10g_results = pd.read_csv('metrics\generalisation\Gabor_VGG16_1_generalise_s42.csv') # Figure 7 of Evans
cifar10_perturbed_results = pd.read_csv('metrics\perturbed_cifar\Gabor_VGG16_1_perturb_cifar10_s42.csv') # Figure 6 of Evans
cifar10g_perturbed_results_contours_inverted = pd.read_csv('metrics\perturbed_generalisation\Gabor_VGG16_1_perturb_contours_inverted_s42.csv') # Figure 8 of Evans
cifar10g_perturbed_results_contours = pd.read_csv('metrics\perturbed_generalisation\Gabor_VGG16_1_perturb_contours_s42.csv') # Figure 8 of Evans
cifar10g_perturbed_results_line_drawings_inverted = pd.read_csv('metrics\perturbed_generalisation\Gabor_VGG16_1_perturb_line_drawings_inverted_s42.csv') # Figure 8 of Evans
cifar10g_perturbed_results_line_drawings = pd.read_csv('metrics\perturbed_generalisation\Gabor_VGG16_1_perturb_line_drawings_s42.csv') # Figure 8 of Evans
cifar10g_perturbed_results_silhouettes = pd.read_csv('metrics\perturbed_generalisation\Gabor_VGG16_1_perturb_silhouettes_inverted_s42.csv') # Figure 8 of Evans
cifar10g_perturbed_results_silhouettes_inverted = pd.read_csv('metrics\perturbed_generalisation\Gabor_VGG16_1_perturb_silhouettes_s42.csv') # Figure 8 of Evans


# Plot the results
plot_cifar10g_results(cifar10g_results)

plot_perturbed(cifar10_perturbed_results)

plot_perturbed(cifar10g_perturbed_results_contours_inverted )
plot_perturbed(cifar10g_perturbed_results_contours)
plot_perturbed(cifar10g_perturbed_results_line_drawings_inverted)
plot_perturbed(cifar10g_perturbed_results_line_drawings)
plot_perturbed(cifar10g_perturbed_results_silhouettes)
plot_perturbed(cifar10g_perturbed_results_silhouettes_inverted)


