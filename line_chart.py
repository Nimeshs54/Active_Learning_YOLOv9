import matplotlib.pyplot as plt
import seaborn as sns

# Data for Uncertainty Sampling (Average)
training_samples = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
mAP_uncertainty_sampling = [67.9, 69.9, 70.0, 84.6, 90.1, 89.1, 92.7, 94.1, 93.3, 93.9, 94.0, 93.9, 95.5, 95.9, 95.4, 95.8]

# Data for Random Sampling
# mAP_random_sampling = [62.4, 72.7, 79.6, 81.4, 84.5, 87.8, 85.3, 86.7, 88.9, 91.7, 91.6, 92.3, 93.6, 93.4, 93.6, 94.3]
mAP_random_sampling = [0.05, 0.03, 63.5, 77.9, 81.5, 84.2, 83.7, 87.7, 88.3, 89.2, 88.6, 91.5, 91.6, 95.1, 94.7, 95.1]

# Data for Uncertainty Sampling (Max)
mAP_uncertainty_max = [71.6, 79.6, 84.1, 85.6, 91.4, 90.3, 90.1, 92.2, 92.2, 93.5, 93.7, 94.5, 95.6, 95.6, 95.0, 95.2]

# Data for Uncertainty Sampling (Sum)
mAP_uncertainty_sum = [56.3, 59.9, 75.7, 76.0, 81.0, 85.9, 86.5, 89.6, 88.5, 89.5, 90.2, 92.6, 93.1, 93.7, 95.1, 95.7]

# Set the style and context for the plot
sns.set(style="whitegrid", context="talk")

# Plotting the data
plt.figure(figsize=(14, 8))

# Colors as requested
plt.plot(training_samples, mAP_uncertainty_sampling, marker='o', linestyle='-', color='#006400', label='Pipline using US-Avg')
plt.plot(training_samples, mAP_uncertainty_max, marker='^', linestyle='-', color='#00008b', label='Pipline using US-Max')        
plt.plot(training_samples, mAP_uncertainty_sum, marker='d', linestyle='-', color='#ff0000', label='Pipline using US-Sum')

# plt.plot(training_samples, mAP_uncertainty_sampling, marker='o', linestyle='-', color='#008B8B', label='Uncertainty Sampling (Average)')
# plt.plot(training_samples, mAP_uncertainty_max, marker='^', linestyle='-', color='#B22222', label='Uncertainty Sampling (Max)')        
# plt.plot(training_samples, mAP_uncertainty_sum, marker='d', linestyle='-', color='#FFD700', label='Uncertainty Sampling (Sum)')

# Distinct color for Random Sampling
plt.plot(training_samples, mAP_random_sampling, marker='s', linestyle='-', color='#DAA520', label='w/o phase 1')

# # Adding a bold vertical dashed line at 400 training samples
# plt.axvline(x=400, color='lightcoral', linestyle='--', linewidth=2.5)
# plt.text(410, 70, '50% Data Portion', color='lightcoral', fontsize=14, verticalalignment='center', fontweight='bold')

# Adding titles and labels
plt.title('Comparison of Full Pipeline vs. Phase 2 Only', fontsize=18, fontweight='bold')
plt.xlabel('Number of Training Samples', fontsize=16)
plt.ylabel('mAP Percentage', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Adding legend
plt.legend(fontsize=14)

# Adding grid for better readability
plt.grid(True, linestyle='--', linewidth=0.5)

# Adjust layout for saving
plt.tight_layout()

# Save the plot as a PDF file
plt.savefig('sampling_comparison.pdf', format='pdf')

# Display the plot
plt.show()



# import matplotlib.pyplot as plt
# import seaborn as sns

# # Data for Uncertainty Sampling (Average)
# training_samples = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
# mAP_uncertainty_sampling = [67.9, 69.9, 70.0, 84.6, 90.1, 89.1, 92.7, 94.1, 93.3, 93.9, 94.0, 93.9, 95.5, 95.9, 95.4, 95.8]

# # Data for Random Sampling
# mAP_random_sampling = [62.4, 72.7, 79.6, 81.4, 84.5, 87.8, 85.3, 86.7, 88.9, 91.7, 91.6, 92.3, 93.6, 93.4, 93.6, 94.3]

# # Data for Uncertainty Sampling (Max)
# mAP_uncertainty_max = [71.6, 79.6, 84.1, 85.6, 91.4, 90.3, 90.1, 92.2, 92.2, 93.5, 93.7, 94.5, 95.6, 95.6, 95.0, 95.2]

# # Data for Uncertainty Sampling (Sum)
# mAP_uncertainty_sum = [56.3, 59.9, 75.7, 76.0, 81.0, 85.9, 86.5, 89.6, 88.5, 89.5, 90.2, 92.6, 93.1, 93.7, 95.1, 95.7]

# # Set the style and context for the plot
# sns.set(style="whitegrid", context="talk")

# # Plotting the data
# plt.figure(figsize=(14, 8))

# # Colors as requested
# plt.plot(training_samples, mAP_uncertainty_sampling, marker='o', linestyle='-', color='#006400', label='Uncertainty Sampling (Average)')  # Dark Green
# plt.plot(training_samples, mAP_uncertainty_max, marker='^', linestyle='-', color='#00008b', label='Uncertainty Sampling (Max)')         # Dark Blue
# plt.plot(training_samples, mAP_uncertainty_sum, marker='d', linestyle='-', color='#d2691e', label='Uncertainty Sampling (Sum)')         # Dark Orange

# # Distinct color for Random Sampling
# plt.plot(training_samples, mAP_random_sampling, marker='s', linestyle='-', color='#ff0000', label='Random Sampling')  # Red

# # Adding a bold vertical dashed line at 400 training samples
# plt.axvline(x=400, color='lightcoral', linestyle='--', linewidth=2.5)
# plt.text(410, 70, '50% Data Portion', color='lightcoral', fontsize=14, verticalalignment='center', fontweight='bold')

# # Adding titles and labels
# plt.title('Comparison of Uncertainty Sampling Methods and Random Sampling', fontsize=18, fontweight='bold')
# plt.xlabel('Number of Training Samples', fontsize=16)
# plt.ylabel('mAP Percentage', fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# # Adding legend
# plt.legend(fontsize=14)

# # Adding grid for better readability
# plt.grid(True, linestyle='--', linewidth=0.5)

# # Display the plot
# plt.tight_layout()
# plt.show()







# # # # # # import matplotlib.pyplot as plt
# # # # # # import seaborn as sns

# # # # # # # Data for the plots
# # # # # # training_samples = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

# # # # # # # mAP values for each method
# # # # # # mAP_uncertainty_avg = [67.9, 69.9, 70.0, 84.6, 90.1, 89.1, 92.7, 94.1, 93.3, 93.9, 94.0, 93.9, 95.5, 95.9, 95.4, 95.8]
# # # # # # mAP_uncertainty_max = [71.6, 79.6, 84.1, 85.6, 91.4, 90.3, 90.1, 92.2, 92.2, 93.5, 93.7, 94.5, 95.6, 95.6, 95.0, 95.2]
# # # # # # mAP_uncertainty_sum = [56.3, 59.9, 75.7, 76.0, 81.0, 85.9, 86.5, 89.6, 88.5, 89.5, 90.2, 92.6, 93.1, 93.7, 95.1, 95.7]
# # # # # # mAP_random_sampling = [62.4, 72.7, 79.6, 81.4, 84.5, 87.8, 85.3, 86.7, 88.9, 91.7, 91.6, 92.3, 93.6, 93.4, 93.6, 94.3]

# # # # # # # Set the style and context for the plots
# # # # # # sns.set(style="whitegrid", context="talk")

# # # # # # # Create a figure with 3 subplots
# # # # # # fig, axes = plt.subplots(3, 1, figsize=(14, 18))

# # # # # # # Plot 1: Uncertainty Sampling (Average) vs Random Sampling
# # # # # # axes[0].plot(training_samples, mAP_uncertainty_avg, marker='o', linestyle='-', color='#006400', label='Uncertainty Sampling (Average)')  # Dark Green
# # # # # # axes[0].plot(training_samples, mAP_random_sampling, marker='s', linestyle='-', color='#ff0000', label='Random Sampling')  # Red
# # # # # # axes[0].axvline(x=400, color='lightcoral', linestyle='--', linewidth=2.5)
# # # # # # axes[0].text(410, 70, '50% Data Portion', color='lightcoral', fontsize=14, verticalalignment='center', fontweight='bold')
# # # # # # axes[0].set_title('Uncertainty Sampling (Average) vs Random Sampling', fontsize=18, fontweight='bold')
# # # # # # axes[0].set_xlabel('Number of Training Samples', fontsize=16)
# # # # # # axes[0].set_ylabel('mAP Percentage', fontsize=16)
# # # # # # axes[0].legend(fontsize=14)
# # # # # # axes[0].grid(True, linestyle='--', linewidth=0.5)

# # # # # # # Plot 2: Uncertainty Sampling (Max) vs Random Sampling
# # # # # # axes[1].plot(training_samples, mAP_uncertainty_max, marker='^', linestyle='-', color='#00008b', label='Uncertainty Sampling (Max)')  # Dark Blue
# # # # # # axes[1].plot(training_samples, mAP_random_sampling, marker='s', linestyle='-', color='#ff0000', label='Random Sampling')  # Red
# # # # # # axes[1].axvline(x=400, color='lightcoral', linestyle='--', linewidth=2.5)
# # # # # # axes[1].text(410, 70, '50% Data Portion', color='lightcoral', fontsize=14, verticalalignment='center', fontweight='bold')
# # # # # # axes[1].set_title('Uncertainty Sampling (Max) vs Random Sampling', fontsize=18, fontweight='bold')
# # # # # # axes[1].set_xlabel('Number of Training Samples', fontsize=16)
# # # # # # axes[1].set_ylabel('mAP Percentage', fontsize=16)
# # # # # # axes[1].legend(fontsize=14)
# # # # # # axes[1].grid(True, linestyle='--', linewidth=0.5)

# # # # # # # Plot 3: Uncertainty Sampling (Sum) vs Random Sampling
# # # # # # axes[2].plot(training_samples, mAP_uncertainty_sum, marker='d', linestyle='-', color='#d2691e', label='Uncertainty Sampling (Sum)')  # Dark Orange
# # # # # # axes[2].plot(training_samples, mAP_random_sampling, marker='s', linestyle='-', color='#ff0000', label='Random Sampling')  # Red
# # # # # # axes[2].axvline(x=400, color='lightcoral', linestyle='--', linewidth=2.5)
# # # # # # axes[2].text(410, 70, '50% Data Portion', color='lightcoral', fontsize=14, verticalalignment='center', fontweight='bold')
# # # # # # axes[2].set_title('Uncertainty Sampling (Sum) vs Random Sampling', fontsize=18, fontweight='bold')
# # # # # # axes[2].set_xlabel('Number of Training Samples', fontsize=16)
# # # # # # axes[2].set_ylabel('mAP Percentage', fontsize=16)
# # # # # # axes[2].legend(fontsize=14)
# # # # # # axes[2].grid(True, linestyle='--', linewidth=0.5)

# # # # # # # Adjust layout for better spacing
# # # # # # plt.tight_layout()

# # # # # # # Display the plots
# # # # # # plt.show()






# import matplotlib.pyplot as plt
# import seaborn as sns

# # Data for Uncertainty Sampling (Algorithm 1)
# training_samples = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
# mAP_uncertainty_sampling = [67.9, 69.9, 70.0, 84.6, 90.1, 89.1, 92.7, 94.1, 93.3, 93.9, 94.0, 93.9, 95.5, 95.9, 95.4, 95.8]

# # Data for Random Sampling (Algorithm 2)
# mAP_random_sampling = [62.4, 72.7, 79.6, 81.4, 84.5, 87.8, 85.3, 86.7, 88.9, 91.7, 91.6, 92.3, 93.6, 93.4, 93.6, 94.3]

# # Set the style and context for the plot
# sns.set(style="whitegrid", context="talk")

# # Plotting the data
# plt.figure(figsize=(14, 8))
# plt.plot(training_samples, mAP_uncertainty_sampling, marker='o', linestyle='-', color='#006400', label='Uncertainty Sampling (Avg)')
# plt.plot(training_samples, mAP_random_sampling, marker='s', linestyle='-', color='#ff0000', label='Random Sampling')

# # Adding a bold vertical dashed line at 400 training samples
# plt.axvline(x=400, color='lightcoral', linestyle='--', linewidth=2.5)
# plt.text(410, 70, '50% Data Portion', color='lightcoral', fontsize=14, verticalalignment='center', fontweight='bold')

# # Adding titles and labels
# plt.title('Comparison of Uncertainty - Avg and Random Sampling', fontsize=18, fontweight='bold')
# plt.xlabel('Number of Training Samples', fontsize=16)
# plt.ylabel('mAP Percentage', fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# # Adding legend
# plt.legend(fontsize=14)

# # Adding grid for better readability
# plt.grid(True, linestyle='--', linewidth=0.5)

# # Save the plot as a PDF
# plt.tight_layout()
# plt.savefig('line_chart_comparison.pdf', format='pdf')

# # Display the plot (optional)
# plt.show()




# # # # # # import matplotlib.pyplot as plt
# # # # # # import seaborn as sns

# # # # # # # Data for Uncertainty Sampling (Algorithm 1)
# # # # # # training_samples = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

# # # # # # mAP_uncertainty_sampling = {
# # # # # #     'BLUE_LID​': [81.5, 69.9, 70.0, 84.6, 90.1, 89.1, 92.7, 94.1, 93.3, 93.9, 94.0, 93.9, 95.5, 95.9, 95.4, 95.8],
# # # # # #     'BLUE_TRAILER​': [65.2, 67.5, 68.3, 82.5, 88.7, 87.6, 91.3, 93.0, 92.2, 92.8, 93.0, 92.8, 94.6, 95.1, 94.8, 95.2],
# # # # # #     'CABIN': [70.1, 72.3, 73.0, 86.0, 91.2, 90.0, 93.2, 94.7, 94.0, 94.5, 94.7, 94.5, 96.0, 96.4, 96.0, 96.5],
# # # # # #     'FRONT_CHASSIS': [60.0, 62.0, 63.0, 77.0, 83.0, 82.0, 86.0, 88.0, 87.5, 88.0, 88.5, 88.0, 89.5, 90.0, 89.5, 90.5],
# # # # # #     'REAR_CHASSIS​': [68.0, 70.5, 71.0, 85.0, 90.0, 89.0, 92.5, 94.0, 93.0, 93.5, 94.0, 93.5, 95.0, 95.5, 95.0, 95.5],
# # # # # #     'YELLOW_LID​': [66.0, 68.0, 69.0, 83.0, 88.0, 87.0, 90.5, 92.0, 91.5, 92.0, 92.5, 92.0, 93.5, 94.0, 93.5, 94.0],
# # # # # #     'YELLOW_TRAILER': [64.0, 66.0, 67.0, 81.0, 86.0, 85.0, 88.5, 90.0, 89.5, 90.0, 90.5, 90.0, 91.5, 92.0, 91.5, 92.0],
# # # # # #     'GLUE_GUN': [61.0, 63.0, 64.0, 78.0, 84.0, 83.0, 87.0, 89.0, 88.5, 89.0, 89.5, 89.0, 90.5, 91.0, 90.5, 91.0]
# # # # # # }

# # # # # # # Data for Random Sampling (Algorithm 2)
# # # # # # mAP_random_sampling = {
# # # # # #     'BLUE_LID​': [62.4, 72.7, 79.6, 81.4, 84.5, 87.8, 85.3, 86.7, 88.9, 91.7, 91.6, 92.3, 93.6, 93.4, 93.6, 94.3],
# # # # # #     'BLUE_TRAILER​': [60.0, 70.0, 77.0, 79.0, 82.0, 85.0, 83.0, 84.5, 86.0, 88.5, 88.0, 89.0, 90.5, 90.0, 90.5, 91.0],
# # # # # #     'CABIN': [65.0, 75.0, 80.0, 83.0, 86.0, 89.0, 87.0, 88.5, 90.0, 92.0, 91.5, 92.5, 94.0, 93.5, 94.0, 94.5],
# # # # # #     'FRONT_CHASSIS': [55.0, 65.0, 72.0, 74.0, 77.0, 80.0, 78.0, 79.5, 81.0, 83.0, 82.5, 83.5, 85.0, 84.5, 85.0, 85.5],
# # # # # #     'REAR_CHASSIS​': [63.0, 73.0, 78.0, 81.0, 84.0, 87.0, 85.0, 86.5, 88.0, 90.0, 89.5, 90.5, 92.0, 91.5, 92.0, 92.5],
# # # # # #     'YELLOW_LID​': [61.0, 71.0, 76.0, 79.0, 82.0, 85.0, 83.0, 84.5, 86.0, 88.0, 87.5, 88.5, 90.0, 89.5, 90.0, 90.5],
# # # # # #     'YELLOW_TRAILER': [59.0, 69.0, 74.0, 77.0, 80.0, 83.0, 81.0, 82.5, 84.0, 86.0, 85.5, 86.5, 88.0, 87.5, 88.0, 88.5],
# # # # # #     'GLUE_GUN': [56.0, 66.0, 71.0, 74.0, 77.0, 80.0, 78.0, 79.5, 81.0, 83.0, 82.5, 83.5, 85.0, 84.5, 85.0, 85.5]
# # # # # # }

# # # # # # # Set the style and context for the plot
# # # # # # sns.set(style="whitegrid", context="talk")

# # # # # # # Plotting the data for each class
# # # # # # fig, axes = plt.subplots(4, 2, figsize=(16, 20), sharex=True, sharey=True)
# # # # # # axes = axes.flatten()

# # # # # # for i, class_name in enumerate(mAP_uncertainty_sampling.keys()):
# # # # # #     ax = axes[i]
# # # # # #     ax.plot(training_samples, mAP_uncertainty_sampling[class_name], marker='o', linestyle='-', color='#1f77b4', label='Uncertainty Sampling')
# # # # # #     ax.plot(training_samples, mAP_random_sampling[class_name], marker='s', linestyle='-', color='#ff7f0e', label='Random Sampling')
# # # # # #     ax.axvline(x=400, color='lightcoral', linestyle='--', linewidth=2.5)
# # # # # #     ax.text(410, min(mAP_uncertainty_sampling[class_name]), '50% Data Portion', color='lightcoral', fontsize=12, verticalalignment='center', fontweight='bold')
# # # # # #     ax.set_title(class_name, fontsize=16, fontweight='bold')
# # # # # #     ax.set_xlabel('Training Samples', fontsize=14)
# # # # # #     ax.set_ylabel('mAP (%)', fontsize=14)
# # # # # #     ax.legend(fontsize=12)
# # # # # #     ax.grid(True, linestyle='--', linewidth=0.5)
# # # # # #     ax.tick_params(axis='both', which='major', labelsize=12)

# # # # # # # Adjust layout
# # # # # # plt.tight_layout()
# # # # # # plt.show()


# # # # # # import matplotlib.pyplot as plt
# # # # # # import seaborn as sns

# # # # # # # Data for Uncertainty Sampling (Algorithm 1)
# # # # # # training_samples = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

# # # # # # mAP_uncertainty_sampling = {
# # # # # #     'Class 1': [67.9, 69.9, 70.0, 84.6, 90.1, 89.1, 92.7, 94.1, 93.3, 93.9, 94.0, 93.9, 95.5, 95.9, 95.4, 95.8],
# # # # # #     'Class 2': [65.2, 67.5, 68.3, 82.5, 88.7, 87.6, 91.3, 93.0, 92.2, 92.8, 93.0, 92.8, 94.6, 95.1, 94.8, 95.2],
# # # # # #     'Class 3': [70.1, 72.3, 73.0, 86.0, 91.2, 90.0, 93.2, 94.7, 94.0, 94.5, 94.7, 94.5, 96.0, 96.4, 96.0, 96.5],
# # # # # #     'Class 4': [60.0, 62.0, 63.0, 77.0, 83.0, 82.0, 86.0, 88.0, 87.5, 88.0, 88.5, 88.0, 89.5, 90.0, 89.5, 90.5],
# # # # # #     'Class 5': [68.0, 70.5, 71.0, 85.0, 90.0, 89.0, 92.5, 94.0, 93.0, 93.5, 94.0, 93.5, 95.0, 95.5, 95.0, 95.5],
# # # # # #     'Class 6': [66.0, 68.0, 69.0, 83.0, 88.0, 87.0, 90.5, 92.0, 91.5, 92.0, 92.5, 92.0, 93.5, 94.0, 93.5, 94.0],
# # # # # #     'Class 7': [64.0, 66.0, 67.0, 81.0, 86.0, 85.0, 88.5, 90.0, 89.5, 90.0, 90.5, 90.0, 91.5, 92.0, 91.5, 92.0],
# # # # # #     'Class 8': [61.0, 63.0, 64.0, 78.0, 84.0, 83.0, 87.0, 89.0, 88.5, 89.0, 89.5, 89.0, 90.5, 91.0, 90.5, 91.0]
# # # # # # }

# # # # # # # Data for Random Sampling (Algorithm 2)
# # # # # # mAP_random_sampling = {
# # # # # #     'Class 1': [62.4, 72.7, 79.6, 81.4, 84.5, 87.8, 85.3, 86.7, 88.9, 91.7, 91.6, 92.3, 93.6, 93.4, 93.6, 94.3],
# # # # # #     'Class 2': [60.0, 70.0, 77.0, 79.0, 82.0, 85.0, 83.0, 84.5, 86.0, 88.5, 88.0, 89.0, 90.5, 90.0, 90.5, 91.0],
# # # # # #     'Class 3': [65.0, 75.0, 80.0, 83.0, 86.0, 89.0, 87.0, 88.5, 90.0, 92.0, 91.5, 92.5, 94.0, 93.5, 94.0, 94.5],
# # # # # #     'Class 4': [55.0, 65.0, 72.0, 74.0, 77.0, 80.0, 78.0, 79.5, 81.0, 83.0, 82.5, 83.5, 85.0, 84.5, 85.0, 85.5],
# # # # # #     'Class 5': [63.0, 73.0, 78.0, 81.0, 84.0, 87.0, 85.0, 86.5, 88.0, 90.0, 89.5, 90.5, 92.0, 91.5, 92.0, 92.5],
# # # # # #     'Class 6': [61.0, 71.0, 76.0, 79.0, 82.0, 85.0, 83.0, 84.5, 86.0, 88.0, 87.5, 88.5, 90.0, 89.5, 90.0, 90.5],
# # # # # #     'Class 7': [59.0, 69.0, 74.0, 77.0, 80.0, 83.0, 81.0, 82.5, 84.0, 86.0, 85.5, 86.5, 88.0, 87.5, 88.0, 88.5],
# # # # # #     'Class 8': [56.0, 66.0, 71.0, 74.0, 77.0, 80.0, 78.0, 79.5, 81.0, 83.0, 82.5, 83.5, 85.0, 84.5, 85.0, 85.5]
# # # # # # }

# # # # # # # Set the style and context for the plot
# # # # # # sns.set(style="whitegrid", context="talk")

# # # # # # # Define a color palette
# # # # # # colors = sns.color_palette("tab10", 8)

# # # # # # # Plotting the data
# # # # # # plt.figure(figsize=(16, 10))

# # # # # # for i, (class_name, mAP_values) in enumerate(mAP_uncertainty_sampling.items()):
# # # # # #     plt.plot(training_samples, mAP_values, marker='o', linestyle='-', color=colors[i], label=f'{class_name} (Uncertainty Sampling)')

# # # # # # for i, (class_name, mAP_values) in enumerate(mAP_random_sampling.items()):
# # # # # #     plt.plot(training_samples, mAP_values, marker='s', linestyle='--', color=colors[i], label=f'{class_name} (Random Sampling)')

# # # # # # # Adding a bold vertical dashed line at 400 training samples
# # # # # # plt.axvline(x=400, color='lightcoral', linestyle='--', linewidth=2.5)
# # # # # # plt.text(410, 65, '50% Data Portion', color='lightcoral', fontsize=14, verticalalignment='center', fontweight='bold')

# # # # # # # Adding titles and labels
# # # # # # plt.title('Comparison of mAP Results for All Classes', fontsize=18, fontweight='bold')
# # # # # # plt.xlabel('Number of Training Samples', fontsize=16)
# # # # # # plt.ylabel('mAP Percentage', fontsize=16)
# # # # # # plt.xticks(fontsize=14)
# # # # # # plt.yticks(fontsize=14)

# # # # # # # Adding legend
# # # # # # plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

# # # # # # # Adding grid for better readability
# # # # # # plt.grid(True, linestyle='--', linewidth=0.5)

# # # # # # # Display the plot
# # # # # # plt.tight_layout()
# # # # # # plt.show()


# # # # # # import matplotlib.pyplot as plt
# # # # # # import seaborn as sns
# # # # # # import pandas as pd

# # # # # # # Training samples
# # # # # # training_samples = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

# # # # # # # Data for Uncertainty Sampling (Algorithm 1)
# # # # # # mAP_uncertainty_sampling = {
# # # # # #     'BLUE_LID': [67.9, 69.9, 70.0, 84.6, 90.1, 89.1, 92.7, 94.1, 93.3, 93.9, 94.0, 93.9, 95.5, 95.9, 95.4, 95.8],
# # # # # #     'BLUE_TRAILER': [65.2, 67.5, 68.3, 82.5, 88.7, 87.6, 91.3, 93.0, 92.2, 92.8, 93.0, 92.8, 94.6, 95.1, 94.8, 95.2],
# # # # # #     'CABIN': [70.1, 72.3, 73.0, 86.0, 91.2, 90.0, 93.2, 94.7, 94.0, 94.5, 94.7, 94.5, 96.0, 96.4, 96.0, 96.5],
# # # # # #     'FRONT_CHASSIS': [60.0, 62.0, 63.0, 77.0, 83.0, 82.0, 86.0, 88.0, 87.5, 88.0, 88.5, 88.0, 89.5, 90.0, 89.5, 90.5],
# # # # # #     'REAR_CHASSIS': [68.0, 70.5, 71.0, 85.0, 90.0, 89.0, 92.5, 94.0, 93.0, 93.5, 94.0, 93.5, 95.0, 95.5, 95.0, 95.5],
# # # # # #     'YELLOW_LID': [66.0, 68.0, 69.0, 83.0, 88.0, 87.0, 90.5, 92.0, 91.5, 92.0, 92.5, 92.0, 93.5, 94.0, 93.5, 94.0],
# # # # # #     'YELLOW_TRAILER': [64.0, 66.0, 67.0, 81.0, 86.0, 85.0, 88.5, 90.0, 89.5, 90.0, 90.5, 90.0, 91.5, 92.0, 91.5, 92.0],
# # # # # #     'GLUE_GUN': [61.0, 63.0, 64.0, 78.0, 84.0, 83.0, 87.0, 89.0, 88.5, 89.0, 89.5, 89.0, 90.5, 91.0, 90.5, 91.0]
# # # # # # }

# # # # # # # Data for Random Sampling (Algorithm 2)
# # # # # # mAP_random_sampling = {
# # # # # #     'BLUE_LID': [62.4, 72.7, 79.6, 81.4, 84.5, 87.8, 85.3, 86.7, 88.9, 91.7, 91.6, 92.3, 93.6, 93.4, 93.6, 94.3],
# # # # # #     'BLUE_TRAILER': [60.0, 70.0, 77.0, 79.0, 82.0, 85.0, 83.0, 84.5, 86.0, 88.5, 88.0, 89.0, 90.5, 90.0, 90.5, 91.0],
# # # # # #     'CABIN': [65.0, 75.0, 80.0, 83.0, 86.0, 89.0, 87.0, 88.5, 90.0, 92.0, 91.5, 92.5, 94.0, 93.5, 94.0, 94.5],
# # # # # #     'FRONT_CHASSIS': [55.0, 65.0, 72.0, 74.0, 77.0, 80.0, 78.0, 79.5, 81.0, 83.0, 82.5, 83.5, 85.0, 84.5, 85.0, 85.5],
# # # # # #     'REAR_CHASSIS': [63.0, 73.0, 78.0, 81.0, 84.0, 87.0, 85.0, 86.5, 88.0, 90.0, 89.5, 90.5, 92.0, 91.5, 92.0, 92.5],
# # # # # #     'YELLOW_LID': [61.0, 71.0, 76.0, 79.0, 82.0, 85.0, 83.0, 84.5, 86.0, 88.0, 87.5, 88.5, 90.0, 89.5, 90.0, 90.5],
# # # # # #     'YELLOW_TRAILER': [59.0, 69.0, 74.0, 77.0, 80.0, 83.0, 81.0, 82.5, 84.0, 86.0, 85.5, 86.5, 88.0, 87.5, 88.0, 88.5],
# # # # # #     'GLUE_GUN': [56.0, 66.0, 71.0, 74.0, 77.0, 80.0, 78.0, 79.5, 81.0, 83.0, 82.5, 83.5, 85.0, 84.5, 85.0, 85.5]
# # # # # # }

# # # # # # # Combine data into a DataFrame for plotting
# # # # # # data = []
# # # # # # for cls in mAP_uncertainty_sampling.keys():
# # # # # #     for i, samples in enumerate(training_samples):
# # # # # #         data.append((samples, cls, mAP_uncertainty_sampling[cls][i], 'Uncertainty Sampling'))
# # # # # #         data.append((samples, cls, mAP_random_sampling[cls][i], 'Random Sampling'))

# # # # # # df = pd.DataFrame(data, columns=['Training Samples', 'Class', 'mAP', 'Algorithm'])

# # # # # # # Set the style
# # # # # # sns.set(style="whitegrid", context="talk")

# # # # # # # Create a FacetGrid
# # # # # # g = sns.FacetGrid(df, col="Class", hue="Algorithm", col_wrap=4, height=4, aspect=1.2, palette=["#1f77b4", "#ff7f0e"])

# # # # # # # Map the lineplot
# # # # # # g.map(sns.lineplot, "Training Samples", "mAP", marker='o')

# # # # # # # Highlight top performers
# # # # # # for ax, cls in zip(g.axes.flat, mAP_uncertainty_sampling.keys()):
# # # # # #     top_value = max(max(mAP_uncertainty_sampling[cls]), max(mAP_random_sampling[cls]))
# # # # # #     ax.axhline(top_value, ls='--', color='gray', alpha=0.7)

# # # # # # # Add a single legend
# # # # # # g.add_legend(title='Algorithm')

# # # # # # # Improve aesthetics
# # # # # # g.set_titles("{col_name}", size=12, weight='bold')
# # # # # # g.set_axis_labels("Training Samples", "mAP (%)")
# # # # # # for ax in g.axes.flat:
# # # # # #     ax.grid(True, linestyle='--', linewidth=0.5)
# # # # # #     for label in ax.get_xticklabels() + ax.get_yticklabels():
# # # # # #         label.set_fontsize(10)

# # # # # # plt.subplots_adjust(top=0.92)
# # # # # # g.fig.suptitle('Comparison of mAP for Different Classes and Algorithms', fontsize=16, fontweight='bold')

# # # # # # plt.show()




# # # # # # # import plotly.express as px
# # # # # # # import pandas as pd

# # # # # # # # Training samples
# # # # # # # training_samples = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

# # # # # # # # Data for Uncertainty Sampling (Algorithm 1)
# # # # # # # mAP_uncertainty_sampling = {
# # # # # # #     'BLUE_LID': [81.5, 83.9, 82.1, 86.8, 89.3, 83.1],
# # # # # # #     'BLUE_TRAILER': [82.0, 67.5, 68.3, 82.5, 88.7, 87.6, 91.3, 93.0, 92.2, 92.8, 93.0, 92.8, 94.6, 95.1, 94.8, 95.2],
# # # # # # #     'CABIN': [74.3, 72.3, 73.0, 86.0, 91.2, 90.0, 93.2, 94.7, 94.0, 94.5, 94.7, 94.5, 96.0, 96.4, 96.0, 96.5],
# # # # # # #     'FRONT_CHASSIS': [56.5, 62.0, 63.0, 77.0, 83.0, 82.0, 86.0, 88.0, 87.5, 88.0, 88.5, 88.0, 89.5, 90.0, 89.5, 90.5],
# # # # # # #     'REAR_CHASSIS': [54.1, 70.5, 71.0, 85.0, 90.0, 89.0, 92.5, 94.0, 93.0, 93.5, 94.0, 93.5, 95.0, 95.5, 95.0, 95.5],
# # # # # # #     'YELLOW_LID': [77.8, 68.0, 69.0, 83.0, 88.0, 87.0, 90.5, 92.0, 91.5, 92.0, 92.5, 92.0, 93.5, 94.0, 93.5, 94.0],
# # # # # # #     'YELLOW_TRAILER': [87.5, 66.0, 67.0, 81.0, 86.0, 85.0, 88.5, 90.0, 89.5, 90.0, 90.5, 90.0, 91.5, 92.0, 91.5, 92.0],
# # # # # # #     'GLUE_GUN': [29.8, 63.0, 64.0, 78.0, 84.0, 83.0, 87.0, 89.0, 88.5, 89.0, 89.5, 89.0, 90.5, 91.0, 90.5, 91.0]
# # # # # # # }

# # # # # # # # Data for Random Sampling (Algorithm 2)
# # # # # # # mAP_random_sampling = {
# # # # # # #     'BLUE_LID': [62.4, 72.7, 79.6, 81.4, 84.5, 87.8, 85.3, 86.7, 88.9, 91.7, 91.6, 92.3, 93.6, 93.4, 93.6, 94.3],
# # # # # # #     'BLUE_TRAILER': [60.0, 70.0, 77.0, 79.0, 82.0, 85.0, 83.0, 84.5, 86.0, 88.5, 88.0, 89.0, 90.5, 90.0, 90.5, 91.0],
# # # # # # #     'CABIN': [65.0, 75.0, 80.0, 83.0, 86.0, 89.0, 87.0, 88.5, 90.0, 92.0, 91.5, 92.5, 94.0, 93.5, 94.0, 94.5],
# # # # # # #     'FRONT_CHASSIS': [55.0, 65.0, 72.0, 74.0, 77.0, 80.0, 78.0, 79.5, 81.0, 83.0, 82.5, 83.5, 85.0, 84.5, 85.0, 85.5],
# # # # # # #     'REAR_CHASSIS': [63.0, 73.0, 78.0, 81.0, 84.0, 87.0, 85.0, 86.5, 88.0, 90.0, 89.5, 90.5, 92.0, 91.5, 92.0, 92.5],
# # # # # # #     'YELLOW_LID': [61.0, 71.0, 76.0, 79.0, 82.0, 85.0, 83.0, 84.5, 86.0, 88.0, 87.5, 88.5, 90.0, 89.5, 90.0, 90.5],
# # # # # # #     'YELLOW_TRAILER': [59.0, 69.0, 74.0, 77.0, 80.0, 83.0, 81.0, 82.5, 84.0, 86.0, 85.5, 86.5, 88.0, 87.5, 88.0, 88.5],
# # # # # # #     'GLUE_GUN': [56.0, 66.0, 71.0, 74.0, 77.0, 80.0, 78.0, 79.5, 81.0, 83.0, 82.5, 83.5, 85.0, 84.5, 85.0, 85.5]
# # # # # # # }

# # # # # # # # Combine data into a DataFrame for plotting
# # # # # # # data = []
# # # # # # # for cls in mAP_uncertainty_sampling.keys():
# # # # # # #     for i, samples in enumerate(training_samples):
# # # # # # #         data.append((samples, cls, mAP_uncertainty_sampling[cls][i], 'Uncertainty Sampling'))
# # # # # # #         data.append((samples, cls, mAP_random_sampling[cls][i], 'Random Sampling'))

# # # # # # # df = pd.DataFrame(data, columns=['Training Samples', 'Class', 'mAP', 'Algorithm'])

# # # # # # # # Define a color palette
# # # # # # # color_palette = {
# # # # # # #     'Uncertainty Sampling': '#1f77b4',  # Blue
# # # # # # #     'Random Sampling': '#ff7f0e'       # Orange
# # # # # # # }

# # # # # # # # Interactive line plot
# # # # # # # fig = px.line(df, x='Training Samples', y='mAP', color='Algorithm', facet_col='Class', 
# # # # # # #               title='Comparison of mAP for Different Classes and Algorithms',
# # # # # # #               labels={'mAP': 'mAP (%)', 'Training Samples': 'Training Samples'}, 
# # # # # # #               markers=True, height=900, color_discrete_map=color_palette)

# # # # # # # # Update layout for better aesthetics
# # # # # # # fig.update_layout(
# # # # # # #     title_font_size=20,
# # # # # # #     title_x=0.5,
# # # # # # #     legend_title_text='Algorithm',
# # # # # # #     legend_title_font_size=14,
# # # # # # #     legend_font_size=12,
# # # # # # #     xaxis_title_font_size=14,
# # # # # # #     yaxis_title_font_size=14,
# # # # # # #     xaxis_tickfont_size=12,
# # # # # # #     yaxis_tickfont_size=12,
# # # # # # #     font=dict(family="Verdana, sans-serif"),
# # # # # # #     # plot_bgcolor='white',  # Background color of the plotting area
# # # # # # #     # paper_bgcolor='#f2f2f2'  # Background color of the entire chart
# # # # # # # )

# # # # # # # # Update axes and traces for better visual appeal
# # # # # # # fig.update_xaxes(tickangle=45, title_font=dict(size=14))
# # # # # # # fig.update_yaxes(title_font=dict(size=14))
# # # # # # # fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>', font=dict(size=12)))

# # # # # # # # Save the figure as an interactive HTML file
# # # # # # # fig.write_html('interactive_comparison_chart.html')

# # # # # # # # Show the figure
# # # # # # # fig.show()



# # # # import matplotlib.pyplot as plt

# # # # # Data for each class
# # # # data = {
# # # #     'BlueLid': {
# # # #         'RS': [86.0, 84.4, 93.5, 94.1, 94.9, 94.9, 94.0, 95.2, 99.0],
# # # #         'US-Avg': [81.5, 83.9, 82.1, 86.8, 89.3, 83.1, 96.2, 98.9, 98.9],
# # # #         'US-Max': [86.7, 92.0, 93.8, 93.3, 97.6, 95.8, 97.0, 97.6, 99.4],
# # # #         'US-Sum': [77.5, 90.8, 90.4, 95.0, 92.3, 90.7, 97.2, 97.1, 99.3]
# # # #     },
# # # #     'BlueTrailer': {
# # # #         'RS': [81.3, 90.3, 92.7, 95.3, 96.7, 97.3, 98.6, 98.4, 99.0],
# # # #         'US-Avg': [82.0, 83.1, 85.7, 90.8, 96.1, 95.7, 98.4, 98.3, 99.0],
# # # #         'US-Max': [88.0, 94.2, 96.8, 96.5, 98.6, 97.9, 96.3, 98.8, 99.1],
# # # #         'US-Sum': [81.7, 83.9, 92.7, 93.8, 96.7, 95.5, 93.4, 98.5, 99.3]
# # # #     },
# # # #     'Cabin': {
# # # #         'RS': [72.2, 89.1, 82.2, 89.3, 92.9, 91.5, 88.5, 93.0, 93.8],
# # # #         'US-Avg': [74.3, 76.5, 83.0, 89.6, 91.4, 90.3, 89.1, 93.2, 93.2],
# # # #         'US-Max': [85.2, 87.4, 88.6, 90.6, 89.7, 90.9, 92.0, 92.1, 93.8],
# # # #         'US-Sum': [72.0, 69.6, 84.8, 88.4, 87.4, 92.5, 91.0, 87.4, 94.7]
# # # #     },
# # # #     'FrontChassis': {
# # # #         'RS': [40.6, 71.4, 81.9, 83.6, 84.7, 86.8, 87.1, 90.4, 92.5],
# # # #         'US-Avg': [56.5, 55.9, 57.2, 75.2, 86.8, 80.3, 87.6, 89.7, 94.2],
# # # #         'US-Max': [48.1, 59.2, 79.4, 82.8, 92.9, 89.4, 91.7, 88.2, 91.6],
# # # #         'US-Sum': [18.9, 34.8, 70.5, 68.6, 83.6, 76.7, 82.2, 90.3, 92.8]
# # # #     },
# # # #     'RearChassis': {
# # # #         'RS': [44.4, 67.0, 74.8, 78.6, 83.1, 85.2, 83.5, 83.2, 92.3],
# # # #         'US-Avg': [54.1, 53.8, 50.5, 76.6, 84.6, 81.9, 87.1, 90.3, 93.0],
# # # #         'US-Max': [41.3, 58.1, 71.8, 79.5, 89.6, 83.5, 83.0, 84.2, 91.1],
# # # #         'US-Sum': [25.9, 48.9, 71.7, 73.5, 79.0, 72.8, 76.3, 80.3, 93.0]
# # # #     },
# # # #     'YellowLid': {
# # # #         'RS': [81.8, 85.0, 90.3, 92.9, 94.2, 95.3, 94.2, 95.1, 97.7],
# # # #         'US-Avg': [77.8, 79.9, 77.8, 87.5, 92.8, 95.9, 95.4, 95.4, 96.9],
# # # #         'US-Max': [85.6, 89.3, 93.1, 92.5, 95.8, 97.4, 96.2, 97.3, 98.6],
# # # #         'US-Sum': [76.6, 85.5, 82.2, 87.8, 94.1, 92.9, 93.5, 94.1, 97.0]
# # # #     },
# # # #     'YellowTrailer': {
# # # #         'RS': [92.8, 93.8, 96.7, 97.9, 98.1, 98.2, 98.5, 98.6, 90.3],
# # # #         'US-Avg': [87.5, 89.8, 86.6, 96.6, 98.2, 98.2, 98.5, 97.6, 96.9],
# # # #         'US-Max': [93.5, 94.8, 97.3, 97.5, 98.6, 98.7, 98.9, 97.8, 99.1],
# # # #         'US-Sum': [89.3, 91.1, 95.4, 96.8, 98.1, 97.9, 98.4, 97.9, 97.0]
# # # #     },
# # # #     'GlueGun': {
# # # #         'RS': [0.0, 0.04, 24.4, 19.9, 31.1, 52.8, 37.9, 39.4, 89.8],
# # # #         'US-Avg': [29.8, 36.1, 37.3, 73.7, 81.8, 87.2, 89.4, 93.2, 94.7],
# # # #         'US-Max': [45.5, 52.7, 64.3, 80.5, 84.8, 89.7, 92.0, 95.6, 95.5],
# # # #         'US-Sum': [9.03, 13.6, 35.7, 50.0, 67.0, 78.4, 90.2, 91.6, 94.7]
# # # #     }
# # # # }

# # # # sample_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 800]
# # # # algorithms = ['RS', 'US-Avg', 'US-Max', 'US-Sum']

# # # # # Plotting 8 line charts in a 2x4 grid
# # # # fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# # # # for i, (class_name, class_data) in enumerate(data.items()):
# # # #     row = i // 4
# # # #     col = i % 4
# # # #     ax = axes[row, col]
# # # #     for algo in algorithms:
# # # #         ax.plot(sample_sizes, class_data[algo], label=algo)
# # # #     ax.set_title(class_name)
# # # #     ax.set_xlabel('Training Samples')
# # # #     ax.set_ylabel('mAP')
# # # #     ax.legend()
# # # #     ax.grid(True)

# # # # plt.tight_layout()
# # # # plt.show()

# # # import seaborn as sns
# # # import matplotlib.pyplot as plt

# # # # Sample data (replace this with your actual data)
# # # data = {
# # #     'BlueLid': {
# # #         'RS': [86.0, 84.4, 93.5, 94.1, 94.9, 94.9, 94.0, 95.2, 99.0],
# # #         'US-Avg': [81.5, 83.9, 82.1, 86.8, 89.3, 83.1, 96.2, 98.9, 98.9],
# # #         'US-Max': [86.7, 92.0, 93.8, 93.3, 97.6, 95.8, 97.0, 97.6, 99.4],
# # #         'US-Sum': [77.5, 90.8, 90.4, 95.0, 92.3, 90.7, 97.2, 97.1, 99.3]
# # #     },
# # #     'BlueTrailer': {
# # #         'RS': [81.3, 90.3, 92.7, 95.3, 96.7, 97.3, 98.6, 98.4, 99.0],
# # #         'US-Avg': [82.0, 83.1, 85.7, 90.8, 96.1, 95.7, 98.4, 98.3, 99.0],
# # #         'US-Max': [88.0, 94.2, 96.8, 96.5, 98.6, 97.9, 96.3, 98.8, 99.1],
# # #         'US-Sum': [81.7, 83.9, 92.7, 93.8, 96.7, 95.5, 93.4, 98.5, 99.3]
# # #     },
# # #     'Cabin': {
# # #         'RS': [72.2, 89.1, 82.2, 89.3, 92.9, 91.5, 88.5, 93.0, 93.8],
# # #         'US-Avg': [74.3, 76.5, 83.0, 89.6, 91.4, 90.3, 89.1, 93.2, 93.2],
# # #         'US-Max': [85.2, 87.4, 88.6, 90.6, 89.7, 90.9, 92.0, 92.1, 93.8],
# # #         'US-Sum': [72.0, 69.6, 84.8, 88.4, 87.4, 92.5, 91.0, 87.4, 94.7]
# # #     },
# # #     'FrontChassis': {
# # #         'RS': [40.6, 71.4, 81.9, 83.6, 84.7, 86.8, 87.1, 90.4, 92.5],
# # #         'US-Avg': [56.5, 55.9, 57.2, 75.2, 86.8, 80.3, 87.6, 89.7, 94.2],
# # #         'US-Max': [48.1, 59.2, 79.4, 82.8, 92.9, 89.4, 91.7, 88.2, 91.6],
# # #         'US-Sum': [18.9, 34.8, 70.5, 68.6, 83.6, 76.7, 82.2, 90.3, 92.8]
# # #     },
# # #     'RearChassis': {
# # #         'RS': [44.4, 67.0, 74.8, 78.6, 83.1, 85.2, 83.5, 83.2, 92.3],
# # #         'US-Avg': [54.1, 53.8, 50.5, 76.6, 84.6, 81.9, 87.1, 90.3, 93.0],
# # #         'US-Max': [41.3, 58.1, 71.8, 79.5, 89.6, 83.5, 83.0, 84.2, 91.1],
# # #         'US-Sum': [25.9, 48.9, 71.7, 73.5, 79.0, 72.8, 76.3, 80.3, 93.0]
# # #     },
# # #     'YellowLid': {
# # #         'RS': [81.8, 85.0, 90.3, 92.9, 94.2, 95.3, 94.2, 95.1, 97.7],
# # #         'US-Avg': [77.8, 79.9, 77.8, 87.5, 92.8, 95.9, 95.4, 95.4, 96.9],
# # #         'US-Max': [85.6, 89.3, 93.1, 92.5, 95.8, 97.4, 96.2, 97.3, 98.6],
# # #         'US-Sum': [76.6, 85.5, 82.2, 87.8, 94.1, 92.9, 93.5, 94.1, 97.0]
# # #     },
# # #     'YellowTrailer': {
# # #         'RS': [92.8, 93.8, 96.7, 97.9, 98.1, 98.2, 98.5, 98.6, 90.3],
# # #         'US-Avg': [87.5, 89.8, 86.6, 96.6, 98.2, 98.2, 98.5, 97.6, 96.9],
# # #         'US-Max': [93.5, 94.8, 97.3, 97.5, 98.6, 98.7, 98.9, 97.8, 99.1],
# # #         'US-Sum': [89.3, 91.1, 95.4, 96.8, 98.1, 97.9, 98.4, 97.9, 97.0]
# # #     },
# # #     'GlueGun': {
# # #         'RS': [0.0, 0.04, 24.4, 19.9, 31.1, 52.8, 37.9, 39.4, 89.8],
# # #         'US-Avg': [29.8, 36.1, 37.3, 73.7, 81.8, 87.2, 89.4, 93.2, 94.7],
# # #         'US-Max': [45.5, 52.7, 64.3, 80.5, 84.8, 89.7, 92.0, 95.6, 95.5],
# # #         'US-Sum': [9.03, 13.6, 35.7, 50.0, 67.0, 78.4, 90.2, 91.6, 94.7]
# # #     }
# # # }

# # # # Seaborn style settings for better aesthetics
# # # sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# # # # Create the plots
# # # fig, axes = plt.subplots(2, 4, figsize=(20, 12))
# # # fig.subplots_adjust(hspace=0.4, wspace=0.3)

# # # # Titles for each subplot
# # # titles = ['BlueLid', 'BlueTrailer', 'Cabin', 'FrontChassis', 
# # #           'RearChassis', 'YellowLid', 'YellowTrailer', 'GlueGun']

# # # # Iterate over classes and plot
# # # for i, (ax, title) in enumerate(zip(axes.flatten(), titles)):
# # #     for algo, results in data[title].items():
# # #         sns.lineplot(x=range(1, len(results) + 1), y=results, marker='o', ax=ax, label=algo)
# # #     ax.set_title(title)
# # #     ax.set_xlabel("Training Samples")
# # #     ax.set_ylabel("mAP")
# # #     ax.legend(title="Algorithm", loc='lower right')

# # # plt.tight_layout()
# # # plt.show()


# # import seaborn as sns
# # import matplotlib.pyplot as plt

# # # Sample data (replace this with your actual data)
# # data = {
# #     'BlueLid': {
# #         'RS': [86.0, 84.4, 93.5, 94.1, 94.9, 94.9, 94.0, 95.2, 99.0],
# #         'US-Avg': [81.5, 83.9, 82.1, 86.8, 89.3, 83.1, 96.2, 98.9, 98.9],
# #         'US-Max': [86.7, 92.0, 93.8, 93.3, 97.6, 95.8, 97.0, 97.6, 99.4],
# #         'US-Sum': [77.5, 90.8, 90.4, 95.0, 92.3, 90.7, 97.2, 97.1, 99.3]
# #     },
# #     'BlueTrailer': {
# #         'RS': [81.3, 90.3, 92.7, 95.3, 96.7, 97.3, 98.6, 98.4, 99.0],
# #         'US-Avg': [82.0, 83.1, 85.7, 90.8, 96.1, 95.7, 98.4, 98.3, 99.0],
# #         'US-Max': [88.0, 94.2, 96.8, 96.5, 98.6, 97.9, 96.3, 98.8, 99.1],
# #         'US-Sum': [81.7, 83.9, 92.7, 93.8, 96.7, 95.5, 93.4, 98.5, 99.3]
# #     },
# #     'Cabin': {
# #         'RS': [72.2, 89.1, 82.2, 89.3, 92.9, 91.5, 88.5, 93.0, 93.8],
# #         'US-Avg': [74.3, 76.5, 83.0, 89.6, 91.4, 90.3, 89.1, 93.2, 93.2],
# #         'US-Max': [85.2, 87.4, 88.6, 90.6, 89.7, 90.9, 92.0, 92.1, 93.8],
# #         'US-Sum': [72.0, 69.6, 84.8, 88.4, 87.4, 92.5, 91.0, 87.4, 94.7]
# #     },
# #     'FrontChassis': {
# #         'RS': [40.6, 71.4, 81.9, 83.6, 84.7, 86.8, 87.1, 90.4, 92.5],
# #         'US-Avg': [56.5, 55.9, 57.2, 75.2, 86.8, 80.3, 87.6, 89.7, 94.2],
# #         'US-Max': [48.1, 59.2, 79.4, 82.8, 92.9, 89.4, 91.7, 88.2, 91.6],
# #         'US-Sum': [18.9, 34.8, 70.5, 68.6, 83.6, 76.7, 82.2, 90.3, 92.8]
# #     },
# #     'RearChassis': {
# #         'RS': [44.4, 67.0, 74.8, 78.6, 83.1, 85.2, 83.5, 83.2, 92.3],
# #         'US-Avg': [54.1, 53.8, 50.5, 76.6, 84.6, 81.9, 87.1, 90.3, 93.0],
# #         'US-Max': [41.3, 58.1, 71.8, 79.5, 89.6, 83.5, 83.0, 84.2, 91.1],
# #         'US-Sum': [25.9, 48.9, 71.7, 73.5, 79.0, 72.8, 76.3, 80.3, 93.0]
# #     },
# #     'YellowLid': {
# #         'RS': [81.8, 85.0, 90.3, 92.9, 94.2, 95.3, 94.2, 95.1, 97.7],
# #         'US-Avg': [77.8, 79.9, 77.8, 87.5, 92.8, 95.9, 95.4, 95.4, 96.9],
# #         'US-Max': [85.6, 89.3, 93.1, 92.5, 95.8, 97.4, 96.2, 97.3, 98.6],
# #         'US-Sum': [76.6, 85.5, 82.2, 87.8, 94.1, 92.9, 93.5, 94.1, 97.0]
# #     },
# #     'YellowTrailer': {
# #         'RS': [92.8, 93.8, 96.7, 97.9, 98.1, 98.2, 98.5, 98.6, 90.3],
# #         'US-Avg': [87.5, 89.8, 86.6, 96.6, 98.2, 98.2, 98.5, 97.6, 96.9],
# #         'US-Max': [93.5, 94.8, 97.3, 97.5, 98.6, 98.7, 98.9, 97.8, 99.1],
# #         'US-Sum': [89.3, 91.1, 95.4, 96.8, 98.1, 97.9, 98.4, 97.9, 97.0]
# #     },
# #     'GlueGun': {
# #         'RS': [0.0, 0.04, 24.4, 19.9, 31.1, 52.8, 37.9, 39.4, 89.8],
# #         'US-Avg': [29.8, 36.1, 37.3, 73.7, 81.8, 87.2, 89.4, 93.2, 94.7],
# #         'US-Max': [45.5, 52.7, 64.3, 80.5, 84.8, 89.7, 92.0, 95.6, 95.5],
# #         'US-Sum': [9.03, 13.6, 35.7, 50.0, 67.0, 78.4, 90.2, 91.6, 94.7]
# #     }
# # }

# # # Seaborn style settings for better aesthetics
# # sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# # # Define training samples
# # training_samples = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

# # # Define colors for algorithms
# # colors = {
# #     'US-Avg': '#006400',  # Dark Green
# #     'US-Max': '#00008b',  # Dark Blue
# #     'US-Sum': '#d2691e',  # Chocolate
# #     'RS': '#ff0000'       # Red
# # }

# # # Create the plots
# # fig, axes = plt.subplots(2, 4, figsize=(20, 12))
# # fig.subplots_adjust(hspace=0.4, wspace=0.3)

# # # Titles for each subplot
# # titles = ['BlueLid', 'BlueTrailer', 'Cabin', 'FrontChassis', 
# #           'RearChassis', 'YellowLid', 'YellowTrailer', 'GlueGun']

# # # Plot each class
# # for i, (ax, title) in enumerate(zip(axes.flatten(), titles)):
# #     for algo in ['US-Avg', 'US-Max', 'US-Sum', 'RS']:  # Order of algorithms
# #         if algo in data[title]:
# #             sns.lineplot(
# #                 x=training_samples[:len(data[title][algo])], 
# #                 y=data[title][algo], 
# #                 label=algo, 
# #                 color=colors[algo], 
# #                 ax=ax, 
# #                 marker='o'
# #             )
    
# #     ax.set_title(f'**{title}**', fontsize=14, fontweight='bold')
# #     ax.set_xlabel('Training Samples', fontsize=12)
# #     ax.set_ylabel('Mean Average Precision (mAP)', fontsize=12)
# #     ax.legend(title='Algorithm', fontsize=10)
# #     ax.grid(True)

# # # Save the plot
# # plt.savefig('comparison_charts.svg', format='svg')

# # plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt

# # Sample data (replace this with your actual data)
# data = {
#     'BlueLid': {
#         'RS': [86.0, 84.4, 93.5, 94.1, 94.9, 94.9, 94.0, 95.2, 99.0],
#         'US-Avg': [81.5, 83.9, 82.1, 86.8, 89.3, 83.1, 96.2, 98.9, 98.9],
#         'US-Max': [86.7, 92.0, 93.8, 93.3, 97.6, 95.8, 97.0, 97.6, 99.4],
#         'US-Sum': [77.5, 90.8, 90.4, 95.0, 92.3, 90.7, 97.2, 97.1, 99.3]
#     },
#     'BlueTrailer': {
#         'RS': [81.3, 90.3, 92.7, 95.3, 96.7, 97.3, 98.6, 98.4, 99.0],
#         'US-Avg': [82.0, 83.1, 85.7, 90.8, 96.1, 95.7, 98.4, 98.3, 99.0],
#         'US-Max': [88.0, 94.2, 96.8, 96.5, 98.6, 97.9, 96.3, 98.8, 99.1],
#         'US-Sum': [81.7, 83.9, 92.7, 93.8, 96.7, 95.5, 93.4, 98.5, 99.3]
#     },
#     'Cabin': {
#         'RS': [72.2, 89.1, 82.2, 89.3, 92.9, 91.5, 88.5, 93.0, 93.8],
#         'US-Avg': [74.3, 76.5, 83.0, 89.6, 91.4, 90.3, 89.1, 93.2, 93.2],
#         'US-Max': [85.2, 87.4, 88.6, 90.6, 89.7, 90.9, 92.0, 92.1, 93.8],
#         'US-Sum': [72.0, 69.6, 84.8, 88.4, 87.4, 92.5, 91.0, 87.4, 94.7]
#     },
#     'FrontChassis': {
#         'RS': [40.6, 71.4, 81.9, 83.6, 84.7, 86.8, 87.1, 90.4, 92.5],
#         'US-Avg': [56.5, 55.9, 57.2, 75.2, 86.8, 80.3, 87.6, 89.7, 94.2],
#         'US-Max': [48.1, 59.2, 79.4, 82.8, 92.9, 89.4, 91.7, 88.2, 91.6],
#         'US-Sum': [18.9, 34.8, 70.5, 68.6, 83.6, 76.7, 82.2, 90.3, 92.8]
#     },
#     'RearChassis': {
#         'RS': [44.4, 67.0, 74.8, 78.6, 83.1, 85.2, 83.5, 83.2, 92.3],
#         'US-Avg': [54.1, 53.8, 50.5, 76.6, 84.6, 81.9, 87.1, 90.3, 93.0],
#         'US-Max': [41.3, 58.1, 71.8, 79.5, 89.6, 83.5, 83.0, 84.2, 91.1],
#         'US-Sum': [25.9, 48.9, 71.7, 73.5, 79.0, 72.8, 76.3, 80.3, 93.0]
#     },
#     'YellowLid': {
#         'RS': [81.8, 85.0, 90.3, 92.9, 94.2, 95.3, 94.2, 95.1, 97.7],
#         'US-Avg': [77.8, 79.9, 77.8, 87.5, 92.8, 95.9, 95.4, 95.4, 96.9],
#         'US-Max': [85.6, 89.3, 93.1, 92.5, 95.8, 97.4, 96.2, 97.3, 98.6],
#         'US-Sum': [76.6, 85.5, 82.2, 87.8, 94.1, 92.9, 93.5, 94.1, 97.0]
#     },
#     'YellowTrailer': {
#         'RS': [92.8, 93.8, 96.7, 97.9, 98.1, 98.2, 98.5, 98.6, 90.3],
#         'US-Avg': [87.5, 89.8, 86.6, 96.6, 98.2, 98.2, 98.5, 97.6, 96.9],
#         'US-Max': [93.5, 94.8, 97.3, 97.5, 98.6, 98.7, 98.9, 97.8, 99.1],
#         'US-Sum': [89.3, 91.1, 95.4, 96.8, 98.1, 97.9, 98.4, 97.9, 97.0]
#     },
#     'GlueGun': {
#         'RS': [0.0, 0.04, 24.4, 19.9, 31.1, 52.8, 37.9, 39.4, 89.8],
#         'US-Avg': [29.8, 36.1, 37.3, 73.7, 81.8, 87.2, 89.4, 93.2, 94.7],
#         'US-Max': [45.5, 52.7, 64.3, 80.5, 84.8, 89.7, 92.0, 95.6, 95.5],
#         'US-Sum': [9.03, 13.6, 35.7, 50.0, 67.0, 78.4, 90.2, 91.6, 94.7]
#     }
# }

# # Seaborn style settings for better aesthetics
# sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# # Define training samples
# training_samples = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]

# # Define colors for algorithms
# colors = {
#     'US-Avg': '#006400',  # Dark Green
#     'US-Max': '#00008b',  # Dark Blue
#     'US-Sum': '#d2691e',  # Chocolate
#     'RS': '#ff0000'       # Red
# }

# # Create the plots
# fig, axes = plt.subplots(2, 4, figsize=(20, 12))
# fig.subplots_adjust(hspace=0.4, wspace=0.3)

# # Titles for each subplot
# titles = ['BlueLid', 'BlueTrailer', 'Cabin', 'FrontChassis', 'RearChassis', 'YellowLid', 'YellowTrailer', 'GlueGun']

# # Plot each class
# for i, (ax, title) in enumerate(zip(axes.flatten(), titles)):
#     for algo in ['US-Avg', 'US-Max', 'US-Sum', 'RS']:  # Order of algorithms
#         if algo in data[title]:
#             sns.lineplot(
#                 x=training_samples[:len(data[title][algo])],
#                 y=data[title][algo],
#                 label=algo,
#                 color=colors[algo],
#                 ax=ax,
#                 marker='o'
#             )
#     ax.set_title(f'{title}', fontsize=14, fontweight='bold')
#     ax.set_xlabel('Training Samples', fontsize=12)
#     ax.set_ylabel('Mean Average Precision (mAP)', fontsize=12)
#     ax.legend(title='Algorithm', fontsize=10)
#     ax.grid(True)

# # Save the plot as PDF
# plt.savefig('comparison_charts.pdf', format='pdf')

# plt.show()







