# author: Soumen Mohanty
# email: sm8966@nyu.edu
# version: 0.1 (2nd August 2023)

# This script analyzes and plots the volume data of a patient. 
# It takes as input a CSV file containing the volume data of a patient and outputs a PDF file containing the plots.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from statsmodels.nonparametric.smoothers_lowess import lowess
from tabulate import tabulate

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze and plot volume data")
    parser.add_argument("-i", "--input", action='append', dest="input_files", help="Input file with patient's volume data (.stat format)")
    parser.add_argument("-a", "--age", type=float, action='append', dest="ages", help="Age of the patient (in years)")
    parser.add_argument("-g", "--gender", choices=['M', 'F'], action='append', dest="genders", help="Gender of the patient (M or F)")
    parser.add_argument("-o", "--output", dest="output_file", default="hcpa_analysis.pdf", help="Output PDF file name")
    return parser.parse_args()

def load_data(file_path):
    """Loads data from the CSV file and processes the 'age' column."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

    df['age'] = df['age'] / 12  # Convert age from months to years
    df['age'] = (df['age'] // 5) * 5  # Convert age to half decades

    return df

import re

def extract_features(file_name):
    feature_dict = {}
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('# Measure'):
                parts = line.split(',')
                feature_name = re.sub('[- ]', '_', parts[0].split(' ')[2])
                # Add 'x' before numerical prefixes
                feature_name = re.sub(r'\b(\d)', r'x\1', feature_name)
                feature_value = float(parts[1])
                feature_dict[feature_name] = feature_value
    return feature_dict

def plot_point(age, value, color='blue', marker='o', linestyle='--', markersize=4):
    plt.plot(age, value, color+marker+linestyle, markersize=markersize)

def create_distribution_plots(df, axs):
    """Creates age and sex distribution plots."""
    # Sex distribution pie chart
    sex_counts = df['sex'].value_counts()
    axs[0].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', startangle=90, colors=['pink', 'lightblue'])
    axs[0].set_title('Sex Distribution')
    
    # Age distribution bar chart
    age_counts = df['age'].value_counts().sort_index()
    axs[1].bar(age_counts.index, age_counts.values, width=8)
    axs[1].set_title('Age Distribution')
    axs[1].set_xlabel('Age (in decades)')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

def create_percentile_plots(df, feature, percentiles, color_dict, ax, gender, patient_data_list=None, smoothing_frac=0.3):
    # Filter data by gender
    df_gender = df[df['sex'] == gender]

    for percentile in percentiles:
        percentile_values = df_gender.groupby('age')[feature].apply(lambda x: np.percentile(x, percentile))
        linewidth = 2.5 if percentile == 50 else 1.0
        # Convert the index to float
        x_values = percentile_values.index.to_numpy(dtype=float)
        # Apply LOESS smoothing
        smoothed_values = lowess(percentile_values.values, x_values, frac=smoothing_frac)

        ax.plot(smoothed_values[:, 0], smoothed_values[:, 1], linewidth=linewidth, color=color_dict[percentile], alpha=0.7)
        ax.text(smoothed_values[-1, 0], smoothed_values[-1, 1], f'{percentile}', va='center', color=color_dict[percentile])
    
    # Add patient data if available and gender matches
    if patient_data_list is not None:
        patient_ages = []
        patient_values = []
        for patient_data in patient_data_list:
            if feature in patient_data and patient_data['sex'] == gender:
                ax.scatter(patient_data['age'], patient_data[feature], color='green', marker='x', s=40)
                patient_ages.append(patient_data['age'])
                patient_values.append(patient_data[feature])

        # x_zoom_min = 40  # minimum x value for zoom region
        # x_zoom_max = 50  # maximum x value for zoom region
        # y_zoom_min = 1000  # minimum y value for zoom region (you'll want to set this to something appropriate for your data)
        # y_zoom_max = 2000  # maximum y value for zoom region (you'll want to set this to something appropriate for your data)

        # plt.xlim(x_zoom_min, x_zoom_max)
        # plt.ylim(y_zoom_min, y_zoom_max)

        if len(patient_ages) > 1:
            # Calculate the line of best fit parameters
            m, b = np.polyfit(patient_ages, patient_values, 1)

            # Determine the range of x values
            x_min = min(patient_ages) - 10
            x_max = max(patient_ages) + 10

            # Calculate the corresponding y values
            x_values = np.linspace(x_min, x_max, 100)
            y_values = m * x_values + b

            # Plot the line
            ax.plot(x_values, y_values, color='lightgreen', linestyle='--')

def main():
    # Parse arguments
    args = parse_args()

    file_path = 'hcpa.csv'
    df = load_data(file_path)

    if df is None:
        return

    # features = df.columns[3:]

    features = [
        'Left_Cerebral_Cortex',
        'Right_Cerebral_Cortex',
        'Left_Cerebral_White_Matter',
        'Right_Cerebral_White_Matter',
        'Left_Cerebellum_Cortex',
        'Right_Cerebellum_Cortex',
        'Left_Cerebellum_White_Matter',
        'Right_Cerebellum_White_Matter',
        'Left_Hippocampus',
        'Right_Hippocampus',
        'Left_Amygdala',
        'Right_Amygdala',
        'Left_VentralDC',
        'Right_VentralDC',
        'Left_Putamen',
        'Right_Putamen',
        'Left_Accumbens_area',
        'Right_Accumbens_area',
        'Brain_Stem',
        'Right_Pallidum',
        'Left_Caudate',
        'Right_Thalamus',
        'Left_Pallidum',
        'Right_Caudate',
        'Left_Thalamus',
        'Right_Thalamus',
        'Left_Lateral_Ventricle',
        'Right_Lateral_Ventricle',
        'Left_Inf_Lat_Vent',
        'Right_Inf_Lat_Vent',
        'x3rd_Ventricle', 
        'x4th_Ventricle', 
        'x5th_Ventricle',
        'CSF'
    ]

    percentiles = [5, 25, 50, 75, 95]
    color_dict = {5: 'orange', 25: 'darkorange', 50: 'red', 75: 'darkorange', 95: 'orange'}

    # Extract features from patient files
    patient_data_list = []
    if args.input_files:
        for input_file, age, gender in zip(args.input_files, args.ages, args.genders):
            patient_data = extract_features(input_file)
            patient_data['age'] = age
            patient_data['sex'] = gender
            patient_data_list.append(patient_data)

    timepoint_data = pd.DataFrame.from_dict([patient_data_list[0], patient_data_list[1]])
    # Transpose the DataFrame
    timepoint_data = timepoint_data.transpose()

    # Convert the two columns to numeric, excluding any non-numeric rows, and store in new columns
    timepoint_data['Timepoint 1'] = pd.to_numeric(timepoint_data[0], errors='coerce')
    timepoint_data['Timepoint 2'] = pd.to_numeric(timepoint_data[1], errors='coerce')

    # Drop the original unnamed columns
    timepoint_data = timepoint_data[['Timepoint 1', 'Timepoint 2']]

    # Calculate the percentage change, fixing the parentheses
    timepoint_data['Percentage Change'] = ((timepoint_data['Timepoint 2'] - timepoint_data['Timepoint 1']) / timepoint_data['Timepoint 1']) * 100

    # save the dataframe to a xlsx file
    timepoint_data.to_excel(f"{args.output_file}.pdf")


    with PdfPages(f"{args.output_file}.pdf") as pdf:
        # First page with age and sex distribution
        fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
        create_distribution_plots(df, axs)
        pdf.savefig(fig)
        plt.close(fig)

        # Other pages with percentile plots
        for gender in ['M', 'F']:
            for i in range(0, len(features), 4):
                fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                axs = axs.flatten()

                for j in range(4):
                    if i + j < len(features):
                        create_percentile_plots(df, features[i + j], percentiles, color_dict, axs[j], gender, patient_data_list)
                        axs[j].set_title(f'{features[i + j]} for {gender}')
                        axs[j].set_xlabel('Age (years)')
                        axs[j].set_ylabel('Volume (mm^3))')
                        axs[j].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.3)

                pdf.savefig(fig)
                plt.close(fig)

if __name__ == "__main__":
    main()

# python3 analysis.py -i tp001/samseg.stats -a 45 -g M -i tp002/samseg.stats -a 47 -g M
