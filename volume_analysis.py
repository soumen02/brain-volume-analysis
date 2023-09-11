# author: Soumen Mohanty
# email: sm8966@nyu.edu
# version: 0.1 (2nd August 2023)

# This script analyzes and plots the volume data of a patient. 
# It takes as input a CSV file containing the volume data of a patient and outputs a PDF file containing the plots.

# Usage 
# python3 volume_analysis.py -i ~/Downloads/tp1_samseg.stats -a 61 -g M -i ~/Downloads/tp2_samseg.stats -a 62 -g M -o Opth0001_dementia


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
    parser.add_argument("-o", "--output", dest="output_file", default="analysis", help="Output PDF file name")
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


    # calculate the total brain volume

    df = calculate_Total_Brain_Volume(df)

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

### Make changes to this formula to calculate the total brain volume 
# look at variable 'feature_labels' for the names of the features
def calculate_Total_Brain_Volume(df):
    df['Total_Brain_Volume'] = df['Left_Cerebral_Cortex'] + df['Right_Cerebral_Cortex'] + df['Left_Cerebral_White_Matter'] + df['Right_Cerebral_White_Matter']
    return df

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

def get_percentile_values(df_gender, feature, percentile):
    """Calculate the percentile values for the given feature and percentile."""
    return df_gender.groupby('age')[feature].apply(lambda x: np.percentile(x, percentile))

def plot_percentiles(ax, smoothed_values, percentile, color):
    """Plot the smoothed percentiles on the given axes."""
    ax.plot(smoothed_values[:, 0], smoothed_values[:, 1], linewidth=1.0, color=color, alpha=0.5, linestyle='--')
    # Add legend for the percentile within the plot itself instead of at the top of the figure 
    ax.text(smoothed_values[-1, 0], smoothed_values[-1, 1], f'{percentile}th', color=color, fontsize=8)


def plot_patient_data(ax, patient_data_list, feature, gender):
    """Plot the patient data if available and gender matches."""
    patient_ages = []
    patient_values = []
    for patient_data in patient_data_list:
        if feature in patient_data and patient_data['sex'] == gender:
            ax.scatter(patient_data['age'], patient_data[feature], color='green', marker='x', s=40)
            patient_ages.append(patient_data['age'])
            patient_values.append(patient_data[feature])
    return patient_ages, patient_values

def create_percentile_plots(df, feature, percentiles, color_dict, ax, gender, patient_data_list=None, smoothing_frac=0.35):
    # Check input arguments
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected df to be a DataFrame.")
    if feature not in df.columns:
        raise ValueError(f"Feature {feature} not found in DataFrame.")
    if not all(0 <= percentile <= 100 for percentile in percentiles):
        raise ValueError("Percentiles must be between 0 and 100.")

    # Filter data by gender
    df_gender = df[df['sex'] == gender]

    for percentile in percentiles:
        percentile_values = get_percentile_values(df_gender, feature, percentile)
        x_values = percentile_values.index.to_numpy(dtype=float)

        # Apply LOESS smoothing
        smoothed_values = lowess(percentile_values.values, x_values, frac=smoothing_frac)

        # Plot the smoothed percentile curve
        plot_percentiles(ax, smoothed_values, percentile, color_dict[percentile])

    # Plot patient data if available and gender matches
    if patient_data_list is not None:
        patient_ages, patient_values = plot_patient_data(ax, patient_data_list, feature, gender)

        # Plot line connecting patient data points
        if len(patient_ages) > 1:
            ax.plot(patient_ages, patient_values, color='green', linestyle='-', linewidth=1.5)

        # Show percentage change in feature value
        if len(patient_ages) > 1:
            percent_change = (patient_values[-1] - patient_values[0]) / patient_values[0] * 100
            ax.text(patient_ages[-1] + 1, patient_values[-1], f'{percent_change:.1f}%', va='center', color='green')
            
            ### Delete this part if you don't want to crop the x-axis to the patient data
            try: 
                # extract the ages from patient_ages which is a list in the format [age1, age2, age3, ...]
                minage = patient_ages[0]
                maxage = patient_ages[-1]

                ax.set_xlim(left=minage-5, right=maxage+10)
            except:
                pass
    
    # Get the current upper limit of the y-axis
    ymin, ymax = ax.get_ylim()

    # Set the lower limit to 0, keeping the upper limit the same
    ax.set_ylim(bottom=0, top=ymax*1.2)


def create_features_and_labels():
    # Mapping from CSV feature names to display labels
    feature_labels = {
        'Total_Brain_Volume': 'Total Brain Volume',
        'Left_Cerebral_Cortex': 'Left Cerebral Cortex',
        'Right_Cerebral_Cortex': 'Right Cerebral Cortex',
        'Left_Cerebral_White_Matter': 'Left Cerebral White Matter',
        'Right_Cerebral_White_Matter': 'Right Cerebral White Matter',
        'Left_Cerebellum_Cortex': 'Left Cerebellum Cortex',
        'Right_Cerebellum_Cortex': 'Right Cerebellum Cortex',
        'Left_Cerebellum_White_Matter': 'Left Cerebellum White Matter',
        'Right_Cerebellum_White_Matter': 'Right Cerebellum White Matter',
        'Left_Hippocampus': 'Left Hippocampus',
        'Right_Hippocampus': 'Right Hippocampus',
        'Left_Amygdala': 'Left Amygdala',
        'Right_Amygdala': 'Right Amygdala',
        'Left_VentralDC': 'Left VentralDC',
        'Right_VentralDC': 'Right VentralDC',
        'Left_Putamen': 'Left Putamen',
        'Right_Putamen': 'Right Putamen',
        'Left_Accumbens_area': 'Left Accumbens Area',
        'Right_Accumbens_area': 'Right Accumbens Area',
        'Brain_Stem': 'Brain Stem',
        'Right_Pallidum': 'Right Pallidum',
        'Left_Caudate': 'Left Caudate',
        'Right_Thalamus': 'Right Thalamus',
        'Left_Pallidum': 'Left Pallidum',
        'Right_Caudate': 'Right Caudate',
        'Left_Thalamus': 'Left Thalamus',
        'Right_Thalamus': 'Right Thalamus',
        'Left_Lateral_Ventricle': 'Left Lateral Ventricle',
        'Right_Lateral_Ventricle': 'Right Lateral Ventricle',
        'Left_Inf_Lat_Vent': 'Left Inf Lat Vent',
        'Right_Inf_Lat_Vent': 'Right Inf Lat Vent',
        'x3rd_Ventricle': '3rd Ventricle',
        'x4th_Ventricle': '4th Ventricle',
        'x5th_Ventricle': '5th Ventricle',
        'CSF': 'CSF',
    }
    return list(feature_labels.keys()), feature_labels

def main():
    # Parse arguments
    args = parse_args()

    file_path = 'baseline.csv'
    df = load_data(file_path)

    if df is None:
        return

    features, feature_labels = create_features_and_labels()

    # Mapping from CSV gender values to display labels
    gender_labels = {
        'M': 'Males',
        'F': 'Females',
    }       

    percentiles = [5, 25, 50, 75, 95]
    color_dict = {5: 'orange', 25: 'darkorange', 50: 'red', 75: 'darkorange', 95: 'orange'}

    # Extract features from patient files
    patient_data_list = []
    if args.input_files:
        for input_file, age, gender in zip(args.input_files, args.ages, args.genders):
            patient_data = extract_features(input_file)
            patient_data['age'] = age
            patient_data['sex'] = gender
            calculate_Total_Brain_Volume(patient_data)
            patient_data_list.append(patient_data)

    # Determine the number of input files (zero, one, or two)
    num_input_files = len(patient_data_list)

    if num_input_files == 0:
        # No input files: Print percentile plots for both genders

        with PdfPages(f"{args.output_file}.pdf") as pdf:
            # First page with age and sex distribution
            fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
            create_distribution_plots(df, axs)
            pdf.savefig(fig)
            plt.close(fig)

            for gender in ['M', 'F']:
                for i in range(0, len(features), 4):
                    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                    axs = axs.flatten()

                    for j in range(4):
                        if i + j < len(features):
                            create_percentile_plots(df, features[i + j], percentiles, color_dict, axs[j], gender)
                            
                            feature_name_csv = features[i + j]
                            feature_label = feature_labels.get(feature_name_csv, feature_name_csv)
                            gender_label = gender_labels.get(gender, gender)
                            axs[j].set_title(f'{feature_label} for {gender_label}')
                            axs[j].set_xlabel('Age (years)')
                            axs[j].set_ylabel('Volume (mm$^3$)')
                            axs[j].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.3)

                    pdf.savefig(fig)
                    plt.close(fig)

    elif num_input_files == 1:
        # One input file: Plot a single point on percentile plots
        gender = patient_data_list[0]['sex']

        with PdfPages(f"{args.output_file}.pdf") as pdf:
            # First page with age and sex distribution
            fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
            create_distribution_plots(df, axs)
            pdf.savefig(fig)
            plt.close(fig)

            for i in range(0, len(features), 4):
                fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                axs = axs.flatten()

                for j in range(4):
                    if i + j < len(features):
                        create_percentile_plots(df, features[i + j], percentiles, color_dict, axs[j], gender, patient_data_list)
                            
                        feature_name_csv = features[i + j]
                        feature_label = feature_labels.get(feature_name_csv, feature_name_csv)
                        gender_label = gender_labels.get(gender, gender)
                        axs[j].set_title(f'{feature_label} for {gender_label}')
                        axs[j].set_xlabel('Age (years)')
                        axs[j].set_ylabel('Volume (mm$^3$)')
                        axs[j].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.3)

                pdf.savefig(fig)
                plt.close(fig)


    elif num_input_files == 2:

        patient_data_list.sort(key=lambda x: x['age'])

        gender1 = patient_data_list[0]['sex']
        gender2 = patient_data_list[1]['sex']

        if gender1 != gender2:
            raise ValueError("The genders of the two timepoints are different! Make sure you are comparing the same patient.")

        # Create a DataFrame from the patient data
        timepoint_data = pd.DataFrame.from_dict([patient_data_list[0], patient_data_list[1]])

        # Transpose the DataFrame and set the index as the features
        timepoint_data = timepoint_data.transpose()
        # print(timepoint_data[0])
        timepoint_data.index.name = 'Feature'

        # Convert the two columns to numeric, excluding any non-numeric rows, and store in new columns
        timepoint_data['Timepoint 1'] = pd.to_numeric(timepoint_data[0], errors='coerce')
        timepoint_data['Timepoint 2'] = pd.to_numeric(timepoint_data[1], errors='coerce')

        # Drop the original unnamed columns
        timepoint_data = timepoint_data[['Timepoint 1', 'Timepoint 2']]

        extraFeatures = ['Total_Brain_Volume', 'Lesions']

        withlesions = extraFeatures + features

        timepoint_data = timepoint_data.reindex(withlesions)

        # Calculate the percentage change, fixing the parentheses
        timepoint_data['Percentage Change'] = ((timepoint_data['Timepoint 2'] - timepoint_data['Timepoint 1']) / timepoint_data['Timepoint 1']) * 100

        # round the percentage change to 1 decimal place
        timepoint_data['Percentage Change'] = timepoint_data['Percentage Change'].round(1)

        # Save the DataFrame to an Excel file
        timepoint_data.to_excel(f"{args.output_file}.xlsx")


        with PdfPages(f"{args.output_file}.pdf") as pdf:
            # First page with age and sex distribution
            fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
            create_distribution_plots(df, axs)
            pdf.savefig(fig)
            plt.close(fig)

            # Plot for Lesions if available
            if 'Lesions' in timepoint_data.index:
                # drop the percentage change column in order to plot the lesions
                percentagechange = timepoint_data.loc['Lesions']['Percentage Change']
                timepoint_data = timepoint_data.drop(columns=['Percentage Change'])

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(['Timepoint 1', 'Timepoint 2'], timepoint_data.loc['Lesions'], marker='x', color='green', linewidth=2, markersize=10, label='Lesions', linestyle='dashed', markeredgewidth=2, markeredgecolor='green', markerfacecolor='white', alpha=0.8, zorder=3, clip_on=False, solid_capstyle='round', dash_capstyle='round')
                ax.set_title('Lesions')
                ax.set_ylabel('Volume (mm$^3$)')
                ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.3)
                        # Get the current upper limit of the y-axis
                ymin, ymax = ax.get_ylim()

                # Set the lower limit to 0, keeping the upper limit the same
                ax.set_ylim(bottom=0, top=ymax*1.2)
                # write the percentage change on the plot 
                ax.annotate(f"{percentagechange}%", xy=(1, timepoint_data.loc['Lesions']['Timepoint 2']), xytext=(8, 0), textcoords='offset points', color='green', fontsize=16)
                pdf.savefig(fig)
                plt.close(fig)

            # Other pages with percentile plots
            for gender in patient_data['sex']:
                for i in range(0, len(features), 4):
                    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                    axs = axs.flatten()

                    for j in range(4):
                        if i + j < len(features):
                            create_percentile_plots(df, features[i + j], percentiles, color_dict, axs[j], gender, patient_data_list)
                            

                            feature_name_csv = features[i + j]
                            feature_label = feature_labels.get(feature_name_csv, feature_name_csv)  # Use the CSV name as a fallback
                            gender_label = gender_labels.get(gender, gender)  # Use the CSV value as a fallback
                            axs[j].set_title(f'{feature_label} for {gender_label}')
                            # axs[j].set_title(f'{features[i + j]} for {gender}')
                            axs[j].set_xlabel('Age (years)')
                            axs[j].set_ylabel('Volume (mm$^3$)')
                            axs[j].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.3)

                    pdf.savefig(fig)
                    plt.close(fig)

    else:
        raise ValueError("Too many input files! Please provide only zero, one or two input files.")

if __name__ == "__main__":
    main()

