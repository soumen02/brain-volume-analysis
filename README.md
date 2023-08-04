# Volume Analysis Script

### Description

This script analyzes and plots the volume data of a patient, taking a CSV file containing volume data as input and outputting a PDF file with plots. It includes functionality to process the data, create distribution plots, calculate percentiles, and generate a PDF report.

### Usage

Run the script with the following command:

```bash
python3 volume_analysis.py -i input_file1 -a age1 -g gender1 -i input_file2 -a age2 -g gender2 -o output_file

``````

- -i, --input: Input file with patient's volume data (.stat format)
- -a, --age: Age of the patient (in years)
- -g, --gender: Gender of the patient (M or F)
- -o, --output: Output PDF file name (default "analysis")

### Sample Files 
[Sample input file](Sample Output/Opth0001_dementia.pdf)