import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot(saving, xaxis, yaxis_left, yaxis_right=None, yaxis_right_2=None, extensions=None, *varargin):
    
    def check_directory(directory_path):
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"The folder '{directory_path}' does not exists.")
        return [file for file in os.listdir(directory_path) if file.endswith(".csv")]

    def read_csv_data(csv_path, xaxis, yaxis_left, yaxis_right=None, yaxis_right_2=None, start_index=None):
        df = pd.read_csv(csv_path)
        x = df[xaxis]
        y1 = df[yaxis_left]
        y2 = df[yaxis_right] if yaxis_right else None
        y3 = df[yaxis_right_2] if yaxis_right_2 else None

        if start_index is not None:
            x = x[start_index:]
            y1 = y1[start_index:]
            if y2 is not None:
                y2 = y2[start_index:]
            if y3 is not None:
                y3 = y3[start_index:]
        return x, y1, y2, y3

    def plot_text(ax, values, keywords, text_lim=50):
        mytext = ''.join([f"{key}:{values[key]} - " for key in keywords if key in values])
        if len(mytext) > text_lim:
            mytext2 = mytext[text_lim:]
            mytext = mytext[:text_lim]
            if len(mytext2) > text_lim:
                mytext3 = mytext2[text_lim+11:]
                mytext2 = mytext2[:text_lim+11]
            pos_mytext = pos_mytext2 = pos_mytext3 = 0.375
            ax.text(0.5-pos_mytext, 1, mytext, fontsize=14, transform=ax.transAxes)
            ax.text(0.5-pos_mytext2, 0.95, mytext2, fontsize=14, transform=ax.transAxes)
            ax.text(0.5-pos_mytext3, 0.9, mytext3, fontsize=14, transform=ax.transAxes)
        else:
            ax.text(0.5 - len(mytext) / 200, 0.95, mytext, fontsize=14, transform=ax.transAxes)

    def save_plot(fig, csv_file, extensions):
        if isinstance(extensions, list):
            for extension in extensions:
                output_file = os.path.splitext(csv_file)[0] + '.' + extension
                fig.savefig(output_file, bbox_inches='tight')
        else:
            output_file = os.path.splitext(csv_file)[0] + '.' + extensions
            fig.savefig(output_file, bbox_inches='tight')

    directory_path = saving["root_path"]
    csv_files = check_directory(directory_path)

    for csv_file in csv_files:
        csv_path = os.path.join(directory_path, csv_file)
        values, keywords = extract_values_from_filename(csv_file)
        start_index = varargin[0] if len(varargin) == 1 else None
        x, y1, y2, y3 = read_csv_data(csv_path, xaxis, yaxis_left, yaxis_right, yaxis_right_2, start_index)

        fig, ax1 = plt.subplots(figsize=(10, 6.1423))

        ax1.plot(x, y1, color='tab:blue', linestyle='dashed', linewidth=2)
        ax1.set_xlabel(xaxis)
        ax1.set_ylabel(yaxis_left, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        if y2 is not None:
            ax2 = ax1.twinx()
            ax2.plot(x, y2, color='tab:red')
            ax2.set_ylabel(yaxis_right, color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')

        if y3 is not None:
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            ax3.plot(x, y3, color='tab:green')
            ax3.set_ylabel(yaxis_right_2, color='tab:green')
            ax3.tick_params(axis='y', labelcolor='tab:green')

        plot_text(fig, values, keywords)
        save_plot(fig, csv_file, extensions)
        plt.close(fig)

# Example of the extract_values_from_filename function
def extract_values_from_filename(filename):
    # Example implementation, customize as needed
    values = {'example_key': 'example_value'}
    keywords = ['example_key']
    return values, keywords
