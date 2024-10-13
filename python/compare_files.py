import tkinter as tk
from tkinter import filedialog, messagebox

def select_file1():
    file1.set(filedialog.askopenfilename(title="Select the first file"))

def select_file2():
    file2.set(filedialog.askopenfilename(title="Select the second file"))

def compare_files():
    file_path1 = file1.get()
    file_path2 = file2.get()

    try:
        if not file_path1 or not file_path2:
            messagebox.showerror("Error", "Please select both files to compare.")
            return
        
        # Open both files and read their lines
        with open(file_path1, 'r') as f1, open(file_path2, 'r') as f2:
            file_content1 = f1.readlines()
            file_content2 = f2.readlines()

        differences = []  # List to store the line numbers of differences
        max_lines = max(len(file_content1), len(file_content2))

        # Compare line by line
        for i in range(max_lines):
            line_f1 = file_content1[i].rstrip() if i < len(file_content1) else "[Missing line]"
            line_f2 = file_content2[i].rstrip() if i < len(file_content2) else "[Missing line]"
            
            if line_f1 != line_f2:
                differences.append(f"{i+1}")

        # Display the result
        if differences:
            result = ", ".join(differences)
            messagebox.showinfo("Result", f"The files differ at the following lines:\n\n{result}")
        else:
            messagebox.showinfo("Result", "The files are identical.")
    
    except FileNotFoundError as e:
        messagebox.showerror("Error", f"File not found: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Initialize the Tkinter GUI
window = tk.Tk()
window.title("File Comparator")

# Variables to store the file paths
file1 = tk.StringVar()
file2 = tk.StringVar()

# Interface for selecting the first file
tk.Label(window, text="File 1:").grid(row=0, column=0, padx=10, pady=10)
tk.Entry(window, textvariable=file1, width=50).grid(row=0, column=1, padx=10, pady=10)
tk.Button(window, text="Select", command=select_file1).grid(row=0, column=2, padx=10, pady=10)

# Interface for selecting the second file
tk.Label(window, text="File 2:").grid(row=1, column=0, padx=10, pady=10)
tk.Entry(window, textvariable=file2, width=50).grid(row=1, column=1, padx=10, pady=10)
tk.Button(window, text="Select", command=select_file2).grid(row=1, column=2, padx=10, pady=10)

# Button to compare the files
tk.Button(window, text="Compare", command=compare_files).grid(row=2, column=1, padx=10, pady=20)

# Main loop of the GUI
window.mainloop()
