import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
from dixon import dixon_test


class ScrollableFrame(ttk.Frame):
    
    """A scrollable frame class"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # Create a canvas and scrollbars
        self.canvas = tk.Canvas(self)
        self.scrollbar_y = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)

        # Create the scrollable frame
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Configure the canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        # Configure canvas resize
        self.canvas.bind('<Configure>', self.on_canvas_configure)

        # Layout
        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Bind mouse wheel
        self.bind_mouse_wheel()

    def on_canvas_configure(self, event):
        """Handle canvas resize"""
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

    def bind_mouse_wheel(self):
        """Bind mouse wheel to scrolling"""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _on_mousewheel_horizontal(event):
            self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", _on_mousewheel_horizontal)


class DataTransformationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Transformation Tool")
        self.root.geometry("1200x800")

        # Data storage
        self.df = None
        self.selected_column = None
        self.excel_sheets = {}
        self.current_sheet = None

        self.create_widgets()

    def dixon_test(data, alpha=0.05):
        """
            Perform Dixon's Q test for outliers.
            Parameters:
                data (array-like): 1D array of numeric data.
                alpha (float): Significance level (default is 0.05).
            Returns:
                outliers (list): List of detected outliers.
        """
        data = np.sort(data)
        n = len(data)

        if n < 3 or n > 30:
                raise ValueError("Dixon's test is only valid for sample sizes between 3 and 30.")

                # Q critical values table for alpha=0.05
        q_critical = {
                3: 0.941, 4: 0.765, 5: 0.642, 6: 0.560, 7: 0.507, 8: 0.468,
                9: 0.437, 10: 0.412, 11: 0.392, 12: 0.376, 13: 0.361, 14: 0.349,
                15: 0.338, 16: 0.329, 17: 0.320, 18: 0.313, 19: 0.306, 20: 0.300,
                21: 0.295, 22: 0.290, 23: 0.285, 24: 0.281, 25: 0.277, 26: 0.273,
                27: 0.270, 28: 0.267, 29: 0.263, 30: 0.260
            }

        q_min = (data[1] - data[0]) / (data[-1] - data[0])
        q_max = (data[-1] - data[-2]) / (data[-1] - data[0])

        outliers = []
        if q_min > q_critical[n]:
            outliers.append(data[0])
        if q_max > q_critical[n]:
            outliers.append(data[-1])

        return outliers
    

    def create_widgets(self):
        # Wrap everything in a ScrollableFrame
        self.scroll_frame = ScrollableFrame(self.root)
        self.scroll_frame.pack(fill="both", expand=True)
        
        # Main frame with padding and weights (now inside scrollable_frame)
        main_frame = ttk.Frame(self.scroll_frame.scrollable_frame, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Data source selection
        ttk.Label(main_frame, text="Select Data Source:").grid(row=0, column=0, pady=5, sticky=tk.W)

        data_buttons_frame = ttk.Frame(main_frame)
        data_buttons_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W)

        ttk.Button(data_buttons_frame, text="Use Example Data",
                  command=self.load_example_data).grid(row=0, column=0, padx=5)

        ttk.Button(data_buttons_frame, text="Load CSV File",
                  command=lambda: self.load_file("csv")).grid(row=0, column=1, padx=5)

        ttk.Button(data_buttons_frame, text="Load Excel File",
                  command=lambda: self.load_file("excel")).grid(row=0, column=2, padx=5)

        # Sheet selection frame (initially hidden)
        self.sheet_frame = ttk.LabelFrame(main_frame, text="Sheet Selection", padding="5")
        self.sheet_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        self.sheet_frame.grid_remove()  # Hide initially

        ttk.Label(self.sheet_frame, text="Select Sheet:").grid(row=0, column=0, padx=5)
        self.sheet_var = tk.StringVar()
        self.sheet_combo = ttk.Combobox(self.sheet_frame, textvariable=self.sheet_var)
        self.sheet_combo.grid(row=0, column=1, padx=5)
        self.sheet_combo.bind('<<ComboboxSelected>>', self.on_sheet_select)

        # Data preview
        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding="5")
        preview_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        # Create Treeview for data preview
        self.tree = ttk.Treeview(preview_frame, height=5)
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Add scrollbar to Treeview
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Column selection
        self.column_frame = ttk.LabelFrame(main_frame, text="Column Selection", padding="5")
        self.column_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(self.column_frame, text="Select column to transform:").grid(row=0, column=0, padx=5)
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(self.column_frame, textvariable=self.column_var)
        self.column_combo.grid(row=0, column=1, padx=5)
        self.column_combo.bind('<<ComboboxSelected>>', self.on_column_select)

        # Transformation selection
        transform_frame = ttk.LabelFrame(main_frame, text="Transformations", padding="5")
        transform_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(transform_frame, text="Base:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.log_base_var = tk.StringVar(value="2")  # Valeur par défaut
        ttk.Entry(transform_frame, textvariable=self.log_base_var, width=5).grid(row=1, column=1, padx=5)

        ttk.Button(transform_frame, text="Normalize",
                  command=lambda: self.apply_transformation("normalize")).grid(row=0, column=0, padx=5)
        ttk.Button(transform_frame, text="Standardize",
                  command=lambda: self.apply_transformation("standardize")).grid(row=0, column=1, padx=5)
        ttk.Button(transform_frame, text="Log10",
                  command=lambda: self.apply_transformation("log")).grid(row=0, column=2, padx=5)
        ttk.Button(transform_frame, text="Log_n",
                    command=lambda: self.apply_transformation("log_n")).grid(row=1, column=2, padx=5)
        ttk.Button(transform_frame, text="Dixon Test",
                    command=self.apply_dixon_test).grid(row=2, column=0, padx=5)



        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        self.results_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create a frame for the plots
        self.plot_frame = ttk.Frame(self.results_frame)
        self.plot_frame.grid(row=0, column=0, pady=5)

    def load_file(self, file_type):
        if file_type == "csv":
            filetypes = [("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        else:  # excel
            filetypes = [("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]

        file_path = filedialog.askopenfilename(filetypes=filetypes)

        if file_path:
            try:
                if file_type == "csv":
                    self.df = pd.read_csv(file_path, delimiter='\t')
                    self.excel_sheets = {}
                    self.sheet_frame.grid_remove()
                    self.process_dataframe(self.df)
                else:  # excel
                    self.excel_sheets = pd.read_excel(file_path, sheet_name=None)
                    self.sheet_combo['values'] = list(self.excel_sheets.keys())
                    if len(self.excel_sheets) > 0:
                        self.sheet_combo.set(list(self.excel_sheets.keys())[0])
                        self.df = self.excel_sheets[self.sheet_combo.get()]
                        self.process_dataframe(self.df)
                    self.sheet_frame.grid()

                messagebox.showinfo("Success", "File loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file: {str(e)}")

    def process_dataframe(self, df):
        try:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')

            numeric_columns = ['ouv', 'haut', 'bas', 'clot', 'vol']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            self.update_treeview()
            self.update_column_list()

        except Exception as e:
            messagebox.showerror("Error", f"Error processing data: {str(e)}")

    def on_sheet_select(self, event):
        selected_sheet = self.sheet_var.get()
        if selected_sheet in self.excel_sheets:
            self.df = self.excel_sheets[selected_sheet]
            self.process_dataframe(self.df)

    def on_column_select(self, event):
        """Handle column selection"""
        self.selected_column = self.column_var.get()

    def update_treeview(self):
        self.tree.delete(*self.tree.get_children())

        if self.df is not None:
            self.tree["columns"] = list(self.df.columns)
            self.tree["show"] = "headings"

            for column in self.df.columns:
                self.tree.heading(column, text=column)
                max_width = max(
                    len(str(column)),
                    self.df[column].astype(str).str.len().max() if len(self.df) > 0 else 0
                )
                self.tree.column(column, width=min(max_width * 10, 150))

            for i in range(min(5, len(self.df))):
                values = self.df.iloc[i].tolist()
                values = [f"{x:.2f}" if isinstance(x, (float, np.float64)) else str(x) for x in values]
                self.tree.insert("", "end", values=values)

    def update_column_list(self):
        if self.df is not None:
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            self.column_combo['values'] = list(numeric_columns)
            if len(numeric_columns) > 0:
                self.column_combo.set(numeric_columns[0])
                self.selected_column = numeric_columns[0]

    def load_example_data(self):
        data = {
            'date': pd.date_range(start='2024-01-01', periods=5),
            'ouv': [160, 170, 180, 190, 185],
            'haut': [165, 175, 185, 195, 190],
            'bas': [155, 165, 175, 185, 180],
            'clot': [162, 172, 182, 192, 187],
            'vol': [1000, 1200, 1100, 1300, 1150],
            'devise': ['EUR', 'EUR', 'EUR', 'EUR', 'EUR']
        }
        self.df = pd.DataFrame(data)
        self.excel_sheets = {}
        self.sheet_frame.grid_remove()
        self.update_treeview()
        self.update_column_list()
        messagebox.showinfo("Success", "Example data loaded successfully!")
        
    
    def normalize_data(self, X):
        try:
            X = X.astype(float)
            X_min = np.min(X)
            X_max = np.max(X)
            return (X - X_min) / (X_max - X_min)
        except Exception as e:
            messagebox.showerror("Error", f"Error in normalization: {str(e)}")
            return None

    def standardize_data(self, X):
        try:
            X = X.astype(float)
            return (X - np.mean(X)) / np.std(X)
        except Exception as e:
            messagebox.showerror("Error", f"Error in standardization: {str(e)}")
            return None

    def log_transform(self, X):
        try:
            X = X.astype(float)
            if np.any(X <= 0):
                messagebox.showerror("Error", "Log transformation requires positive values!")
                return None
            return np.log10(X)
        except Exception as e:
            messagebox.showerror("Error", f"Error in log transformation: {str(e)}")
            return None

    def log_n_transform(self, X, base):
        try:
            X = X.astype(float)
            if np.any(X <= 0):
                messagebox.showerror("Error", "Log_n transformation requires positive values!")
                return None
            return np.log(X) / np.log(base)
        except Exception as e:
            messagebox.showerror("Error", f"Error in log_n transformation: {str(e)}")
            return None

    def apply_transformation(self, transform_type):
        if self.df is None or self.selected_column is None:
            messagebox.showerror("Error", "Please load data and select a column first!")
            return

        try:
            original_data = self.df[self.selected_column].values

            if transform_type == "normalize":
                transformed_data = self.normalize_data(original_data)
                title = "Normalized"
            elif transform_type == "standardize":
                transformed_data = self.standardize_data(original_data)
                title = "Standardized"
            elif transform_type == "log":
                transformed_data = self.log_transform(original_data)
                title = "Log-transformed"
            elif transform_type == "log_n":
                base = float(self.log_base_var.get())
                transformed_data = self.log_n_transform(original_data, base)
                title = f"Log_{base}-transformed"
            elif transform_type == "dixon_test":
                outliers = dixon_test(original_data)
                if outliers:
                    messagebox.showinfo("Dixon Test Result", f"Outliers detected: {outliers}")
                else:
                    messagebox.showinfo("Dixon Test Result", "No outliers detected.")


            if transformed_data is not None:
                self.plot_results(original_data, transformed_data, title)

        except Exception as e:
            messagebox.showerror("Error", f"Error applying transformation: {str(e)}")
            
    
    def apply_dixon_test(self):
        if self.df is None or self.selected_column is None:
            messagebox.showerror("Error", "Please load data and select a column first!")
            return

        try:
            data = self.df[self.selected_column].dropna().values
            outliers = dixon_test(data)
        
            if outliers:
                messagebox.showinfo("Dixon Test Result", f"Detected outliers: {outliers}")
            else:
                messagebox.showinfo("Dixon Test Result", "No outliers detected.")
        except Exception as e:
            messagebox.showerror("Error", f"Error applying Dixon test: {str(e)}")

    def plot_results(self, original_data, transformed_data, title):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Plot original data with statistical indicators
        self._plot_histogram_with_stats(ax1, original_data, 'Original Distribution\n{}'.format(self.selected_column))
        
        # Plot transformed data with statistical indicators
        self._plot_histogram_with_stats(ax2, transformed_data, '{} Distribution\n{}'.format(title, self.selected_column))

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)

        # Add statistics text below the plots
        self._add_statistics_text(original_data, transformed_data)

    def _plot_histogram_with_stats(self, ax, data, title):
        # Calculate statistics
        mean = np.mean(data)
        median = np.median(data)
        mode = float(pd.Series(data).mode().iloc[0])
        std = np.std(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)

        # Plot histogram with KDE
        n, bins, patches = ax.hist(data, bins=30, density=False, alpha=0.7, color='lightblue')
        kde_x = np.linspace(min(data), max(data), 100)
        kde = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x) * len(data) * (bins[1] - bins[0]), color='blue', linewidth=1)

        # Add vertical lines for statistics
        ax.axvline(mean, color='red', linestyle='--', label=f'Moyenne = {mean:.2f}')
        ax.axvline(median, color='blue', linestyle='-', label=f'Médiane = {median:.2f}')
        ax.axvline(mode, color='purple', linestyle=':', label=f'Mode = {mode:.2f}')
        
        # Add standard deviation bounds
        ax.axvline(mean + std, color='orange', linestyle='--', label=f'Moyenne + σ = {mean + std:.2f}')
        ax.axvline(mean - std, color='orange', linestyle='--', label=f'Moyenne - σ = {mean - std:.2f}')
        
        # Add quartiles
        ax.axvline(q1, color='green', linestyle=':', label=f'1er quartile = {q1:.2f}')
        ax.axvline(q3, color='green', linestyle=':', label=f'3e quartile = {q3:.2f}')

        ax.set_title(title)
        ax.set_xlabel('Valeurs')
        ax.set_ylabel('Fréquence')
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    def _add_statistics_text(self, original_data, transformed_data):
        stats_frame = ttk.Frame(self.plot_frame)
        stats_frame.grid(row=1, column=0, pady=10)

        original_stats = f"""Original Data Statistics:
Mean: {np.mean(original_data):.2f}
Std: {np.std(original_data):.2f}
Min: {np.min(original_data):.2f}
Max: {np.max(original_data):.2f}
Q1: {np.percentile(original_data, 25):.2f}
Q3: {np.percentile(original_data, 75):.2f}"""

        transformed_stats = f"""Transformed Data Statistics:
Mean: {np.mean(transformed_data):.2f}
Std: {np.std(transformed_data):.2f}
Min: {np.min(transformed_data):.2f}
Max: {np.max(transformed_data):.2f}
Q1: {np.percentile(transformed_data, 25):.2f}
Q3: {np.percentile(transformed_data, 75):.2f}"""

        ttk.Label(stats_frame, text=original_stats, justify=tk.LEFT).grid(row=0, column=0, padx=20)
        ttk.Label(stats_frame, text=transformed_stats, justify=tk.LEFT).grid(row=0, column=1, padx=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataTransformationGUI(root)
    root.mainloop()