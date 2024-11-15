# --- IMPORTS ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
from scipy.stats import skew, shapiro
import statsmodels.api as sm

# --- SCROLLABLE FRAME CLASS ---
class ScrollableFrame(ttk.Frame):
    """A scrollable frame class"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        self.canvas = tk.Canvas(self)
        self.scrollbar_y = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
        self.canvas.bind('<Configure>', self.on_canvas_configure)

        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

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

# --- MAIN GUI CLASS ---
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

    # Shapiro-Wilk asymétrie et aplatissement
    def apply_shapiro_test(self):
        """
        Perform Shapiro-Wilk test for normality and visualize the results with detailed statistics
        """
        if self.df is None or self.selected_column is None:
            messagebox.showerror("Error", "Please load data and select a column first!")
            return

        try:
            # Get the data and remove any NaN values
            data = self.df[self.selected_column].dropna().values

            # Perform Shapiro-Wilk test
            statistic, p_value = shapiro(data)

            # Calculate additional statistics
            skewness = skew(data)
            kurtosis = stats.kurtosis(data)
            mean = np.mean(data)
            std = np.std(data)
            n = len(data)

            # Create figure for visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Histogram with normal distribution overlay
            n_bins = min(int(np.sqrt(n)), 30)  # Optimal number of bins
            n, bins, patches = ax1.hist(data, bins=n_bins, density=True, 
                                      alpha=0.7, color='lightblue', 
                                      label='Observed Data')

            # Add normal distribution curve
            x = np.linspace(mean - 4*std, mean + 4*std, 100)
            normal_dist = stats.norm.pdf(x, mean, std)
            ax1.plot(x, normal_dist, 'r-', lw=2, 
                    label=f'Normal Distribution\nμ={mean:.2f}\nσ={std:.2f}')

            ax1.set_title('Distribution Plot with Normal Curve')
            ax1.set_xlabel('Values')
            ax1.set_ylabel('Density')
            ax1.legend(loc='upper right')

            # Add vertical lines for key statistics
            ax1.axvline(mean, color='red', linestyle='--', alpha=0.5)
            ax1.axvline(mean - std, color='orange', linestyle=':', alpha=0.5)
            ax1.axvline(mean + std, color='orange', linestyle=':', alpha=0.5)

            # Q-Q plot with improved formatting
            sm.qqplot(data, line='45', ax=ax2, markerfacecolor='lightblue', 
                     alpha=0.7, markeredgecolor='blue')
            ax2.set_title('Q-Q Plot')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Clear previous plots
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Add the new plot to the GUI
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0)

            # Add detailed statistics text with proper formatting
            stats_frame = ttk.Frame(self.plot_frame)
            stats_frame.grid(row=1, column=0, pady=10)

            result_text = f"""
    ─── Shapiro-Wilk Test Results ───────────────────────
    Test Statistic: {statistic:.4f}
    P-value: {p_value:.4f}

    ─── Distribution Metrics ──────────────────────────
    • Sample Size: {n} observations
    • Mean: {mean:.4f}
    • Standard Deviation: {std:.4f}
    • Skewness: {skewness:.4f} {'(positively skewed)' if skewness > 0 else '(negatively skewed)' if skewness < 0 else '(symmetric)'}
    • Kurtosis: {kurtosis:.4f} {'(leptokurtic)' if kurtosis > 0 else '(platykurtic)' if kurtosis < 0 else '(mesokurtic)'}

    ─── Interpretation ────────────────────────────────
    {'The data significantly deviates from normal distribution (p < 0.05)' 
    if p_value < 0.05 else 'The data appears to follow a normal distribution (p ≥ 0.05)'}

    ─── Additional Information ─────────────────────────
    • 68% of data falls between: {mean-std:.4f} and {mean+std:.4f}
    • 95% of data falls between: {mean-2*std:.4f} and {mean+2*std:.4f}
    • 99.7% of data falls between: {mean-3*std:.4f} and {mean+3*std:.4f}
    """

            ttk.Label(stats_frame, text=result_text, justify=tk.LEFT,
                     font=('Consolas', 10)).grid(row=0, column=0, padx=20)

            # Show summary message box with proper formatting
            messagebox.showinfo("Shapiro-Wilk Test Result", 
                f"Test Results:\n"
                f"• Statistic: {statistic:.4f}\n"
                f"• P-value: {p_value:.4f}\n\n"
                f"{'❌ Data is NOT normally distributed (p < 0.05)' if p_value < 0.05 else '✓ Data appears to be normally distributed (p ≥ 0.05)'}")

        except Exception as e:
            messagebox.showerror("Error", f"Error applying Shapiro-Wilk test: {str(e)}")
    
    def apply_dixon_test(self, data, alpha=0.05):
        """
        Applique le test de Dixon pour la détection des valeurs aberrantes

        Parameters:
            data (array-like): Données à tester
            alpha (float): Niveau de signification (par défaut 0.05)

        Returns:
            tuple: (résultat du test, valeurs aberrantes trouvées, indices des valeurs aberrantes)
        """
        try:
            # Trier les données
            sorted_data = np.sort(data)
            n = len(sorted_data)

            # Calculer les statistiques de Dixon
            if n >= 3 and n <= 30:  # Le test de Dixon est valide pour ces tailles d'échantillon
                # Test pour la plus petite valeur
                r10_min = (sorted_data[1] - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])
                # Test pour la plus grande valeur
                r10_max = (sorted_data[-1] - sorted_data[-2]) / (sorted_data[-1] - sorted_data[0])

                # Valeurs critiques pour α = 0.05
                critical_values = {
                    3: 0.941, 4: 0.765, 5: 0.642, 6: 0.560, 7: 0.507,
                    8: 0.468, 9: 0.437, 10: 0.412, 11: 0.392, 12: 0.376,
                    13: 0.361, 14: 0.349, 15: 0.338, 16: 0.329, 17: 0.320,
                    18: 0.313, 19: 0.306, 20: 0.300, 21: 0.295, 22: 0.290,
                    23: 0.285, 24: 0.281, 25: 0.277, 26: 0.273, 27: 0.269,
                    28: 0.266, 29: 0.263, 30: 0.260
                }

                critical_value = critical_values[n]

                outliers = []
                outlier_indices = []

                # Vérifier les valeurs aberrantes
                if r10_min > critical_value:
                    outliers.append(sorted_data[0])
                    outlier_indices.append(np.where(data == sorted_data[0])[0][0])

                if r10_max > critical_value:
                    outliers.append(sorted_data[-1])
                    outlier_indices.append(np.where(data == sorted_data[-1])[0][0])

                return True, outliers, outlier_indices
            else:
                return False, [], []

        except Exception as e:
            messagebox.showerror("Error", f"Erreur dans le test de Dixon: {str(e)}")
            return False, [], []
        
    def visualize_dixon_test(self):
        """Visualise les résultats du test de Dixon"""
        if self.df is None or self.selected_column is None:
            messagebox.showerror("Error", "Veuillez charger des données et sélectionner une colonne d'abord!")
            return

        try:
            # Obtenir les données
            data = self.df[self.selected_column].dropna().values

            # Appliquer le test de Dixon
            test_successful, outliers, outlier_indices = self.apply_dixon_test(data)

            if not test_successful:
                messagebox.showwarning("Attention", 
                    "Le test de Dixon nécessite entre 3 et 30 observations valides.")
                return

            # Créer la visualisation
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Boxplot
            ax1.boxplot(data)
            if outliers:
                ax1.plot([1] * len(outliers), outliers, 'ro', label='Valeurs aberrantes')
            ax1.set_title('Boxplot avec valeurs aberrantes')
            ax1.legend()

            # Scatter plot
            x = range(len(data))
            ax2.scatter(x, data, c='blue', alpha=0.5)
            if outliers:
                for idx in outlier_indices:
                    ax2.scatter(idx, data[idx], c='red', s=100, 
                              label='Valeur aberrante' if idx == outlier_indices[0] else "")
            ax2.set_title('Distribution des données')
            ax2.set_xlabel('Index')
            ax2.set_ylabel(self.selected_column)
            ax2.legend()

            plt.tight_layout()

            # Nettoyer le frame des résultats
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Ajouter le nouveau plot à l'interface
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0)

            # Afficher les statistiques
            if outliers:
                stats_text = f"""
                Résultats du test de Dixon:

                Nombre de valeurs aberrantes détectées: {len(outliers)}
                Valeurs aberrantes: {', '.join([f'{x:.2f}' for x in outliers])}
                Positions: {', '.join(map(str, outlier_indices))}
                """
            else:
                stats_text = "Aucune valeur aberrante détectée selon le test de Dixon."

            stats_label = ttk.Label(self.plot_frame, text=stats_text, justify=tk.LEFT)
            stats_label.grid(row=1, column=0, pady=10)

            # Message de résultats
            if outliers:
                messagebox.showinfo("Résultats du test de Dixon",
                                  f"Nombre de valeurs aberrantes détectées: {len(outliers)}\n" +
                                  f"Valeurs: {', '.join([f'{x:.2f}' for x in outliers])}")
            else:
                messagebox.showinfo("Résultats du test de Dixon",
                                  "Aucune valeur aberrante détectée.")

        except Exception as e:
            messagebox.showerror("Error", f"Erreur lors de la visualisation: {str(e)}")


    def create_widgets(self):
        self.scroll_frame = ScrollableFrame(self.root)
        self.scroll_frame.pack(fill="both", expand=True)

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

        # Sheet selection frame
        self.sheet_frame = ttk.LabelFrame(main_frame, text="Sheet Selection", padding="5")
        self.sheet_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        self.sheet_frame.grid_remove()

        ttk.Label(self.sheet_frame, text="Select Sheet:").grid(row=0, column=0, padx=5)
        self.sheet_var = tk.StringVar()
        self.sheet_combo = ttk.Combobox(self.sheet_frame, textvariable=self.sheet_var)
        self.sheet_combo.grid(row=0, column=1, padx=5)
        self.sheet_combo.bind('<<ComboboxSelected>>', self.on_sheet_select)

        # Data preview
        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding="5")
        preview_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        self.tree = ttk.Treeview(preview_frame, height=5)
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E))

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
        ttk.Button(transform_frame, text="Log10 Transform",
                  command=lambda: self.apply_transformation("log10")).grid(row=0, column=2, padx=5)
        ttk.Button(transform_frame, text="Natural Log Transform",
                  command=lambda: self.apply_transformation("log_n")).grid(row=0, column=3, padx=5)
        ttk.Button(transform_frame, text="Log_x",
                command=lambda: self.apply_transformation("log_x")).grid(row=0, column=4, padx=5)
        ttk.Button(transform_frame, text="Test de Dixon",
           command=self.visualize_dixon_test).grid(row=2, column=0, padx=5)
        ttk.Button(transform_frame, text="Shapiro-Wilk Test",
                  command=self.apply_shapiro_test).grid(row=2, column=1, padx=5)

        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        self.results_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.plot_frame = ttk.Frame(self.results_frame)
        self.plot_frame.grid(row=0, column=0, pady=5)

    def load_file(self, file_type):
        filetypes = [("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")] if file_type == "csv" \
            else [("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]

        file_path = filedialog.askopenfilename(filetypes=filetypes)

        if file_path:
            try:
                if file_type == "csv":
                    self.df = pd.read_csv(file_path, delimiter='\t')
                    self.excel_sheets = {}
                    self.sheet_frame.grid_remove()
                else:
                    self.excel_sheets = pd.read_excel(file_path, sheet_name=None)
                    self.sheet_combo['values'] = list(self.excel_sheets.keys())
                    if len(self.excel_sheets) > 0:
                        self.sheet_combo.set(list(self.excel_sheets.keys())[0])
                        self.df = self.excel_sheets[self.sheet_combo.get()]
                    self.sheet_frame.grid()

                self.process_dataframe(self.df)
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

    def log_n_transform(self, X):
        try:
            X = X.astype(float)
            if np.any(X <= 0):
                messagebox.showerror("Error", "Natural log transformation requires positive values!")
                return None
            return np.log(X)
        except Exception as e:
            messagebox.showerror("Error", f"Error in natural log transformation: {str(e)}")
            return None
    
    def log_X_transform(self, X, base):
        try:
            X = X.astype(float)
            if np.any(X <= 0):
                messagebox.showerror("Error", "Log_X transformation requires positive values!")
                return None
            return np.log(X) / np.log(base)
        except Exception as e:
            messagebox.showerror("Error", f"Error in log_X transformation: {str(e)}")
            return None
        
    def apply_transformation(self, transform_type):
        """
        Apply the selected transformation to the data and display results
        
        Parameters:
            transform_type (str): Type of transformation to apply ('normalize', 'standardize', 'log10', or 'log_n')
        """
        if self.df is None or self.selected_column is None:
            messagebox.showerror("Error", "Please load data and select a column first!")
            return
    
        try:
            # Get original data
            original_data = self.df[self.selected_column].dropna().values
            
            # Verify we have data to transform
            if len(original_data) == 0:
                messagebox.showerror("Error", "No valid data to transform!")
                return
    
            # Apply the selected transformation
            if transform_type == "normalize":
                transformed_data = self.normalize_data(original_data)
                title = "Normalized"
                include_qqplot = True
                
            elif transform_type == "standardize":
                transformed_data = self.standardize_data(original_data)
                title = "Standardized"
                include_qqplot = True
                
            elif transform_type == "log10":
                transformed_data = self.log_transform(original_data)
                title = "Log10-transformed"
                include_qqplot = False
                
            elif transform_type == "log_n":
                transformed_data = self.log_n_transform(original_data)
                title = "Natural Log-transformed"
                include_qqplot = False
            
            elif transform_type == "log_x":
                base = float(self.log_base_var.get())
                transformed_data = self.log_X_transform(original_data, base)
                title = f"Log_{base}-transformed"
                include_qqplot = False
                
            else:
                messagebox.showerror("Error", f"Unknown transformation type: {transform_type}")
                return
    
            # If transformation was successful, display results
            if transformed_data is not None:
                # Add transformed data as new column
                new_col_name = f"{self.selected_column}_{transform_type}"
                self.df[new_col_name] = transformed_data
                
                # Update the column list in the GUI
                self.update_column_list()
                
                # Plot the results
                self.plot_results(original_data, transformed_data, title, include_qqplot)
                
                # Show success message
                messagebox.showinfo("Success", 
                                  f"Transformation applied successfully!\n"
                                  f"New column '{new_col_name}' has been added to the dataset.")
    
        except Exception as e:
            messagebox.showerror("Error", f"Error applying transformation: {str(e)}")

    def plot_results(self, original_data, transformed_data, title, include_qqplot):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        if include_qqplot:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.3])
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        self._plot_histogram_with_stats_and_box(ax1, original_data, 'Original Distribution\n{}'.format(self.selected_column))
        self._plot_histogram_with_stats_and_box(ax2, transformed_data, '{} Distribution\n{}'.format(title, self.selected_column))

        if include_qqplot:
            sm.qqplot(original_data, line='45', ax=ax3)
            ax3.set_title('Q-Q Plot - Original Data')
            sm.qqplot(transformed_data, line='45', ax=ax4)
            ax4.set_title('Q-Q Plot - Transformed Data')

        plt.subplots_adjust(left=0.07, right=0.85, wspace=0.5)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)

        self._add_statistics_text(original_data, transformed_data)

    def _plot_histogram_with_stats_and_box(self, ax, data, title):
        """
        Create a detailed histogram with statistical overlay and boxplot

        Parameters:
            ax (matplotlib.axes.Axes): The axes to plot on
            data (numpy.ndarray): The data to plot
            title (str): The title for the plot
        """
        # Calculate statistics
        mean = np.mean(data)
        median = np.median(data)
        mode = float(pd.Series(data).mode().iloc[0])
        std = np.std(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        skewness = skew(data)

        # Create histogram
        n_bins = min(int(np.sqrt(len(data))), 30)
        n, bins, patches = ax.hist(data, bins=n_bins, density=False,
                                 alpha=0.7, color='lightblue', 
                                 label='Data Distribution')

        # Add kernel density estimation
        kde_x = np.linspace(min(data), max(data), 100)
        kde = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x) * len(data) * (bins[1] - bins[0]), 
                color='blue', linewidth=1, label='Density Curve')

        # Add boxplot
        bp = ax.boxplot(data, vert=False, positions=[max(n)], 
                       widths=[max(n)/4], patch_artist=True)

        # Color the boxplot
        plt.setp(bp['boxes'], facecolor='lightgreen', alpha=0.6)
        plt.setp(bp['medians'], color='darkgreen')
        plt.setp(bp['fliers'], marker='o', markerfacecolor='red', alpha=0.6)

        # Add vertical lines for key statistics
        ax.axvline(mean, color='red', linestyle='--', 
                   label=f'Mean = {mean:.2f}')
        ax.axvline(median, color='blue', linestyle='-', 
                   label=f'Median = {median:.2f}')
        ax.axvline(mode, color='purple', linestyle=':', 
                   label=f'Mode = {mode:.2f}')
        ax.axvline(mean + std, color='orange', linestyle='--', 
                   label=f'Mean ± σ ({std:.2f})')
        ax.axvline(mean - std, color='orange', linestyle='--')
        ax.axvline(q1, color='green', linestyle=':', 
                   label=f'Quartiles (Q1={q1:.2f}, Q3={q3:.2f})')
        ax.axvline(q3, color='green', linestyle=':')

        # Formatting
        ax.set_title(f"{title}\nSkewness: {skewness:.2f}")
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), 
                 fontsize='small', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Add some padding to the x-axis
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min - 0.05 * (x_max - x_min), 
                    x_max + 0.05 * (x_max - x_min))
    
    def _add_statistics_text(self, original_data, transformed_data):
        stats_frame = ttk.Frame(self.plot_frame)
        stats_frame.grid(row=1, column=0, pady=10)

        original_stats = f"""Original Data Statistics:
Mean: {np.mean(original_data):.2f}
Std: {np.std(original_data):.2f}
Min: {np.min(original_data):.2f}
Max: {np.max(original_data):.2f}
Q1: {np.percentile(original_data, 25):.2f}
Q3: {np.percentile(original_data, 75):.2f}
Kurtosis: {stats.kurtosis(original_data):.2f}
Skewness: {skew(original_data):.2f}
Population: {len(original_data):.2f}"""

        transformed_stats = f"""Transformed Data Statistics:
Mean: {np.mean(transformed_data):.2f}
Std: {np.std(transformed_data):.2f}
Min: {np.min(transformed_data):.2f}
Max: {np.max(transformed_data):.2f}
Q1: {np.percentile(transformed_data, 25):.2f}
Q3: {np.percentile(transformed_data, 75):.2f}
Kurtosis: {stats.kurtosis(transformed_data):.2f}
Skewness: {skew(transformed_data):.2f}
Population: {len(transformed_data):.2f}"""

        ttk.Label(stats_frame, text=original_stats, justify=tk.LEFT).grid(row=0, column=0, padx=20)
        ttk.Label(stats_frame, text=transformed_stats, justify=tk.LEFT).grid(row=0, column=1, padx=20)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DataTransformationGUI(root)
    root.mainloop()
