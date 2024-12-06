# --- IMPORTS ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
from scipy.stats import skew, shapiro, t
import statsmodels.api as sm

# --- SCROLLABLE FRAME CLASS ---
class ScrollableFrame(ttk.Frame):
    """A scrollable frame class that enables vertical and horizontal scrolling"""
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
        """Handle canvas resize event by updating the window width"""
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

    def bind_mouse_wheel(self):
        """Bind mouse wheel events for both vertical and horizontal scrolling"""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _on_mousewheel_horizontal(event):
            self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", _on_mousewheel_horizontal)

# --- STATISTICAL TEST FUNCTIONS ---
def apply_dixon_test(data, alpha=0.05):
    """
    Applique le test de Dixon pour la détection des valeurs aberrantes.
    
    Parameters:
        data (array-like): Données à tester
        alpha (float): Niveau de signification (par défaut 0.05)
    
    Returns:
        tuple: (test_successful, outliers, outlier_indices)
    """
    try:
        # Trier les données
        sorted_data = np.sort(data)
        n = len(sorted_data)

        if n >= 3 and n <= 30:  
            # Test pour la plus petite et la plus grande valeur
            r10_min = (sorted_data[1] - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])
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

def apply_grubbs_test(data, alpha=0.05):
    """
    Applique le test de Grubbs pour détecter les valeurs aberrantes.
    
    Parameters:
        data (array-like): Données à tester
        alpha (float): Niveau de signification
    
    Returns:
        tuple: (is_outlier, outlier, outlier_index)
    """
    try:
        # Calculer la moyenne et l'écart type
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        # Calculer les statistiques G pour min et max
        G_min = (mean - np.min(data)) / std
        G_max = (np.max(data) - mean) / std
        
        # Valeurs critiques du test de Grubbs
        t_value = t.ppf(1 - alpha / (2 * n), n - 2)
        G_crit = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))
        
        # Trouver la valeur la plus extrême
        if G_max > G_min:
            G = G_max
            outlier = np.max(data)
            idx = np.argmax(data)
        else:
            G = G_min
            outlier = np.min(data)
            idx = np.argmin(data)
        
        # Tester si c'est une valeur aberrante
        is_outlier = G > G_crit
        
        return is_outlier, outlier, idx
    
    except Exception as e:
        messagebox.showerror("Error", f"Erreur dans le test de Grubbs: {str(e)}")
        return False, None, None


def normalize_data(data, method='zscore'):
    """
    Normalise les données selon différentes méthodes.

    Parameters:
        data (array-like): Données à normaliser
        method (str): Méthode de normalisation ('zscore', 'minmax', 'robust', 'decimal')

    Returns:
        array-like: Données normalisées
    """
    try:
        data = np.array(data, dtype=float)
        
        if method == 'zscore':
            return (data - np.mean(data)) / np.std(data)
        elif method == 'minmax':
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        elif method == 'robust':
            median = np.median(data)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            return (data - median) / iqr
        elif method == 'decimal':
            return data / 10**len(str(int(np.max(data))))
        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")
            
    except Exception as e:
        messagebox.showerror("Error", f"Erreur de normalisation: {str(e)}")
        return None
# --- MAIN GUI CLASS ---
class DataTransformationGUI:
    def __init__(self, root):
        """Initialize the main GUI application"""
        self.root = root
        self.root.title("Data Transformation Tool")
        self.root.geometry("1200x800")

        # Data storage initialization
        self.df = None
        self.selected_column = None
        self.excel_sheets = {}
        self.current_sheet = None
        
        # Create the GUI widgets
        self.create_widgets()

    def create_widgets(self):
        """Create and arrange all GUI widgets"""
        # Create main scrollable frame
        self.scroll_frame = ScrollableFrame(self.root)
        self.scroll_frame.pack(fill="both", expand=True)

        main_frame = ttk.Frame(self.scroll_frame.scrollable_frame, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Data source selection section
        ttk.Label(main_frame, text="Select Data Source:").grid(row=0, column=0, pady=5, sticky=tk.W)

        data_buttons_frame = ttk.Frame(main_frame)
        data_buttons_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W)

        ttk.Button(data_buttons_frame, text="Use Example Data",
                  command=self.load_example_data).grid(row=0, column=0, padx=5)
        ttk.Button(data_buttons_frame, text="Load CSV File",
                  command=lambda: self.load_file("csv")).grid(row=0, column=1, padx=5)
        ttk.Button(data_buttons_frame, text="Load Excel File",
                  command=lambda: self.load_file("excel")).grid(row=0, column=2, padx=5)

        # Sheet selection frame (for Excel files)
        self.sheet_frame = ttk.LabelFrame(main_frame, text="Sheet Selection", padding="5")
        self.sheet_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        self.sheet_frame.grid_remove()  # Hidden by default

        ttk.Label(self.sheet_frame, text="Select Sheet:").grid(row=0, column=0, padx=5)
        self.sheet_var = tk.StringVar()
        self.sheet_combo = ttk.Combobox(self.sheet_frame, textvariable=self.sheet_var)
        self.sheet_combo.grid(row=0, column=1, padx=5)
        self.sheet_combo.bind('<<ComboboxSelected>>', self.on_sheet_select)

        # Data preview section
        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding="5")
        preview_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        self.tree = ttk.Treeview(preview_frame, height=5)
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E))

        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Column selection section
        self.column_frame = ttk.LabelFrame(main_frame, text="Column Selection", padding="5")
        self.column_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(self.column_frame, text="Select column to transform:").grid(row=0, column=0, padx=5)
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(self.column_frame, textvariable=self.column_var)
        self.column_combo.grid(row=0, column=1, padx=5)
        self.column_combo.bind('<<ComboboxSelected>>', self.on_column_select)

        # Create transformation buttons frame
        transform_frame = self.create_transformation_frame(main_frame)
        transform_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        self.results_frame.grid(row=6, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.plot_frame = ttk.Frame(self.results_frame)
        self.plot_frame.grid(row=0, column=0, pady=5)

    def create_transformation_frame(self, parent):
        """Create the frame containing transformation buttons"""
        transform_frame = ttk.LabelFrame(parent, text="Transformations", padding="5")

        # Base input for logarithmic transformations
        ttk.Label(transform_frame, text="Base:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.log_base_var = tk.StringVar(value="2")
        ttk.Entry(transform_frame, textvariable=self.log_base_var, width=5).grid(row=1, column=1, padx=5)

        # First row of transformation buttons
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

        # Second row of statistical test buttons
        ttk.Button(transform_frame, text="Test de Dixon",
            command=self.visualize_dixon_test).grid(row=2, column=0, padx=5)
        ttk.Button(transform_frame, text="Test de Grubbs",
            command=self.remove_outliers_grubbs).grid(row=2, column=1, padx=5)
        ttk.Button(transform_frame, text="Shapiro-Wilk Test",
            command=self.apply_shapiro_test).grid(row=2, column=2, padx=5)
        ttk.Button(transform_frame, text="Advanced Normalize",
            command=self.show_normalize_dialog).grid(row=2, column=3, padx=5)
        ttk.Button(transform_frame, text="Clean Data",
            command=self.apply_clean_data).grid(row=2, column=4, padx=5)

        return transform_frame
    
    def load_file(self, file_type):
        """
        Charge un fichier CSV ou Excel.
        
        Parameters:
            file_type (str): Type de fichier ('csv' ou 'excel')
        """
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

    def load_example_data(self):
        """Charge un jeu de données exemple"""
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

    def process_dataframe(self, df): #check
        """
        Traite le DataFrame chargé pour assurer la compatibilité des types de données.
        
        Parameters:
            df (pandas.DataFrame): DataFrame à traiter
        """
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

    def update_treeview(self): #check 
        """Met à jour l'affichage des données dans le Treeview"""
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
        """Met à jour la liste des colonnes disponibles dans le ComboBox"""
        if self.df is not None:
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            self.column_combo['values'] = list(numeric_columns)
            if len(numeric_columns) > 0:
                self.column_combo.set(numeric_columns[0])
                self.selected_column = numeric_columns[0]

    def on_sheet_select(self, event):
        """Gestionnaire d'événement pour la sélection d'une feuille Excel"""
        selected_sheet = self.sheet_var.get()
        if selected_sheet in self.excel_sheets:
            self.df = self.excel_sheets[selected_sheet]
            self.process_dataframe(self.df)

    def on_column_select(self, event):
        """Gestionnaire d'événement pour la sélection d'une colonne"""
        self.selected_column = self.column_var.get()

    def show_normalize_dialog(self):
        """Affiche la boîte de dialogue pour la normalisation avancée"""
        if not self.selected_column:
            messagebox.showerror("Error", "Please select a column first")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Advanced Normalize")

        ttk.Label(dialog, text="Select normalization method:").pack(pady=5)
        method = tk.StringVar(value='zscore')

        for val in ['zscore', 'minmax', 'robust', 'decimal']:
            ttk.Radiobutton(dialog, text=val, value=val, 
                           variable=method).pack()

        ttk.Button(dialog, text="Apply",
                   command=lambda: self.apply_advanced_normalize(method.get(), 
                                                              dialog)).pack(pady=10)

    def apply_transformation(self, transform_type):
        """
        Applique une transformation aux données et affiche les résultats
        
        Parameters:
            transform_type (str): Type de transformation ('normalize', 'standardize', 'log10', 'log_n', 'log_x')
        """
        if self.df is None or self.selected_column is None:
            messagebox.showerror("Error", "Please load data and select a column first!")
            return
    
        try:
            # Obtenir les données originales
            original_data = self.df[self.selected_column].dropna().values
            
            if len(original_data) == 0:
                messagebox.showerror("Error", "No valid data to transform!")
                return
    
            # Appliquer la transformation sélectionnée
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
    
            # Si la transformation a réussi, afficher les résultats
            if transformed_data is not None:
                new_col_name = f"{self.selected_column}_{transform_type}"
                self.df[new_col_name] = transformed_data
                self.update_column_list()
                self.plot_results(original_data, transformed_data, title, include_qqplot)
                messagebox.showinfo("Success", 
                                  f"Transformation applied successfully!\n"
                                  f"New column '{new_col_name}' has been added to the dataset.")
    
        except Exception as e:
            messagebox.showerror("Error", f"Error applying transformation: {str(e)}")

    def standardize_data(self, X):
        """Standardise les données"""
        try:
            X = X.astype(float)
            return (X - np.mean(X)) / np.std(X)
        except Exception as e:
            messagebox.showerror("Error", f"Error in standardization: {str(e)}")
            return None

    def log_transform(self, X):
        """Transformation logarithmique base 10"""
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
        """Transformation logarithmique naturelle"""
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
        """Transformation logarithmique avec base personnalisée"""
        try:
            X = X.astype(float)
            if np.any(X <= 0):
                messagebox.showerror("Error", "Log transformation requires positive values!")
                return None
            return np.log(X) / np.log(base)
        except Exception as e:
            messagebox.showerror("Error", f"Error in log transformation: {str(e)}")
            return None

    def clean_data(self, data):
        """Nettoie les données en gérant les valeurs manquantes et aberrantes"""
        try:
            cleaned = data.copy()

            # Colonnes numériques
            numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # Gestion des valeurs aberrantes
                Q1 = cleaned[col].quantile(0.25)
                Q3 = cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                cleaned.loc[cleaned[col] < lower, col] = cleaned[col].median()
                cleaned.loc[cleaned[col] > upper, col] = cleaned[col].median()

                # Gestion des valeurs manquantes
                cleaned[col].fillna(cleaned[col].median(), inplace=True)

            # Colonnes catégorielles
            categorical_cols = cleaned.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                cleaned[col].fillna(cleaned[col].mode()[0], inplace=True)
                cleaned[col] = cleaned[col].str.upper()

            return cleaned

        except Exception as e:
            messagebox.showerror("Error", f"Error cleaning data: {str(e)}")
            return None
    
    def apply_shapiro_test(self):
        """Applique le test de Shapiro-Wilk et visualise les résultats"""
        if self.df is None or self.selected_column is None:
            messagebox.showerror("Error", "Please load data and select a column first!")
            return

        try:
            data = self.df[self.selected_column].dropna().values
            statistic, p_value = shapiro(data)

            # Calculer les statistiques additionnelles
            skewness = skew(data)
            kurtosis = stats.kurtosis(data)
            mean = np.mean(data)
            std = np.std(data)
            n = len(data)

            # Créer la visualisation
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Histogramme avec courbe normale
            n_bins = min(int(np.sqrt(n)), 30)
            n, bins, patches = ax1.hist(data, bins=n_bins, density=True, 
                                      alpha=0.7, color='lightblue', 
                                      label='Observed Data')

            x = np.linspace(mean - 4*std, mean + 4*std, 100)
            normal_dist = stats.norm.pdf(x, mean, std)
            ax1.plot(x, normal_dist, 'r-', lw=2, 
                    label=f'Normal Distribution\nμ={mean:.2f}\nσ={std:.2f}')

            ax1.set_title('Distribution Plot with Normal Curve')
            ax1.set_xlabel('Values')
            ax1.set_ylabel('Density')
            ax1.legend(loc='upper right')

            # Q-Q plot
            sm.qqplot(data, line='45', ax=ax2, markerfacecolor='lightblue', 
                     alpha=0.7, markeredgecolor='blue')
            ax2.set_title('Q-Q Plot')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Mettre à jour l'interface
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0)

            # Afficher les statistiques détaillées
            stats_text = f"""
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
            stats_frame = ttk.Frame(self.plot_frame)
            stats_frame.grid(row=1, column=0, pady=10)
            ttk.Label(stats_frame, text=stats_text, justify=tk.LEFT,
                     font=('Consolas', 10)).grid(row=0, column=0, padx=20)

        except Exception as e:
            messagebox.showerror("Error", f"Error in Shapiro-Wilk test: {str(e)}")

    def apply_clean_data(self):
        """Nettoie les données en gérant les valeurs manquantes et aberrantes"""
        try:
            self.df = self.clean_data(self.df)
            if self.df is not None:
                self.update_treeview()
                self.update_column_list()
                messagebox.showinfo("Success", "Data cleaned successfully")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def remove_outliers_grubbs(self):
        """Détecte et supprime les valeurs aberrantes avec le test de Grubbs"""
        if self.df is None or self.selected_column is None:
            messagebox.showerror("Error", "Please load data and select a column first!")
            return

        try:
            data = self.df[self.selected_column].dropna().values
            outliers = []
            outlier_indices = []
            
            # Application itérative du test
            while True:
                is_outlier, outlier, idx = apply_grubbs_test(data)
                if not is_outlier:
                    break
                    
                outliers.append(outlier)
                outlier_indices.append(idx)
                data = np.delete(data, idx)
            
            if outliers:
                new_col_name = f"{self.selected_column}_no_outliers"
                self.df[new_col_name] = self.df[self.selected_column].copy()
                self.df.loc[self.df[self.selected_column].isin(outliers), new_col_name] = np.nan
                
                self.update_column_list()
                self.visualize_outliers_removal(self.df[self.selected_column].values, 
                                             outliers, 
                                             outlier_indices)
                
                messagebox.showinfo("Results",
                                  f"Number of outliers detected: {len(outliers)}\n" +
                                  f"Outlier values: {', '.join([f'{x:.2f}' for x in outliers])}")
            else:
                messagebox.showinfo("Results", "No outliers detected using Grubbs test.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error in Grubbs test: {str(e)}")

    def plot_results(self, original_data, transformed_data, title, include_qqplot):
        """
        Visualise les résultats d'une transformation avec des graphiques détaillés.
        
        Parameters:
            original_data (array-like): Données originales
            transformed_data (array-like): Données transformées
            title (str): Titre de la transformation
            include_qqplot (bool): Inclure ou non les Q-Q plots
        """
        # Nettoyer le frame des résultats
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

        # Créer les histogrammes
        self._plot_histogram_with_stats_and_box(ax1, original_data, 
                                              'Original Distribution\n{}'.format(self.selected_column))
        self._plot_histogram_with_stats_and_box(ax2, transformed_data, 
                                              '{} Distribution\n{}'.format(title, self.selected_column))

        # Ajouter les Q-Q plots si nécessaire
        if include_qqplot:
            sm.qqplot(original_data, line='45', ax=ax3)
            ax3.set_title('Q-Q Plot - Original Data')
            sm.qqplot(transformed_data, line='45', ax=ax4)
            ax4.set_title('Q-Q Plot - Transformed Data')

        plt.subplots_adjust(left=0.07, right=0.85, wspace=0.5)

        # Ajouter le plot à l'interface
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)

        # Ajouter les statistiques
        self._add_statistics_text(original_data, transformed_data)

    def visualize_outliers_removal(self, data, outliers, outlier_indices):
        """
        Visualise les données avant et après la suppression des valeurs aberrantes.
        
        Parameters:
            data (array-like): Données originales
            outliers (list): Liste des valeurs aberrantes
            outlier_indices (list): Indices des valeurs aberrantes
        """
        # Nettoyer le frame des résultats
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Créer la visualisation
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Boxplot original avec valeurs aberrantes
        bp1 = ax1.boxplot(data)
        if outliers:
            ax1.plot([1] * len(outliers), outliers, 'ro', label='Outliers')
        ax1.set_title('Original Boxplot\nwith Outliers')
        ax1.legend()

        # Boxplot sans valeurs aberrantes
        clean_data = np.delete(data, outlier_indices)
        ax2.boxplot(clean_data)
        ax2.set_title('Cleaned Boxplot\nwithout Outliers')

        # Distribution plot
        x = range(len(data))
        ax3.scatter(x, data, c='blue', alpha=0.5, label='Normal Values')
        if outliers:
            for idx in outlier_indices:
                ax3.scatter(idx, data[idx], c='red', s=100, 
                          label='Outlier' if idx == outlier_indices[0] else "")
        ax3.set_title('Data Distribution')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Value')
        ax3.legend()

        plt.tight_layout()

        # Ajouter le plot à l'interface
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)

        # Ajouter les statistiques détaillées
        stats_text = f"""
        ─── Outlier Detection Statistics ───────────────────────
        • Total observations: {len(data)}
        • Number of outliers: {len(outliers)}
        • Percentage of outliers: {(len(outliers)/len(data))*100:.2f}%
        
        ─── Original Data Statistics ──────────────────────────
        • Mean: {np.mean(data):.2f}
        • Std Dev: {np.std(data):.2f}
        • Median: {np.median(data):.2f}
        • Range: {np.ptp(data):.2f}
        
        ─── Cleaned Data Statistics ──────────────────────────
        • Mean: {np.mean(clean_data):.2f}
        • Std Dev: {np.std(clean_data):.2f}
        • Median: {np.median(clean_data):.2f}
        • Range: {np.ptp(clean_data):.2f}
        
        ─── Outlier Values ────────────────────────────────────
        {', '.join([f'{x:.2f}' for x in outliers]) if outliers else 'None detected'}
        """

        stats_label = ttk.Label(self.plot_frame, text=stats_text, 
                              justify=tk.LEFT, font=('Consolas', 10))
        stats_label.grid(row=1, column=0, pady=10)

    def _plot_histogram_with_stats_and_box(self, ax, data, title):
        """
        Crée un histogramme détaillé avec superposition statistique et boxplot.
        
        Parameters:
            ax (matplotlib.axes.Axes): Axes pour le plot
            data (array-like): Données à visualiser
            title (str): Titre du graphique
        """
        # Calculer les statistiques
        mean = np.mean(data)
        median = np.median(data)
        mode = float(pd.Series(data).mode().iloc[0])
        std = np.std(data)
        q1, q3 = np.percentile(data, [25, 75])
        skewness = skew(data)

        # Créer l'histogramme
        n_bins = min(int(np.sqrt(len(data))), 30)
        n, bins, patches = ax.hist(data, bins=n_bins, density=False,
                                 alpha=0.7, color='lightblue', 
                                 label='Data Distribution')

        # Ajouter l'estimation par noyau
        kde_x = np.linspace(min(data), max(data), 100)
        kde = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x) * len(data) * (bins[1] - bins[0]), 
                color='blue', linewidth=1, label='Density Curve')

        # Ajouter le boxplot
        bp = ax.boxplot(data, vert=False, positions=[max(n)], 
                       widths=[max(n)/4], patch_artist=True)
        plt.setp(bp['boxes'], facecolor='lightgreen', alpha=0.6)
        plt.setp(bp['medians'], color='darkgreen')
        plt.setp(bp['fliers'], marker='o', markerfacecolor='red', alpha=0.6)

        # Ajouter les lignes verticales pour les statistiques clés
        ax.axvline(mean, color='red', linestyle='--', 
                   label=f'Mean = {mean:.2f}')
        ax.axvline(median, color='green', linestyle='-', 
                   label=f'Median = {median:.2f}')
        ax.axvline(mode, color='purple', linestyle=':', 
                   label=f'Mode = {mode:.2f}')
        ax.axvline(mean + std, color='orange', linestyle='--', 
                   label=f'Mean ± σ ({std:.2f})')
        ax.axvline(mean - std, color='orange', linestyle='--')

        # Mise en forme
        ax.set_title(f"{title}\nSkewness: {skewness:.2f}")
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), 
                 fontsize='small', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    def visualize_dixon_test(self):
        """Visualise les résultats du test de Dixon"""
        if self.df is None or self.selected_column is None:
            messagebox.showerror("Error", "Veuillez charger des données et sélectionner une colonne d'abord!")
            return

        try:
            # Obtenir les données
            data = self.df[self.selected_column].dropna().values

            # Appliquer le test de Dixon
            test_successful, outliers, outlier_indices = apply_dixon_test(data)

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

    def save_to_csv(self):
        """Sauvegarde les données transformées dans un fichier CSV"""
        if self.df is None:
            messagebox.showerror("Error", "Aucune donnée à sauvegarder!")
            return

        try:
            # Ouvrir la boîte de dialogue pour choisir l'emplacement et le nom du fichier
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Sauvegarder les données"
            )

            if file_path:
                # Sauvegarder le DataFrame au format CSV
                self.df.to_csv(file_path, index=False)
                messagebox.showinfo("Succès", "Données sauvegardées avec succès!")

        except Exception as e:
            messagebox.showerror("Error", f"Erreur lors de la sauvegarde: {str(e)}")

    def _add_statistics_text(self, original_data, transformed_data):
        """
        Ajoute un texte détaillé des statistiques pour les données originales et transformées.
        
        Parameters:
            original_data (array-like): Données originales
            transformed_data (array-like): Données transformées
        """
        stats_frame = ttk.Frame(self.plot_frame)
        stats_frame.grid(row=1, column=0, pady=10)

        # Préparer les textes des statistiques
        original_stats = self._format_statistics_text("Original Data", original_data)
        transformed_stats = self._format_statistics_text("Transformed Data", transformed_data)

        # Afficher les statistiques
        ttk.Label(stats_frame, text=original_stats, justify=tk.LEFT,
                 font=('Consolas', 10)).grid(row=0, column=0, padx=20)
        ttk.Label(stats_frame, text=transformed_stats, justify=tk.LEFT,
                 font=('Consolas', 10)).grid(row=0, column=1, padx=20)

    def _format_statistics_text(self, title, data):
        """
        Formate les statistiques descriptives en texte.
        
        Parameters:
            title (str): Titre des statistiques
            data (array-like): Données à analyser
            
        Returns:
            str: Texte formaté des statistiques
        """
        return f"""{title} Statistics:
Mean: {np.mean(data):.4f}
Median: {np.median(data):.4f}
Std Dev: {np.std(data):.4f}
Min: {np.min(data):.4f}
Max: {np.max(data):.4f}
Q1: {np.percentile(data, 25):.4f}
Q3: {np.percentile(data, 75):.4f}
Skewness: {skew(data):.4f}
Kurtosis: {stats.kurtosis(data):.4f}
N: {len(data)}"""

# --- MAIN EXECUTION ---
def set_style():

    """Configure le style de l'interface graphique"""
    style = ttk.Style()
    
    # Configuration générale
    style.configure(".", font=('Helvetica', 10))
    style.configure("TLabel", padding=3)
    style.configure("TButton", padding=6)
    style.configure("TLabelframe", padding=6)
    
    # Style pour les frames principaux
    style.configure("Main.TFrame", padding=10)
    
    # Style pour les cadres de résultats
    style.configure("Results.TLabelframe", padding=10)
    
    # Style pour les boutons de transformation
    style.configure("Transform.TButton",
                   padding=8,
                   font=('Helvetica', 10, 'bold'))
    
    return style

def create_menu(root, app):
    """Crée la barre de menu de l'application"""
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # Menu File
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Load CSV", command=lambda: app.load_file("csv"))
    file_menu.add_command(label="Load Excel", command=lambda: app.load_file("excel"))
    file_menu.add_command(label="Load Example Data", command=app.load_example_data)
    file_menu.add_command(label="Save to CSV", command=app.save_to_csv)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    # Menu Help
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", 
                         command=lambda: show_about_dialog(root))

def show_about_dialog(root):
    """Affiche la boîte de dialogue À propos"""
    about_text = """Data Transformation Tool
Version 1.0

Un outil pour l'analyse et la transformation de données avec:
- Tests statistiques (Shapiro-Wilk, Dixon, Grubbs)
- Transformations de données
- Détection des valeurs aberrantes
- Visualisations interactives
Sous licence GPL 3.0 
© 2024"""
    
    messagebox.showinfo("À propos", about_text)

def main():
    """Fonction principale pour démarrer l'application"""
    try:
        # Créer la fenêtre principale
        root = tk.Tk()
        root.title("Data Transformation Tool")
        
        # Définir l'icône de l'application (si disponible)
        try:
            root.iconbitmap("icon.ico")
        except:
            pass
        
        # Configurer le style
        style = set_style()
        
        # Créer l'application
        app = DataTransformationGUI(root)
        
        # Ajouter le menu
        create_menu(root, app)
        
        # Configurer la taille minimale de la fenêtre
        root.minsize(800, 600)
        
        # Centrer la fenêtre
        window_width = 1200
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Démarrer la boucle principale
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Error", f"Une erreur s'est produite au démarrage: {str(e)}")
        raise

if __name__ == "__main__":
    main()
