from Biblio import * 
from EditableTable import *
from ModelEvaluationInterface import *


class DataControlInterface(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        sidebar_frame = ttk.Frame(self, relief=tk.SUNKEN, padding=2, style='My.TFrame') 
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)

        import_button = ttk.Button(sidebar_frame, text="Import File", command=self.import_data, width=30, style='My.TButton')
        import_button.pack(pady=10)

        create_button = ttk.Button(sidebar_frame, text="Create New Table", command=self.create_new_table, width=30, style='My.TButton')
        create_button.pack(pady=10)

        self.editable_table = EditableTable(self)
        self.editable_table.pack(side=tk.LEFT, expand=True, fill="both", padx=10, pady=10)

        visualization_button = ttk.Button(sidebar_frame, text="Data Visualization", command=self.perform_visualization, width=30, style='My.TButton')
        visualization_button.pack(pady=10)

        self.data_management_button = ttk.Button(sidebar_frame, text="Data Management", command=self.data_management_dialog, width=30, style='My.TButton')
        self.data_management_button.pack(pady=10, padx=10)

        ml_button = ttk.Button(sidebar_frame, text="Machine Learning Algorithms", command=self.run_ml_algorithms, width=30, style='My.TButton')
        ml_button.pack(pady=10)
        
        self.ml_algorithms_window = None
        
    def data_management_dialog(self):
        if self.editable_table.df is None or self.editable_table.df.empty:
            messagebox.showinfo("Information", "Please import data or create a new table.")
            return

        self.ml_algorithms_window = None

        self.data_management_window = tk.Toplevel(self)
        self.data_management_window.protocol("WM_DELETE_WINDOW", self.data_management_window.destroy)

        # Header frame with buttons
        header_frame = ttk.Frame(self.data_management_window)
        header_frame.pack(pady=10)

        # Button for normalization
        normalization_button = ttk.Button(header_frame, text="Normalization", command=self.show_normalization_options, width=15, style='My.TButton')
        normalization_button.grid(row=0, column=0, padx=0)

        # Button for handling null values
        null_values_button = ttk.Button(header_frame, text="Manage Null Values", command=self.show_null_values_options, width=20, style='My.TButton')
        null_values_button.grid(row=0, column=1, padx=0)

        # Button for encoding categorical variables
        encode_categorical_button = ttk.Button(header_frame, text="Encode Categorical Variables", command=self.encode_categorical_variables, width=30, style='My.TButton')
        encode_categorical_button.grid(row=0, column=2, padx=0)

        # Frame to hold normalization, null values, and encoding options
        self.options_frame = ttk.Frame(self.data_management_window)
        self.options_frame.pack(pady=10)
        
    def perform_visualization(self):
        if self.editable_table.df is None or self.editable_table.df.empty:
            messagebox.showinfo("Information", "Veuillez importer des données ou créer une nouvelle table.")
            return

        self.ml_algorithms_window = None

        # Create a new window for visualization
        visualization_window = tk.Toplevel(self)
        visualization_window.title("Visualisation des Données")
        visualization_window.geometry("800x700")  # Set the window size

        # Create a Notebook to organize visualizations in tabs
        notebook = ttk.Notebook(visualization_window)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Example: Bar chart
        bar_chart_frame = ttk.Frame(notebook)
        notebook.add(bar_chart_frame, text="Bar Chart")

        # Determine appropriate data for bar chart
        numeric_columns = self.editable_table.df.select_dtypes(include=[np.number]).columns
        if numeric_columns.empty:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, "No numeric columns available for the bar chart.", ha="center")
        else:
            plt.figure(figsize=(6, 4))
            plt.bar(numeric_columns, self.editable_table.df[numeric_columns].iloc[0])
            plt.title("Example Bar Chart")
            plt.xlabel("Columns")
            plt.ylabel("Values")

        canvas_chart = FigureCanvasTkAgg(plt.gcf(), master=bar_chart_frame)
        canvas_chart.draw()
        canvas_chart.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add a brief definition in English
        definition_label = tk.Label(bar_chart_frame, text="Bar charts are used to visualize the distribution of values "
                                                          "across different numeric columns in the dataset.")
        definition_label.pack(pady=10)

        # Example: Null values chart
        null_values_frame = ttk.Frame(notebook)
        notebook.add(null_values_frame, text="Null Values Chart")

        null_values_chart = plt.figure(figsize=(6, 4))
        self.editable_table.df.isnull().sum().plot(kind='bar')
        plt.title("Null Values Chart")

        canvas_null = FigureCanvasTkAgg(null_values_chart, master=null_values_frame)
        canvas_null.draw()
        canvas_null.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add a brief definition in English
        definition_label = tk.Label(null_values_frame, text="Null values charts help identify columns with missing data "
                                                            "in the dataset.")
        definition_label.pack(pady=10)

        # Example: Duplicate values chart for all rows
        duplicate_values_frame = ttk.Frame(notebook)
        notebook.add(duplicate_values_frame, text="Duplicate Values Chart")

        # Use duplicated() to find duplicate rows
        duplicate_rows = self.editable_table.df[self.editable_table.df.duplicated()]

        # Plotting the counts of duplicate rows
        plt.figure(figsize=(6, 4))
        duplicate_rows.count().plot(kind='bar')
        plt.title("Duplicate Values Chart")

        canvas_duplicate = FigureCanvasTkAgg(plt.gcf(), master=duplicate_values_frame)
        canvas_duplicate.draw()
        canvas_duplicate.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add a brief definition in English
        definition_label = tk.Label(duplicate_values_frame, text="Duplicate values charts show the count of rows "
                                                                 "with identical data in the dataset.")
        definition_label.pack(pady=10)

        # Example: Data Summary Tables
        data_summary_frame = ttk.Frame(notebook)
        notebook.add(data_summary_frame, text="Data Summary Tables")

        # Display a summary of the dataset
        data_summary_text = tk.Text(data_summary_frame, wrap=tk.WORD, height=10, width=60)
        data_summary_text.pack(pady=10, padx=10)
        data_summary_text.insert(tk.END, str(self.editable_table.df.describe()))

        # Add a brief definition in English
        definition_label = tk.Label(data_summary_frame, text="Data summary tables provide statistical summaries "
                                                             "of the dataset, including mean, min, max, and more.")
        definition_label.pack(pady=10)

        # Example: Correlation Matrix
        correlation_matrix_frame = ttk.Frame(notebook)
        notebook.add(correlation_matrix_frame, text="Correlation Matrix")

        # Select numeric columns for correlation matrix
        numeric_columns_for_correlation = self.editable_table.df.select_dtypes(include=[np.number]).columns
        if numeric_columns_for_correlation.empty:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, "No numeric columns available for the correlation matrix.", ha="center")
        else:
            # Calculate the correlation matrix for numeric columns only
            correlation_matrix = self.editable_table.df[numeric_columns_for_correlation].corr()

            # Plot the correlation matrix using a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
            plt.title("Correlation Matrix")

        canvas_correlation = FigureCanvasTkAgg(plt.gcf(), master=correlation_matrix_frame)
        canvas_correlation.draw()
        canvas_correlation.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add a brief definition in English
        definition_label = tk.Label(correlation_matrix_frame, text="Correlation matrices visualize the correlation "
                                                                    "between different numeric columns in the dataset.")
        definition_label.pack(pady=10)

        # Example: Box Plots
        box_plots_frame = ttk.Frame(notebook)
        notebook.add(box_plots_frame, text="Box Plots")

        # Select numeric columns for box plots
        numeric_columns_for_boxplots = self.editable_table.df.select_dtypes(include=[np.number]).columns
        if numeric_columns_for_boxplots.empty:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, "No numeric columns available for box plots.", ha="center")
        else:
            # Create box plots for each numeric column
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=self.editable_table.df[numeric_columns_for_boxplots])
            plt.title("Box Plots")

        canvas_box_plots = FigureCanvasTkAgg(plt.gcf(), master=box_plots_frame)
        canvas_box_plots.draw()
        canvas_box_plots.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add a brief definition in English
        definition_label = tk.Label(box_plots_frame, text="Box plots provide a graphical summary of the distribution "
                                                           "of data, including the median, quartiles, and outliers.")
        definition_label.pack(pady=10)
        visualization_window.mainloop()
        
    def encode_categorical_variables(self):
        # Destroy existing options frame content
        for widget in self.options_frame.winfo_children():
            widget.destroy()

        # Get a list of categorical columns
        categorical_columns = self.editable_table.df.select_dtypes(include=['object']).columns

        if not categorical_columns.any():
            cat_label = ttk.Label(self.options_frame, text="No categorical variables to encode.", font=('Helvetica', 12))
            cat_label.pack(pady=10)
            return

        # Label for encoding method
        encoding_label = ttk.Label(self.options_frame, text="Encoding Method:", font=('Helvetica', 16, 'bold'))
        encoding_label.pack(pady=5)

        # Options for encoding method
        encoding_options = ["Label Encoding", "One-Hot Encoding"]
        encoding_var = tk.StringVar(self.options_frame)
        encoding_var.set(encoding_options[0])  # Initial value
        encoding_menu = ttk.Combobox(self.options_frame, textvariable=encoding_var, values=encoding_options, state="readonly")
        encoding_menu.pack(pady=10)

        # Button to apply encoding
        apply_button = ttk.Button(self.options_frame, text="Apply", command=lambda: self.apply_encoding(encoding_var.get()))
        apply_button.pack(pady=10)

    def apply_encoding(self, encoding_method):
        # Get a list of categorical columns
        categorical_columns = self.editable_table.df.select_dtypes(include=['object']).columns

        if not categorical_columns.any():
            messagebox.showinfo("Information", "No categorical variables to encode.")
            return

        # Apply the selected encoding method
        if encoding_method == "Label Encoding":
            le = LabelEncoder()
            self.editable_table.df[categorical_columns] = self.editable_table.df[categorical_columns].apply(le.fit_transform)

        elif encoding_method == "One-Hot Encoding":
            self.editable_table.df = pd.get_dummies(self.editable_table.df, columns=categorical_columns)

        # Update the table with the modified data
        self.editable_table.create_data(self.editable_table.df)
        messagebox.showinfo("Success", "Categorical variables encoded successfully.")
        self.data_management_window.destroy()

    def show_normalization_options(self):
        # Destroy existing options frame content
        for widget in self.options_frame.winfo_children():
            widget.destroy()

        # Label for the selection of the normalization method
        normalization_label = ttk.Label(self.options_frame, text="Normalization Methods:", font=('Helvetica', 16, 'bold'))
        normalization_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Note for classification tasks
        classification_note = ttk.Label(self.options_frame, text="Note: For classification tasks, include the target variable for normalization.", font=('Helvetica', 12, 'bold'), foreground='Red')
        classification_note.grid(row=1, column=0, columnspan=2, pady=5)

        # Entry for the target variable (optional)
        target_label = ttk.Label(self.options_frame, text="Target variable name (optional) :", font=('Arial', 12))
        target_label.grid(row=2, column=0, pady=5)

        target_entry = ttk.Entry(self.options_frame)
        target_entry.grid(row=2, column=1, pady=5)

        # Label for normalization options
        normalization_options_label = ttk.Label(self.options_frame, text="Normalization method :", font=('Arial', 12))
        normalization_options_label.grid(row=3, column=0, pady=10)

        # Options for the normalization method
        normalization_options = ["Standardization (Z-score)", "Min-Max Scaling"]
        normalization_var = tk.StringVar(self.options_frame)
        normalization_var.set(normalization_options[0])  # Initial value
        normalization_menu = ttk.Combobox(self.options_frame, textvariable=normalization_var, values=normalization_options, state="readonly")
        normalization_menu.grid(row=3, column=1, pady=10)

        # Button to apply normalization
        apply_button = ttk.Button(self.options_frame, text="Apply", command=lambda: self.apply_normalization(target_entry.get(), normalization_method=normalization_var.get()))
        apply_button.grid(row=4, column=0, columnspan=2, pady=10)

    def show_null_values_options(self):
        # Destroy existing options frame content
        for widget in self.options_frame.winfo_children():
            widget.destroy()

        if self.editable_table.df.isnull().sum().sum() == 0:
            null_info_label = ttk.Label(self.options_frame, text="No null values in columns.", font=('Helvetica', 12))
            null_info_label.pack(pady=10)
            return

        # Get information about null values before applying the method
        null_info_before = self.editable_table.df.isnull().sum()
        null_columns_before = null_info_before[null_info_before > 0].index.tolist()

        # Show the user how many null values and which columns are affected before applying the method
        if null_info_before.sum() > 0:
            null_info_str = "Before applying the method:\n\n"
            null_info_str += f"Total number of null values: {null_info_before.sum()}\n"
            null_info_str += "Columns with null values:\n"
            for col in null_columns_before:
                null_info_str += f"\n- {col}: {null_info_before[col]} null values"

            null_info_label = ttk.Label(self.options_frame, text=null_info_str, font=('Helvetica', 12), foreground='red')
            null_info_label.pack(pady=10)

        missing_values_options = [
            "Remove rows with missing values",
            "Remove columns with missing values",
            "Remove specified columns with missing values",
            "Remove rows and columns with missing values",
            "Replace with mean",
            "Replace with median",
            "Replace with a constant value",
            "Replace with the previous value",
            "Replace with the next value"
        ]

        missing_values_var = tk.StringVar(self.options_frame)
        missing_values_var.set(missing_values_options[0])  # Initial value
        missing_values_menu = ttk.Combobox(self.options_frame, textvariable=missing_values_var, values=missing_values_options, state="readonly")
        missing_values_menu.pack(pady=10)

        # Entry for specifying columns to remove (for the new option)
        self.columns_listbox_label = ttk.Label(self.options_frame, text="Select columns to remove:")
        self.columns_listbox = tk.Listbox(self.options_frame, selectmode=tk.MULTIPLE, height=5)
        self.columns_listbox_label.pack_forget()
        self.columns_listbox.pack_forget()

        # Entry for constant value (for methods that require a constant value)
        self.constant_value_entry_label = ttk.Label(self.options_frame, text="Constant value:")
        self.constant_value_entry = ttk.Entry(self.options_frame)
        self.constant_value_entry_label.pack_forget()
        self.constant_value_entry.pack_forget()

        # Button to apply null values handling
        apply_button = ttk.Button(self.options_frame, text="Apply", command=lambda: self.apply_null_values_handling(
            missing_values_var.get(), self.columns_listbox.curselection(), self.constant_value_entry.get()))
        apply_button.pack(pady=10)

    def on_missing_values_option_selected(self, selected_option_var):
        selected_option = selected_option_var.get()

        if selected_option == "Remove specified columns with missing values":
            self.columns_listbox_label.pack(pady=5)
            self.columns_listbox.delete(0, tk.END)  # Clear previous selections
            for col in self.editable_table.df.columns:
                if self.editable_table.df[col].isnull().any():
                    self.columns_listbox.insert(tk.END, col)
            self.columns_listbox.pack(pady=5)
        else:
            self.columns_listbox_label.pack_forget()
            self.columns_listbox.pack_forget()

        if selected_option == "Replace with a constant value":
            self.constant_value_entry_label.pack(pady=5)
            self.constant_value_entry.pack(pady=5)
        else:
            self.constant_value_entry_label.pack_forget()
            self.constant_value_entry.pack_forget()
                
    def apply_null_values_handling(self, method, selected_columns_indices, constant_value):
        # Show the user how many lines or columns will be affected by the method
        lines_columns_info = ""

        if method.startswith("Remove"):
            lines_columns_info = f"Number of rows before: {len(self.editable_table.df)}\nNumber of columns before: {len(self.editable_table.df.columns)}"

        # Ask for confirmation before applying the method
        confirmation = messagebox.askyesno("Confirmation", f"Are you sure you want to apply this method?\n{lines_columns_info}\nThis can affect the data permanently.")

        if not confirmation:
            return

        # Handle missing values based on the selected method
        if method == "Remove rows with missing values":
            self.editable_table.df = self.editable_table.df.dropna(subset=self.editable_table.df.columns)

        elif method == "Remove columns with missing values":
            self.editable_table.df = self.editable_table.df.dropna(axis=1)

        elif method == "Remove specified columns with missing values":
            # Get the actual column names from the selected indices
            selected_column_names = [self.columns_listbox.get(i) for i in selected_columns_indices]

            # Create a mapping between column names and their indices
            column_indices_mapping = {col: idx for idx, col in enumerate(self.editable_table.df.columns)}

            # Map the selected column names to their actual indices
            specified_columns_list = [column_indices_mapping[col] for col in selected_column_names]

            # Drop the specified columns from the DataFrame
            self.editable_table.df = self.editable_table.df.drop(self.editable_table.df.columns[specified_columns_list], axis=1)

        elif method == "Remove rows and columns with missing values":
            self.editable_table.df = self.editable_table.df.dropna()

        elif method == "Replace with mean":
            # Check for categorical variables before replacing with mean
            categorical_columns = self.editable_table.df.select_dtypes(include=['object']).columns
            if categorical_columns.any():
                messagebox.showinfo("Information", "There are categorical variables. Replacing with mean will be applied after encoding categorical variables with LabelEncoder.")
                le = LabelEncoder()
                self.editable_table.df[categorical_columns] = self.editable_table.df[categorical_columns].apply(le.fit_transform)
            self.editable_table.df = self.editable_table.df.fillna(self.editable_table.df.mean())

        elif method == "Replace with median":
            # Check for categorical variables before replacing with median
            categorical_columns = self.editable_table.df.select_dtypes(include=['object']).columns
            if categorical_columns.any():
                messagebox.showinfo("Information", "There are categorical variables. Replacing with median will be applied after encoding categorical variables with LabelEncoder.")
                le = LabelEncoder()
                self.editable_table.df[categorical_columns] = self.editable_table.df[categorical_columns].apply(le.fit_transform)
            self.editable_table.df = self.editable_table.df.fillna(self.editable_table.df.median())

        elif method == "Replace with a constant value" and constant_value:
            self.editable_table.df = self.editable_table.df.fillna(float(constant_value))

        elif method == "Replace with the previous value":
            self.editable_table.df = self.editable_table.df.fillna(method='ffill')

        elif method == "Replace with the next value":
            self.editable_table.df = self.editable_table.df.fillna(method='bfill')

        # Get information about null values after applying the method
        null_info_after = self.editable_table.df.isnull().sum()
        null_columns_after = null_info_after[null_info_after > 0].index.tolist()

        # Show the user how many null values and which columns are affected after applying the method
        if null_info_after.sum() > 0:
            null_info_str = "After applying the method:\n\n"
            null_info_str += f"Total number of null values: {null_info_after.sum()}\n"
            null_info_str += "Columns with null values:\n"
            for col in null_columns_after:
                null_info_str += f"\n- {col}: {null_info_after[col]} null values"

            null_info_label = ttk.Label(self.options_frame, text=null_info_str, font=('Helvetica', 12), foreground='green')
            null_info_label.pack(pady=10)

        # Update the table with the modified data
        self.editable_table.create_data(self.editable_table.df)
        messagebox.showinfo("Success", "Method applied successfully.")

        # Destroy the data_management_window
        self.data_management_window.destroy()

    def apply_normalization(self, target_variable, normalization_method):
        try:
            if target_variable and target_variable not in self.editable_table.df.columns:
                raise ValueError(f"The target variable '{target_variable}' does not exist in the DataFrame.")

            # Handle normalization
            numeric_columns = self.editable_table.df.select_dtypes(include=[np.number]).columns
            if target_variable in numeric_columns:
                numeric_columns = numeric_columns.drop(target_variable) if target_variable else numeric_columns

            if normalization_method == "Standardization (Z-score)":
                scaler = StandardScaler()
                self.editable_table.df[numeric_columns] = scaler.fit_transform(self.editable_table.df[numeric_columns])
            elif normalization_method == "Min-Max Scaling":
                self.editable_table.df[numeric_columns] = (self.editable_table.df[numeric_columns] - self.editable_table.df[numeric_columns].min()) / (self.editable_table.df[numeric_columns].max() - self.editable_table.df[numeric_columns].min())

            # Update the table with the modified data
            self.editable_table.create_data(self.editable_table.df)
            messagebox.showinfo("Success", "Normalization applied successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error during normalization: {str(e)}")
        finally:
            self.data_management_window.destroy()

    def import_data(self):
        file_path = filedialog.askopenfilename(title="Choisissez un fichier", filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")])

        if file_path:
            try:
                new_data = pd.read_csv(file_path)
                self.editable_table.create_data(new_data)
                messagebox.showinfo("Succès", "Données importées avec succès.")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'importation des données :\n{str(e)}")
                print(str(e))

    def create_new_table(self):
        # Create a new top-level window for input
        input_dialog_window = tk.Toplevel(self)

        # Step 1: Ask for the number of rows
        rows_var = tk.IntVar()
        ttk.Label(input_dialog_window, text="Nombre de lignes :", font=('Helvetica', 13, 'bold')).pack(pady=10)
        ttk.Entry(input_dialog_window, textvariable=rows_var).pack(pady=10)

        # Function to proceed to the next step
        def next_step():
            input_dialog_window.destroy()
            self.create_table_step2(rows_var.get())

        # Button to proceed to the next step
        ttk.Button(input_dialog_window, text="Suivant", command=next_step).pack(pady=10)

    def create_table_step2(self, rows):
        # Create a new top-level window for input
        input_dialog_window = tk.Toplevel(self)

        # Step 2: Ask for the number of columns
        columns_var = tk.IntVar()
        ttk.Label(input_dialog_window, text="Nombre de colonnes :", font=('Helvetica', 13, 'bold')).pack(pady=10)
        ttk.Entry(input_dialog_window, textvariable=columns_var).pack(pady=10)

        # Function to proceed to the next step
        def next_step():
            input_dialog_window.destroy()
            self.create_table_step3(rows, columns_var.get())

        # Button to proceed to the next step
        ttk.Button(input_dialog_window, text="Suivant", command=next_step).pack(pady=10)

    def create_table_step3(self, rows, columns):
        # Create a new top-level window for input
        input_dialog_window = tk.Toplevel(self)

        # Step 3: Ask for column names and types
        entries_frame = ttk.Frame(input_dialog_window)
        entries_frame.pack(pady=10)

        # List to store column information
        column_info = []

        for i in range(columns):
            ttk.Label(entries_frame, text=f"Colonne {i + 1} :", font=('Helvetica', 12)).grid(row=i, column=0, padx=5, pady=5, sticky=tk.E)
            col_name_entry = tk.Entry(entries_frame)
            col_name_entry.grid(row=i, column=1, padx=5, pady=5)

            col_type_options = ["int", "float", "str", "datetime"] 
            col_type_var = tk.StringVar(input_dialog_window)
            col_type_var.set(col_type_options[0])
            col_type_menu = ttk.OptionMenu(entries_frame, col_type_var, *col_type_options)
            col_type_menu.grid(row=i, column=2, padx=5, pady=5)

            column_info.append((col_name_entry, col_type_var))

        # Function to create the table
        def create_table():
            if all(entry.get() for entry, _ in column_info):
                # Create a new DataFrame with the specified number of rows and columns
                columns_names, columns_types = zip(*[(entry.get(), col_type.get()) for entry, col_type in column_info])
                new_data = pd.DataFrame(index=range(rows), columns=columns_names)

                # Convert each column with NaN handling
                for col, col_type in zip(columns_names, columns_types):
                    if col_type == "float":
                        new_data[col] = new_data[col].astype(float)
                    elif col_type == "int":
                        new_data[col] = new_data[col].astype(float).astype(int, errors='ignore')
                    else:
                        new_data[col] = new_data[col].astype(col_type)

                self.editable_table.create_data(new_data)
                messagebox.showinfo("Succès", "Nouvelle table créée avec succès.")
                input_dialog_window.destroy()
            else:
                messagebox.showwarning("Attention", "Veuillez saisir tous les noms de colonnes.")

        # Button to apply the selected methods
        ttk.Button(input_dialog_window, text="Appliquer", command=create_table).pack(pady=10)

    def run_ml_algorithms(self):
        if self.editable_table.df is None or self.editable_table.df.empty:
            messagebox.showinfo("Information", "Please import data or create a new table before using machine learning algorithms.")
            return

        if self.ml_algorithms_window is None:
            self.ml_algorithms_window = tk.Toplevel(self)
            self.ml_algorithms_window.title("Machine Learning Algorithms")
            self.show_algorithm_selection()  
    
    def show_algorithm_selection(self):
        for widget in self.ml_algorithms_window.winfo_children():
            widget.destroy()

        # Label for algorithm selection
        label = ttk.Label(self.ml_algorithms_window, text="Select the Machine Learning Algorithm:", font=('Helvetica', 13, 'bold'))
        label.pack(pady=10)

        # OptionMenu for algorithm selection
        algorithm_options = ["Linear Regression", "Decision Trees (Regression)", "Decision Trees (Classification)", "Support Vector Machine (Regression)", "Support Vector Machine (Classification)", "K-means", "K-Nearest Neighbors (Regression)", "K-Nearest Neighbors (Classification)", "Random Forest (Regression)", "Random Forest (Classification)", "Artificial Neural Networks (Regression)", "Artificial Neural Networks (Classification)"]
        self.algorithm_var = tk.StringVar(self.ml_algorithms_window)
        self.algorithm_var.set(algorithm_options[0])
        algorithm_menu = ttk.Combobox(self.ml_algorithms_window, textvariable=self.algorithm_var, values=algorithm_options, state='readonly')
        algorithm_menu.pack(pady=10)

        # Entry for target variable selection
        target_label = ttk.Label(self.ml_algorithms_window, text="Target Variable Name: ", font=('Helvetica', 13, 'bold'))
        target_label.pack(pady=5)

        target_entry = ttk.Entry(self.ml_algorithms_window)
        target_entry.pack(pady=10)

        # Button to show parameters and allow modifications
        show_parameters_button = ttk.Button(self.ml_algorithms_window, text="Show Parameters", command=self.show_parameters)
        show_parameters_button.pack(pady=10)

        # Button to run algorithm with default parameters
        run_algorithm_button = ttk.Button(self.ml_algorithms_window, text="Run Algorithm with Default Parameters", command=lambda: self.run_selected_algorithm(self.algorithm_var.get(), target_entry.get()))
        run_algorithm_button.pack(pady=10)

    def show_parameters(self):
        for widget in self.ml_algorithms_window.winfo_children():
            widget.destroy()

        selected_algorithm = self.algorithm_var.get()

        # Label for algorithm selection
        label = ttk.Label(self.ml_algorithms_window, text=f"Selected Algorithm: {selected_algorithm}", font=('Helvetica', 13, 'bold'))
        label.pack(pady=10)

        # Entry for target variable selection
        target_label = ttk.Label(self.ml_algorithms_window, text="Target Variable Name:", font=('Helvetica', 13, 'bold'))
        target_label.pack(pady=5)

        target_entry = ttk.Entry(self.ml_algorithms_window)
        target_entry.pack(pady=10)

        # Specific parameters for Linear Regression
        if selected_algorithm == "Linear Regression":
            # Label for fit_intercept
            fit_intercept_label = ttk.Label(self.ml_algorithms_window, text="Fit Intercept:", font=('Helvetica', 13, 'bold'))
            fit_intercept_label.pack(pady=5)

            # OptionMenu for fit_intercept
            fit_intercept_options = ["True", "False"]
            fit_intercept_var = tk.StringVar(self.ml_algorithms_window)
            fit_intercept_var.set(fit_intercept_options[0])
            fit_intercept_menu = ttk.Combobox(self.ml_algorithms_window, textvariable=fit_intercept_var, values=fit_intercept_options, state='readonly')
            fit_intercept_menu.pack(pady=5)

        # Specific parameters for Decision Trees (Regression) and Decision Trees (Classification)
        elif selected_algorithm in ["Decision Trees (Regression)", "Decision Trees (Classification)"]:
            # Label for max_depth
            max_depth_label = ttk.Label(self.ml_algorithms_window, text="Max Depth:", font=('Helvetica', 13, 'bold'))
            max_depth_label.pack(pady=5)

            # Entry for max_depth
            max_depth_entry = ttk.Entry(self.ml_algorithms_window)
            max_depth_entry.pack(pady=5)

            # Label for min_samples_split
            min_samples_split_label = ttk.Label(self.ml_algorithms_window, text="Min Samples Split:", font=('Helvetica', 13, 'bold'))
            min_samples_split_label.pack(pady=5)

            # Entry for min_samples_split
            min_samples_split_entry = ttk.Entry(self.ml_algorithms_window)
            min_samples_split_entry.pack(pady=5)

            # Label for min_samples_leaf
            min_samples_leaf_label = ttk.Label(self.ml_algorithms_window, text="Min Samples Leaf:", font=('Helvetica', 13, 'bold'))
            min_samples_leaf_label.pack(pady=5)

            # Entry for min_samples_leaf
            min_samples_leaf_entry = ttk.Entry(self.ml_algorithms_window)
            min_samples_leaf_entry.pack(pady=5)

            # Specific parameters for Decision Trees (Classification)
            if selected_algorithm == "Decision Trees (Classification)":
                # Label for criterion
                criterion_label = ttk.Label(self.ml_algorithms_window, text="Criterion:", font=('Helvetica', 13, 'bold'))
                criterion_label.pack(pady=5)

                # OptionMenu for criterion
                criterion_options = ['entropy', 'gini', 'log_loss']
                criterion_var = tk.StringVar(self.ml_algorithms_window)
                criterion_var.set(criterion_options[0])
                criterion_menu = ttk.Combobox(self.ml_algorithms_window, textvariable=criterion_var, values=criterion_options, state='readonly')
                criterion_menu.pack(pady=5)

        # Specific parameters for Support Vector Machine (Regression) and Support Vector Machine (Classification)
        elif selected_algorithm in ["Support Vector Machine (Regression)", "Support Vector Machine (Classification)"]:
            # Label for kernel
            kernel_label = ttk.Label(self.ml_algorithms_window, text="Kernel:", font=('Helvetica', 13, 'bold'))
            kernel_label.pack(pady=5)

            # OptionMenu for kernel
            kernel_options = ['precomputed', 'rbf', 'poly', 'sigmoid', 'linear']
            kernel_var = tk.StringVar(self.ml_algorithms_window)
            kernel_var.set(kernel_options[0])
            kernel_menu = ttk.Combobox(self.ml_algorithms_window, textvariable=kernel_var, values=kernel_options, state='readonly')
            kernel_menu.pack(pady=5)

            # Label for C (regularization parameter)
            c_label = ttk.Label(self.ml_algorithms_window, text="C (Regularization Parameter):", font=('Helvetica', 13, 'bold'))
            c_label.pack(pady=5)

            # Entry for C
            c_entry = ttk.Entry(self.ml_algorithms_window)
            c_entry.pack(pady=5)

        # Specific parameters for K-means
        elif selected_algorithm == "K-means":
            # Label for n_clusters
            n_clusters_label = ttk.Label(self.ml_algorithms_window, text="Number of Clusters:", font=('Helvetica', 13, 'bold'))
            n_clusters_label.pack(pady=5)

            # Entry for n_clusters
            n_clusters_entry = ttk.Entry(self.ml_algorithms_window)
            n_clusters_entry.pack(pady=5)

            # Label for init
            init_label = ttk.Label(self.ml_algorithms_window, text="Initialization Method:", font=('Helvetica', 13, 'bold'))
            init_label.pack(pady=5)

            # OptionMenu for init
            init_options = ["k-means++", "random"]
            init_var = tk.StringVar(self.ml_algorithms_window)
            init_var.set(init_options[0])
            init_menu = ttk.Combobox(self.ml_algorithms_window, textvariable=init_var, values=init_options, state='readonly')
            init_menu.pack(pady=5)

        # Specific parameters for K-Nearest Neighbors
        elif selected_algorithm in ["K-Nearest Neighbors (Regression)", "K-Nearest Neighbors (Classification)"]: 
            # Label for n_neighbors
            n_neighbors_label = ttk.Label(self.ml_algorithms_window, text="Number of Neighbors:", font=('Helvetica', 13, 'bold'))
            n_neighbors_label.pack(pady=5)

            # Entry for n_neighbors
            n_neighbors_entry = ttk.Entry(self.ml_algorithms_window)
            n_neighbors_entry.pack(pady=5)

            # Label for algorithm
            algorithm_label = ttk.Label(self.ml_algorithms_window, text="Algorithm:", font=('Helvetica', 13, 'bold'))
            algorithm_label.pack(pady=5)

            # OptionMenu for algorithm
            algorithm_options = ['auto', 'ball_tree', 'brute', 'kd_tree']
            algorithm_var = tk.StringVar(self.ml_algorithms_window)
            algorithm_var.set(algorithm_options[0])
            algorithm_menu = ttk.Combobox(self.ml_algorithms_window, textvariable=algorithm_var, values=algorithm_options, state='readonly')
            algorithm_menu.pack(pady=5)

        # Specific parameters for Random Forest
        elif selected_algorithm in ["Random Forest (Regression)", "Random Forest (Classification)"]:
            # Label for n_estimators
            n_estimators_label = ttk.Label(self.ml_algorithms_window, text="Number of Estimators:", font=('Helvetica', 13, 'bold'))
            n_estimators_label.pack(pady=5)

            # Entry for n_estimators
            n_estimators_entry = ttk.Entry(self.ml_algorithms_window)
            n_estimators_entry.pack(pady=5)

            # Label for max_depth
            max_depth_label = ttk.Label(self.ml_algorithms_window, text="Max Depth:", font=('Helvetica', 13, 'bold'))
            max_depth_label.pack(pady=5)

            # Entry for max_depth
            max_depth_entry = ttk.Entry(self.ml_algorithms_window)
            max_depth_entry.pack(pady=5)

            # Label for min_samples_split
            min_samples_split_label = ttk.Label(self.ml_algorithms_window, text="Min Samples Split:", font=('Helvetica', 13, 'bold'))
            min_samples_split_label.pack(pady=5)

            # Entry for min_samples_split
            min_samples_split_entry = ttk.Entry(self.ml_algorithms_window)
            min_samples_split_entry.pack(pady=5)

        # Specific parameters for Artificial Neural Networks
        elif selected_algorithm in ["Artificial Neural Networks (Regression)", "Artificial Neural Networks (Classification)"]:
            # Label for hidden_layer_sizes
            hidden_layer_sizes_label = ttk.Label(self.ml_algorithms_window, text="Hidden Layer Sizes:", font=('Helvetica', 13, 'bold'))
            hidden_layer_sizes_label.pack(pady=5)

            # Entry for hidden_layer_sizes
            hidden_layer_sizes_entry = ttk.Entry(self.ml_algorithms_window)
            hidden_layer_sizes_entry.pack(pady=5)

            # Label for activation
            activation_label = ttk.Label(self.ml_algorithms_window, text="Activation Function:", font=('Helvetica', 13, 'bold'))
            activation_label.pack(pady=5)

            # OptionMenu for activation
            activation_options = ["identity", "logistic", "tanh", "relu"]
            activation_var = tk.StringVar(self.ml_algorithms_window)
            activation_var.set(activation_options[0])
            activation_menu = ttk.Combobox(self.ml_algorithms_window, textvariable=activation_var, values=activation_options, state='readonly')
            activation_menu.pack(pady=5)

            # Label for solver
            solver_label = ttk.Label(self.ml_algorithms_window, text="Solver:", font=('Helvetica', 13, 'bold'))
            solver_label.pack(pady=5)

            # OptionMenu for solver
            solver_options = ["lbfgs", "sgd", "adam"]
            solver_var = tk.StringVar(self.ml_algorithms_window)
            solver_var.set(solver_options[0])
            solver_menu = ttk.Combobox(self.ml_algorithms_window, textvariable=solver_var, values=solver_options, state='readonly')
            solver_menu.pack(pady=5)

        # Button to run algorithm with modified parameters
        run_algorithm_button = ttk.Button(self.ml_algorithms_window, text="Run Algorithm with Modified Parameters", command=lambda: self.run_algorithm_with_parameters(
            selected_algorithm,
            target_entry.get(),
            bool(fit_intercept_var.get()) if selected_algorithm == "Linear Regression" else None,
            int(max_depth_entry.get()) if selected_algorithm in ["Decision Trees (Regression)", "Decision Trees (Classification)"] else None,
            int(min_samples_split_entry.get()) if selected_algorithm in ["Random Forest (Regression)", "Random Forest (Classification)", "Decision Trees (Regression)", "Decision Trees (Classification)", "Random Forest (Regression)", "Random Forest (Classification)"] else None,
            int(min_samples_leaf_entry.get()) if selected_algorithm in ["Decision Trees (Regression)", "Decision Trees (Classification)"] else None,
            criterion_var.get() if selected_algorithm == "Decision Trees (Classification)" else None,
            kernel_var.get() if selected_algorithm in ["Support Vector Machine (Regression)", "Support Vector Machine (Classification)"] else None,
            float(c_entry.get()) if selected_algorithm in ["Support Vector Machine (Regression)", "Support Vector Machine (Classification)"] else None,
            int(n_clusters_entry.get()) if selected_algorithm == "K-means" else None,
            init_var.get() if selected_algorithm == "K-means" else None,
            int(n_neighbors_entry.get()) if selected_algorithm in ["K-Nearest Neighbors (Regression)", "K-Nearest Neighbors (Classification)"] else None,
            str(algorithm_var.get()) if selected_algorithm in ["K-Nearest Neighbors (Regression)", "K-Nearest Neighbors (Classification)"] else None,
            int(n_estimators_entry.get()) if selected_algorithm in ["Random Forest (Regression)", "Random Forest (Classification)"] else None,
            tuple(map(int, hidden_layer_sizes_entry.get().split(','))) if selected_algorithm in ["Artificial Neural Networks (Regression)", "Artificial Neural Networks (Classification)"] else None,
            activation_var.get() if selected_algorithm in ["Artificial Neural Networks (Regression)", "Artificial Neural Networks (Classification)"] else None,
            solver_var.get() if selected_algorithm in ["Artificial Neural Networks (Regression)", "Artificial Neural Networks (Classification)"] else None
        ))
        run_algorithm_button.pack(pady=10)

        back_button = ttk.Button(self.ml_algorithms_window, text="Back to Algorithm Selection", command=self.show_algorithm_selection)
        back_button.pack(pady=10)

    def run_selected_algorithm(self, selected_algorithm, target_variable):
        # Check if the target variable is provided
        if not target_variable:
            messagebox.showwarning("Attention", "Veuillez saisir le nom de la variable cible.")
            return
        if target_variable not in self.editable_table.df.columns:
            messagebox.showwarning("Attention", f"La variable cible '{target_variable}' n'existe pas dans le DataFrame.")
            return
        
        if self.editable_table.df.isna().any().any():
            messagebox.showwarning("Attention", "Le DataFrame contient des valeurs manquantes (NaN). Veuillez traiter les valeurs manquantes avant de continuer.")
            return

        X = self.editable_table.df.drop(target_variable, axis=1)
        y = self.editable_table.df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
        
        if selected_algorithm in ["Decision Trees (Classification)", "Support Vector Machine (Classification)", "K-Nearest Neighbors (classification)", "Random Forest (classification)", "Artificial Neural Networks (classification)"]:
            if not all(isinstance(value, int) or isinstance(value, np.integer) for value in y):
                messagebox.showerror("Error", "The selected algorithm is for classification, but the target variable contains non-integer values. Please choose a regression algorithm for continuous target variables.")
                return
            
            if not all(isinstance(value, (int, float)) for value in y) or not all(isinstance(value, (int, float)) for column in X.columns for value in X[column]):
                messagebox.showinfo("Message", "Label encoding is recommended for better performance with the selected algorithm.")
                return
            
        if selected_algorithm == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)

        elif selected_algorithm == "Decision Trees (Regression)":
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)

        elif selected_algorithm == "Decision Trees (Classification)":
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)

        elif selected_algorithm == "Support Vector Machine (Regression)":
            model = SVR()
            model.fit(X_train, y_train)

        elif selected_algorithm == "Support Vector Machine (Classification)":
            model = SVC()
            model.fit(X_train, y_train)

        elif selected_algorithm == "K-means":
            model = KMeans()
            model.fit(X_train)

        elif selected_algorithm == "K-Nearest Neighbors (Regression)":
            model = KNeighborsRegressor() 
            model.fit(X_train, y_train)

        elif selected_algorithm == "K-Nearest Neighbors (classification)":
            model = KNeighborsClassifier()  
            model.fit(X_train, y_train)

        elif selected_algorithm == "Random Forest (Regression)":
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            
        elif selected_algorithm == "Random Forest (classification)":
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

        elif selected_algorithm == "Artificial Neural Networks (regression)":
            model = MLPRegressor()
            model.fit(X_train, y_train)

        elif selected_algorithm == "Artificial Neural Networks (classification)":
            model = MLPClassifier()  
            model.fit(X_train, y_train)

        model_evaluation_window = ModelEvaluationInterface(self, selected_algorithm, target_variable, model, X_train, X_test, y_train, y_test)
        self.ml_algorithms_window.destroy()
        self.ml_algorithms_window = None

    def run_algorithm_with_parameters(self, algorithm, target_variable, fit_intercept=None, max_depth=None, min_samples_split=None, min_samples_leaf=None, criterion=None, kernel=None, c=None, n_clusters=None, init=None, n_neighbors=None, algorithm_var=None, n_estimators=None, hidden_layer_sizes=None, activation=None, solver=None):
        # Check if the target variable is provided
        if not target_variable:
            messagebox.showwarning("Attention", "Veuillez saisir le nom de la variable cible.")
            return
        if target_variable not in self.editable_table.df.columns:
            messagebox.showwarning("Attention", f"La variable cible '{target_variable}' n'existe pas dans le DataFrame.")
            return
        if self.editable_table.df.isna().any().any():
            messagebox.showwarning("Attention", "Le DataFrame contient des valeurs manquantes (NaN). Veuillez traiter les valeurs manquantes avant de continuer.")
            return

        X = self.editable_table.df.drop(target_variable, axis=1)
        y = self.editable_table.df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if algorithm in ["Decision Trees (Classification)", "Support Vector Machine (Classification)", "K-Nearest Neighbors (classification)", "Random Forest (classification)", "Artificial Neural Networks (classification)"]:
            if not all(isinstance(value, int) or isinstance(value, np.integer) for value in y):
                messagebox.showerror("Error", "The selected algorithm is for classification, but the target variable contains non-integer values. Please choose a regression algorithm for continuous target variables.")
                return
            
            if not all(isinstance(value, (int, float)) for value in y) or not all(isinstance(value, (int, float)) for column in X.columns for value in X[column]):
                messagebox.showinfo("Message", "Label encoding is recommended for better performance with the selected algorithm.")
                return

        if algorithm == "Linear Regression":
            model = LinearRegression(fit_intercept=fit_intercept)

        elif algorithm == "Decision Trees (Regression)":
            model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

        elif algorithm == "Decision Trees (Classification)":
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion)

        elif algorithm == "Support Vector Machine (Regression)":
            model = SVR(kernel=kernel, C=c)

        elif algorithm == "Support Vector Machine (Classification)":
            model = SVC(kernel=kernel, C=c)

        elif algorithm == "K-means":
            model = KMeans(n_clusters=n_clusters, init=init)

        elif algorithm == "K-Nearest Neighbors (Regression)":
            print(algorithm_var)
            model = KNeighborsRegressor(n_neighbors=n_neighbors, algorithm=algorithm_var)

        elif algorithm == "K-Nearest Neighbors (Classification)":
            model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm_var)

        elif algorithm == "Random Forest (Regression)":
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

        elif algorithm == "Random Forest (Classification)":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

        elif algorithm == "Artificial Neural Networks (Regression)":
            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver)

        elif algorithm == "Artificial Neural Networks (Classification)":
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver)

        model.fit(X_train, y_train)

        model_evaluation_window = ModelEvaluationInterface(self, algorithm, target_variable, model, X_train, X_test, y_train, y_test)
        self.ml_algorithms_window.destroy()
        self.ml_algorithms_window = None
