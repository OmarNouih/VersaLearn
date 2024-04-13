from Table import *
from Biblio import *

class EditableTable(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.parent = parent
        self.message_label = ttk.Label(self, text="No data imported yet.", font=('Helvetica', 12), foreground='black')
        self.message_label.pack(expand=True)
        self.df = pd.DataFrame()

        self.pt = Table(self, showstatusbar= True, width=800, height = 400)
        self.pt.pack(expand=True, fill="both")

    def create_data(self, data):
        for widget in self.winfo_children():
            widget.destroy()
        self.df = pd.DataFrame(data)
        self.pt = Table(self, dataframe=self.df, showtoolbar=False, showstatusbar=True, width=800, height=400)
        self.pt.show()

    def import_data(self):
        file_path = filedialog.askopenfilename(title="Choose a file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])

        if file_path:
            try:
                new_data = pd.read_csv(file_path)
                self.create_data(new_data)
                messagebox.showinfo("Success", "Data imported successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Error during data import:\n{str(e)}")
                print(str(e))

    def show_table(self):
        self.pack(side=tk.LEFT, expand=True, fill="both", padx=10, pady=10)
        self.pt.show()