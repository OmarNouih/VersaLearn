from DataControlInterface import *

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Interface de Contrôle de Données")
    data_control_interface = DataControlInterface(root)
    data_control_interface.pack(expand=True, fill = "both")
    sv_ttk.set_theme("light")
    root.mainloop()