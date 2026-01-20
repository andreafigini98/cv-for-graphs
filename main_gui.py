import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from prova_fixed_dimension import main


def select_files():
    file_path1 = filedialog.askopenfilename(title="Select image")
    if not file_path1:
        return

    file_path2 = filedialog.askopenfilename(title="Select excel")
    if not file_path2:
        return

    # Run main() in a thread to avoid freezing GUI
    threading.Thread(
        target=lambda: run_main(file_path1, file_path2), daemon=True
    ).start()


def run_main(file_image, file_excel):
    # Clear previous messages
    messages_text.config(state="normal")
    messages_text.delete("1.0", tk.END)
    messages_text.config(state="disabled")

    def gui_print(msg):
        messages_text.config(state="normal")
        messages_text.insert(tk.END, msg + "\n")
        messages_text.see(tk.END)  # auto scroll
        messages_text.config(state="disabled")

    # Call main and pass gui_print as debug callback
    main(file_image, file_excel, debug_callback=gui_print)
    gui_print("Processing finished!")


root = tk.Tk()
root.title("File Input App")

tk.Button(root, text="Select files", command=select_files).pack(padx=20, pady=10)

# Text widget for debug messages
messages_text = tk.Text(root, height=20, width=80, state="disabled")
messages_text.pack(padx=10, pady=5)

root.mainloop()
