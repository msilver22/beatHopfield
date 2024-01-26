#Standard libraries
import sounddevice as sd
import numpy as np
import os
import threading
import soundfile as sf
#GUI libraries
import tkinter as tk
from tkinter import ttk
import tkinter.simpledialog
import tkinter.messagebox



class VoiceRecordingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Voice Recording App")

        # Imposta uno stile personalizzato per un tema scuro
        self.style = ttk.Style()
        self.style.configure('TButton', foreground='white', background='#333')  # Testo bianco, sfondo grigio scuro
        self.style.configure('TLabel', foreground='white', background='#222')   # Testo bianco, sfondo grigio più scuro


        # Modifica le dimensioni della finestra
        self.master.geometry("600x400")

        # Ottieni le dimensioni dello schermo
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # Calcola le coordinate x, y per centrare la finestra
        x = (screen_width - 600) // 2
        y = (screen_height - 400) // 2

        # Imposta la posizione della finestra centrata
        self.master.geometry(f"600x400+{x}+{y}")

        #Bottoni interattivi
        self.record_button = ttk.Button(self.master, text="Start Recording", command=self.start_recording)
        self.record_button.pack(pady=10)

        self.stop_button = ttk.Button(self.master, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.play_button = ttk.Button(self.master, text="Play Audio", command=self.play_audio, state=tk.DISABLED)
        self.play_button.pack(pady=10)

        self.confirm_button = ttk.Button(self.master, text="Confirm Audio", command=self.confirm_audio, state=tk.DISABLED)
        self.confirm_button.pack(pady=10)

        self.retry_button = ttk.Button(self.master, text="Retry Recording", command=self.retry_recording, state=tk.DISABLED)
        self.retry_button.pack(pady=10)

        self.status_label = tk.Label(self.master, text="Status: Not recording", background='#222', foreground='white')
        self.status_label.pack(pady=10)

        self.quit_button = ttk.Button(self.master, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)

        self.frames = []
        self.recording_thread = None
        self.playing_thread = None

    def start_recording(self):
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.play_button.config(state=tk.DISABLED)
        self.confirm_button.config(state=tk.DISABLED)
        self.retry_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Recording...")

        # Avvia la registrazione in un thread separato
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

    def record_audio(self):
        # Impostazioni di registrazione audio
        sample_rate = 48000
        duration = 2  # Durata della registrazione in secondi

        # Registrazione audio
        self.frames = []
        with sd.InputStream(callback=self.callback, channels=1, samplerate=sample_rate):
            sd.sleep(duration * 1000)

        # Abilita i pulsanti appropriati dopo la registrazione
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.NORMAL)
        self.confirm_button.config(state=tk.NORMAL)
        self.retry_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Recording completed")

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.frames.append(indata.copy())

    def stop_recording(self):
        # Ferma l'audio
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.NORMAL)
        self.confirm_button.config(state=tk.NORMAL)
        self.retry_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Not recording")

    def play_audio(self):
        # Riproduci l'audio registrato in un thread separato
        self.playing_thread = threading.Thread(target=self.play_audio_thread)
        self.playing_thread.start()

    def play_audio_thread(self):
        # Riproduci l'audio
        sd.play(np.concatenate(self.frames), samplerate=48000)
        sd.wait()

    def confirm_audio(self):
        # Chiedi all'utente di inserire un nome per il file
        file_name = tkinter.simpledialog.askstring("Save Recording", "Enter a name for the recording file:")
        if not file_name:
            # L'utente ha cliccato "Annulla" o ha lasciato il campo vuoto
            return

        # Salva l'audio registrato su un file
        self.save_audio(file_name)

        # Disabilita i pulsanti di conferma e riprova
        self.confirm_button.config(state=tk.DISABLED)
        self.retry_button.config(state=tk.DISABLED)

        # Mostra un messaggio di conferma
        tk.messagebox.showinfo("Success", "Voice recording saved successfully.")

        # Torna allo stato iniziale con solo il pulsante di registrazione attivo
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Ready to record")

    def retry_recording(self):
        # Cancella i frame registrati
        self.frames = []

        # Disabilita i pulsanti di conferma e riprova
        self.confirm_button.config(state=tk.DISABLED)
        self.retry_button.config(state=tk.DISABLED)

        # Avvia una nuova registrazione
        self.start_recording()

    def save_audio(self,file_name):
        folder_path = "/Users/silver22/Desktop/registrazioni"
        os.makedirs(folder_path, exist_ok=True)

        # Genera un percorso completo per il file audio
        file_path = os.path.join(folder_path, f"{file_name}.wav")

        # Salva il file audio utilizzando la libreria soundfile
        sf.write(file_path, np.concatenate(self.frames), 48000, 'PCM_16')

    def quit_app(self):
        # Chiudi l'applicazione
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceRecordingApp(root)
    root.mainloop()
