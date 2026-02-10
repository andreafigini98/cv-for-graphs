## OPEN-CV to digitalize power graphs

# Come eseguire il tool 

Assicurarsi di avere installato Python nel proprio pc. 

Installare tutte le dipendenze presenti nel file “requirements.txt”. E’ possibile installare le dipendenze con un comando del tipo: 

python -m pip install nome_dipendenza 

Per installare Tesseract OCR seguire i seguenti passaggi:  

Scaricare la verisone più recente di Tesseract OCR per Windows dal sito: 

https://github.com/UB-Mannheim/tesseract/wiki 

Installare Tesseract OCR seguendo le istruzioni 

Aggiungere la variabile d’ambiente: recarsi in Impostazioni > Sistema > Informazioni sul sistema > Impostazioni di sistema avanzate > Variabili d'ambiente e modificare la variabile "PATH" per l'utente aggiungendo il percorso di installazione di Tesseract ORC (es. C:\Program Files\Tesseract-OCR) 

Dalla cartella root del progetto, aprire il terminale ed eseguire: 

python main_gui.py 

Importare i file da analizzare cliccando sul pulsante “Select files” nell’ordine: 

File .pdf contenente la rete delle cabine 

File .xlsx contenente l’elenco delle cabine di competenza 

(Il file “PUNTI DI CONFINE.xlsx” deve essere presente nella cartella input_data) 

Il programma si avvierà in automatico.
