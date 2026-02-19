## OPEN-CV to digitalize power graphs

# Installazione

1) Installare python sul proprio pc.
Si consiglia l'installazione tramite windows store

2) scaricare il progetto da github
Click sul pulsante verde con scritto "codice" (o "code" a seconda della lingua), selezionare downlaod zip nel sotto-menú.
Estrarre il contenuto del codice in una nuova cartella sul proprio pc

3)  Installare tutte le dipendenze del progetto: 
aprire Powershell
navigare alla cartella dove si é salvato il codice estratto dallo zip
eseguire il comando seguente per installare tutte le dipendenze del progetto
```pip install -r "requirements.txt"```

4) installare Tesseract OCR  
Scaricare la verisone più recente di Tesseract OCR per Windows dal sito: https://github.com/UB-Mannheim/tesseract/wiki 
Installare Tesseract OCR seguendo le istruzioni 
Aggiungere la variabile d’ambiente: recarsi in Impostazioni > Sistema > Informazioni sul sistema > Impostazioni di sistema avanzate > Variabili d'ambiente e modificare la variabile "PATH" per l'utente aggiungendo il percorso di installazione di Tesseract ORC (es. C:\Program Files\Tesseract-OCR) 

5) Lanicare il programma
Da powershell, sempre rimanendo nella cartella del progetto, eseguire il seguente comando: 
```python main_gui.py ```
Importare i file da analizzare cliccando sul pulsante “Select files” nell’ordine: 
File .pdf contenente la rete delle cabine 
File .xlsx contenente l’elenco delle cabine di competenza 
(Il file “PUNTI DI CONFINE.xlsx” deve essere presente nella cartella input_data) 
Il programma si avvierà in automatico.
