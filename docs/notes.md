# Notizen

**Frage:** Wo KI / Computer Vision an den Kamerastream haengen und wo der Beamer-Inhalt entsteht?

**Antwort:**
- **CV / Erkennung:** Im `while True`-Loop, **nach** `dewarped_img` (oder anderem gewaehltem Eingangsbild), **vor** `cv2.imshow("Beamer_Window", BeamerImage)`.
- **Beamer "bearbeiten":** Alles Sichtbare kommt aus **`BeamerImage`** (`np.zeros`, Zeichnungen, Overlays) - KI-Ergebnisse dort einzeichnen bzw. daraus steuern.

**Frage:** Stimmen die Pixel-Koordinaten von Erkennung und Beamer ueberein?

**Antwort:** Ja. YOLO (und jede andere Erkennung) laeuft auf `dewarped_img` — dem bereits gewarpten Bild in Beamer-Koordinaten. Die Koordinaten die YOLO zurueckgibt sind deshalb direkt identisch mit den Beamer-Pixeln. Pixel (300, 200) bei YOLO = Pixel (300, 200) auf dem Beamer. Kein Umrechnen noetig.

**Frage:** Wo und wie werden die ArUco-Marker erkannt — OpenCV oder manuell fuer die Speziellen?

**Antwort:** Erkennung laeuft ueber **OpenCV** (`cv2.aruco`): vorgegebenes Woerterbuch z.B. `DICT_5X5_50`, dann `detectMarkers` auf dem (gewarpten) Graustufenbild. Welche Marker *bedeuten* was (Ecken 0–3, Overlay bei ID 5 usw.), steht **im Code** als feste IDs / Logik — keine separate Trainings-KI, nur Dictionary + Parameter passend zu den gedruckten/generierten Markern halten.

---

# Pipeline - Schritt fuer Schritt nachmachen

Kamera = Index **1** in den Skripten (`VideoCapture(1)`); bei falscher Quelle 0/2 probieren.
ChArUco-Blatt ggf. ausgedruckt; **Eck-Vorlage** kommt aus Schritt 2a (oder gedruckt statt Beamer).

```bash
# --- 1. Kamera kalibrieren --- SKIP: bool_load_cam_calib=False gesetzt, fuer Top-Down nicht noetig ---

# --- 2a. Eck-Marker-Vorlage fuer den Beamer (4 ArUco, schwarz) ---
cd ../02_homogrphic_transform
# In 01_marker_representation.py Zeile bool_fullscreen:
#   False = normales Fenster (Vorschau / Fenster auf den Beamer-Monitor ziehen)
#   True  = sofort Vollbild (direkt komplette Beamerflaeche)
python 01_marker_representation.py
# -> schreibt BeamerImage.png hier; Q schliesst. Ohne Beamer: PNG z.B. in Viewer Vollbild auf Projektor.
# Alternative: dieselben Marker gedruckt kleben — ID0=oben-links, ID1=oben-rechts, ID2=unten-links, ID3=unten-rechts

# --- 2b. Homographie berechnen ---
python 02_calc_pose_trans.py           # Marker sichtbar (Beamer oder Druck), ESC wenn fertig

# --- 3. Projektion starten ---
cd ../03_aruco_tracking
python 02_aruco_plus_transform_on_live_stream.py
# Wichtig: nur "Beamer_Window" braucht man fuer die Ausgabe (auf Beamer / Vollbild).
# "output" = Kontrollansicht (gewarpter Kamerastream mit Markierungen) — kann man ignorieren oder wegstellen.
# Beamer_Window = die "Oberflaeche": was die Kamera in der Projektionsflaeche sieht, wird dort als Projektion abgebildet;
#   ausserhalb des Kamerablicks (wenn du nur den Laptop-Bildschirm anschaust) ist das im Wesentlichen Schwarz plus die Projektionen.
# ESC = beenden
```
