# Beamer Projection Mapping - Anleitung & Verstaendnis

Diese Anleitung erklaert Schritt fuer Schritt, wie die Pipeline funktioniert, was jeder Schritt macht und **warum** er noetig ist.

---

## Ueberblick: Was macht dieses Projekt?

Das Ziel: Eine Kamera erkennt physische Marker (ArUco-Marker) oder mittels Kamerastream und anderen Bibliotheken oder AI sollen Dinge erkannt werden auf einer Flaeche, und ein Beamer projiziert passend dazu digitale Inhalte (Kreise, Bilder, Text) **genau an die richtige Stelle**.

Damit das funktioniert, muessen zwei Probleme geloest werden:

1. **Die Kamera verzerrt das Bild** (Linsenverzerrung) --> Kamera-Kalibrierung (Phase 1) 
2. **Die Kamera schaut aus einem anderen Winkel als der Beamer** --> Homographie / Warping (Phase 2)

Erst wenn beides korrigiert ist, kann man im letzten Schritt (Phase 3) Marker tracken und Dinge praezise projizieren.

---

## Phase 1: Kamera-Kalibrierung (Intrinsische Parameter)

**Ordner:** `01_intrinsic_calibration/`

### Was passiert hier?

Jede Kamera hat eine Linse, und jede Linse verzerrt das Bild ein bisschen. Gerade Linien werden leicht gekruemmt, und verschiedene Bereiche des Bildes werden unterschiedlich stark verzerrt. Das nennt man **Linsenverzerrung** (lens distortion). Das ist ein unvermeidbarer Nebeneffekt des internen Aufbaus jeder Kamera - egal wie teuer oder gut sie ist.

Die Kalibrierung berechnet zwei Dinge:
- **Camera Matrix** (intrinsische Parameter): Beschreibt, wie die Kamera 3D-Punkte auf 2D-Pixel abbildet (Brennweite, optisches Zentrum). Das sind Werte, die von der Kamera-Hardware und ihrer internen Optik abhaengen - quasi ein "Fingerabdruck" der Kamera.
- **Distortion Coefficients**: Beschreiben die genaue Verzerrung der Linse. Auch diese kommen direkt von der physischen Linse der Kamera - jede Kamera hat andere Werte.

### Warum braucht man das?

Ohne Kalibrierung wuerden alle weiteren Berechnungen (Homographie, Marker-Positionen) ungenau sein, weil sie auf einem verzerrten Bild basieren. Konkret: Wenn du versuchst, basierend auf dem Kamerabild etwas auf dem Beamer zu zeichnen und projizieren zu lassen, wuerde es an der falschen Stelle landen - weil die Kamera-Positionen durch die Linsenverzerrung verschoben sind.

**Wichtig:** Verschiedene Stellen im Bild werden **unterschiedlich stark** verzerrt. Der Rand ist typischerweise staerker verzerrt als die Mitte. Deshalb reicht es nicht, einfach das ganze Bild gleichmaessig zu stauchen - man braucht die exakten Koeffizienten.

### Ablauf der Scripts

| Script | Was es macht |
|--------|-------------|
| `00_test_camera_for_resolution.py` | Testet welche Aufloesungen die Kamera unterstuetzt |
| `01_generate_ChAruco.py` | Erzeugt ein ChArUco-Board (Schachbrett mit ArUco-Markern) zum Ausdrucken. Dieses Board wird als Referenzmuster gebraucht: Weil die exakten Abstaende der Felder bekannt sind, kann OpenCV daraus die Linsenverzerrung zurueckrechnen. |
| `02_record_images.py` | Nimmt Fotos vom ChArUco-Board auf (SPACE druecken). Das Board aus verschiedenen Winkeln und Positionen fotografieren! Diese Fotos braucht das naechste Script, um aus moeglichst vielen Perspektiven die Verzerrung der Linse zu berechnen. |

| `03_calibration_ChAruco.py` | Berechnet aus den Fotos die Camera Matrix und Distortion Coefficients |

### Ergebnis

Die Kalibrierungsdaten werden in einer **Pickle-Datei** gespeichert. Pickle ist einfach ein Python-Format zum Speichern von Objekten auf der Festplatte, damit man sie spaeter wieder laden kann, ohne die Kalibrierung erneut durchfuehren zu muessen:

```
calibration/ProCamCalibration.pckl
```

Inhalt: `(calibration, cameraMatrix, distCoeffs, rvecs, tvecs)`

**Das muss man nur einmal machen**, solange man die gleiche Kamera benutzt.

---

## Phase 2: Homographie berechnen (Warping / Dewarping)

**Ordner:** `02_homogrphic_transform/`

### Was passiert hier?

Die Kamera und der Beamer schauen aus **unterschiedlichen Winkeln** auf die gleiche Flaeche. Was die Kamera sieht, ist perspektivisch verzerrt gegenueber dem, was der Beamer projiziert.

Die **Homographie** ist eine mathematische Transformation (eine 3x3-Matrix), die beschreibt, wie man das Kamerabild so umrechnen ("warpen") kann, dass es der Perspektive des Beamers entspricht.

### Warum braucht man das?

Wenn die Kamera einen Marker an Position (300, 200) sieht, heisst das nicht, dass der Beamer an Pixel (300, 200) projizieren soll. Die Kamera schaut von der Seite/oben/unten auf die Flaeche - der Beamer von vorne. Die Homographie rechnet die Kamera-Koordinaten in Beamer-Koordinaten um.

### Wie funktioniert das?

1. Vier ArUco-Marker (ID 0-3) werden an die **Ecken der Projektionsflaeche** geklebt
2. Die Kamera erkennt diese vier Marker
3. Aus den vier Punktpaaren (Kamera-Position --> Beamer-Ecke) wird die Homographie-Matrix berechnet
4. Mit `cv2.warpPerspective()` kann man dann jedes Kamerabild in die Beamer-Perspektive "warpen"

### Ablauf der Scripts

| Script | Was es macht |
|--------|-------------|
| `01_marker_representation.py` | Erzeugt die vier Eck-Marker (ID 0-3) als Bilder zum Ausdrucken |
| `02_calc_pose_trans.py` | Erkennt die vier Eck-Marker per Kamera, berechnet die Homographie-Matrix und speichert sie in pickle datei
| `03_apply_transform_on_live_stream.py` | Zeigt das gewarpte Kamerabild live an (zur Ueberpruefung) |

### Ergebnis

Die Transformation wird in einer **Pickle-Datei** gespeichert:

```
homographic_tranform.pckl
```

Inhalt: `(H, output_width, output_height)` - Die Homographie-Matrix H und die Zielgroesse.

**Das muss man nur neu machen**, wenn sich die Position von Kamera oder Beamer aendert.

---

## Phase 3: ArUco-Tracking & Beamer-Projektion

**Ordner:** `03_aruco_tracking/`

### Was passiert hier?

Hier kommt alles zusammen. Das Hauptscript (`02_aruco_plus_transform_on_live_stream.py`) macht Folgendes:

```
Kamerabild
    |
    v
[1] Undistort (Linsenverzerrung entfernen, mit Daten aus Phase 1)
    |
    v
[2] Warp (Perspektive korrigieren, mit Homographie aus Phase 2)

    Unterschied zu Schritt [1]:
    - Undistort (Phase 1) korrigiert die LINSE: Das Bild ist krumm, weil
      die Linse es verzerrt. Wie ein Foto durch ein Fischauge.
    - Warp (Phase 2) korrigiert die PERSPEKTIVE: Die Kamera schaut schraeg
      auf die Flaeche, der Beamer von vorne. Wie wenn du ein Foto schraeg
      von der Seite machst und es dann gerade ziehen willst.
    |
    v
[3] ArUco-Marker erkennen (im entzerrten + gewarpten Bild)
    |
    v
[4] Auf ein schwarzes Bild zeichnen (Kreise, Text, Bilder) 
    |
    v
[5] Schwarzes Bild an Beamer senden --> Beamer zeigt es an
```

### Das schwarze Bild - wie funktioniert die Projektion?

Das ist der Kern der Projektion. Das passiert alles im Script `03_aruco_tracking/02_aruco_plus_transform_on_live_stream.py` - das ist die Datei, in der du arbeitest wenn du projizieren willst:

1. Es wird ein **komplett schwarzes Bild** erstellt, in der gleichen Groesse wie die Beamer-Aufloesung (1280x720):
   ```python
   BeamerImage = np.zeros(dewarped_img.shape, np.uint8)
   ```

2. Auf dieses schwarze Bild werden mit OpenCV Dinge **gezeichnet** (Kreise, Text, Bilder etc.):
   ```python
   cv2.circle(BeamerImage, position, 30, (0, 255, 255), -1)
   cv2.putText(BeamerImage, "ID 6", position, ...)
   ```

3. Dieses Bild wird in einem **Fenster angezeigt**, das auf dem Beamer liegt:
   ```python
   cv2.imshow("Beamer_Window", BeamerImage)
   ```

4. Wenn das Fenster **fullscreen** auf dem Beamer-Display ist, projiziert der Beamer genau das, was auf dem schwarzen Bild gezeichnet wurde. Das schwarze Bild wird **jeden Frame neu erstellt** (also live, ca. 30x pro Sekunde). Die vier Eck-Marker (ID 0-3) aus Phase 2 werden hier **nicht mehr gebraucht** - die Homographie-Matrix wurde ja bereits berechnet und in der Pickle-Datei gespeichert. Am Rand der Projektionsflaeche muessen also keine Marker mehr kleben. Du brauchst nur noch die Marker, die du tatsaechlich tracken willst (z.B. ID 4, 5, 6...).


**Warum schwarz?** Schwarz = Beamer projiziert nichts (kein Licht). Nur die gezeichneten Elemente sind sichtbar. So sieht man nur die Kreise/Texte auf der physischen Flaeche - ohne stoerenden Hintergrund.

### Warum funktionieren die Positionen korrekt?

Weil in Schritt [2] das Kamerabild bereits in die Beamer-Perspektive gewarpt wurde. Die Marker-Positionen im gewarpten Bild entsprechen direkt den Pixel-Positionen des Beamers. Wenn ein Marker im gewarpten Bild bei (500, 300) erkannt wird, kann man auf dem schwarzen Bild an (500, 300) zeichnen und der Beamer projiziert es genau auf den Marker.

### Ablauf der Scripts

| Script | Was es macht |
|--------|-------------|
| `01_track_aruco_draw_image.py` | Einfache ArUco-Erkennung (Test/Demo) |
| `02_aruco_plus_transform_on_live_stream.py` | **Hauptanwendung**: Volle Pipeline mit Tracking + Projektion |

**Ja, dieses Script funktioniert out-of-the-box** - es erkennt bereits Marker und zeichnet Kreise/Text auf den Beamer. Es ist auch genau die Datei, an der du anknuepfst, wenn du eigene Dinge machen willst. Du musst keinen komplett neuen Code schreiben. Einfach in der `while True`-Schleife (ab Zeile 156) deine eigene Logik einbauen: Marker erkennen, auf `BeamerImage` zeichnen, fertig. Wenn du statt ArUco-Markern andere Erkennung willst (z.B. AI-basiert), ersetzt du nur den Erkennungsteil - das Warping und die Beamer-Ausgabe bleiben gleich.

### Zwei Fenster im Hauptscript

Das Script zeigt zwei Fenster:
- **"Beamer_Window"**: Das schwarze Bild mit den gezeichneten Elementen --> geht an den Beamer
- **"output"**: Das entzerrte + gewarpte Kamerabild mit Markierungen --> fuer dich zur Kontrolle

---

## Zusammenfassung: Die ganze Pipeline

```
EINMALIG:
  Phase 1: Kamera kalibrieren --> ProCamCalibration.pckl
  Phase 2: Homographie berechnen --> homographic_tranform.pckl

BEI JEDEM START:
  Phase 3: Beide Pickle-Dateien laden
           --> Kamerabild undistorten
           --> Kamerabild warpen
           --> Marker erkennen
           --> Auf schwarzes Bild zeichnen
           --> Beamer zeigt es an
```

---

## Wie projiziert man eigene Inhalte mit dem Beamer?

Sobald Phase 1 und 2 abgeschlossen sind, kann man im Hauptscript (`03_aruco_tracking/02_aruco_plus_transform_on_live_stream.py`) eigene Inhalte projizieren:

1. **Marker erkennen und Position holen:**
   ```python
   for marker_id, corner in zip(sorted_ids, sorted_corners):
       center = np.mean(corner[0], axis=0).astype(int)
   ```

2. **Auf das BeamerImage zeichnen** (alles was OpenCV kann):
   ```python
   # Kreis auf Marker-Position
   cv2.circle(BeamerImage, tuple(center), 30, (0, 255, 255), -1)

   # Text
   cv2.putText(BeamerImage, "Hallo", tuple(center), ...)

   # Bild mit Transparenz
   overlay_transparent(BeamerImage, mein_bild, center[0], center[1])
   ```

3. **Beamer-Fenster fullscreen machen** (im Script `bool_fullscreen = True` setzen), das Fenster auf den Beamer-Monitor ziehen, und fertig.

Der Beamer projiziert dann die gezeichneten Elemente praezise auf die Positionen der erkannten Marker.
