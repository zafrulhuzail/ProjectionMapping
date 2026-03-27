import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, max_height=800):
    """Skaliert das Bild, behält aber die Proportionen."""
    h, w = image.shape[:2]
    if h > max_height:
        ratio = max_height / h
        return cv2.resize(image, (int(w * ratio), max_height))
    return image

def analysiere_karten_regionen(img_ref, img_test_ausgerichtet):
    """Vergleicht gezielt das Bauteil und den Text-Bereich."""
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(img_test_ausgerichtet, cv2.COLOR_BGR2GRAY)

    blur_ref = cv2.GaussianBlur(gray_ref, (9, 9), 0)
    blur_test = cv2.GaussianBlur(gray_test, (9, 9), 0)
    
    diff = cv2.absdiff(blur_ref, blur_test)
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    h, w = thresh.shape
    
    # --- NEUE KOORDINATEN FÜR DIE RECHTECKE (ROIs) ---
    # Bauteil-Region (Mitte)
    roi_bauteil_y, roi_bauteil_h = int(h * 0.18), int(h * 0.49) 
    roi_bauteil_x, roi_bauteil_w = int(w * 0.05), int(w * 0.90)

    # Label-Region (Unten) - Jetzt weiter unten auf dem blauen Balken!
    roi_label_y, roi_label_h = int(h * 0.68), int(h * 0.32)
    roi_label_x, roi_label_w = int(w * 0.05), int(w * 0.90)

    # Regionen ausschneiden
    mask_bauteil = thresh[roi_bauteil_y:roi_bauteil_y+roi_bauteil_h, 
                         roi_bauteil_x:roi_bauteil_x+roi_bauteil_w]
    mask_label = thresh[roi_label_y:roi_label_y+roi_label_h, 
                       roi_label_x:roi_label_x+roi_label_w]

    # Durchschnittliche Fehler-Pixel berechnen
    mean_diff_bauteil = np.mean(mask_bauteil)
    mean_diff_label = np.mean(mask_label)

    # --- ENTSCHEIDUNGS-LOGIK & SCHWELLENWERTE ---
    label_threshold = 1.5   # Sehr empfindlich für Text-Fehler
    bauteil_threshold = 40  # Weniger empfindlich für das große Bauteil
    
    ergebnis_text = "Prüfungs-Ergebnis:\n"
    
    if mean_diff_label > label_threshold:
        ergebnis_text += f"- FALSCHER TEXT! (Abweichung: {mean_diff_label:.1f})\n"
    else:
        ergebnis_text += f"- Text ist KORREKT (Abweichung: {mean_diff_label:.1f})\n"
        
    if mean_diff_bauteil > bauteil_threshold:
        ergebnis_text += f"- FEHLER: Bauteil falsch/fehlt! (Abweichung: {mean_diff_bauteil:.1f})"
    else:
        ergebnis_text += f"- Bauteil ist KORREKT (Abweichung: {mean_diff_bauteil:.1f})"

    # --- RECHTECKE EINZEICHNEN ---
    ergebnis_bild = img_test_ausgerichtet.copy()
    
    # Bauteil-Region (Rot bei Fehler, sonst Blau)
    color_component = (0, 0, 255) if mean_diff_bauteil > bauteil_threshold else (255, 0, 0)
    cv2.rectangle(ergebnis_bild, (roi_bauteil_x, roi_bauteil_y), 
                  (roi_bauteil_x+roi_bauteil_w, roi_bauteil_y+roi_bauteil_h), color_component, 2)
    
    # Label-Region (Rot bei Fehler, sonst Grün)
    color_label = (0, 0, 255) if mean_diff_label > label_threshold else (0, 255, 0)
    cv2.rectangle(ergebnis_bild, (roi_label_x, roi_label_y), 
                  (roi_label_x+roi_label_w, roi_label_y+roi_label_h), color_label, 2)
    
    return thresh, ergebnis_bild, ergebnis_text


def teste_karten(pfad_referenz, pfad_test):
    # Bilder laden
    img_ref = cv2.imread(pfad_referenz)
    img_test = cv2.imread(pfad_test)

    if img_ref is None or img_test is None:
        print("Fehler: Mindestens ein Bild konnte nicht geladen werden. Pfade prüfen!")
        return

    # Größe anpassen
    img_ref = resize_image(img_ref)
    img_test = resize_image(img_test)

    # Feature Matching (ORB)
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
    kp_test, des_test = orb.detectAndCompute(gray_test, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_ref, des_test)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.15)]

    if len(good_matches) < 20:
        print("Fehler: Zu wenige Übereinstimmungen für die Ausrichtung.")
        return

    # Homographie (Ausrichten)
    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_test = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(pts_test, pts_ref, cv2.RANSAC, 5.0)

    if matrix is None:
        print("Fehler: Ausrichtung fehlgeschlagen.")
        return

    h, w = gray_ref.shape
    aligned_test = cv2.warpPerspective(img_test, matrix, (w, h))

    # Regionale Analyse durchführen
    thresh, ergebnis_bild, ergebnis_text = analysiere_karten_regionen(img_ref, aligned_test)

    # Plot erstellen
    plt.figure(figsize=(11, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    plt.title("Referenz")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
    plt.title("Original Test-Foto")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(thresh, cmap='gray')
    plt.title("Differenz-Maske")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(ergebnis_bild, cv2.COLOR_BGR2RGB))
    plt.title("Ausgerichtetes Ergebnis mit Regionen")
    plt.axis('off')

    # Ergebnis-Text groß anzeigen
    plt.suptitle(ergebnis_text, fontsize=15, fontweight='bold', color='darkred')

    plt.tight_layout()
    plt.show()


# --- HIER STARTET DAS PROGRAMM ---
pfad_zur_turbinen_referenz = 'referenz_karte.jpg' # DEIN PFAD
pfad_zur_andere_testkarte = 'andere_karte.jpg' # DEIN PFAD

teste_karten(pfad_zur_turbinen_referenz, pfad_zur_andere_testkarte)