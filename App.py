import streamlit as st
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import pytesseract
from PIL import Image
import tempfile
import os
from datetime import timedelta
import librosa
import speech_recognition as sr

st.set_page_config(page_title="Viral Analyzer", page_icon="🎥", layout="centered", initial_sidebar_state="collapsed")

st.title("🎥 TikTok Viral Analyzer V2")
st.markdown("**Analyse Vidéo + Audio complète** — Optimisé iPhone")

uploaded_file = st.file_uploader("Charge ta vidéo (max 50MB recommandé)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(uploaded_file)

    if st.button("🚀 Lancer l'analyse complète", type="primary", use_container_width=True):
        with st.spinner("Analyse vidéo + audio en cours... (peut prendre 1 à 2 minutes)"):
            
            try:
                # === Vidéo Clip ===
                clip = VideoFileClip(video_path)
                duration = clip.duration
                fps = clip.fps or 30

                # === Analyse Vidéo (OpenCV) ===
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                scene_changes = 0
                text_frames = 0
                faces_detected = 0
                prev_frame = None

                for i in range(0, frame_count, max(1, int(fps//2))):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Visages
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0: faces_detected += 1
                    
                    # Texte
                    text = pytesseract.image_to_string(frame)
                    if len(text.strip()) > 15: text_frames += 1
                    
                    # Changements de scène
                    if prev_frame is not None:
                        diff = cv2.absdiff(frame, prev_frame)
                        if np.count_nonzero(diff) > 450000:
                            scene_changes += 1
                    prev_frame = frame.copy()

                cap.release()

                # === Analyse Audio ===
                audio_path = video_path.replace('.mp4', '.wav')
                audio_clip = clip.audio
                if audio_clip:
                    audio_clip.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)
                    
                    # Chargement avec Librosa
                    y, sr_rate = librosa.load(audio_path, sr=16000)
                    
                    # BPM (tempo)
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr_rate)
                    bpm = tempo[0] if isinstance(tempo, np.ndarray) else tempo
                    
                    # Énergie et RMS
                    rms = librosa.feature.rms(y=y)[0]
                    energy_mean = float(np.mean(rms))
                    energy_max = float(np.max(rms))
                    
                    # Transcription voix
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(audio_path) as source:
                        audio_data = recognizer.record(source)
                        try:
                            transcription = recognizer.recognize_google(audio_data, language="fr-FR")
                        except:
                            transcription = "Transcription non disponible ou pas de parole claire."
                else:
                    bpm = 0
                    energy_mean = 0
                    transcription = "Aucun audio détecté."

                # === Calcul Score Global ===
                score = 0
                reasons = []

                # Vidéo
                if 8 <= duration <= 22: 
                    score += 9
                    reasons.append("✅ Durée idéale TikTok")
                elif duration <= 30:
                    score += 7
                    reasons.append("⚠️ Durée acceptable")
                else:
                    score += 4
                    reasons.append("❌ Vidéo trop longue")

                ideal_cuts = duration * 1.1
                if abs(scene_changes - ideal_cuts) < ideal_cuts * 0.5:
                    score += 9
                    reasons.append(f"✅ Rythme dynamique ({scene_changes} cuts)")
                else:
                    score += 6
                    reasons.append(f"⚠️ Rythme moyen ({scene_changes} cuts)")

                # Audio
                if bpm > 80:
                    score += 8
                    reasons.append(f"✅ Tempo entraînant ({int(bpm)} BPM)")
                else:
                    score += 5
                    reasons.append(f"⚠️ Tempo calme ({int(bpm)} BPM)")

                if energy_mean > 0.02:
                    score += 8
                    reasons.append("✅ Bonne énergie sonore")
                else:
                    score += 5
                    reasons.append("⚠️ Audio faible ou plat")

                final_score = min(10, round(score / 4.5, 1))

                # === Résultats ===
                st.success(f"**Potentiel Viral Global : {final_score}/10**")
                
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Durée", str(timedelta(seconds=int(duration))))
                with col2: st.metric("BPM", f"{int(bpm)}")
                with col3: st.metric("Changements", scene_changes)

                st.subheader("📝 Transcription de la voix")
                st.write(transcription)

                st.subheader("📊 Détails complets")
                for r in reasons:
                    st.write(r)

                if final_score >= 8:
                    st.balloons()
                    st.success("**Excellent !** Cette vidéo a un très fort potentiel viral.")
                else:
                    st.warning("**Conseils pour améliorer :**")
                    st.write("- Hook plus fort dans les 3 premières secondes")
                    st.write("- Augmente l’énergie audio (voix + musique)")
                    st.write("- Ajoute du texte visible et rythme plus rapide")

            except Exception as e:
                st.error(f"Erreur : {str(e)}")
            
            finally:
                # Nettoyage
                for path in [video_path, audio_path if 'audio_path' in locals() else '']:
                    if os.path.exists(path):
                        os.unlink(path)

st.caption("V2 Audio + Vidéo • Grok 2026")
