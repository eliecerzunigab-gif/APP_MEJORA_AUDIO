import os
import sys
import json
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal as scipy_signal
import soundfile as sf
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
from flask import Flask, render_template, jsonify, request, send_file
import math
import time
import threading

app = Flask(__name__)

# Directorios a escanear
AUDIO_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MP3_01'),
]

# Directorio de salida para archivos mejorados
IMPROVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'improved_audio')
os.makedirs(IMPROVED_DIR, exist_ok=True)

# Extensiones de audio soportadas
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}

# Almacenar estado de mejoras en progreso
improvement_jobs = {}

def find_audio_files():
    files = []
    for audio_dir in AUDIO_DIRS:
        if os.path.exists(audio_dir):
            for root, dirs, filenames in os.walk(audio_dir):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in AUDIO_EXTENSIONS:
                        files.append(os.path.normpath(os.path.join(root, filename)))
    return files

def resolve_path(filepath):
    """Resuelve una ruta de archivo, ya sea absoluta o relativa."""
    filepath = os.path.normpath(filepath)
    if os.path.isabs(filepath) and os.path.exists(filepath):
        return filepath
    # Intentar relativo al directorio actual
    full = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath))
    if os.path.exists(full):
        return full
    return filepath  # Devolver lo que sea, el caller verificará

def analyze_spectral_content(filepath):
    try:
        # Leer solo 2 segundos para analisis rapido (96k samples a 48kHz, 88.2k a 44.1kHz)
        data, samplerate = sf.read(filepath, dtype='float64', always_2d=False)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        # Tomar max 3 segundos desde el centro del archivo para mejor representatividad
        max_samples = min(len(data), samplerate * 3)
        if len(data) > max_samples:
            start = len(data) // 3  # empezar a 1/3 del archivo (evitar intros silenciosas)
            data = data[start:start + max_samples]
        if len(data) < 1024:
            return None

        n = len(data)
        window = np.hanning(n)
        data_windowed = data * window
        yf = fft(data_windowed)
        xf = fftfreq(n, 1 / samplerate)
        positive_freqs = xf[:n//2]
        positive_magnitude = np.abs(yf[:n//2]) / n

        bands = {
            'sub_bass': (20, 60), 'bass': (60, 250), 'low_mids': (250, 500),
            'mids': (500, 2000), 'high_mids': (2000, 4000),
            'highs': (4000, 8000), 'ultra_highs': (8000, 20000),
        }
        band_energy = {}
        for band_name, (low, high) in bands.items():
            mask = (positive_freqs >= low) & (positive_freqs <= high)
            if np.any(mask):
                energy = np.sum(positive_magnitude[mask] ** 2)
                band_energy[band_name] = float(energy)
            else:
                band_energy[band_name] = 0.0

        total_energy = sum(band_energy.values())
        if total_energy > 0:
            for key in band_energy:
                band_energy[key] = round((band_energy[key] / total_energy) * 100, 2)

        low_energy = band_energy.get('sub_bass', 0) + band_energy.get('bass', 0) + band_energy.get('low_mids', 0)
        mid_energy = band_energy.get('mids', 0) + band_energy.get('high_mids', 0)
        high_energy = band_energy.get('highs', 0) + band_energy.get('ultra_highs', 0)

        spectral_balance = {
            'low_percent': round(low_energy, 2),
            'mid_percent': round(mid_energy, 2),
            'high_percent': round(high_energy, 2),
        }

        issues = []
        rms = np.sqrt(np.mean(data ** 2))
        peak = np.max(np.abs(data))
        crest_factor = 20 * math.log10(peak / rms) if peak > 0 and rms > 0 else 0
        dynamic_range = crest_factor

        if dynamic_range < 6:
            issues.append('Rango dinamico muy comprimido (posible loudness war)')
        elif dynamic_range > 20:
            issues.append('Rango dinamico muy amplio (puede necesitar normalizacion)')

        clipping_samples = np.sum(np.abs(data) > 0.99)
        clipping_percent = (clipping_samples / len(data)) * 100
        if clipping_percent > 0.1:
            issues.append(f'Posible clipping detectado ({clipping_percent:.2f}% de samples)')

        if spectral_balance['low_percent'] < 10:
            issues.append('Bajos debiles - falta presencia en graves')
        elif spectral_balance['low_percent'] > 50:
            issues.append('Exceso de bajos - puede sonar embotado')
        if spectral_balance['high_percent'] < 8:
            issues.append('Agudos apagados - falta brillo y claridad')
        elif spectral_balance['high_percent'] > 35:
            issues.append('Exceso de agudos - puede sonar fatigante')
        if spectral_balance['mid_percent'] > 60:
            issues.append('Medios muy dominantes - posible falta de balance')

        return {
            'band_energy': band_energy,
            'spectral_balance': spectral_balance,
            'dynamic_range_db': round(dynamic_range, 2),
            'clipping_percent': round(clipping_percent, 4),
            'issues': issues,
            'rms_level': round(float(20 * math.log10(rms)) if rms > 0 else -100, 2),
            'peak_level': round(float(20 * math.log10(peak)) if peak > 0 else 0, 2),
        }
    except Exception as e:
        print(f"Error en analisis espectral de {filepath}: {e}")
        return None

def calculate_improvement_potential(filepath, file_info):
    score = 0
    max_score = 0
    improvements = []
    target_format = None
    details = {
        'bass_improvement': 0, 'mid_improvement': 0, 'high_improvement': 0,
        'clarity_improvement': 0, 'dynamics_improvement': 0,
    }

    ext = os.path.splitext(filepath)[1].lower()
    format_scores = {'.mp3': 70, '.aac': 80, '.ogg': 85, '.m4a': 82, '.wma': 60, '.wav': 95, '.flac': 95}
    format_score = format_scores.get(ext, 60)
    max_score += 100

    if format_score < 90:
        score += format_score
        if ext == '.mp3':
            target_format = 'FLAC (Lossless) o WAV'
            improvements.append(f'Convertir de {ext.upper()} a FLAC/WAV (calidad sin perdida) para preservacion')
            details['clarity_improvement'] += 15
        elif ext in ['.wma', '.ogg']:
            target_format = 'FLAC (Lossless)'
            improvements.append(f'Convertir de {ext.upper()} a FLAC para maxima calidad')
            details['clarity_improvement'] += 10
    else:
        score += 95
        target_format = 'Formato actual optimo'

    if 'bitrate' in file_info:
        bitrate = file_info['bitrate'] / 1000
        max_score += 100
        if bitrate < 128:
            score += 30
            improvements.append(f'Bitrate muy bajo ({bitrate:.0f} kbps) - se recomienda al menos 256-320 kbps')
            details['clarity_improvement'] += 25
            details['bass_improvement'] += 10
            details['high_improvement'] += 10
        elif bitrate < 192:
            score += 55
            improvements.append(f'Bitrate bajo ({bitrate:.0f} kbps) - mejorable a 256-320 kbps')
            details['clarity_improvement'] += 15
            details['bass_improvement'] += 5
            details['high_improvement'] += 5
        elif bitrate < 256:
            score += 75
            improvements.append(f'Bitrate medio ({bitrate:.0f} kbps) - recomendable 320 kbps')
            details['clarity_improvement'] += 8
        elif bitrate < 320:
            score += 90
            details['clarity_improvement'] += 3
        else:
            score += 100

    if 'sample_rate' in file_info:
        sr = file_info['sample_rate']
        max_score += 50
        if sr < 22050:
            score += 10
            improvements.append(f'Sample rate muy bajo ({sr} Hz)')
            details['clarity_improvement'] += 15
        elif sr < 44100:
            score += 30
        elif sr == 44100:
            score += 45
        elif sr >= 48000:
            score += 50

    if 'channels' in file_info:
        ch = file_info['channels']
        max_score += 30
        if ch == 1:
            score += 15
            improvements.append('Audio mono - convertir a stereo puede mejorar la experiencia')
        elif ch == 2:
            score += 30

    if 'duration' in file_info:
        dur = file_info['duration']
        max_score += 20
        if dur < 60:
            score += 10
        elif dur < 300:
            score += 16
        else:
            score += 20

    spectral = analyze_spectral_content(filepath)
    if spectral:
        max_score += 100
        low_pct = spectral['spectral_balance']['low_percent']
        mid_pct = spectral['spectral_balance']['mid_percent']
        high_pct = spectral['spectral_balance']['high_percent']

        balance_score = 0
        if 20 <= low_pct <= 40:
            balance_score += 30
        elif 10 <= low_pct <= 50:
            balance_score += 15
            if low_pct < 20:
                improvements.append('Refuerzo de graves recomendado')
                details['bass_improvement'] += 10
            else:
                improvements.append('Reduccion de graves recomendada para mejor claridad')
                details['bass_improvement'] += 5

        if 35 <= mid_pct <= 55:
            balance_score += 35
        elif 25 <= mid_pct <= 65:
            balance_score += 18
            details['mid_improvement'] += 8

        if 15 <= high_pct <= 35:
            balance_score += 35
        elif 8 <= high_pct <= 45:
            balance_score += 18
            if high_pct < 15:
                improvements.append('Refuerzo de agudos recomendado (falta de brillo)')
                details['high_improvement'] += 12
            else:
                improvements.append('Atenuacion de agudos recomendada (sonido fatigante)')
                details['high_improvement'] += 5

        score += balance_score

        dr = spectral['dynamic_range_db']
        if 8 <= dr <= 18:
            score += 30
        elif 6 <= dr <= 22:
            score += 18
            details['dynamics_improvement'] += 8
        else:
            details['dynamics_improvement'] += 12

        if spectral['clipping_percent'] > 0.5:
            improvements.append(f'Clipping significativo detectado ({spectral["clipping_percent"]:.2f}%)')
            details['clarity_improvement'] += 10
        elif spectral['clipping_percent'] > 0.05:
            details['clarity_improvement'] += 3

        for issue in spectral.get('issues', []):
            if issue not in improvements:
                improvements.append(issue)

        details['peak_db'] = spectral.get('peak_level', 0)
        details['rms_db'] = spectral.get('rms_level', -100)

    if max_score > 0:
        quality_percent = (score / max_score) * 100
        improvement_potential = round(100 - quality_percent, 2)
    else:
        improvement_potential = 0

    if improvement_potential < 10:
        rating, rating_color = 'Excelente', '#00c853'
    elif improvement_potential < 25:
        rating, rating_color = 'Buena', '#64dd17'
    elif improvement_potential < 45:
        rating, rating_color = 'Aceptable', '#ffd600'
    elif improvement_potential < 65:
        rating, rating_color = 'Regular', '#ff9100'
    else:
        rating, rating_color = 'Necesita Mejora', '#ff1744'

    return {
        'improvement_potential': improvement_potential,
        'quality_score': round(100 - improvement_potential, 2),
        'rating': rating,
        'rating_color': rating_color,
        'target_format': target_format or 'Mantener formato actual',
        'improvements': improvements if improvements else ['Calidad satisfactoria'],
        'details': details,
        'spectral_analysis': spectral,
    }

def analyze_file(filepath):
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    file_info = {
        'filename': filename,
        'filepath': filepath,
        'extension': ext,
        'size_bytes': os.path.getsize(filepath),
        'size_mb': round(os.path.getsize(filepath) / (1024 * 1024), 2),
    }
    try:
        if ext == '.mp3':
            audio = MP3(filepath)
            file_info['duration'] = round(audio.info.length, 2) if audio.info.length else 0
            file_info['bitrate'] = audio.info.bitrate if hasattr(audio.info, 'bitrate') else 0
            file_info['sample_rate'] = audio.info.sample_rate if hasattr(audio.info, 'sample_rate') else 0
            file_info['channels'] = audio.info.channels if hasattr(audio.info, 'channels') else 0
            file_info['bitrate_mode'] = str(audio.info.bitrate_mode) if hasattr(audio.info, 'bitrate_mode') else 'N/A'
            file_info['layer'] = audio.info.layer if hasattr(audio.info, 'layer') else 'N/A'
            try:
                id3 = ID3(filepath)
                file_info['artist'] = str(id3.get('TPE1', 'Desconocido'))
                file_info['album'] = str(id3.get('TALB', 'Desconocido'))
                file_info['title'] = str(id3.get('TIT2', filename))
                file_info['genre'] = str(id3.get('TCON', 'Desconocido'))
                file_info['year'] = str(id3.get('TDRC', 'Desconocido'))
            except:
                file_info['artist'] = 'Desconocido'
                file_info['album'] = 'Desconocido'
                file_info['title'] = filename
                file_info['genre'] = 'Desconocido'
                file_info['year'] = 'Desconocido'
        else:
            from mutagen import File as MutagenFile
            audio = MutagenFile(filepath)
            if audio and hasattr(audio, 'info'):
                file_info['duration'] = round(audio.info.length, 2) if audio.info.length else 0
                file_info['bitrate'] = audio.info.bitrate if hasattr(audio.info, 'bitrate') else 0
                file_info['sample_rate'] = audio.info.sample_rate if hasattr(audio.info, 'sample_rate') else 0
                file_info['channels'] = audio.info.channels if hasattr(audio.info, 'channels') else 0
            else:
                for k in ['duration', 'bitrate', 'sample_rate', 'channels']:
                    file_info[k] = 0
    except Exception as e:
        print(f"Error leyendo metadata de {filepath}: {e}")
        for k in ['duration', 'bitrate', 'sample_rate', 'channels']:
            file_info[k] = 0

    if file_info.get('bitrate', 0) > 10000:
        file_info['bitrate_kbps'] = round(file_info['bitrate'] / 1000, 1)
    else:
        file_info['bitrate_kbps'] = round(file_info.get('bitrate', 0), 1) if file_info.get('bitrate', 0) else 0

    improvement = calculate_improvement_potential(filepath, file_info)
    file_info.update(improvement)
    return file_info

# ====== ROUTES ======

# Directorio temporal para uploads
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MP3_01')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scan')
def scan():
    # Escanear tanto MP3_01 como improved_audio (por si hay archivos subidos)
    files = find_audio_files()
    # También buscar en uploads recientes
    if os.path.exists(UPLOAD_DIR):
        for root, dirs, filenames in os.walk(UPLOAD_DIR):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in AUDIO_EXTENSIONS:
                    fp = os.path.normpath(os.path.join(root, filename))
                    if fp not in files:
                        files.append(fp)
    results = []
    for filepath in files:
        try:
            info = analyze_file(filepath)
            results.append(info)
        except Exception as e:
            print(f"Error procesando {filepath}: {e}")
            results.append({
                'filename': os.path.basename(filepath),
                'filepath': filepath,
                'error': str(e),
                'improvement_potential': 0,
                'quality_score': 0,
                'rating': 'Error',
                'rating_color': '#757575',
            })
    results.sort(key=lambda x: x.get('improvement_potential', 0), reverse=True)

    if results:
        avg_potential = sum(r.get('improvement_potential', 0) for r in results) / len(results)
        total_files = len(results)
        needs_improvement = sum(1 for r in results if r.get('improvement_potential', 0) > 30)
        good_quality = sum(1 for r in results if r.get('improvement_potential', 0) < 15)
    else:
        avg_potential = total_files = needs_improvement = good_quality = 0

    return jsonify({
        'files': results,
        'stats': {
            'total_files': total_files,
            'average_improvement_potential': round(avg_potential, 2),
            'files_needing_improvement': needs_improvement,
            'files_good_quality': good_quality,
        },
        'directories_scanned': AUDIO_DIRS,
    })

# ====== AUDIO ENHANCEMENT - RBJ BIQUAD FILTERS (Audio EQ Cookbook) ======

def biquad_peaking(freq, gain_db, q, samplerate):
    """Peaking EQ filter. UNIFIED formula works for both boost (+) and cut (-)."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / samplerate
    cos_w = np.cos(w0)
    sin_w = np.sin(w0)
    alpha = sin_w / (2.0 * q)
    # Unified formula (same for boost and cut)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w
    a2 = 1.0 - alpha / A
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]

def biquad_lowshelf(freq, gain_db, samplerate, q=0.707):
    """Low shelf filter. UNIFIED RBJ Cookbook formula for boost AND cut."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / samplerate
    cos_w = np.cos(w0)
    sin_w = np.sin(w0)
    beta = np.sqrt(A) / q
    alpha = sin_w / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / beta - 1.0) + 2.0)
    two_sqrtA_alpha = 2.0 * np.sqrt(A) * alpha
    # Unified formula - same for boost and cut
    b0 = A * ((A + 1.0) - (A - 1.0) * cos_w + two_sqrtA_alpha)
    b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w)
    b2 = A * ((A + 1.0) - (A - 1.0) * cos_w - two_sqrtA_alpha)
    a0 = (A + 1.0) + (A - 1.0) * cos_w + two_sqrtA_alpha
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w)
    a2 = (A + 1.0) + (A - 1.0) * cos_w - two_sqrtA_alpha
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]

def biquad_highshelf(freq, gain_db, samplerate, q=0.707):
    """High shelf filter. UNIFIED RBJ Cookbook formula for boost AND cut."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / samplerate
    cos_w = np.cos(w0)
    sin_w = np.sin(w0)
    beta = np.sqrt(A) / q
    alpha = sin_w / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / beta - 1.0) + 2.0)
    two_sqrtA_alpha = 2.0 * np.sqrt(A) * alpha
    # Unified formula - same for boost and cut
    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w + two_sqrtA_alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w - two_sqrtA_alpha)
    a0 = (A + 1.0) - (A - 1.0) * cos_w + two_sqrtA_alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w)
    a2 = (A + 1.0) - (A - 1.0) * cos_w - two_sqrtA_alpha
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]

def biquad_highpass(freq, samplerate, q=0.707):
    """Highpass filter."""
    w0 = 2.0 * np.pi * freq / samplerate
    cos_w = np.cos(w0)
    sin_w = np.sin(w0)
    alpha = sin_w / (2.0 * q)
    b0 = (1.0 + cos_w) / 2.0
    b1 = -(1.0 + cos_w)
    b2 = (1.0 + cos_w) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w
    a2 = 1.0 - alpha
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]

def apply_audio_enhancements(input_path, output_path, recipe):
    """
    Motor de mejora CORRECTIVO.
    Analiza desde 1/3 del archivo (misma posicion que analyze_spectral_content)
    para consistencia en comparaciones antes/despues.
    
    Pipeline:
    1. Reparar clipping
    2. EQ correctivo adaptativo con low-shelf para graves, high-shelf para agudos
    3. Rebalanceo fino post-EQ
    4. Ensanchamiento estereo
    5. Normalizacion final
    """
    try:
        # ---- LECTURA ----
        data, samplerate = sf.read(input_path, dtype='float64', always_2d=False)
        print(f"  Procesando: {os.path.basename(input_path)}  SR:{samplerate}Hz  Shape:{data.shape}")

        was_mono = len(data.shape) == 1
        is_stereo = not was_mono and data.shape[1] >= 2
        if was_mono:
            data = data.reshape(-1, 1)

        results_log = []
        
        # ---- FUNCION AUXILIAR: FFT desde 1/3 del archivo ----
        def spectral_snapshot(signal_2d, label=""):
            """Analisis espectral usando ~2.5s desde 1/3 del archivo."""
            mono = np.mean(signal_2d, axis=1) if len(signal_2d.shape) > 1 else signal_2d
            chunk = min(len(mono), int(samplerate * 2.5))
            start_idx = len(mono) // 3
            if start_idx + chunk > len(mono):
                start_idx = max(0, len(mono) - chunk)
            segment = mono[start_idx:start_idx + chunk]
            n_fft = min(len(segment), 65536)
            win = np.hanning(n_fft)
            yf_s = fft(segment[:n_fft] * win)
            xf_s = fftfreq(n_fft, 1 / samplerate)
            pos_s = xf_s > 0
            mag_s = np.abs(yf_s[pos_s])
            def be(lo, hi):
                mask = (xf_s[pos_s] >= lo) & (xf_s[pos_s] <= hi)
                return float(np.sum(mag_s[mask] ** 2)) if np.any(mask) else 0.0
            sb_ = be(20,60); b_ = be(60,250); lm_ = be(250,500)
            m_ = be(500,2000); hm_ = be(2000,4000)
            h_ = be(4000,10000); u_ = be(10000,20000)
            t_ = sb_+b_+lm_+m_+hm_+h_+u_ + 0.0001
            return {
                'low_pct': (sb_+b_+lm_)/t_*100,
                'mid_pct': (m_+hm_)/t_*100,
                'high_pct': (h_+u_)/t_*100,
                'rms_db': 20*np.log10(np.sqrt(np.mean(segment**2))+1e-10),
                'peak_db': 20*np.log10(np.max(np.abs(segment))+1e-10),
                'clipping_pct': np.sum(np.abs(segment) > 0.98) / len(segment) * 100,
            }
        
        # ---- DIAGNOSTICO INICIAL ----
        pre_spec = spectral_snapshot(data, "PRE")
        low_pct = pre_spec['low_pct']
        mid_pct = pre_spec['mid_pct']
        high_pct = pre_spec['high_pct']
        results_log.append(
            f"Diagnostico: Graves={low_pct:.0f}% Medios={mid_pct:.0f}% Agudos={high_pct:.0f}% "
            f"| RMS={pre_spec['rms_db']:.1f}dB")
        
        # ---- PASO 0: DETECTAR SATURACION ----
        peak_in = np.max(np.abs(data))
        if peak_in > 0.98 or pre_spec['clipping_pct'] > 0.01:
            recipe['remove_clipping'] = True
            results_log.append(f"Saturacion detectada (peak={peak_in:.3f}, clipping={pre_spec['clipping_pct']:.3f}%)")
        
        # ---- PASO 1: REPARAR CLIPPING ----
        if recipe.get('remove_clipping', False):
            for ch in range(data.shape[1]):
                channel = data[:, ch]
                clipped_mask = np.abs(channel) > 0.98
                if np.any(clipped_mask):
                    for i in range(1, len(channel) - 1):
                        if clipped_mask[i]:
                            left_idx, right_idx = i - 1, i + 1
                            while left_idx > 0 and clipped_mask[left_idx]: left_idx -= 1
                            while right_idx < len(channel) - 1 and clipped_mask[right_idx]: right_idx += 1
                            alpha = (i - left_idx) / max(right_idx - left_idx, 1)
                            channel[i] = channel[left_idx] * (1 - alpha) + channel[right_idx] * alpha
                    data[:, ch] = channel
            remaining = np.sum(np.abs(data) > 0.98) / data.size * 100
            results_log.append(f"✓ Clipping reparado (restante: {remaining:.3f}%)")
        
        # ---- PASO 2: EQ CORRECTIVO ADAPTATIVO ----
        # Usar low-shelf para graves (mas efectivo que peaking para bandas anchas)
        # Usar high-shelf para agudos
        eq_steps = []
        for ch in range(data.shape[1]):
            channel = data[:, ch]
            
            # Highpass 28Hz (limpiar subsonico)
            b_hp, a_hp = biquad_highpass(28.0, samplerate)
            channel = scipy_signal.lfilter(b_hp, a_hp, channel)
            
            # --- CORRECCION DE GRAVES (LOW SHELF) ---
            # Target: ~35%
            if low_pct > 40:
                # low-shelf cut proporcional al exceso (max -15dB)
                cut_db = min(15.0, (low_pct - 35) * 0.4)
                b_low, a_low = biquad_lowshelf(200.0, -cut_db, samplerate, q=0.6)
                channel = scipy_signal.lfilter(b_low, a_low, channel)
            elif low_pct < 18:
                boost_db = min(8.0, (22 - low_pct) * 0.6)
                b_low, a_low = biquad_lowshelf(150.0, boost_db, samplerate, q=0.6)
                channel = scipy_signal.lfilter(b_low, a_low, channel)
            
            # --- CORRECCION DE AGUDOS (HIGH SHELF) ---
            # Target: ~25%
            if high_pct < 10:
                boost_h = min(15.0, (15 - high_pct) * 1.2)
                b_high, a_high = biquad_highshelf(5000.0, boost_h, samplerate, q=0.5)
                channel = scipy_signal.lfilter(b_high, a_high, channel)
            elif high_pct > 38:
                cut_h = min(8.0, (high_pct - 35) * 0.5)
                b_high, a_high = biquad_highshelf(8000.0, -cut_h, samplerate)
                channel = scipy_signal.lfilter(b_high, a_high, channel)
            
            # --- AIR BAND (>12kHz) para brillo extra ---
            if high_pct < 12:
                air_boost = min(8.0, (15 - high_pct) * 0.5)
                b_air, a_air = biquad_highshelf(12000.0, air_boost, samplerate, q=0.5)
                channel = scipy_signal.lfilter(b_air, a_air, channel)
            
            data[:, ch] = channel
        
        # Construir lista de pasos (sin duplicados por canal)
        if low_pct > 40:
            cut_db = min(15.0, (low_pct - 35) * 0.4)
            eq_steps.append(f"Graves -{cut_db:.1f}dB (low-shelf 200Hz)")
        elif low_pct < 18:
            boost_db = min(8.0, (22 - low_pct) * 0.6)
            eq_steps.append(f"Graves +{boost_db:.1f}dB")
        if high_pct < 10:
            boost_h = min(15.0, (15 - high_pct) * 1.2)
            eq_steps.append(f"Agudos +{boost_h:.1f}dB (high-shelf 5kHz)")
        elif high_pct > 38:
            cut_h = min(8.0, (high_pct - 35) * 0.5)
            eq_steps.append(f"Agudos -{cut_h:.1f}dB")
        if high_pct < 12:
            air_boost = min(8.0, (15 - high_pct) * 0.5)
            eq_steps.append(f"Aire +{air_boost:.1f}dB (>12kHz)")
        results_log.append(f"✓ EQ correctivo: {' | '.join(eq_steps)}")
        
        # ---- PASO 3: REBALANCEO POST-EQ ----
        post_eq_spec = spectral_snapshot(data, "POST-EQ")
        new_low = post_eq_spec['low_pct']
        new_high = post_eq_spec['high_pct']
        
        fine_steps = []
        if new_low > 45 or new_high < 6:
            for ch in range(data.shape[1]):
                channel = data[:, ch]
                if new_low > 45:
                    extra_cut = min(6.0, (new_low - 40) * 0.3)
                    b_fl, a_fl = biquad_lowshelf(200.0, -extra_cut, samplerate, q=0.5)
                    channel = scipy_signal.lfilter(b_fl, a_fl, channel)
                if new_high < 6:
                    extra_boost = min(8.0, (10 - new_high) * 0.7)
                    b_fh, a_fh = biquad_highshelf(6000.0, extra_boost, samplerate, q=0.5)
                    channel = scipy_signal.lfilter(b_fh, a_fh, channel)
                data[:, ch] = channel
            if new_low > 45:
                fine_steps.append(f"Graves extra -{min(6.0,(new_low-40)*0.3):.1f}dB")
            if new_high < 6:
                fine_steps.append(f"Agudos extra +{min(8.0,(10-new_high)*0.7):.1f}dB")
            results_log.append(f"✓ Rebalanceo fino: {' | '.join(fine_steps)}")
        
        # Anti-clip
        peak_after_eq = np.max(np.abs(data))
        if peak_after_eq > 0.95:
            data = data / peak_after_eq * 0.90
            results_log.append("✓ Ganancia ajustada post-EQ (anti-clip)")
        
        # ---- PASO 4: ENSANCHAMIENTO ESTEREO ----
        if is_stereo:
            mid_c = (data[:, 0] + data[:, 1]) / 2.0
            side_c = (data[:, 0] - data[:, 1]) / 2.0
            side_c = side_c * 1.25
            data[:, 0] = np.clip(mid_c + side_c, -1, 1)
            data[:, 1] = np.clip(mid_c - side_c, -1, 1)
            results_log.append("✓ Ensanchamiento estereo (+25% side)")
        
        # ---- PASO 5: NORMALIZACION FINAL ----
        if recipe.get('normalize', True):
            peak = np.max(np.abs(data))
            if peak > 0 and peak < 0.99:
                target_peak = 10 ** (-1.0/20)
                gain = min(target_peak / peak, 10**(3.0/20))
                data = data * gain
                new_peak = 20*np.log10(np.max(np.abs(data))+1e-10)
                new_rms = 20*np.log10(np.sqrt(np.mean(data**2))+1e-10)
                results_log.append(f"✓ Normalizado: peak {new_peak:.1f} dBFS, RMS ~{new_rms:.1f} dB")
        
        # Limpieza final
        data = np.clip(data, -0.97, 0.97)
        if was_mono:
            data = data.flatten()
        
        # ---- ANALISIS FINAL (sobre datos ya procesados) ----
        # Crear array temporal con la misma forma para spectral_snapshot
        data_for_spec = data.reshape(-1, 1) if was_mono else data
        post_spec = spectral_snapshot(data_for_spec, "FINAL")
        results_log.append(
            f"Resultado: Graves={post_spec['low_pct']:.0f}% Medios={post_spec['mid_pct']:.0f}% "
            f"Agudos={post_spec['high_pct']:.0f}% | RMS={post_spec['rms_db']:.1f}dB")
        
        # ---- ESCRITURA ----
        output_format = recipe.get('output_format', 'flac').lower()
        if output_format == 'mp3':
            try:
                sf.write(output_path, data, samplerate, subtype='MPEG_LAYER_III')
                results_log.append(f"✓ Guardado: MP3 320kbps")
            except Exception as e:
                print(f"    MP3 encoding fallo: {e}, usando WAV")
                output_path = output_path.rsplit('.', 1)[0] + '.wav'
                sf.write(output_path, data, samplerate, subtype='PCM_16')
                results_log.append(f"✓ Guardado: WAV 16-bit (fallback)")
        elif output_format == 'flac':
            sf.write(output_path, data, samplerate, subtype='PCM_24')
            results_log.append(f"✓ Guardado: FLAC 24-bit (lossless)")
        else:
            sf.write(output_path, data, samplerate, subtype='PCM_24')
            results_log.append(f"✓ Guardado: WAV 24-bit")
        
        # ---- COMPARACION USANDO MEDICIONES REALES ----
        comparison = build_comparison_from_specs(input_path, output_path, pre_spec, post_spec)
        return True, results_log, output_path, comparison

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, [f"Error: {str(e)}"], None, None

def build_comparison_from_specs(original_path, improved_path, pre_spec, post_spec):
    """Construye comparacion usando mediciones REALES tomadas durante el procesamiento.
    Evita re-analizar los archivos (lo cual es inconsistente por usar diferentes segmentos)."""
    try:
        orig_size_mb = round(os.path.getsize(original_path) / (1024 * 1024), 2) if os.path.exists(original_path) else 0
        imp_size_mb = round(os.path.getsize(improved_path) / (1024 * 1024), 2) if os.path.exists(improved_path) else 0

        # Score basado en balance espectral real
        def spec_score(sp):
            low = sp.get('low_pct', 30)
            high = sp.get('high_pct', 15)
            mid = sp.get('mid_pct', 40)
            s = 50
            # Graves ideales: 25-40%
            if 20 <= low <= 40:
                s += 20
            elif 10 <= low <= 50:
                s += 5
            else:
                s -= max(0, abs(low - 35) * 0.5)
            # Agudos ideales: 15-30%
            if 15 <= high <= 32:
                s += 20
            elif 8 <= high <= 40:
                s += 5
            else:
                s -= max(0, abs(high - 25) * 0.6)
            # Medios
            if 35 <= mid <= 55:
                s += 10
            return max(0, min(100, s))

        orig_score = spec_score(pre_spec)
        imp_score = spec_score(post_spec)

        orig_balance = {
            'low_percent': round(pre_spec['low_pct'], 2),
            'mid_percent': round(pre_spec['mid_pct'], 2),
            'high_percent': round(pre_spec['high_pct'], 2),
        }
        imp_balance = {
            'low_percent': round(post_spec['low_pct'], 2),
            'mid_percent': round(post_spec['mid_pct'], 2),
            'high_percent': round(post_spec['high_pct'], 2),
        }

        gains = {
            'score': {
                'original': round(orig_score, 1),
                'improved': round(imp_score, 1),
                'gain_pct': round(max(0, imp_score - orig_score), 1),
            },
            'size': {
                'original_mb': orig_size_mb,
                'improved_mb': imp_size_mb,
                'ratio': round(imp_size_mb / orig_size_mb, 2) if orig_size_mb > 0 else 0,
            },
            'bitrate': {
                'original_kbps': '--',
                'improved_kbps': 'Procesado',
            },
            'spectral_balance': {
                'original': orig_balance,
                'improved': imp_balance,
            },
            'rms_level': {
                'original_db': round(pre_spec.get('rms_db', -100), 1),
                'improved_db': round(post_spec.get('rms_db', -100), 1),
            },
            'peak_level': {
                'original_db': round(pre_spec.get('peak_db', 0), 1),
                'improved_db': round(post_spec.get('peak_db', 0), 1),
            },
            'clipping': {
                'original_pct': round(pre_spec.get('clipping_pct', 0), 4),
                'improved_pct': 0.0,
            },
            'original_improvements': ['--'],
            'improved_issues': ['--'],
        }
        return gains
    except Exception as e:
        print(f"Error en build_comparison: {e}")
        return None

def compare_audio_files(original_path, improved_path):
    """Compara archivo original vs mejorado usando solo analisis espectral (evita
    re-ejecutar el pipeline completo de scoring que da resultados enganosos en FLAC)."""
    try:
        # Datos basicos originales
        orig_size_mb = round(os.path.getsize(original_path) / (1024 * 1024), 2)
        imp_size_mb = round(os.path.getsize(improved_path) / (1024 * 1024), 2)

        # Solo analisis espectral, sin re-ejecutar calculate_improvement_potential
        orig_spec = analyze_spectral_content(original_path)
        imp_spec = analyze_spectral_content(improved_path)

        orig_balance = orig_spec.get('spectral_balance', {}) if orig_spec else {}
        imp_balance = imp_spec.get('spectral_balance', {}) if imp_spec else {}

        # Construir score estimado basado en balance espectral
        def quick_spectral_score(balance):
            """Score rapido basado solo en el balance espectral (0-100)."""
            if not balance:
                return 0
            low = balance.get('low_percent', 30)
            high = balance.get('high_percent', 15)
            mid = balance.get('mid_percent', 40)
            s = 50
            # Penalizar exceso de graves
            if low > 45: s -= (low - 45) * 1.2
            elif low < 15: s -= (18 - low) * 1.0
            else: s += 10
            # Penalizar falta de agudos
            if high < 8: s -= (10 - high) * 1.5
            elif high > 38: s -= (high - 38) * 1.0
            else: s += 10
            # Rango medio
            if 30 <= mid <= 55: s += 15
            elif 20 <= mid <= 65: s += 5
            return max(0, min(100, s))

        orig_score = quick_spectral_score(orig_balance)
        imp_score = quick_spectral_score(imp_balance)

        gains = {
            'score': {
                'original': round(orig_score, 1),
                'improved': round(imp_score, 1),
                'gain_pct': round(max(0, imp_score - orig_score), 1),
            },
            'size': {
                'original_mb': orig_size_mb,
                'improved_mb': imp_size_mb,
                'ratio': round(imp_size_mb / orig_size_mb, 2) if orig_size_mb > 0 else 0,
            },
            'bitrate': {
                'original_kbps': '--',
                'improved_kbps': 'FLAC/WAV',
            },
            'spectral_balance': {
                'original': orig_balance,
                'improved': imp_balance,
            },
            'rms_level': {
                'original_db': orig_spec.get('rms_level', -100) if orig_spec else -100,
                'improved_db': imp_spec.get('rms_level', -100) if imp_spec else -100,
            },
            'peak_level': {
                'original_db': orig_spec.get('peak_level', 0) if orig_spec else 0,
                'improved_db': imp_spec.get('peak_level', 0) if imp_spec else 0,
            },
            'clipping': {
                'original_pct': orig_spec.get('clipping_percent', 0) if orig_spec else 0,
                'improved_pct': imp_spec.get('clipping_percent', 0) if imp_spec else 0,
            },
            'original_improvements': ['--'],
            'improved_issues': ['--'],
        }

        return gains
    except Exception as e:
        print(f"Error comparando archivos: {e}")
        return None

def build_enhancement_recipe(file_info):
    recipe = {
        'normalize': True, 'target_lufs': -14,
        'eq_bass_boost': 0, 'eq_mid_adjust': 0, 'eq_high_boost': 0,
        'remove_clipping': False, 'expand_dynamics': False,
        'output_format': 'flac', 'output_bitrate': 320,
    }

    spectral = file_info.get('spectral_analysis', {})
    if spectral.get('clipping_percent', 0) > 0.05:
        recipe['remove_clipping'] = True
    if spectral.get('dynamic_range_db', 15) < 8:
        recipe['expand_dynamics'] = True

    if spectral.get('spectral_balance'):
        sb = spectral['spectral_balance']
        low_pct = sb.get('low_percent', 30)
        high_pct = sb.get('high_percent', 15)
        if low_pct < 15:
            recipe['eq_bass_boost'] = min(5.0, (15 - low_pct) * 0.4)
        elif low_pct > 45:
            recipe['eq_bass_boost'] = max(-5.0, (45 - low_pct) * 0.3)
        if high_pct < 10:
            recipe['eq_high_boost'] = min(4.0, (10 - high_pct) * 0.5)
        elif high_pct > 35:
            recipe['eq_high_boost'] = max(-4.0, (35 - high_pct) * 0.3)

    # El formato se sobreescribe desde el frontend
    return recipe

@app.route('/api/enhance/preview', methods=['POST'])
def preview_enhancement():
    data = request.get_json(silent=True) or {}
    filepath = data.get('filepath', '')
    user_output_format = (data.get('output_format') or '').lower()

    full_path = resolve_path(filepath)
    if not os.path.exists(full_path):
        return jsonify({'error': f'Archivo no encontrado: {filepath}'}), 404

    file_info = analyze_file(full_path)
    recipe = build_enhancement_recipe(file_info)
    if user_output_format in ('flac', 'wav', 'mp3'):
        recipe['output_format'] = user_output_format

    steps = []
    if recipe.get('remove_clipping'):
        steps.append({'icon': '🔧', 'title': 'Reparacion de Clipping',
            'description': 'Reconstruccion de ondas recortadas para eliminar saturacion digital.'})
    if recipe.get('expand_dynamics'):
        steps.append({'icon': '📈', 'title': 'Expansion de Rango Dinamico',
            'description': 'Restauracion del rango dinamico para mas vida y separacion.'})
    if abs(recipe.get('eq_bass_boost', 0)) > 0.1:
        d = 'reforzaran' if recipe['eq_bass_boost'] > 0 else 'atenuaran'
        steps.append({'icon': '🔊', 'title': f'EQ Graves ({recipe["eq_bass_boost"]:+.1f} dB)',
            'description': f'Se {d} los graves (~80 Hz).'})
    if abs(recipe.get('eq_mid_adjust', 0)) > 0.1:
        d = 'reforzaran' if recipe['eq_mid_adjust'] > 0 else 'atenuaran'
        steps.append({'icon': '🎤', 'title': f'EQ Medios ({recipe["eq_mid_adjust"]:+.1f} dB)',
            'description': f'Se {d} los medios (~1500 Hz).'})
    if abs(recipe.get('eq_high_boost', 0)) > 0.1:
        d = 'reforzaran' if recipe['eq_high_boost'] > 0 else 'atenuaran'
        steps.append({'icon': '✨', 'title': f'EQ Agudos ({recipe["eq_high_boost"]:+.1f} dB)',
            'description': f'Se {d} los agudos (~8000 Hz).'})
    if recipe.get('normalize'):
        steps.append({'icon': '📏', 'title': 'Normalizacion',
            'description': f'Volumen normalizado a niveles optimos.'})

    fmt_names = {'flac': 'FLAC 24-bit (Lossless)', 'wav': 'WAV 24-bit (Alta calidad)', 'mp3': 'MP3 320kbps (Comprimido)'}
    steps.append({'icon': '💾', 'title': 'Formato de Salida',
        'description': fmt_names.get(recipe['output_format'], recipe['output_format'].upper())})

    return jsonify({
        'filename': file_info.get('filename', ''),
        'recipe': recipe,
        'steps': steps,
        'original_score': file_info.get('quality_score', 0),
        'improvement_potential': file_info.get('improvement_potential', 0),
        'estimated_new_score': min(100, file_info.get('quality_score', 70) + file_info.get('improvement_potential', 15) * 0.7),
    })

@app.route('/api/enhance/apply', methods=['POST'])
def apply_enhancement():
    data = request.get_json(silent=True) or {}
    filepaths = data.get('filepaths', [])
    user_output_format = (data.get('output_format') or '').lower()

    if not filepaths:
        return jsonify({'error': 'No se especificaron archivos'}), 400

    job_id = f"job_{int(time.time() * 1000)}"
    improvement_jobs[job_id] = {
        'status': 'processing', 'progress': 0,
        'total': len(filepaths), 'completed': 0,
        'results': [], 'errors': [],
    }

    def process_files():
        for i, filepath in enumerate(filepaths):
            try:
                full_path = resolve_path(filepath)
                if not os.path.exists(full_path):
                    improvement_jobs[job_id]['errors'].append({'filepath': filepath, 'error': 'Archivo no encontrado'})
                    continue

                file_info = analyze_file(full_path)
                recipe = build_enhancement_recipe(file_info)
                if user_output_format in ('flac', 'wav', 'mp3'):
                    recipe['output_format'] = user_output_format

                base_name = os.path.splitext(os.path.basename(full_path))[0]
                ext = recipe['output_format']
                output_filename = f"{base_name}_MEJORADO.{ext}"
                output_path = os.path.join(IMPROVED_DIR, output_filename)

                counter = 1
                while os.path.exists(output_path):
                    output_filename = f"{base_name}_MEJORADO_{counter}.{ext}"
                    output_path = os.path.join(IMPROVED_DIR, output_filename)
                    counter += 1

                success, log, actual_output, comparison = apply_audio_enhancements(full_path, output_path, recipe)
                if actual_output and actual_output != output_path:
                    output_path = actual_output
                    output_filename = os.path.basename(actual_output)

                improvement_jobs[job_id]['completed'] += 1
                improvement_jobs[job_id]['progress'] = int((improvement_jobs[job_id]['completed'] / len(filepaths)) * 100)

                if success:
                    improvement_jobs[job_id]['results'].append({
                        'original': os.path.basename(full_path),
                        'output': output_filename,
                        'output_path': output_path,
                        'log': log,
                        'download_url': f'/api/download/{output_filename}',
                        'comparison': comparison,
                    })
                else:
                    improvement_jobs[job_id]['errors'].append({
                        'filepath': filepath,
                        'error': log[0] if log else 'Error desconocido',
                    })
            except Exception as e:
                improvement_jobs[job_id]['errors'].append({'filepath': filepath, 'error': str(e)})

        improvement_jobs[job_id]['status'] = 'completed' if not improvement_jobs[job_id]['errors'] else 'completed_with_errors'

    thread = threading.Thread(target=process_files)
    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id': job_id, 'status': 'started',
        'total_files': len(filepaths),
        'output_directory': IMPROVED_DIR,
    })

@app.route('/api/enhance/status/<job_id>')
def enhancement_status(job_id):
    job = improvement_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Trabajo no encontrado'}), 404
    return jsonify(job)

@app.route('/api/download/<filename>')
def download_file(filename):
    filepath = os.path.join(IMPROVED_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename)
    return jsonify({'error': 'Archivo no encontrado'}), 404

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Endpoint para subir archivos de audio via drag & drop o selector."""
    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return jsonify({'error': 'No se enviaron archivos'}), 400
    
    saved = []
    errors = []
    for file in uploaded_files:
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in AUDIO_EXTENSIONS:
                errors.append(f'{file.filename}: formato no soportado')
                continue
            # Guardar en MP3_01 (directorio de uploads)
            safe_name = os.path.basename(file.filename)
            dest = os.path.join(UPLOAD_DIR, safe_name)
            # Evitar sobrescritura
            counter = 1
            while os.path.exists(dest):
                base, ext_f = os.path.splitext(safe_name)
                dest = os.path.join(UPLOAD_DIR, f"{base}_{counter}{ext_f}")
                counter += 1
            file.save(dest)
            saved.append(os.path.normpath(dest))
    
    return jsonify({
        'uploaded': len(saved),
        'errors': errors,
        'files': saved,
        'message': f'{len(saved)} archivo(s) subidos exitosamente' + (f'. {len(errors)} errores.' if errors else '')
    })

@app.route('/api/improved_files')
def list_improved_files():
    files = []
    if os.path.exists(IMPROVED_DIR):
        for f in os.listdir(IMPROVED_DIR):
            fp = os.path.join(IMPROVED_DIR, f)
            if os.path.isfile(fp):
                files.append({
                    'filename': f,
                    'size_mb': round(os.path.getsize(fp) / (1024 * 1024), 2),
                    'download_url': f'/api/download/{f}',
                    'created': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(fp))),
                })
    files.sort(key=lambda x: os.path.getmtime(os.path.join(IMPROVED_DIR, x['filename'])), reverse=True)
    return jsonify({'files': files, 'directory': IMPROVED_DIR})

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    print("=" * 60)
    print("🔊 Analizador de Calidad de Audio - Dashboard")
    print("=" * 60)
    print(f"Directorios: {AUDIO_DIRS}")
    print(f"Mejoras guardadas en: {IMPROVED_DIR}")
    print("=" * 60)
    # Puerto: usar variable de entorno PORT (Render/Railway) o 5000 por defecto
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', '').lower() == 'true'
    print(f"Servidor en http://localhost:{port}")
    # Auto-abrir navegador solo en modo .exe (no en desarrollo con Flask reloader)
    if not debug_mode and not os.environ.get('WERKZEUG_RUN_MAIN'):
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
