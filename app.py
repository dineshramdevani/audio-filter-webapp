import os
import numpy as np
from flask import Flask, render_template, request, send_file, send_from_directory
# Set non-GUI backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, bessel, ellip, freqs, lti, impulse, lfilter
import soundfile as sf
from fpdf import FPDF


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

current_audio = {"input": None, "output": None, "fs": None}

def reset_output_folder():
    for f in os.listdir(OUTPUT_FOLDER):
        os.remove(os.path.join(OUTPUT_FOLDER, f))

def generate_filter_response(method, ftype, order, low, high, ripple, rs):
    Wn = [2 * np.pi * low, 2 * np.pi * high] if ftype in ["bandpass", "bandstop"] else 2 * np.pi * low
    if method == "ellip":
        b, a = ellip(order, ripple, rs, Wn, btype=ftype, analog=True)
    elif method == "cheby1":
        b, a = cheby1(order, ripple, Wn, btype=ftype, analog=True)
    elif method == "bessel":
        b, a = bessel(order, Wn, btype=ftype, analog=True)
    else:
        b, a = butter(order, Wn, btype=ftype, analog=True)
    return b, a

def plot_and_save(fig, name):
    path = os.path.join(OUTPUT_FOLDER, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

@app.route('/', methods=['GET', 'POST'])
def index():
    reset_output_folder()
    global current_audio
    graphs = []
    error = None

    if request.method == 'POST':
        try:
            method = request.form['method']
            ftype = request.form['filter_type']
            order = int(request.form['order'])
            cutoff = float(request.form['cutoff'])
            cutoff2 = request.form.get('cutoff2', '')
            low = cutoff
            high = float(cutoff2) if cutoff2 else None
            ripple = float(request.form.get('ripple', 1))
            rs = float(request.form.get('stopband', 40))

            file = request.files.get('file')  # 🔧 fixed input name
            audio_loaded = False

            if file and file.filename.endswith('.wav'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                data, fs = sf.read(filepath)
                if len(data.shape) > 1:
                    data = data[:, 0]
                current_audio["input"] = data
                current_audio["fs"] = fs
                audio_loaded = True

            b, a = generate_filter_response(method, ftype, order, low, high or low, ripple, rs)
            w, h = freqs(b, a, worN=np.logspace(1, 5, 1000))
            fig = plt.figure(figsize=(8, 3))
            plt.semilogx(w, 20 * np.log10(np.abs(h)))
            plt.title("Filter Frequency Response")
            plt.grid(True, which='both')
            plot_and_save(fig, "filter_response")
            graphs.append("filter_response.png")

            sys = lti(b, a)
            t, y = impulse(sys)
            fig = plt.figure(figsize=(8, 3))
            plt.plot(t, y)
            plt.title("Filter Impulse Response")
            plt.grid(True)
            plot_and_save(fig, "impulse_response")
            graphs.append("impulse_response.png")

            if audio_loaded:
                nyq = 0.5 * current_audio["fs"]
                norm_Wn = [low / nyq, high / nyq] if ftype in ["bandpass", "bandstop"] else low / nyq

                if method == "ellip":
                    b, a = ellip(order, ripple, rs, norm_Wn, btype=ftype)
                elif method == "cheby1":
                    b, a = cheby1(order, ripple, norm_Wn, btype=ftype)
                elif method == "bessel":
                    b, a = bessel(order, norm_Wn, btype=ftype)
                else:
                    b, a = butter(order, norm_Wn, btype=ftype)

                filtered = lfilter(b, a, current_audio["input"])
                current_audio["output"] = filtered

                time = np.arange(len(current_audio["input"])) / current_audio["fs"]

                fig = plt.figure(figsize=(8, 3))
                plt.plot(time, current_audio["input"])
                plt.title("Input Audio - Time Domain")
                plt.grid(True)
                plot_and_save(fig, "input_time")
                graphs.append("input_time.png")

                fig = plt.figure(figsize=(8, 3))
                freq = np.fft.rfftfreq(len(current_audio["input"]), d=1 / current_audio["fs"])
                fft_input = np.fft.rfft(current_audio["input"])
                plt.semilogx(freq, 20 * np.log10(np.abs(fft_input) + 1e-10))
                plt.title("Input Audio - Frequency Domain")
                plt.grid(True, which='both')
                plot_and_save(fig, "input_freq")
                graphs.append("input_freq.png")

                fig = plt.figure(figsize=(8, 3))
                plt.plot(time, filtered)
                plt.title("Filtered Audio - Time Domain")
                plt.grid(True)
                plot_and_save(fig, "output_time")
                graphs.append("output_time.png")

                fig = plt.figure(figsize=(8, 3))
                freq = np.fft.rfftfreq(len(filtered), d=1 / current_audio["fs"])
                fft_out = np.fft.rfft(filtered)
                plt.semilogx(freq, 20 * np.log10(np.abs(fft_out) + 1e-10))
                plt.title("Filtered Audio - Frequency Domain")
                plt.grid(True, which='both')
                plot_and_save(fig, "output_freq")
                graphs.append("output_freq.png")

        except Exception as e:
            error = str(e)

    return render_template("index.html", graph_files=graphs, input_audio=True if current_audio["input"] is not None else None, output_audio=True if current_audio["output"] is not None else None, error=error)

@app.route('/play/<type>')
def play_audio(type):
    if type == "input" and current_audio["input"] is not None:
        temp_path = os.path.join(OUTPUT_FOLDER, "input.wav")
        sf.write(temp_path, current_audio["input"], current_audio["fs"])
        return send_file(temp_path)
    elif type == "output" and current_audio["output"] is not None:
        temp_path = os.path.join(OUTPUT_FOLDER, "output.wav")
        sf.write(temp_path, current_audio["output"], current_audio["fs"])
        return send_file(temp_path)
    return "Audio not available", 404

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/download_pdf')
def download_pdf():
    pdf = FPDF()
    for fname in sorted(os.listdir(OUTPUT_FOLDER)):
        if fname.endswith(".png"):
            pdf.add_page()
            img_path = os.path.join(OUTPUT_FOLDER, fname)
            pdf.image(img_path, x=10, y=10, w=180)
    out_pdf_path = os.path.join(OUTPUT_FOLDER, "graphs.pdf")
    pdf.output(out_pdf_path)
    return send_file(out_pdf_path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # required by Render
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))


