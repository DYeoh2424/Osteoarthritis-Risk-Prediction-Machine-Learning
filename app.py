import os
import sqlite3
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta
from flask import render_template, request, session, redirect, url_for, flash, send_file, make_response, Flask, jsonify, render_template_string
from io import BytesIO
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
import joblib
import tensorflow as tf
from openai import OpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import cv2
import json
import pdfkit
import base64
import re
import zlib
from flask_session import Session


app = Flask(__name__)
load_dotenv()
app.secret_key = os.getenv('SECRET_KEY') or 'defaultsecretkey'
app.config.update({
    'SESSION_TYPE': 'filesystem',
    'SESSION_FILE_DIR': './flask_sessions',
    'SESSION_COOKIE_NAME': 'flask_sess',
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SECURE': True,
    'PERMANENT_SESSION_LIFETIME': timedelta(hours=1)
})
Session(app)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# SQLite setup
db_path = 'users.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)''')
c.execute('''CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    filename TEXT,
    prediction TEXT,
    timestamp TEXT
)''')
conn.commit()
conn.close()

# load model
svm_model = joblib.load('svm_best_pipeline.pkl')
feature_extractor = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
class_names = ['No Risk', 'Mild Risk', 'Moderate Risk', 'Severe Risk', 'Most Serious Risk']

# Load classification report for Test set
with open("svm_classification_reports.json") as f:
    saved_reports = json.load(f)

def format_classification_report(report_json):
    lines = []
    header = f"{'':<10}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"
    lines.append(header)
    lines.append('-' * len(header))

    for label in ['0', '1', '2', '3', '4']:
        if label in report_json:
            row = report_json[label]
            line = f"{label:<10}{row['precision']:>10.2f}{row['recall']:>10.2f}{row['f1-score']:>10.2f}{int(row['support']):>10}"
            lines.append(line)

    lines.append('')
    for avg in ['accuracy', 'macro avg', 'weighted avg']:
        if avg in report_json:
            row = report_json[avg]
            if avg == 'accuracy':
                line = f"{'accuracy':<10}{'':>10}{'':>10}{'':>10}{report_json['accuracy']*100:>9.2f}%"
            else:
                line = f"{avg:<10}{row['precision']:>10.2f}{row['recall']:>10.2f}{row['f1-score']:>10.2f}{int(row['support']):>10}"
            lines.append(line)

    lines.append(f"\nTest Accuracy: {report_json['accuracy']:.4f}")
    return '\n'.join(lines)

def get_recommendation(label):
    prompt = f"""
    You are a medical assistant trained to provide clear, medically detailed, and actionable osteoarthritis recommendations 
    using the Kellgren-Lawrence (KL) grading system. Your explanation should be suitable for both public understanding and medical students.

    Instructions:
    - Briefly explain the clinical meaning of KL Grade {label} (include features like joint space narrowing, osteophytes, bone deformity, etc.).
    - Describe the symptoms a patient may experience at this stage.
    - List specific recommended actions the patient should take (e.g., types of exercise, lifestyle changes).
    - Mention relevant medical treatments or interventions (e.g., medication, therapy, surgery) and when they should be considered.
    - Keep the response friendly, but informative and structured in 2–4 well-developed paragraphs.

    KL Grading Reference:
    - Grade 0: No risk – healthy joint, no signs of osteoarthritis.
    - Grade 1: Mild risk – very early signs, such as small osteophytes.
    - Grade 2: Moderate risk – definite osteophytes, possible joint space narrowing.
    - Grade 3: Severe risk – clear joint space narrowing, multiple osteophytes, pain and stiffness.
    - Grade 4: Most serious risk – severe damage, joint deformity, total space loss.

    Generate a detailed, structured explanation for KL Grade {label}.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides medical recommendations."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def generate_gradcam(image_array, model, last_conv_layer_name="top_conv", pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def wrap_text(text, width=95):
    lines = []
    for paragraph in text.split("\n"):
        while len(paragraph) > width:
            split_at = paragraph.rfind(' ', 0, width)
            if split_at == -1:
                split_at = width
            lines.append(paragraph[:split_at])
            paragraph = paragraph[split_at:].lstrip()
        if paragraph:
            lines.append(paragraph)
    return lines

def add_page_number(pdf_canvas: canvas):
    page_num = pdf_canvas.getPageNumber()
    text = f"Page {page_num}"
    pdf_canvas.setFont("Helvetica", 9)
    pdf_canvas.setFillColorRGB(0.4, 0.4, 0.4)
    pdf_canvas.drawRightString(570, 20, text)

@app.route('/', methods=['GET', 'POST'])
def index():
    session.setdefault('logged_in', False)
    session.setdefault('latest_prediction', None)
    session.setdefault('latest_recommendation', None)
    session.setdefault('latest_image', None)
    session.setdefault('latest_gradcam', None)
    session.setdefault('latest_summary', None)
    session.setdefault('latest_trend', None)
    session.setdefault('chart_image_data', None)
    
    if 'logged_in' not in session:
            session['logged_in'] = False
            
    prediction = None
    recommendation = None
    uploaded_image = None
    gradcam_image = None
    classification_summary = None
    trend_summary = None
    history_labels = []
    history_times = []
    show_history = False

    if session['logged_in']:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT prediction, timestamp FROM history WHERE username=? ORDER BY timestamp ASC", (session['username'],))
        rows = c.fetchall()
        conn.close()

        if rows:
            prediction_to_index = {name: idx for idx, name in enumerate(class_names)}
            history_labels = [prediction_to_index.get(row[0], 0) for row in rows]
            history_times = [row[1] for row in rows]
            show_history = True

    if request.method == 'POST':
        # Check if user submitted prediction form
        if 'file' in request.files:
            file = request.files['file']
            if file:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                image = Image.open(filepath).convert('RGB').resize((224, 224))
                x = np.expand_dims(np.array(image), axis=0)
                x = preprocess_input(x)
                features = feature_extractor.predict(x)
                pred_label = int(svm_model.predict(features)[0])

                # Grad-CAM generation
                heatmap = generate_gradcam(x, feature_extractor)
                original_image = cv2.imread(filepath)
                original_image = cv2.resize(original_image, (224, 224))
                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

                cam_path = os.path.join(UPLOAD_FOLDER, 'gradcam_' + file.filename)
                cv2.imwrite(cam_path, superimposed_img)

                prediction = class_names[pred_label]
                recommendation = get_recommendation(pred_label)
                classification_summary = format_classification_report(saved_reports.get("Test", {}))
                trend_summary = request.form.get("trend_summary")
                if 'chart_image' in request.form:
                    chart_data = request.form['chart_image']
                    chart_data = chart_data.split(',')[1] if ',' in chart_data else chart_data                    
                    compressed = zlib.compress(chart_data.encode('utf-8'))
                    session['chart_image_data'] = compressed

                # Save all into session
                session['latest_prediction'] = prediction
                session['latest_recommendation'] = recommendation
                session['latest_image'] = filepath
                session['latest_gradcam'] = cam_path
                session['latest_summary'] = classification_summary
                session['latest_trend'] = trend_summary
                
                session.modified = True
                session.permanent = True 

                if session['logged_in']:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    conn = sqlite3.connect(db_path)
                    c = conn.cursor()
                    c.execute("INSERT INTO history (username, filename, prediction, timestamp) VALUES (?, ?, ?, ?)",
                              (session['username'], file.filename, prediction, timestamp))
                    conn.commit()
                    conn.close()
                    show_history = True
                    
    if not prediction:
        prediction = session.get('latest_prediction')
    if not recommendation:
        recommendation = session.get('latest_recommendation')
    if not classification_summary:
        classification_summary = session.get('latest_summary')
    if not trend_summary:
        trend_summary = session.get('latest_trend')
    if not uploaded_image:
        uploaded_image = session.get('latest_image')
    if not gradcam_image:
        gradcam_image = session.get('latest_gradcam')
        
    session.modified = True
            
    return render_template(
        'index.html',
        prediction=prediction,
        recommendation=recommendation,
        uploaded_image=uploaded_image,
        gradcam_image=gradcam_image,
        classification_summary=classification_summary,
        trend_summary=trend_summary,
        history_labels=history_labels,
        history_times=history_times,
        show_history=show_history
    )

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
            return redirect(url_for('register'))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
        except sqlite3.IntegrityError:
            flash("Username already exists.", "error")
            return redirect(url_for('register'))
        conn.close()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        row = c.fetchone()
        conn.close()
        if row and check_password_hash(row[0], password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash("Login failed. Please check your credentials.", "error")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/download', methods=['POST'])
def download_report():
    if 'chart_data_compressed' in session:
        try:
            compressed = session['chart_data_compressed']            
            chart_data = zlib.decompress(compressed).decode('utf-8')
            chart_data = f"data:image/png;base64,{chart_data}"
        except Exception as e:
            print(f"Chart decompression failed: {str(e)}")
            chart_data = None
    else:
        chart_data = None

    prediction = session.get('latest_prediction')
    recommendation = session.get('latest_recommendation')
    classification_summary = session.get('latest_summary')
    gradcam_path = session.get('latest_gradcam')
    original_path = session.get('latest_image')
    trend_summary = session.get('latest_trend') 
    chart_data = session.get('chart_image_data')
    
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    # Logo
    logo_path = "C:\\Users\\User\\OneDrive - Asia Pacific University\\FYP\\static\\logo.png"
    try:
        # Draw logo at top-left, enlarged
        p.drawImage(logo_path, 50, height - 80, width=80, height=50, preserveAspectRatio=True, mask='auto')
    except:
        pass

    p.setFont("Helvetica-Bold", 20)
    p.drawCentredString(width / 2, y - 20, "Osteoarthritis Report")
    y -= 100

    # Username
    p.setFont("Helvetica", 12)
    user_text = f"Username: {session.get('username')}" if session.get('logged_in') else "Guest User"
    p.drawString(50, y, user_text)
    y -= 10
    p.line(50, y, width - 50, y)
    y -= 30

    # Load session data
    prediction = session.get('latest_prediction')
    recommendation = session.get('latest_recommendation')
    classification_summary = session.get('latest_summary')
    gradcam_path = session.get('latest_gradcam')
    original_path = session.get('latest_image')

    if session.get('logged_in'):
        trend_summary = request.form.get('trend_summary')
        chart_data = request.form.get('chart_image')
    else:
        trend_summary = None
        chart_data = None

    if prediction:
        # Prediction
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y, f"Prediction: {prediction}")
        y -= 20

        # Recommendation
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y, "Recommendation:")
        y -= 15
        p.setFont("Helvetica", 10)
        for line in wrap_text(recommendation):
            if y < 50:
                add_page_number(p)
                p.showPage()
                y = height - 50
            p.drawString(50, y, line)
            y -= 12
        y -= 10

        # Classification Report Block
        p.setFillColor(colors.darkblue)
        p.rect(45, y - 20, width - 90, 20, fill=1, stroke=0)
        p.setFillColor(colors.white)
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y - 16, "Classification Report")
        y -= 35

        # Prepare table data
        table_data = [['Classes', 'Precision', 'Recall', 'F1-Score', 'Support']]
        lines = classification_summary.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('-'):
                continue

            parts = re.split(r'\s{2,}', line)

            if len(parts) == 5:
                table_data.append(parts)
            elif "accuracy" in line and "%" in line:
                match = re.search(r'accuracy\s+([\d.]+)%', line)
                if match:
                    table_data.append(['Accuracy', match.group(1), '', '', ''])
            elif "macro avg" in line:
                table_data.append(['Macro Avg'] + parts[1:])
            elif "weighted avg" in line:
                table_data.append(['Weighted Avg'] + parts[1:])
            elif "Test Accuracy" in line:
                match = re.search(r'Accuracy:\s+([\d.]+)', line)
                if match:
                    table_data.append(['Test Accuracy', match.group(1), '', '', ''])

        # Create and style the table
        table = Table(table_data, colWidths=[100, 80, 80, 80, 80], hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
        ]))

        table_width, table_height = table.wrapOn(p, width - 100, y)
        if y - table_height < 50:
            add_page_number(p)
            p.showPage()
            y = height - 50

        # Draw table on canvas
        table.drawOn(p, 50, y - table_height)
        y -= table_height + 20
            
        # Page 2: X-ray and GradCAM
        add_page_number(p)
        p.showPage()
        y = height - 50

        # Original Image
        if original_path and os.path.exists(original_path):
            p.setFillColor(colors.darkblue)
            p.rect(45, y - 20, width - 90, 20, fill=1, stroke=0)
            p.setFillColor(colors.white)
            p.setFont("Helvetica-Bold", 12)
            p.drawString(50, y - 16, "Original X-ray")
            y -= 240
            p.drawImage(original_path, 50, y, width=200, preserveAspectRatio=True, mask='auto')
            y -= 20

        # Grad-CAM Image
        if gradcam_path and os.path.exists(gradcam_path):
            if y < 300:
                add_page_number(p)
                p.showPage()
                y = height - 50
            p.setFillColor(colors.darkblue)
            p.rect(45, y - 20, width - 90, 20, fill=1, stroke=0)
            p.setFillColor(colors.white)
            p.setFont("Helvetica-Bold", 12)
            p.drawString(50, y - 16, "Grad-CAM Heatmap")
            y -= 240
            p.drawImage(gradcam_path, 50, y, width=200, preserveAspectRatio=True, mask='auto')
            y -= 20

        # Page 3: Chart
        if chart_data:
            add_page_number(p)
            p.showPage()
            y = height - 50
            p.setFont("Helvetica-Bold", 14)
            p.drawString(50, y, "Prediction History Chart")
            y -= 30

            chart_bytes = base64.b64decode(chart_data.split(',')[1])

            rect_x = 45
            rect_y = y - 420
            rect_width = 500
            rect_height = 380
            p.setFillColorRGB(0, 0, 0)
            p.rect(rect_x, rect_y, rect_width, rect_height, fill=1, stroke=0)

            img_reader = ImageReader(BytesIO(chart_bytes))
            p.drawImage(
                img_reader,
                rect_x + 10,
                rect_y + 10,
                width=rect_width - 20,
                height=rect_height - 20,
                preserveAspectRatio=True,
                mask='auto'
            )
            y = rect_y - 30

            # Trend Summary after chart
            if trend_summary:
                p.setFont("Helvetica-Bold", 12)
                p.drawString(50, y, "Trend Summary:")
                y -= 15
                p.setFont("Helvetica", 10)
                for line in wrap_text(trend_summary):
                    if y < 50:
                        add_page_number(p)
                        p.showPage()
                        y = height - 50
                    p.drawString(50, y, line)
                    y -= 12
                y -= 10

    add_page_number(p)
    p.showPage()
    p.save()
    buffer.seek(0)
    
    session.modified = True

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Osteoarthritis_Report.pdf",
        mimetype="application/pdf"
    )
    
if __name__ == '__main__':
    app.run(debug=True)