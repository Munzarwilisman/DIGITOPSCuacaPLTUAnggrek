# full_app_with_fixed_pdf.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import plotly.express as px
import anthropic
import os
from io import BytesIO
import requests
import folium
from streamlit_folium import st_folium
from bs4 import BeautifulSoup
import time
import base64

# ReportLab / PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, Frame, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.utils import ImageReader

# Google Sheets integration
import gspread
from google.oauth2.service_account import Credentials

# ====== LOAD DATA ======
@st.cache_data
def load_google_sheet(sheet_name):
    try:
        url = "https://docs.google.com/spreadsheets/d/15d3TQ1fCuWtjXzu2vIfUPFdqK54UDn0A8xcIZPNM99k/export?format=xlsx"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(
                f"Gagal mengunduh file dari Google Sheet untuk sheet {sheet_name}. "
                f"Status code: {response.status_code}"
            )
        
        # Membaca langsung sheet tertentu
        return pd.read_excel(BytesIO(response.content), sheet_name=sheet_name, engine="openpyxl")
    
    except Exception as e:
        st.error(f"⚠️ Gagal memuat data dari sheet {sheet_name}: {str(e)}")
        return pd.DataFrame()

# ================================
# Ambil data curah hujan
# ================================
def get_rainfall_data():
    """Ambil data curah hujan dari Google Sheets"""
    try:
        df = load_google_sheet("Curah Hujan")
        
        if df.empty:
            st.error("⚠️ Data kosong atau gagal dimuat dari Google Sheets.")
            return None
        
        # Pastikan kolom yang diperlukan ada
        if 'TANGGAL' in df.columns and 'Curah Hujan' in df.columns:
            return df
        else:
            st.error("⚠️ Format data tidak sesuai. Pastikan ada kolom 'TANGGAL' dan 'Curah Hujan'.")
            return None

    except Exception as e:
        st.error(f"⚠️ Error mengambil data curah hujan: {e}")
        return None

# ================================
# Utility / Data preparation
# ================================
def clean_data(df):
    df['Curah Hujan'] = df['Curah Hujan'].replace(['#NA', '-', 'NA', np.nan], 0)
    df['Curah Hujan'] = df['Curah Hujan'].astype(str).str.replace(',', '.').astype(float)
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['TANGGAL'])
    df.set_index('TANGGAL', inplace=True)
    return df

def aggregate_data(df, method):
    if method == "Jumlah (Sum)":
        return df.resample('M').sum()
    elif method == "Rata-rata (Mean)":
        return df.resample('M').mean()
    else:
        return df.resample('M').sum()

# ================================
# EDA / Forecasting / Evaluation
# ================================
def perform_eda(df, agg_method):
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    st.write("**Statistik Deskriptif:**")
    st.write(df.describe())

    # --- Time Series Curah Hujan ---
    st.write("**Time Series Curah Hujan:**")
    df_plot = df.reset_index()  # pastikan kolom TANGGAL tersedia
    fig = px.line(
        df_plot,
        x="TANGGAL",
        y="Curah Hujan",
        markers=True,
        title="Time Series Curah Hujan Gorontalo"
    )
    fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Distribusi Curah Hujan ---
    st.write("**Distribusi Curah Hujan:**")
    fig = px.histogram(
        df_plot,
        x="Curah Hujan",
        nbins=50,
        title="Distribusi Curah Hujan"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Analisis Bulanan ---
    st.write("**Analisis Bulanan:**")
    monthly = aggregate_data(df, agg_method).reset_index()
    fig = px.bar(
        monthly,
        x="TANGGAL",
        y="Curah Hujan",
        title=f"Curah Hujan Bulanan ({agg_method})"
    )
    fig.update_layout(
        xaxis_title="Bulan",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
# Forecast
def forecast_rainfall(df, model_type, forecast_period, agg_method):
    monthly_data = aggregate_data(df, agg_method)
    if model_type == 'ARIMA':
        model = ARIMA(monthly_data, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_period)
    elif model_type == 'SARIMA':
        model = SARIMAX(monthly_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=forecast_period)
    elif model_type == 'Prophet':
        prophet_df = monthly_data.reset_index()
        prophet_df.columns = ['ds', 'y']
        model = Prophet(yearly_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_period, freq='M')
        forecast_df = model.predict(future)
        forecast = forecast_df.tail(forecast_period)['yhat']
    return forecast

def plot_forecast(df, forecast, model_type, agg_method):
    monthly_data = aggregate_data(df, agg_method)
    last_date = monthly_data.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=len(forecast), freq='M')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
    forecast_df.set_index('Date', inplace=True)

    fig = px.line(x=monthly_data.index, y=monthly_data['Curah Hujan'], labels={'x':'Tanggal','y':'Curah Hujan'}, title=f'Forecast Curah Hujan - Model {model_type} ({agg_method})')
    fig.add_scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast')
    st.plotly_chart(fig, use_container_width=True)

def evaluate_model(df, model_type, agg_method):
    monthly_data = aggregate_data(df, agg_method)
    train_size = int(len(monthly_data) * 0.8)
    train, test = monthly_data.iloc[:train_size], monthly_data.iloc[train_size:]

    if model_type == 'ARIMA':
        model = ARIMA(train, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
    elif model_type == 'SARIMA':
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=len(test))
    elif model_type == 'Prophet':
        prophet_train = train.reset_index()
        prophet_train.columns = ['ds', 'y']
        model = Prophet(yearly_seasonality=True)
        model.fit(prophet_train)
        future = model.make_future_dataframe(periods=len(test), freq='M')
        forecast_df = model.predict(future)
        forecast = forecast_df.tail(len(test))['yhat'].values

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))

    st.write(f"**Evaluasi Model {model_type} ({agg_method}):**")
    st.write(f"- MAE (Mean Absolute Error): {mae:.2f}")
    st.write(f"- RMSE (Root Mean Squared Error): {rmse:.2f}")

    threshold_mae = 0.2 * test['Curah Hujan'].mean()
    if mae < threshold_mae:
        st.success("Model ini memiliki akurasi yang baik berdasarkan MAE.")
    else:
        st.warning("Model ini memiliki kesalahan prediksi yang cukup tinggi berdasarkan MAE. Pertimbangkan model lain atau preprocessing tambahan.")

# ================================
# AI Analysis (Claude/Anthropic)
# ================================
import pandas as pd
import anthropic

def analyze_tides_for_cleaning(tide_df):
    """
    Analisis pasang surut untuk mencari waktu surut terendah,
    pasang tertinggi, dan rekomendasi jam cleaning intake (08:00 - 16:00).
    """
    try:
        df = tide_df.copy()
        df['Waktu'] = pd.to_datetime(df['Waktu'])
        df = df.dropna(subset=['Ketinggian (m)'])

        # Cari surut terendah
        min_row = df.loc[df['Ketinggian (m)'].idxmin()]
        surut_terendah = min_row['Waktu']
        tinggi_min = min_row['Ketinggian (m)']

        # Cari pasang tertinggi
        max_row = df.loc[df['Ketinggian (m)'].idxmax()]
        pasang_tertinggi = max_row['Waktu']
        tinggi_max = max_row['Ketinggian (m)']

        # Filter rekomendasi jam 08:00 – 16:00
        work_hours_df = df[(df['Waktu'].dt.hour >= 8) & (df['Waktu'].dt.hour <= 16)]
        rekomendasi = None
        if not work_hours_df.empty:
            rekomendasi_row = work_hours_df.loc[work_hours_df['Ketinggian (m)'].idxmin()]
            rekomendasi = {
                "waktu": rekomendasi_row['Waktu'],
                "ketinggian": rekomendasi_row['Ketinggian (m)']
            }

        return {
            "surut_terendah": (surut_terendah, tinggi_min),
            "pasang_tertinggi": (pasang_tertinggi, tinggi_max),
            "rekomendasi_cleaning": rekomendasi
        }
    except Exception as e:
        return {"error": str(e)}


def ai_analysis_bmkg(bmkg_df, api_key, tide_df=None):
    client = anthropic.Anthropic(api_key=api_key)

    try:
        temp_stats = bmkg_df['Suhu (°C)'].describe().to_string()
    except Exception:
        temp_stats = 'Tidak ada data suhu'

    try:
        hum_stats = bmkg_df['Kelembapan (%)'].describe().to_string()
    except Exception:
        hum_stats = 'Tidak ada data kelembapan'
    
    # Analisis data pasang surut (jika ada)
    tide_info = ""
    if tide_df is not None and not tide_df.empty:
        try:
            tide_stats = tide_df['Ketinggian (m)'].describe().to_string()
            tide_analysis = analyze_tides_for_cleaning(tide_df)

            rekom_text = ""
            if "error" not in tide_analysis:
                surut = tide_analysis["surut_terendah"]
                pasang = tide_analysis["pasang_tertinggi"]
                rekom = tide_analysis["rekomendasi_cleaning"]

                rekom_text = f"""
Waktu Surut Terendah: {surut[0].strftime("%d-%m-%Y %H:%M")} ({surut[1]:.2f} m)
Waktu Pasang Tertinggi: {pasang[0].strftime("%d-%m-%Y %H:%M")} ({pasang[1]:.2f} m)
"""

                if rekom:
                    rekom_text += f"Rekomendasi Cleaning Intake: sekitar {rekom['waktu'].strftime('%H:%M')} (tinggi air {rekom['ketinggian']:.2f} m)"
                else:
                    rekom_text += "Tidak ada waktu surut signifikan antara jam 08:00 - 16:00."

            tide_info = f"\n\nData Pasang Surut:\n{tide_stats}\n{rekom_text}"
        except Exception:
            tide_info = "\n\nData Pasang Surut: Tidak ada data pasang surut"

    # Ambil contoh baris data
    sample_rows = bmkg_df.head(10).to_csv(index=False)

    # Prompt untuk Claude
    prompt = f"""
Saya memiliki data prakiraan BMKG dengan statistik sebagai berikut:

Suhu (°C):\n{temp_stats}

Kelembapan (%):\n{hum_stats}
{tide_info}

Contoh baris data (CSV):\n{sample_rows}

Tolong berikan analisis profesional dalam bahasa Indonesia dengan struktur:

## 1. Ringkasan Umum (Rangkuman dari data ringkasan statistik cuaca, tren suhu & kelembaban, Perkiraan hari basah / kering dan analisis pasang surut )
## 2. Tren Suhu & Kelembapan
## 3. Perkiraan Hari Basah / Kering (beri alasan)
## 4. Potensi Dampak terhadap Operasional PLTU (Batubara, Pendinginan, SDM, Kualitas Udara, dll.)
## 5. Rekomendasi Perencanaan Untuk Coal Handling dan Ash Handling (detail dan praktis untuk Antisipasi musim kering dan jika musim hujan)
## 6. Analisis Pasang Surut dan Dampaknya pada Operasional PLTU (jika data tersedia) yang berisi menjelaskan waktu surut terendah dan waktu pukul berapa akan pasang serta berikan waktu atau jam terbaik melakukan cleaning sampah di intake yaitu pada waktu air surut (rekomendasikan di jam surut antara jam 8 pagi - 16.00 siang)

Buat setiap bagian menjadi paragraf yang agak panjang (2-6 kalimat) dan di bawah setiap bagian berikan bullet point setidaknya 3-6 item tindakan/observasi yang jelas. Gunakan format Markdown rapi dan jangan potong respons.
"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2500,  # cukup panjang untuk menampung semua bagian
        temperature=0.25,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return message.content[0].text
    except Exception:
        return str(message)


# ================================
# Enhanced PDF Generator (Professional Layout) - Fixed Version
# ================================
def generate_pdf_report_bmkg(bmkg_df, ai_markdown=None, logo_bytes=None, tide_df=None, rainfall_img_url=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=36, leftMargin=36,
        topMargin=100, bottomMargin=36
    )

    # Enhanced Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", 
        parent=styles['Heading1'], 
        alignment=TA_CENTER, 
        fontSize=16, 
        spaceAfter=6,
        textColor=colors.HexColor("#1f77b4")
    )
    h2_style = ParagraphStyle(
        "H2", 
        parent=styles['Heading2'], 
        alignment=TA_LEFT, 
        fontSize=12, 
        spaceAfter=6,
        textColor=colors.HexColor("#1f77b4"),
        spaceBefore=12
    )
    h3_style = ParagraphStyle(
        "H3", 
        parent=styles['Heading3'], 
        alignment=TA_LEFT, 
        fontSize=10, 
        spaceAfter=6,
        textColor=colors.HexColor("#2ca02c")
    )
    normal_justify = ParagraphStyle(
        "Justify", 
        parent=styles['Normal'], 
        alignment=TA_JUSTIFY, 
        fontSize=9, 
        leading=12
    )
    normal_left = ParagraphStyle(
        "Left", 
        parent=styles['Normal'], 
        alignment=TA_LEFT, 
        fontSize=9, 
        leading=12
    )
    warning_style = ParagraphStyle(
        "Warning",
        parent=styles['Normal'],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.HexColor("#d62728"),
        backColor=colors.HexColor("#fff0f0"),
        borderPadding=10,
        borderColor=colors.HexColor("#d62728"),
        borderWidth=1
    )
    bullet_style = ParagraphStyle(
        "Bullet", 
        parent=styles['Normal'], 
        leftIndent=12, 
        bulletIndent=6, 
        alignment=TA_LEFT, 
        fontSize=9,
        bulletFontName='Helvetica-Bold'
    )

    elements = []

    # Professional Header with improved styling
    def draw_header(canvas, doc_obj):
        page_w, page_h = A4
        
        # Draw background rectangle for header
        canvas.setFillColor(colors.HexColor("#f0f8ff"))
        canvas.rect(0, page_h - 70, page_w, 70, fill=1, stroke=0)
        
        # Logo
        if logo_bytes:
            try:
                img_reader = ImageReader(BytesIO(logo_bytes))
                canvas.drawImage(img_reader, 36, page_h - 50, width=60, height=40, preserveAspectRatio=True, mask='auto')
            except Exception as e:
                print(f"Error drawing logo: {e}")
        
        # Title with improved styling
        canvas.setFont("Helvetica-Bold", 16)
        canvas.setFillColor(colors.HexColor("#1f77b4"))
        canvas.drawCentredString(page_w / 2.0, page_h - 40, "PRAKIRAAN DAN ANALISA CUACA")
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawCentredString(page_w / 2.0, page_h - 60, "PLTU ANGGREK")
        
        # Decorative line
        canvas.setStrokeColor(colors.HexColor("#1f77b4"))
        canvas.setLineWidth(1.5)
        canvas.line(36, page_h - 70, page_w - 36, page_h - 70)

    # Add creation timestamp
    created = Paragraph(
        f"<font size=8 color='#666666'>Dicetak: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</font>", 
        normal_left
    )
    elements.append(created)
    elements.append(Spacer(1, 12))

    # AI Analysis Section - Placed First
    elements.append(Paragraph(
        "Analisis - 🤖 DIGITOPS PLTU ANGGREK", 
        ParagraphStyle(
            "AITitle",
            parent=h2_style,
            textColor=colors.HexColor("#d62728"),
            spaceBefore=0
        )
    ))
    
    # Check if AI analysis is available and valid
    if ai_markdown and ai_markdown.strip() and not ai_markdown.startswith("(Al analysis gagal dibuat)"):
        try:
            sections = []
            current_title, current_lines = None, []
            for raw_line in ai_markdown.splitlines():
                line = raw_line.rstrip()
                if line.strip().startswith("##"):
                    if current_title or current_lines:
                        sections.append((current_title or "", "\n".join(current_lines).strip()))
                        current_lines = []
                    current_title = line.lstrip("#").strip()
                else:
                    current_lines.append(line)
            if current_title or current_lines:
                sections.append((current_title or "", "\n".join(current_lines).strip()))
                
            # Process AI content without using a Table for the entire content
            ai_content = []
            for title, body in sections:
                if title:
                    ai_content.append(Paragraph(f"<b>{title}</b>", h3_style))
                paras = [p.strip() for p in body.split("\n\n") if p.strip()]
                for p in paras:
                    lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
                    bullets = [ln[1:].strip() for ln in lines if ln.startswith("-")]
                    if bullets:
                        intro_lines = [ln for ln in lines if not ln.startswith("-")]
                        for il in intro_lines:
                            ai_content.append(Paragraph(il, normal_justify))
                        for b in bullets:
                            ai_content.append(Paragraph("• " + b, bullet_style))
                    else:
                        ai_content.append(Paragraph(p.replace("\n", " "), normal_justify))
                ai_content.append(Spacer(1, 6))
            
            # Add AI content directly to elements instead of using a large table
            # This allows proper page breaking
            for item in ai_content:
                elements.append(item)
            
        except Exception as e:
            # Fallback if AI processing fails
            error_msg = f"⚠️ Terjadi kesalahan dalam memproses analisis AI: {str(e)}"
            elements.append(Paragraph(error_msg, warning_style))
    else:
        # AI analysis is not available or failed
        if not ai_markdown or ai_markdown.startswith("(Al analysis gagal dibuat)"):
            error_text = """
            <b>⚠️ Analisis AI Tidak Tersedia</b><br/>
            <br/>
            Sistem tidak dapat menghasilkan analisis AI pada saat ini. Kemungkinan penyebab:
            <br/>• Model AI sedang dalam pemeliharaan
            <br/>• Data input tidak mencukupi untuk analisis
            <br/>• Koneksi ke layanan AI terputus
            <br/>
            <br/>Silakan hubungi tim DIGITOPS untuk bantuan lebih lanjut.
            """
        else:
            error_text = "⚠️ Analisis AI tidak tersedia atau format tidak dikenali."
        
        elements.append(Paragraph(error_text, warning_style))
    
    elements.append(Spacer(1, 18))

    # Page break before visualizations
    elements.append(PageBreak())
    
    # Visualizations Section Header
    elements.append(Paragraph("Visualisasi Data Cuaca", h2_style))
    
    # Statistik tabel
    elements.append(Paragraph("Ringkasan Statistik Cuaca", h3_style))
    try:
        stats = bmkg_df[["Suhu (°C)", "Kelembapan (%)", "Kecepatan Angin (km/j)"]].describe().round(2)
    except Exception:
        stats = bmkg_df.describe().round(2)

    # Header tabel
    table_data = [["", "Suhu (°C)", "Kelembapan (%)", "Kecepatan Angin (km/j)"]]

    # Isi tabel
    for idx, row in stats.iterrows():
        table_data.append([
            str(idx),
            str(row.get("Suhu (°C)", "-")),
            str(row.get("Kelembapan (%)", "-")),
            str(row.get("Kecepatan Angin (km/j)", "-"))
        ])

    # Buat tabel dengan 4 kolom
    tbl = Table(table_data, colWidths=[70, 90, 90, 110])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f77b4")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f8f8")]),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 12))

    # Grafik Suhu & Kelembapan
    elements.append(Paragraph("Grafik Suhu & Kelembapan", h3_style))
    try:
        fig, ax1 = plt.subplots(figsize=(5, 3))
        plot_df = bmkg_df.sort_values("Datetime").copy()
        
        # Use more professional styling for the chart
        plt.style.use('seaborn-v0_8-whitegrid')
        ax1.plot(plot_df["Datetime"], plot_df["Suhu (°C)"], color="tab:red", marker="o", 
                linewidth=1.5, markersize=3, label="Suhu (°C)")
        ax1.set_xlabel("Tanggal", fontsize=8)
        ax1.set_ylabel("Suhu (°C)", color="tab:red", fontsize=9)
        ax1.tick_params(axis='x', rotation=30, labelsize=7)
        ax1.tick_params(axis='y', labelsize=7, labelcolor="tab:red")
        
        ax2 = ax1.twinx()
        ax2.plot(plot_df["Datetime"], plot_df["Kelembapan (%)"], color="tab:blue", marker="x", 
                linewidth=1.5, markersize=3, label="Kelembapan (%)")
        ax2.set_ylabel("Kelembapan (%)", color="tab:blue", fontsize=9)
        ax2.tick_params(axis='y', labelsize=7, labelcolor="tab:blue")
        
        # Add a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
        
        fig.tight_layout()
        chart_buf = BytesIO()
        fig.savefig(chart_buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        chart_buf.seek(0)
        elements.append(Image(chart_buf, width=400, height=220))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("<i>Gambar 1: Tren Suhu dan Kelembapan</i>", 
                                ParagraphStyle("Caption", parent=normal_left, fontSize=7, textColor=colors.gray)))
    except Exception as e:
        elements.append(Paragraph(f"⚠️ Grafik gagal dibuat: {str(e)}", normal_left))
    elements.append(Spacer(1, 12))

    # Add tidal data section if available
    if tide_df is not None and not tide_df.empty:
        elements.append(Paragraph("Data Pasang Surut", h3_style))
        try:
            # Hapus kolom 'Tipe' jika ada
            tide_df_stats = tide_df.drop(columns=['Tipe'], errors='ignore')

            # Hitung statistik deskriptif
            stats = tide_df_stats[['Ketinggian (m)']].describe().round(2)

            # Buat header tabel
            table_data = [["", "Ketinggian (m)"]]

            # Isi tabel statistik
            for idx, row in stats.iterrows():
                table_data.append([str(idx), str(row['Ketinggian (m)'])])

            # Buat tabel horizontal
            tbl = Table(table_data, colWidths=[80, 100])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f77b4")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 8),
                ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f8f8")]),
            ]))
            elements.append(tbl)
            elements.append(Spacer(1, 12))
        except Exception as e:
            elements.append(Paragraph(f"⚠️ Statistik pasang surut gagal ditampilkan: {e}", normal_left))
    
    # Tidal chart
    if tide_df is not None and not tide_df.empty:
        from scipy.signal import argrelextrema

        elements.append(Paragraph("Grafik Pasang Surut", h3_style))
        try:
            tide_plot_df = tide_df.copy()
            tide_plot_df['Waktu'] = pd.to_datetime(tide_plot_df['Waktu'], errors='coerce')
            tide_plot_df['Ketinggian (m)'] = pd.to_numeric(tide_plot_df['Ketinggian (m)'], errors='coerce')

            # Use a more professional style
            plt.style.use('seaborn-v0_8-whitegrid')
            fig_tide, ax_tide = plt.subplots(figsize=(5, 3))

            if 'Tipe' in tide_plot_df.columns:
                for tipe in tide_plot_df['Tipe'].unique():
                    subset = tide_plot_df[tide_plot_df['Tipe'] == tipe]
                    color = "tab:green" if "High" in tipe else "tab:red"
                    marker = "^" if "High" in tipe else "v"
                    ax_tide.plot(subset['Waktu'], subset['Ketinggian (m)'], 
                                marker=marker, color=color, linestyle='-', 
                                markersize=6, label=tipe)

                    # Label jam hanya untuk titik High & Low
                    for x, y in zip(subset['Waktu'], subset['Ketinggian (m)']):
                        ax_tide.annotate(
                            x.strftime("%H:%M"),
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 8),
                            ha='center',
                            fontsize=6,
                            rotation=45
                        )
            else:
                ax_tide.plot(
                    tide_plot_df['Waktu'], tide_plot_df['Ketinggian (m)'],
                    color="tab:blue", marker='o', linewidth=1.5, markersize=3
                )

                # Cari puncak (maks) & lembah (min)
                y_array = tide_plot_df['Ketinggian (m)'].values
                max_idx = argrelextrema(y_array, np.greater)[0]
                min_idx = argrelextrema(y_array, np.less)[0]
                extrema_idx = np.sort(np.concatenate([max_idx, min_idx]))

                # Tambahkan label jam hanya di puncak & lembah
                for idx in extrema_idx:
                    x = tide_plot_df['Waktu'].iloc[idx]
                    y = tide_plot_df['Ketinggian (m)'].iloc[idx]
                    ax_tide.annotate(
                        x.strftime("%H:%M"),
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha='center',
                        fontsize=6,
                        rotation=45
                    )

            ax_tide.set_xlabel("Waktu", fontsize=8)
            ax_tide.set_ylabel("Ketinggian (m)", fontsize=9)
            ax_tide.tick_params(axis='x', rotation=30, labelsize=7)
            ax_tide.tick_params(axis='y', labelsize=7)
            ax_tide.grid(True, linestyle='--', alpha=0.7)
            if 'Tipe' in tide_plot_df.columns:
                ax_tide.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

            fig_tide.tight_layout()
            tide_chart_buf = BytesIO()
            fig_tide.savefig(tide_chart_buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig_tide)
            tide_chart_buf.seek(0)

            elements.append(Image(tide_chart_buf, width=400, height=220))
            elements.append(Spacer(1, 6))
            elements.append(Paragraph("<i>Gambar 2: Grafik Pasang Surut</i>", 
                                    ParagraphStyle("Caption", parent=normal_left, fontSize=7, textColor=colors.gray)))
        except Exception as e:
            elements.append(Paragraph(f"⚠️ Grafik pasang surut gagal dibuat: {e}", normal_left))

    # Add rainfall prediction image if available
    if rainfall_img_url:
        try:
            elements.append(Paragraph("INFOGRAFIS BMKG", h3_style))
            # Download the image
            response = requests.get(rainfall_img_url, timeout=10)
            if response.status_code == 200:
                img_data = BytesIO(response.content)
                elements.append(Image(img_data, width=400, height=300))
                elements.append(Spacer(1, 6))
                elements.append(Paragraph("<i>Gambar 3: Infografis Prediksi Curah Hujan BMKG</i>", 
                                        ParagraphStyle("Caption", parent=normal_left, fontSize=7, textColor=colors.gray)))
                elements.append(Spacer(1, 12))
            else:
                elements.append(Paragraph("⚠️ Gambar prediksi hujan tidak dapat diunduh.", normal_left))
        except Exception:
            elements.append(Paragraph("⚠️ Gambar prediksi hujan tidak dapat ditampilkan.", normal_left))

    # Professional Footer
    elements.append(Spacer(1, 12))
    footer_text = """
    <font size=8 color='#666666'>
    <b>Catatan:</b> Laporan otomatis berdasarkan data BMKG. Verifikasi lapangan disarankan untuk keputusan operasional.<br/>
    Dokumen ini diproduksi oleh Sistem DIGITOPS PLTU Anggrek
    </font>
    """
    elements.append(Paragraph(footer_text, normal_left))

    # Build the document with header on all pages
    doc.build(elements, onFirstPage=draw_header, onLaterPages=draw_header)
    buffer.seek(0)
    return buffer.read()

# ================================
# BMKG tab (dengan pilihan WorldTides atau Stormglass)
# ================================
def get_worldtides_data(api_key, lat, lon, days=1):
    """
    Ambil data pasang surut dari WorldTides API
    """
    try:
        url = f"https://www.worldtides.info/api/v3?extremes&lat={lat}&lon={lon}&days={days}&key={api_key}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "extremes" in data:
            records = []
            for e in data["extremes"]:
                records.append({
                    "Waktu": pd.to_datetime(e["date"]),
                    "Tipe": e["type"],
                    "Ketinggian (m)": e["height"]
                })
            return pd.DataFrame(records)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal ambil data WorldTides: {e}")
        return pd.DataFrame()

def get_stormglass_data(api_key, lat, lng, days=1):
    """
    Ambil data pasang surut dari Stormglass API
    """
    try:
        start = pd.Timestamp.now().floor("D")
        end = start + pd.Timedelta(days=days)

        url = f"https://api.stormglass.io/v2/tide/sea-level/point?lat={lat}&lng={lng}&start={int(start.timestamp())}&end={int(end.timestamp())}"

        headers = {"Authorization": api_key}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "data" in data:
            tide_data = data["data"]
            records = []
            for entry in tide_data:
                records.append({
                    "Waktu": pd.to_datetime(entry["time"]),
                    "Ketinggian (m)": entry["sg"]
                })
            return pd.DataFrame(records)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal ambil data Stormglass: {e}")
        return pd.DataFrame()

def get_marine_api_data(lat, lng, past_days=1, forecast_days=3, timezone="auto"):
    """
    Ambil data pasang surut dari Marine API (Open-Meteo) - semua data per jam
    """
    try:
        url = (
            f"https://marine-api.open-meteo.com/v1/marine?"
            f"latitude={lat}&longitude={lng}"
            f"&hourly=sea_level_height_msl"
            f"&past_days={past_days}&forecast_days={forecast_days}"
            f"&timezone={timezone}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if "hourly" in data:
            hourly_data = data["hourly"]
            time_list = hourly_data.get("time", [])
            sea_level_list = hourly_data.get("sea_level_height_msl", [])
            
            records = []
            for i in range(len(time_list)):
                records.append({
                    "Waktu": pd.to_datetime(time_list[i]),
                    "Ketinggian (m)": sea_level_list[i]
                })
            
            df = pd.DataFrame(records)

            # Pastikan data minimal sampai forecast_days
            today = pd.Timestamp.now().normalize()
            end_date = today + pd.Timedelta(days=forecast_days)
            df = df[(df["Waktu"] >= today) & (df["Waktu"] < end_date)]

            return df.reset_index(drop=True)
        
        else:
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Gagal ambil data Marine API: {e}")
        return pd.DataFrame()



def show_bmkg_tab():
    st.subheader("🌦️ Data Prakiraan Cuaca BMKG")
    wilayah_adm4 = st.text_input("Masukkan kode wilayah BMKG (adm4)", "75.05.03.2001")

    # Ambil logo otomatis dari Google Drive
    gdrive_url = "https://drive.google.com/uc?id=1jrDgo2FOMoZuuDv6ZX5OV2agcEImmsTf"
    logo_bytes = None
    try:
        response = requests.get(gdrive_url, timeout=10)
        if response.status_code == 200:
            logo_bytes = response.content
            st.image(logo_bytes, caption="Logo dari Google Drive", width=150)
            st.session_state["logo_bytes"] = logo_bytes
            st.success("✅ Logo berhasil dimuat otomatis dari Google Drive")
        else:
            st.error(f"Gagal mengambil logo dari Google Drive. Status: {response.status_code}")
    except Exception as e:
        st.error(f"⚠️ Error mengambil logo: {e}")

    # Jika ingin fallback ke upload manual jika logo_bytes None
    if logo_bytes is None:
        logo_file = st.file_uploader("Upload Logo untuk Laporan (jika otomatis gagal)", type=["png", "jpg", "jpeg"], key="logo_uploader")
        if logo_file is not None:
            logo_bytes = logo_file.read()
            st.session_state["logo_bytes"] = logo_bytes
            st.success(f"Logo berhasil diupload secara manual: {logo_file.name}")

    logo_bytes = st.session_state.get("logo_bytes", None)

    try:
        url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={wilayah_adm4}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        lokasi = data.get("lokasi", {})
        st.markdown(f"**Lokasi:** {lokasi.get('desa','N/A')}, {lokasi.get('kecamatan','N/A')}, {lokasi.get('kotkab','N/A')} - {lokasi.get('provinsi','N/A')}")

        cuaca_data = data.get("data", [])
        records = []
        for hari in cuaca_data:
            for prakiraan_harian in hari.get("cuaca", []):
                for cuaca in prakiraan_harian:
                    records.append({
                        "Datetime": cuaca.get("local_datetime"),
                        "Cuaca": cuaca.get("weather_desc"),
                        "Suhu (°C)": cuaca.get("t"),
                        "Kelembapan (%)": cuaca.get("hu"),
                        "Kecepatan Angin (km/j)": cuaca.get("ws"),
                        "Arah Angin": cuaca.get("wd"),
                        "Jarak Pandang": cuaca.get("vs_text"),
                    })

        if records:
            df_bmkg = pd.DataFrame(records)
            df_bmkg["Datetime"] = pd.to_datetime(df_bmkg["Datetime"], errors="coerce")
            st.dataframe(df_bmkg)

            # Grafik suhu & kelembapan
            fig = px.line(
                df_bmkg.sort_values("Datetime"), 
                x="Datetime", y=["Suhu (°C)", "Kelembapan (%)"], 
                markers=True, title="Perubahan Suhu & Kelembapan"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("🤖 Analisis AI DIGITOPS untuk Data BMKG")
            api_key = st.text_input("Masukkan API Key Claude AI", type='password', key='bmkg_api')
            ai_result = None
            
            # Pilihan API Pasang Surut
            st.markdown("---")
            st.subheader("🌊 Data Pasang Surut")
            
            tide_api_choice = st.radio(
                "Pilih API Pasang Surut:",
                ["WorldTides", "Stormglass", "Marine API (Open-Meteo)"],
                horizontal=True,
                key="tide_api_choice"
            )
            
            df_tide = None
            
            if tide_api_choice == "WorldTides":
                api_key_wt = st.text_input("Masukkan API Key WorldTides", type="password", key="worldtides_api")
                if api_key_wt:
                    try:
                        lat, lng = 0.9666, 122.9583  # Lokasi PLTU Anggrek Ilangata
                        df_tide = get_worldtides_data(api_key_wt, lat, lng, days=1)

                        if not df_tide.empty:
                            st.dataframe(df_tide)
                            fig_tide = px.line(
                                df_tide, x="Waktu", y="Ketinggian (m)", color="Tipe",
                                title="Grafik Pasang Surut PLTU Anggrek (WorldTides)",
                                markers=True
                            )
                            st.plotly_chart(fig_tide, use_container_width=True)
                        else:
                            st.warning("⚠️ Tidak ada data pasang surut dari WorldTides.")
                    except Exception as e:
                        st.error(f"Gagal mengambil data pasang surut: {e}")
            
            elif tide_api_choice == "Stormglass":
                api_key_sg = st.text_input("Masukkan API Key Stormglass", type="password", key="stormglass_api")
                if api_key_sg:
                    try:
                        lat, lng = 0.9666, 122.9583  # Lokasi PLTU Anggrek Ilangata
                        df_tide = get_stormglass_data(api_key_sg, lat, lng, days=1)

                        if not df_tide.empty:
                            st.dataframe(df_tide)
                            fig_tide = px.line(
                                df_tide, x="Waktu", y="Ketinggian (m)",
                                title="Grafik Pasang Surut PLTU Anggrek (Stormglass)",
                                markers=True
                            )
                            st.plotly_chart(fig_tide, use_container_width=True)
                        else:
                            st.warning("⚠️ Tidak ada data pasang surut dari Stormglass.")
                    except Exception as e:
                        st.error(f"Gagal mengambil data pasang surut: {e}")
            
            else:  # Marine API (Open-Meteo)
                try:
                    lat, lng = 0.8506, 122.7963  # Koordinat PLTU Anggrek
                    df_tide = get_marine_api_data(lat, lng, past_days=1, forecast_days=1)

                    if not df_tide.empty:
                        st.dataframe(df_tide)

                        fig_tide = px.line(
                            df_tide,
                            x="Waktu",
                            y="Ketinggian (m)",
                            title="Grafik Pasang Surut PLTU Anggrek (Marine API)",
                            markers=True
                        )

                        # Agar grafik lebih rapi
                        fig_tide.update_traces(line=dict(width=2))
                        fig_tide.update_layout(
                            xaxis_title="Waktu",
                            yaxis_title="Ketinggian (m)",
                            template="plotly_white"
                        )

                        st.plotly_chart(fig_tide, use_container_width=True)
                    else:
                        st.warning("⚠️ Tidak ada data pasang surut dari Marine API.")
                except Exception as e:
                    st.error(f"Gagal mengambil data pasang surut: {e}")


            # Prediksi Hujan Bulanan (langsung tampilkan dari URL)
            st.markdown("---")
            st.subheader("🌧️ INFOGRAFIS CUACA (BMKG)")

            rainfall_img_url = "https://peta-maritim.bmkg.go.id/marine-data/doc/cuaca/pelabuhan/AB004.png"
            try:
                st.image(rainfall_img_url, caption="Prediksi Hujan Bulanan BMKG", use_container_width=True)
            except Exception as e:
                st.error(f"Gagal memuat INFOGRAFIS CUACA: {e}")

            # Jalankan Analisis AI
            if api_key and st.button("Jalankan Analisis AI (BMKG)"):
                with st.spinner('Menghubungi Claude untuk analisis...'):
                    try:
                        ai_result = ai_analysis_bmkg(df_bmkg, api_key, df_tide)
                        st.markdown(ai_result)
                    except Exception as e:
                        st.error(f"Gagal menjalankan analisis AI: {e}")

            # Generate PDF
            st.markdown("---")
            st.subheader("📄 Generate PDF Report")
            if st.button("Buat & Unduh PDF Laporan BMKG"):
                with st.spinner('Membuat PDF...'):
                    try:
                        if api_key and ai_result is None:
                            try:
                                ai_result = ai_analysis_bmkg(df_bmkg, api_key, df_tide)
                            except Exception:
                                ai_result = "(AI analysis gagal dibuat)"
                        pdf_bytes = generate_pdf_report_bmkg(
                            df_bmkg, ai_markdown=ai_result, 
                            logo_bytes=logo_bytes, tide_df=df_tide,
                            rainfall_img_url=rainfall_img_url
                        )
                        
                        today = pd.Timestamp.now().strftime("%d%m%Y")
                        file_name = f"{today}_LAPORAN_CUACA_PLTU_ANGGREK.pdf"
                        
                        st.success('PDF berhasil dibuat.')
                        st.download_button(
                            'Download Laporan PDF BMKG', 
                            data=pdf_bytes, 
                            file_name=file_name, 
                            mime='application/pdf'
                        )
                    except Exception as e:
                        st.error(f"Gagal membuat PDF: {e}")
        else:
            st.warning("⚠️ Tidak ada data prakiraan cuaca dari BMKG.")
    except Exception as e:
        st.error(f"Gagal mengambil data BMKG: {e}")

# ================================
# Streamlit UI: main
# ================================
st.set_page_config(page_title="Forecast Curah Hujan Gorontalo", page_icon="🌧️", layout="wide")
st.title("🌧️ Dashboard Forecast Curah Hujan Gorontalo")
st.markdown("Aplikasi ini mengambil data curah hujan langsung dari Google Sheets...")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Tombol untuk mengambil data dari Google Sheets
    if st.button("🔄 Ambil Data dari Google Sheets"):
        with st.spinner('Mengambil data dari Google Sheets...'):
            df = get_rainfall_data()
            if df is not None:
                df = clean_data(df)
                st.session_state.df = df
                st.success("Data berhasil diambil dari Google Sheets!")
            else:
                st.error("Gagal mengambil data dari Google Sheets")

    # Cek apakah data sudah ada di session state
    if 'df' not in st.session_state:
        st.info("Silakan klik tombol di atas untuk mengambil data dari Google Sheets")
        st.stop()
    else:
        df = st.session_state.df
        st.subheader("Pratinjau Data")
        st.dataframe(df.head())

    st.divider()
    agg_method = st.radio("Metode Agregasi Bulanan", ["Jumlah (Sum)", "Rata-rata (Mean)"], index=0)
    st.subheader("Parameter Forecasting")
    model_type = st.selectbox("Pilih Model Forecasting", ['ARIMA', 'SARIMA', 'Prophet'], index=2)
    forecast_period = st.slider("Jumlah Periode untuk Forecast (bulan)", 1, 24, 12)
    st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 EDA", "🔮 Forecasting", "📊 Evaluasi Model", "🤖 Analisis AI", "💾 Data", "🌦️ BMKG" ])

with tab1:
    perform_eda(df, agg_method)

with tab2:
    st.subheader("Forecasting Curah Hujan")
    if st.button("Jalankan Forecasting"):
        with st.spinner(f"Melakukan forecasting dengan model {model_type}..."):
            forecast = forecast_rainfall(df, model_type, forecast_period, agg_method)
            plot_forecast(df, forecast, model_type, agg_method)
            st.success("Forecasting selesai!")
            st.subheader("Hasil Forecasting")
            forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_period, freq='M')
            forecast_df = pd.DataFrame({'Bulan': forecast_dates.strftime('%B %Y'), 'Curah Hujan (mm)': forecast.round(2)})
            st.dataframe(forecast_df)

with tab3:
    st.subheader("Evaluasi Model Forecasting")
    eval_model = st.selectbox("Pilih Model untuk Evaluasi", ['ARIMA', 'SARIMA', 'Prophet'], index=2)
    if st.button("Evaluasi Model"):
        with st.spinner(f"Menjalankan evaluasi untuk model {eval_model}..."):
            evaluate_model(df, eval_model, agg_method)

with tab4:
    st.subheader("Analisis AI dengan Claude AI")
    api_key = st.text_input("Masukkan API Key Claude AI", type="password")
    if api_key:
        if st.button("Jalankan Analisis AI"):
            with st.spinner("Menganalisis data dengan AI..."):
                try:
                    analysis = ai_analysis_bmkg(df, api_key)
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"Error dalam analisis AI: {e}")
    else:
        st.warning("Masukkan API Key Claude AI untuk mengaktifkan analisis AI")

with tab5:
    st.subheader("Data Mentah")
    st.dataframe(df)
    csv = df.to_csv(sep=';', index=True).encode('utf-8')
    st.download_button("Download Data Bersih (CSV)", data=csv, file_name='curah_hujan_gorontalo_clean.csv', mime='text/csv')

with tab6:
    show_bmkg_tab()

st.divider()
st.markdown("""
<div style="text-align: center; color: grey; font-size: small;">
Dashboard Forecast Curah Hujan Gorontalo © 2024 | Dibangun dengan Streamlit | Data dari Google Sheets
</div>
""", unsafe_allow_html=True)
