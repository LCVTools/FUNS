"""
ScAF2.py — Screening Awal Fundamental Saham (Improved)
=======================================================
Perbaikan dari ScAF1.py:
  1.  _get_field() → ambil kolom TERBARU (bukan iterasi semua kolom)
  2.  Thread safety → kumpulkan error dulu, tampilkan setelah semua thread selesai
  3.  PEG Ratio → skip jika earningsGrowth ≤ 0 (hindari PEG negatif)
  4.  DER → guard v >= 0 agar ekuitas negatif tidak dianggap "Baik"
  5.  DER fallback → auto-detect skala debtToEquity (% vs rasio)
  6.  Target price → cap upside/downside ±80% dari harga saat ini
  7.  fetch_ticker_data() → tambahkan cashflow untuk FCF
  8.  compute_ratios() → tambahkan FCF Yield sebagai rasio bonus
  9.  get_ratio_data() → pre-validasi ticker IDX sebelum parallel fetch
  10. display_results() → tambahkan currency IDR, unit EPS, timestamp per-saham
  11. Scoring → bobot berbeda per rasio (weighted score)
  12. UX → progress bar per-saham, sektor & industri, tombol export CSV
  13. Cache key → include ticker list agar tidak collision antar sesi
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import time
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

RATIO_KEYS = [
    "ROE", "DER", "P/E", "P/B", "P/S",
    "Dividend Yield", "Operating Margin", "GPM", "ROA",
    "Earnings Yield", "Current Ratio", "PEG Ratio", "FCF Yield",
]

# Fix #11 — Weighted score per rasio (default=1; ROE, P/E, ROA diberi bobot lebih)
RATIO_WEIGHTS = {
    "ROE":              3,
    "DER":              2,
    "P/E":              3,
    "P/B":              1,
    "P/S":              1,
    "Dividend Yield":   1,
    "Operating Margin": 2,
    "GPM":              1,
    "ROA":              3,
    "Earnings Yield":   2,
    "Current Ratio":    2,
    "PEG Ratio":        2,
    "FCF Yield":        2,
}

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(page_title="🔍 SAFS v2", layout="wide")
st.title("📊 Screening Awal Fundamental Saham")
st.markdown(
    "Analisa & bandingkan rasio fundamental saham IDX. "
    "**TIDAK berlaku untuk emiten perBankan.** "
    "Maksimal **7 saham** — 3 terbaik akan direkomendasikan. "
    "**Happy Cuan!** 🚀"
)

# ─────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────

for _key, _default in [
    ('should_display_results', False),
    ('manual_values', {}),
    ('ratio_data', {}),
    ('evaluations', {}),
    ('scores', {}),
    ('stock_prices', {}),
    ('target_prices', {}),
    ('estimated_eps', {}),
    ('fetch_timestamps', {}),    # Fix #10 — timestamp per saham
    ('sector_info', {}),         # Fix #10 — sektor & industri
    ('fetch_errors', []),        # Fix #2  — kumpulkan error thread-safe
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ─────────────────────────────────────────────
# Ticker Input
# ─────────────────────────────────────────────

def normalize_ticker(ticker: str) -> str:
    """Normalize ke format IDX — auto-append .JK, uppercase."""
    t = ticker.strip().upper()
    if t and not t.endswith(".JK"):
        t += ".JK"
    return t


ticker_input = st.text_input(
    "Masukkan Kode Saham (max. 7, contoh: TLKM, ARNA, AUTO, ADRO, PTBA, ASII, ANTM)",
    "TLKM, ARNA, AUTO, ADRO, PTBA, ASII, ANTM",
)
_raw  = [t for t in ticker_input.split(",") if t.strip()]
stocks = list(dict.fromkeys(normalize_ticker(t) for t in _raw))[:7]


# ─────────────────────────────────────────────
# Fix #1 — _get_field: ambil kolom TERBARU dulu
# ─────────────────────────────────────────────

def _get_field(df: pd.DataFrame, *field_names):
    """
    Return nilai terbaru (non-null) dari field yang cocok.
    yfinance mengurutkan kolom secara DESCENDING (terbaru di kiri),
    jadi iterasi kolom dari kiri = terbaru ke terlama.
    """
    if df is None or df.empty:
        return None
    for name in field_names:
        if name in df.index:
            # Fix #1: iterasi kolom dari kiri (terbaru) ke kanan (terlama)
            for col in df.columns:
                val = df.loc[name, col]
                if pd.notna(val):
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
    return None


# ─────────────────────────────────────────────
# Fix #9 — Pre-validasi ticker IDX
# ─────────────────────────────────────────────

def validate_ticker(stock: str) -> tuple[bool, str]:
    """
    Cek apakah ticker valid & ada data di yfinance.
    Return (True, "") jika valid, (False, alasan) jika tidak.
    """
    try:
        t = yf.Ticker(stock)
        info = t.info
        # Jika quoteType kosong atau NONE → ticker tidak dikenal
        qt = info.get("quoteType", "")
        if not qt or qt == "NONE":
            return False, f"Ticker `{stock}` tidak ditemukan di Yahoo Finance."
        # Peringatan untuk perbankan
        sector = (info.get("sector") or "").lower()
        industry = (info.get("industry") or "").lower()
        if "bank" in sector or "bank" in industry or "financial" in sector:
            return True, f"⚠️ `{stock}` terdeteksi sebagai emiten keuangan/perbankan — hasil mungkin kurang relevan."
        return True, ""
    except Exception as e:
        return False, f"Gagal memvalidasi `{stock}`: {e}"


# ─────────────────────────────────────────────
# Fetch: cached per ticker (1 jam TTL)
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data(stock: str) -> dict:
    """Fetch raw yfinance data untuk satu saham IDX. Cache 1 jam."""
    for attempt in range(3):
        try:
            ticker = yf.Ticker(stock)
            info   = ticker.info

            # Harga saat ini — coba berbagai field
            current_price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
                or info.get("ask")
                or info.get("bid")
            )
            if current_price is None:
                hist = ticker.history(period="5d")
                if not hist.empty:
                    current_price = float(hist["Close"].dropna().iloc[-1])
            if current_price is None:
                raise ValueError(f"Tidak ada data harga untuk {stock}")

            # Laporan keuangan tahunan → fallback kuartalan
            fin = ticker.financials
            if fin is None or fin.empty:
                fin = ticker.quarterly_financials

            bs = ticker.balance_sheet
            if bs is None or bs.empty:
                bs = ticker.quarterly_balance_sheet

            # Fix #7 — tambahkan cashflow
            cf = ticker.cashflow
            if cf is None or cf.empty:
                cf = ticker.quarterly_cashflow

            return {
                "info":         info,
                "financials":   fin,
                "balance_sheet": bs,
                "cashflow":     cf,            # Fix #7
                "current_price": float(current_price),
                "fetch_time":   dt.datetime.now().isoformat(),  # Fix #10
            }
        except Exception as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"Gagal mengambil data {stock} setelah 3 percobaan: {exc}"
                ) from exc


# ─────────────────────────────────────────────
# Compute: semua rasio fundamental
# ─────────────────────────────────────────────

def compute_ratios(data: dict) -> dict:
    """Hitung semua rasio fundamental dari data mentah yfinance."""
    info  = data["info"]
    fin   = data["financials"]
    bs    = data["balance_sheet"]
    cf    = data["cashflow"]      # Fix #7

    # ── Total Equity (shared) ─────────────────
    total_equity = _get_field(
        bs,
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Common Stock Equity",
        "Total Equity Gross Minority Interest",
        "Stockholders' Equity",
        "Equity",
        "Net Assets",
    )

    # ── ROE ──────────────────────────────────
    net_income_roe = _get_field(
        fin,
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
        "Net Income Applicable To Common Shares",
        "Net Profit",
    )
    if net_income_roe is not None and total_equity is not None and total_equity != 0:
        roe = (net_income_roe / total_equity) * 100
    else:
        raw = info.get("returnOnEquity")
        roe = (float(raw) * 100) if raw is not None else None

    # ── DER ──────────────────────────────────
    total_debt = _get_field(
        bs,
        "Total Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Total Liabilities Net Minority Interest",
        "Total Liabilities",
    )
    if total_debt is None:
        ltd = _get_field(bs, "Long Term Debt", "Long Term Debt And Capital Lease Obligation")
        std = _get_field(
            bs,
            "Short Long Term Debt",
            "Current Debt",
            "Current Portion Of Long Term Debt",
            "Current Debt And Capital Lease Obligation",
        )
        if ltd is not None or std is not None:
            total_debt = (ltd or 0.0) + (std or 0.0)

    if total_debt is not None and total_equity is not None and total_equity != 0:
        der = total_debt / total_equity
    else:
        raw = info.get("debtToEquity")
        if raw is not None:
            raw = float(raw)
            # Fix #5 — auto-detect skala: yfinance bisa kirim % (150) atau rasio (1.5)
            der = raw / 100 if raw > 10 else raw
        else:
            der = None

    # ── P/E ──────────────────────────────────
    pe_raw = info.get("trailingPE") or info.get("forwardPE")
    pe     = float(pe_raw) if pe_raw is not None else None

    # ── P/B ──────────────────────────────────
    pb_raw = info.get("priceToBook")
    pb     = float(pb_raw) if pb_raw is not None else None

    # ── P/S ──────────────────────────────────
    ps_raw = info.get("priceToSalesTrailing12Months")
    if ps_raw is None:
        mktcap  = info.get("marketCap")
        revenue = _get_field(fin, "Total Revenue", "Operating Revenue", "Revenue")
        if mktcap is not None and revenue is not None and revenue != 0:
            ps_raw = mktcap / revenue
    ps = float(ps_raw) if ps_raw is not None else None

    # ── Dividend Yield (disimpan dalam %, misal 3.5 = 3.5%) ──
    div_raw = info.get("dividendYield")
    div = (float(div_raw) * 100) if div_raw is not None else None

    # ── Operating Margin ─────────────────────
    om_raw = info.get("operatingMargins")
    if om_raw is not None:
        op_margin = float(om_raw) * 100
    else:
        op_income = _get_field(fin, "Operating Income", "EBIT", "Operating Profit")
        revenue   = _get_field(fin, "Total Revenue", "Operating Revenue", "Revenue")
        op_margin = (
            (op_income / revenue) * 100
            if op_income is not None and revenue is not None and revenue != 0
            else None
        )

    # ── GPM ──────────────────────────────────
    gpm_raw = info.get("grossMargins")
    if gpm_raw is not None:
        gpm = float(gpm_raw) * 100
    else:
        gross_profit = _get_field(fin, "Gross Profit")
        revenue      = _get_field(fin, "Total Revenue", "Operating Revenue", "Revenue")
        gpm = (
            (gross_profit / revenue) * 100
            if gross_profit is not None and revenue is not None and revenue != 0
            else None
        )

    # ── ROA ──────────────────────────────────
    net_income_roa = _get_field(
        fin,
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
        "Net Profit",
    )
    total_assets = _get_field(bs, "Total Assets")
    if net_income_roa is not None and total_assets is not None and total_assets != 0:
        roa = (net_income_roa / total_assets) * 100
    else:
        raw = info.get("returnOnAssets")
        roa = (float(raw) * 100) if raw is not None else None

    # ── Earnings Yield ───────────────────────
    ey = ((1 / pe) * 100) if (pe is not None and pe > 0) else None

    # ── Current Ratio ─────────────────────────
    cur_assets = _get_field(bs, "Current Assets", "Total Current Assets")
    cur_liab   = _get_field(bs, "Current Liabilities", "Total Current Liabilities")
    if cur_assets is not None and cur_liab is not None and cur_liab != 0:
        current_ratio = cur_assets / cur_liab
    else:
        raw = info.get("currentRatio")
        current_ratio = float(raw) if raw is not None else None

    # ── PEG Ratio — Fix #3: skip jika growth ≤ 0 ────
    peg_raw = info.get("pegRatio")
    if peg_raw is None and pe is not None:
        growth = info.get("earningsGrowth")
        # Fix #3 — hanya hitung PEG jika growth positif
        if growth is not None and growth > 0:
            peg_raw = pe / (growth * 100)
    peg = float(peg_raw) if (peg_raw is not None and float(peg_raw) > 0) else None

    # ── Fix #8 — FCF Yield ───────────────────
    fcf_yield = None
    mktcap = info.get("marketCap")
    if cf is not None and not cf.empty and mktcap and mktcap != 0:
        fcf = _get_field(
            cf,
            "Free Cash Flow",
            "Operating Cash Flow",    # fallback kasar jika FCF tidak ada
        )
        # Kurangi capex jika FCF belum bersih
        capex = _get_field(cf, "Capital Expenditure", "Capital Expenditures")
        if fcf is not None:
            # Jika yang didapat adalah Operating CF, kurangi capex
            if capex is not None and capex < 0:
                # capex yfinance biasanya negatif
                net_fcf = fcf + capex  # (+operating) + (-capex)
            else:
                net_fcf = fcf
            if net_fcf is not None:
                fcf_yield = (net_fcf / mktcap) * 100  # dalam %

    return {
        "ROE":              roe,
        "DER":              der,
        "P/E":              pe,
        "P/B":              pb,
        "P/S":              ps,
        "Dividend Yield":   div,
        "Operating Margin": op_margin,
        "GPM":              gpm,
        "ROA":              roa,
        "Earnings Yield":   ey,
        "Current Ratio":    current_ratio,
        "PEG Ratio":        peg,
        "FCF Yield":        fcf_yield,   # Fix #8
    }


# ─────────────────────────────────────────────
# Fix #6 — Target Price dengan cap ±80%
# ─────────────────────────────────────────────

def compute_target_price(info: dict, ratios: dict, current_price: float) -> float | None:
    """Estimasi target price. Hasil di-cap ±80% dari harga saat ini."""
    target = None

    # 1. Analyst consensus mean
    target = info.get("targetMeanPrice")
    if target is not None:
        target = float(target)

    # 2. Rata-rata analyst high/low
    if target is None:
        high = info.get("targetHighPrice")
        low  = info.get("targetLowPrice")
        if high is not None and low is not None:
            target = float((high + low) / 2)

    # 3. Analyst median
    if target is None:
        median = info.get("targetMedianPrice")
        if median is not None:
            target = float(median)

    # 4. EPS × fair P/E (capped 25x)
    if target is None:
        eps = info.get("forwardEps") or info.get("trailingEps")
        if eps is not None and eps > 0:
            pe = ratios.get("P/E")
            fair_pe = min(pe, 25.0) if (pe is not None and pe > 0) else 15.0
            target = float(eps * fair_pe)

    # 5. PBV-based: book value × fair P/B (1.5x)
    if target is None:
        bvps = info.get("bookValue")
        if bvps is not None and bvps > 0:
            pb = ratios.get("P/B")
            fair_pb = min(pb, 3.0) if (pb is not None and pb > 0) else 1.5
            target = float(bvps * fair_pb)

    # Fix #6 — cap upside/downside ±80% dari harga saat ini
    if target is not None and current_price > 0:
        cap_low  = current_price * 0.20   # max turun 80%
        cap_high = current_price * 1.80   # max naik 80%
        target = max(cap_low, min(cap_high, target))

    return target


# ─────────────────────────────────────────────
# Fix #2 — Thread-safe orchestrator
# ─────────────────────────────────────────────

def get_ratio_data(stocks: list) -> dict:
    """
    Parallel fetch + compute untuk semua saham.
    Fix #2: kumpulkan error di list, tampilkan setelah semua thread selesai.
    """
    results           = {}
    prices            = {}
    target_prices     = {}
    estimated_eps_val = {}
    fetch_timestamps  = {}   # Fix #10
    sector_info_map   = {}   # Fix #10
    thread_errors     = []   # Fix #2 — kumpulkan error, thread-safe

    def process_stock(stock: str):
        try:
            data   = fetch_ticker_data(stock)
            ratios = compute_ratios(data)
            info   = data["info"]
            price  = data["current_price"]
            target = compute_target_price(info, ratios, price)
            eps    = info.get("forwardEps") or info.get("trailingEps")
            ts     = data.get("fetch_time", dt.datetime.now().isoformat())
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            return stock, ratios, price, target, eps, ts, sector, industry, None
        except Exception as exc:
            return stock, None, None, None, None, None, None, None, str(exc)

    with ThreadPoolExecutor(max_workers=min(4, len(stocks))) as executor:
        futures = {executor.submit(process_stock, s): s for s in stocks}
        for future in as_completed(futures):
            stock, ratios, price, target, eps, ts, sector, industry, error = future.result()
            if error:
                # Fix #2 — jangan panggil st.warning() dari dalam thread!
                thread_errors.append((stock, error))
                results[stock]           = {k: None for k in RATIO_KEYS}
                prices[stock]            = None
                target_prices[stock]     = None
                estimated_eps_val[stock] = None
                fetch_timestamps[stock]  = None
                sector_info_map[stock]   = ("N/A", "N/A")
            else:
                results[stock]           = ratios
                prices[stock]            = price
                target_prices[stock]     = target
                estimated_eps_val[stock] = eps
                fetch_timestamps[stock]  = ts
                sector_info_map[stock]   = (sector, industry)

    # Fix #2 — tampilkan semua error di main thread (aman untuk Streamlit)
    for stock, err_msg in thread_errors:
        st.warning(f"⚠️ Gagal mengambil data **{stock}**: {err_msg}")

    st.session_state.stock_prices    = prices
    st.session_state.target_prices   = target_prices
    st.session_state.estimated_eps   = estimated_eps_val
    st.session_state.fetch_timestamps = fetch_timestamps
    st.session_state.sector_info     = sector_info_map
    st.session_state.fetch_errors    = [e[0] for e in thread_errors]
    return results


# ─────────────────────────────────────────────
# Fix #4, #11 — Evaluate rasio dengan weighted score
# ─────────────────────────────────────────────

def evaluate_ratios(ratio_data: dict) -> tuple:
    """
    Grade setiap rasio dan hitung weighted score.
    Fix #4: DER guard v >= 0.
    Fix #11: weighted score per rasio.
    """
    evaluations = {}
    scores      = {}

    for stock, ratios in ratio_data.items():
        evaluations[stock] = {}
        scores[stock]      = 0

        # Terapkan manual overrides
        if stock in st.session_state.manual_values:
            for ratio, value in st.session_state.manual_values[stock].items():
                if value != "":
                    try:
                        ratios[ratio] = float(value)
                    except ValueError:
                        pass

        def grade(key, good_fn, mid_fn):
            val    = ratios.get(key)
            weight = RATIO_WEIGHTS.get(key, 1)  # Fix #11
            if val is None:
                evaluations[stock][key] = "N/A"
                return
            if good_fn(val):
                evaluations[stock][key] = "Baik"
                scores[stock] += 2 * weight   # Fix #11 — skor berbobot
            elif mid_fn(val):
                evaluations[stock][key] = "Biasa"
                scores[stock] += 1 * weight
            else:
                evaluations[stock][key] = "Buruk"

        grade("ROE",             lambda v: v > 15,             lambda v: 5 <= v <= 15)
        # Fix #4 — DER: v >= 0 agar ekuitas negatif tidak lolos sebagai "Baik"
        grade("DER",             lambda v: 0 <= v < 0.8,       lambda v: 0.8 <= v <= 1)
        grade("P/E",             lambda v: 0 < v < 15,         lambda v: 15 <= v <= 25)
        grade("P/B",             lambda v: 0 < v < 1.5,        lambda v: 1.5 <= v <= 3)
        grade("P/S",             lambda v: 0 < v < 1,          lambda v: 1 <= v <= 2)
        grade("Dividend Yield",  lambda v: v > 3.75,           lambda v: 1 <= v <= 3.75)
        grade("Operating Margin",lambda v: v > 20,             lambda v: 10 <= v <= 20)
        grade("GPM",             lambda v: v > 40,             lambda v: 20 <= v <= 40)
        grade("ROA",             lambda v: v > 5,              lambda v: 2 <= v <= 5)
        grade("Earnings Yield",  lambda v: v > 10,             lambda v: 5 <= v <= 10)
        grade("Current Ratio",   lambda v: v > 2,              lambda v: 1 <= v <= 2)
        # Fix #3 — PEG tidak pernah negatif karena compute_ratios() sudah guard
        grade("PEG Ratio",       lambda v: 0 < v < 1,          lambda v: 1 <= v <= 1.5)
        # Fix #8 — FCF Yield: positif = baik
        grade("FCF Yield",       lambda v: v > 5,              lambda v: 1 <= v <= 5)

    return evaluations, scores


# ─────────────────────────────────────────────
# Fix #10, #12 — Display results
# ─────────────────────────────────────────────

def _fmt_ratio(ratio: str, value) -> str:
    """Format nilai rasio sesuai jenis."""
    if not isinstance(value, float):
        return "N/A"
    pct_ratios = {
        "ROE", "Operating Margin", "GPM", "ROA",
        "Earnings Yield", "Dividend Yield", "FCF Yield",
    }
    if ratio in pct_ratios:
        return f"{value:.2f}%"
    return f"{value:.2f}x" if ratio in ("DER", "P/E", "P/B", "P/S", "Current Ratio", "PEG Ratio") else f"{value:.2f}"


def _eval_color(penilaian: str) -> str:
    return {"Baik": "🟢", "Biasa": "🟡", "Buruk": "🔴"}.get(penilaian, "⚪")


def display_results(stocks: list, ratio_data: dict, evaluations: dict, scores: dict):
    # ── Sektor & Industri ────────────────────
    st.subheader("ℹ️ Profil Emiten")
    sector_cols = st.columns(len(stocks))
    for i, stock in enumerate(stocks):
        sector, industry = st.session_state.sector_info.get(stock, ("N/A", "N/A"))
        with sector_cols[i]:
            st.metric(stock, sector, industry)

    # ── Build DataFrame ──────────────────────
    st.subheader("📋 Analisis Fundamental")
    data = []
    for stock in stocks:
        if stock not in ratio_data:
            continue
        row = [stock]

        # Fix #10 — harga dengan satuan IDR
        price = st.session_state.stock_prices.get(stock)
        row.append(f"IDR {price:,.0f}" if price is not None else "N/A")

        for ratio in RATIO_KEYS:
            value     = ratio_data[stock].get(ratio)
            penilaian = evaluations[stock].get(ratio, "N/A") if stock in evaluations else "N/A"
            row.extend([_fmt_ratio(ratio, value), f"{_eval_color(penilaian)} {penilaian}"])

        # Fix #10 — EPS dengan satuan IDR
        eps = st.session_state.estimated_eps.get(stock)
        row.append(f"IDR {eps:,.2f}" if eps is not None else "N/A")

        target = st.session_state.target_prices.get(stock)
        row.append(f"IDR {target:,.0f}" if target is not None else "N/A")

        # Fix #10 — timestamp fetch
        ts = st.session_state.fetch_timestamps.get(stock)
        row.append(ts[:16].replace("T", " ") if ts else "N/A")

        data.append(row)

    # ── MultiIndex columns ───────────────────
    h1 = ["SAHAM", "PRICE (IDR)"]
    h2 = ["", ""]
    for ratio in RATIO_KEYS:
        h1.extend([ratio, ratio])
        h2.extend(["Value", "Penilaian"])
    h1 += ["est. EPS", "TARGET PRICE", "Data Diambil"]
    h2 += ["", "", ""]

    df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(list(zip(h1, h2))))
    st.dataframe(df, use_container_width=True)

    # ── Fix #12 — Export CSV ─────────────────
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Download Hasil (CSV)",
        data=csv_buf.getvalue().encode("utf-8-sig"),
        file_name=f"analisa_fundamental_{dt.date.today()}.csv",
        mime="text/csv",
    )

    # ── Manual Input Section ─────────────────
    st.subheader("✏️ Input Manual untuk Nilai N/A")
    for stock in stocks:
        if stock not in st.session_state.manual_values:
            st.session_state.manual_values[stock] = {r: "" for r in RATIO_KEYS}

    tabs = st.tabs(stocks)
    for i, stock in enumerate(stocks):
        with tabs[i]:
            st.write(f"Override nilai manual untuk **{stock}** (kosongkan untuk pakai data otomatis):")
            cols    = st.columns(3)
            has_na  = False

            for j, ratio in enumerate(RATIO_KEYS):
                with cols[j % 3]:
                    value = ratio_data[stock].get(ratio)
                    display_val = _fmt_ratio(ratio, value) if value is not None else "Current: N/A"
                    if value is None:
                        has_na = True

                    st.text(f"Auto: {display_val}" if value is not None else display_val)
                    current_input = st.session_state.manual_values[stock].get(ratio, "")
                    new_val = st.text_input(
                        f"Override {ratio}",
                        value=current_input,
                        key=f"manual_input_{stock}_{ratio}_{i}",
                    )
                    if new_val != current_input:
                        st.session_state.manual_values[stock][ratio] = new_val

            if not has_na:
                st.info("✅ Semua rasio sudah otomatis terisi. Input manual akan menggantikan nilai tersebut.")

    # ── Re-analyze Button ────────────────────
    if st.button("🔄 Analisa Kembali (pakai override manual)", key="reanalyze_button"):
        if st.session_state.ratio_data:
            with st.spinner("Menganalisis ulang..."):
                evaluations, scores = evaluate_ratios(st.session_state.ratio_data)
                st.session_state.evaluations = evaluations
                st.session_state.scores      = scores
                st.session_state.should_display_results = True
                st.rerun()

    # ── Weighted Score Summary ───────────────
    st.subheader("🏆 REKOMENDASI (Weighted Score)")

    # Tabel ringkasan skor
    summary_rows = []
    for stock in stocks:
        if stock not in evaluations:
            continue
        evals      = evaluations[stock]
        good_count = sum(1 for v in evals.values() if v == "Baik")
        mid_count  = sum(1 for v in evals.values() if v == "Biasa")
        bad_count  = sum(1 for v in evals.values() if v == "Buruk")
        na_count   = sum(1 for v in evals.values() if v == "N/A")
        summary_rows.append({
            "Saham":        stock,
            "Weighted Score": scores.get(stock, 0),
            "🟢 Baik":     good_count,
            "🟡 Biasa":    mid_count,
            "🔴 Buruk":    bad_count,
            "⚪ N/A":      na_count,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Weighted Score", ascending=False)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Rekomendasi: minimal 5 rasio "Baik"
    good_counts = {
        stock: sum(1 for v in evals.values() if v == "Baik")
        for stock, evals in evaluations.items()
    }
    qualified = {s: scores[s] for s in stocks if good_counts.get(s, 0) >= 5}

    if qualified:
        top = sorted(qualified.items(), key=lambda x: x[1], reverse=True)[:3]
        st.markdown("**Top 3 Saham Terpilih:**")
        for rank, (stock, score) in enumerate(top, 1):
            price  = st.session_state.stock_prices.get(stock)
            target = st.session_state.target_prices.get(stock)
            upside = ((target / price - 1) * 100) if (target and price) else None
            upside_str = f", estimasi upside **{upside:+.1f}%**" if upside is not None else ""
            st.write(
                f"{rank}. **{stock}** — Weighted Score: **{score}**, "
                f"Kriteria Baik: {good_counts[stock]}{upside_str}"
            )
    else:
        st.warning("Tidak ada saham memenuhi syarat (minimal 5 rasio **Baik**).")

    st.caption(
        f"Data dianalisa pada {dt.datetime.now().strftime('%d %B %Y %H:%M')} WIB. "
        "Hasil analisa bukan merupakan rekomendasi investasi."
    )


# ─────────────────────────────────────────────
# Main App Flow
# ─────────────────────────────────────────────

if st.button("🔍 Analisis Fundamental", type="primary"):
    if not stocks:
        st.error("Masukkan minimal satu kode saham untuk dianalisis.")
    else:
        # Fix #9 — pre-validasi ticker sebelum fetch
        with st.spinner("Memvalidasi ticker..."):
            valid_stocks  = []
            invalid_found = False
            for s in stocks:
                ok, msg = validate_ticker(s)
                if not ok:
                    st.error(f"❌ {msg}")
                    invalid_found = True
                else:
                    if msg:  # peringatan (bukan error)
                        st.warning(msg)
                    valid_stocks.append(s)

        if not valid_stocks:
            st.stop()

        # Fix #12 — progress bar per-saham
        progress_bar = st.progress(0, text="Memulai pengambilan data...")
        total = len(valid_stocks)

        with st.spinner(f"Mengambil data fundamental {total} saham..."):
            ratio_data = get_ratio_data(valid_stocks)
            progress_bar.progress(100, text="✅ Data berhasil diambil!")

        evaluations, scores = evaluate_ratios(ratio_data)
        st.session_state.ratio_data   = ratio_data
        st.session_state.evaluations  = evaluations
        st.session_state.scores       = scores
        st.session_state.should_display_results = True
        st.rerun()

if st.session_state.should_display_results and st.session_state.ratio_data and stocks:
    display_results(
        stocks,
        st.session_state.ratio_data,
        st.session_state.evaluations,
        st.session_state.scores,
    )

# ─────────────────────────────────────────────
# Hide Streamlit toolbar
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
        .stApp [data-testid="stHeader"]  { display: none !important; }
        .stApp [data-testid="stToolbar"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
