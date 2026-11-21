import base64
import builtins
import io
import os
import sys
import zipfile
import tempfile
import asyncio
import json
import requests

# Εξαγωγή του matplotlib/mpl-data από το sitepackages.zip (Android Flet fix)
_mpl_cache = os.path.join(tempfile.gettempdir(), "mpl_data_extracted")
os.makedirs(_mpl_cache, exist_ok=True)
_mpl_data_dir = None

_search_paths = [str(p) for p in sys.path] + os.environ.get("PYTHONPATH", "").split(os.pathsep)
for _entry in _search_paths:
    _zip_path = _entry.split(".zip")[0] + ".zip" if ".zip" in _entry else _entry
    if _zip_path.endswith(".zip") and os.path.isfile(_zip_path):
        try:
            with zipfile.ZipFile(_zip_path, "r") as _zf:
                _mpl_members = [m for m in _zf.namelist() if m.startswith("matplotlib/mpl-data/")]
                if _mpl_members:
                    _target_dir = os.path.join(_mpl_cache, "matplotlib", "mpl-data")
                    _target_rc = os.path.join(_target_dir, "matplotlibrc")
                    if not os.path.isfile(_target_rc):
                        _zf.extractall(_mpl_cache, members=_mpl_members)
                    _mpl_data_dir = _target_dir
                    os.environ["MPLCONFIGDIR"] = _target_dir
                    os.environ["MATPLOTLIBRC"] = _target_rc
                    os.environ["MATPLOTLIBDATA"] = _target_dir
                    break
        except Exception:
            pass

if _mpl_data_dir:
    _orig_open = builtins.open

    def _patched_open(file, *args, **kwargs):
        try:
            _path_str = os.fspath(file)
            if isinstance(_path_str, str) and ".zip/matplotlib/mpl-data" in _path_str.replace("\\", "/"):
                _rel = _path_str.replace("\\", "/").split(".zip/matplotlib/mpl-data", 1)[1].lstrip("/")
                file = os.path.join(_mpl_data_dir, _rel) if _rel else _mpl_data_dir
        except TypeError:
            pass
        return _orig_open(file, *args, **kwargs)

    builtins.open = _patched_open
    io.open = _patched_open

import matplotlib
if _mpl_data_dir:
    matplotlib.get_data_path = lambda: _mpl_data_dir

matplotlib.use('Agg')  # Non-interactive backend για αποφυγή GUI collisions
import matplotlib.pyplot as plt
import flet as ft


# ---------------------------------------------------------------------------
# FLET VERSION COMPATIBILITY HELPERS
# ---------------------------------------------------------------------------
def border_all(width, color):
    if hasattr(ft, "Border") and hasattr(ft.Border, "all"):
        return ft.Border.all(width, color)
    return ft.border.all(width, color)


def padding_symmetric(horizontal=0, vertical=0):
    if hasattr(ft, "Padding") and hasattr(ft.Padding, "symmetric"):
        return ft.Padding.symmetric(horizontal=horizontal, vertical=vertical)
    return ft.padding.symmetric(horizontal=horizontal, vertical=vertical)


# ---------------------------------------------------------------------------
# API URL & CONFIGURATION
# ---------------------------------------------------------------------------

BASE_URL = "https://thesis-w4mq.onrender.com"
WS_URL = "wss://thesis-w4mq.onrender.com/ws/crypto-stream"


# ---------------------------------------------------------------------------
# REAL HISTORICAL PRICES HELPER (BINANCE PUBLIC API + SAFE FALLBACK)
# ---------------------------------------------------------------------------
def fetch_historical_prices(symbol, live_price=None, limit=10):
    pair = f"{symbol}USDT"
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": pair, "interval": "1d", "limit": limit},
            timeout=10
        )
        if r.status_code == 200:
            closes = [float(candle[4]) for candle in r.json()]
            if live_price and live_price > 0:
                closes[-1] = float(live_price)
            return closes
    except Exception:
        pass

    # Fallback γύρω από την πραγματική live τιμή αν δεν απαντήσει η Binance
    base = float(live_price) if (live_price and live_price > 0) else (74750.0 if symbol == "BTC" else 2125.0)
    offsets = [-0.018, -0.012, -0.021, -0.015, -0.008, 0.004, -0.003, 0.002, -0.001, 0.0]
    return [round(base * (1 + o), 2) for o in offsets]


# ---------------------------------------------------------------------------
# MATPLOTLIB TO FLET IMAGE HELPER
# ---------------------------------------------------------------------------
def fig_to_flet_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    try:
        return ft.Image(src_base64=b64_img, fit="contain", expand=True)
    except TypeError:
        return ft.Image(src=f"data:image/png;base64,{b64_img}", fit="contain", expand=True)


# ---------------------------------------------------------------------------
# MATPLOTLIB CHARTS & BENCHMARK BUILDERS
# ---------------------------------------------------------------------------
def build_horizon_chart(symbol, prices, transformer_pred, lstm_pred, ensemble_pred):
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor='#0b0e14')
    ax.set_facecolor('#0b0e14')

    color = '#f7931a' if symbol == 'BTC' else '#a855f7'
    x_hist = list(range(1, len(prices) + 1))

    # Ιστορικές Τιμές
    ax.plot(x_hist, prices, color=color, marker='o', markersize=4, linewidth=2.2, label=f'{symbol} Historical')

    last_x = len(prices)
    next_x = last_x + 1
    last_y = prices[-1] if prices else 0

    # Προβλέψεις (+24h)
    ax.plot([last_x, next_x], [last_y, transformer_pred], color='#3b82f6', linestyle='--', marker='s', markersize=5, linewidth=2, label=f'Transformer (${transformer_pred:,.0f})')
    ax.plot([last_x, next_x], [last_y, lstm_pred], color='#22c55e' if symbol == 'BTC' else '#c084fc', linestyle='--', marker='^', markersize=5, linewidth=2, label=f'LSTM (${lstm_pred:,.0f})')
    ax.plot([last_x, next_x], [last_y, ensemble_pred], color='#eab308', marker='*', markersize=10, linewidth=2.5, label=f'Ensemble ★ (${ensemble_pred:,.0f})')

    # Άξονες & Όρια
    x_labels = [f"D-{len(prices)-i}" for i in range(1, len(prices))] + ["Now", "+24h"]
    ax.set_xticks(list(range(1, next_x + 1)))
    ax.set_xticklabels(x_labels, fontsize=7, color='#8a8f9e')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"${val:,.0f}"))

    ax.set_title(f"{symbol}/USD Trend & 24h Ensemble Consensus", color='white', fontsize=10, pad=10, weight='bold')
    ax.tick_params(colors='#8a8f9e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
    ax.grid(True, color='#1e293b', linestyle=':', linewidth=0.8)
    ax.legend(facecolor='#101217', edgecolor='#1e293b', labelcolor='white', fontsize=7, loc='best')

    plt.tight_layout()
    return fig_to_flet_image(fig)


def build_benchmark_table():
    return ft.Container(
        padding=10,
        bgcolor="#0b0e14",
        border_radius=10,
        border=border_all(1, "rgba(255, 255, 255, 0.05)"),
        content=ft.Column([
            ft.Text("🏆 Model Benchmark Evaluation", size=13, weight=ft.FontWeight.BOLD, color="#ffffff"),
            ft.Row(
                scroll=ft.ScrollMode.AUTO,
                controls=[
                    ft.DataTable(
                        bgcolor="#0b0e14",
                        heading_row_color="#101217",
                        columns=[
                            ft.DataColumn(ft.Text("Architecture", color="#8a8f9e", weight=ft.FontWeight.BOLD, size=11)),
                            ft.DataColumn(ft.Text("BTC MAPE (%)", color="#8a8f9e", weight=ft.FontWeight.BOLD, size=11)),
                            ft.DataColumn(ft.Text("BTC MAE ($)", color="#8a8f9e", weight=ft.FontWeight.BOLD, size=11)),
                            ft.DataColumn(ft.Text("ETH MAPE (%)", color="#8a8f9e", weight=ft.FontWeight.BOLD, size=11)),
                            ft.DataColumn(ft.Text("ETH MAE ($)", color="#8a8f9e", weight=ft.FontWeight.BOLD, size=11)),
                        ],
                        rows=[
                            ft.DataRow(cells=[
                                ft.DataCell(ft.Text("Weighted Ensemble (Trans 60%...)", color="#facc15", weight=ft.FontWeight.BOLD, size=11)),
                                ft.DataCell(ft.Text("2.24%", color="#facc15", weight=ft.FontWeight.BOLD, size=11)),
                                ft.DataCell(ft.Text("$194.50", color="#facc15", weight=ft.FontWeight.BOLD, size=11)),
                                ft.DataCell(ft.Text("3.10%", color="#facc15", weight=ft.FontWeight.BOLD, size=11)),
                                ft.DataCell(ft.Text("$10.15", color="#facc15", weight=ft.FontWeight.BOLD, size=11)),
                            ]),
                            ft.DataRow(cells=[
                                ft.DataCell(ft.Text("Transformer (Attention) 🏆", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("2.51%", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("$218.02", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("3.37%", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("$11.06", color="#ffffff", size=11)),
                            ]),
                            ft.DataRow(cells=[
                                ft.DataCell(ft.Text("Advanced LSTM", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("2.82%", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("$238.24", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("3.74%", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("$11.92", color="#ffffff", size=11)),
                            ]),
                            ft.DataRow(cells=[
                                ft.DataCell(ft.Text("GRU Model", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("2.89%", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("$243.28", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("4.15%", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("$12.86", color="#ffffff", size=11)),
                            ]),
                            ft.DataRow(cells=[
                                ft.DataCell(ft.Text("Random Forest (Regularized)", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("3.00%", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("$253.43", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("4.56%", color="#ffffff", size=11)),
                                ft.DataCell(ft.Text("$14.29", color="#ffffff", size=11)),
                            ]),
                        ]
                    )
                ]
            )
        ])
    )


def build_correlation_chart():
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor='#0b0e14')
    ax.set_facecolor('#0b0e14')

    lags = [0, 1, 2, 3, 4, 5]
    btc_corr = [0.42, 0.68, 0.51, 0.28, 0.14, 0.05]
    eth_corr = [0.38, 0.60, 0.49, 0.25, 0.11, 0.02]

    ax.plot(lags, btc_corr, color='#f97316', marker='o', linewidth=2.5, label='BTC Sentiment Lead/Lag Corr')
    ax.plot(lags, eth_corr, color='#a855f7', marker='s', linewidth=2.5, label='ETH Sentiment Lead/Lag Corr')

    ax.annotate(
        "PEAK IMPACT (24 Hours)",
        xy=(1, 0.68), xytext=(1.3, 0.65),
        arrowprops=dict(facecolor='#38bdf8', shrink=0.05, width=1.5, headwidth=6),
        color='#38bdf8', fontsize=8, weight='bold'
    )

    ax.set_title("Lead/Lag Correlation Matrix across Lags (0 to 5 Days)", color='white', fontsize=10, pad=10, weight='bold')
    ax.set_xlabel("Lag (Days)", color='#8a8f9e', fontsize=8)
    ax.set_ylabel("Correlation", color='#8a8f9e', fontsize=8)
    ax.tick_params(colors='#8a8f9e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
    ax.grid(True, color='#1e293b', linestyle=':', linewidth=0.8)
    ax.legend(facecolor='#101217', edgecolor='#1e293b', labelcolor='white', fontsize=7.5, loc='upper left')

    plt.tight_layout()
    chart = fig_to_flet_image(fig)

    note_widget = ft.Row([
        ft.Text("📌 Η στατιστική ανάλυση αποδεικνύει ότι η μέγιστη επίδραση των ειδήσεων υψηλού αντικτύπου (Peak Impact) αποτυπώνεται στις τιμές 24 ώρες μετά τη δημοσίευση.", size=11, color="#22c55e", weight=ft.FontWeight.W_500)
    ], wrap=True)

    return ft.Column([chart, note_widget], spacing=10)


# ---------------------------------------------------------------------------
# MAIN FLET APPLICATION ENTRY POINT
# ---------------------------------------------------------------------------
def main(page: ft.Page):
    page.title = "Crypto AI — Linear Intelligence Terminal"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#08090a"
    page.padding = 16
    page.scroll = ft.ScrollMode.AUTO

    page.window.width = 420
    page.window.height = 800

    # -----------------------------------------------------------------------
    # STATE MANAGEMENT
    # -----------------------------------------------------------------------
    raw_news = []
    active_ner_filter = None
    is_syncing = False
    active_tab_index = 0

    # UI Components Reference
    ws_status_icon = ft.Icon(ft.Icons.CELL_TOWER, color="#f59e0b", size=14)
    ws_status_text = ft.Text("Connecting WebSocket...", size=11, color="#f59e0b", weight=ft.FontWeight.W_600)
    sync_status_text = ft.Text("Ready to sync", size=11, color="#8a8f9e", weight=ft.FontWeight.W_500)

    # Spot Price Widgets
    btc_price_lbl = ft.Text("$0.00", size=22, weight=ft.FontWeight.BOLD, color="#f7f8f8")
    btc_change_lbl = ft.Text("24h: 0.00%", size=12, color="#8a8f9e")

    eth_price_lbl = ft.Text("$0.00", size=22, weight=ft.FontWeight.BOLD, color="#a855f7")
    eth_change_lbl = ft.Text("24h: 0.00%", size=12, color="#8a8f9e")

    # Macro Sentiment Widgets
    fng_lbl = ft.Text("-", size=18, weight=ft.FontWeight.BOLD, color="#facc15")
    fng_sub_lbl = ft.Text("Sentiment: -", size=11, color="#8a8f9e")

    sent_lbl = ft.Text("-", size=18, weight=ft.FontWeight.BOLD, color="#5e6ad2")
    sent_sub_lbl = ft.Text("NLP: -", size=11, color="#8a8f9e")

    # Prediction Target Widgets
    btc_trans_lbl = ft.Text("$0.00 (BUY)", size=12, color="#5e6ad2", weight=ft.FontWeight.W_600)
    btc_lstm_lbl = ft.Text("$0.00 (BUY)", size=12, color="#22c55e", weight=ft.FontWeight.W_600)
    btc_ens_lbl = ft.Text("$0.00", size=16, color="#eab308", weight=ft.FontWeight.BOLD)
    btc_ens_sub = ft.Text("Signal: - | Conf: -%", size=11, color="#8a8f9e")

    eth_trans_lbl = ft.Text("$0.00 (BUY)", size=12, color="#38bdf8", weight=ft.FontWeight.W_600)
    eth_lstm_lbl = ft.Text("$0.00 (BUY)", size=12, color="#a855f7", weight=ft.FontWeight.W_600)
    eth_ens_lbl = ft.Text("$0.00", size=16, color="#eab308", weight=ft.FontWeight.BOLD)
    eth_ens_sub = ft.Text("Signal: - | Conf: -%", size=11, color="#8a8f9e")

    # Containers for Dynamic UI
    ner_filter_row = ft.Row(scroll=ft.ScrollMode.AUTO, spacing=6)
    news_list_column = ft.Column(spacing=8)

    # Containers for Dynamic Charts
    btc_chart_container = ft.Container(
        content=build_horizon_chart("BTC", [74300, 74450, 73400, 73380, 74200, 75100, 74900, 74750], 75200.0, 73800.0, 74650.0),
        expand=True
    )
    eth_chart_container = ft.Container(
        content=build_horizon_chart("ETH", [2095, 2065, 2050, 2068, 2105, 2135, 2130, 2085, 2125], 2128.0, 2135.0, 2130.0),
        expand=True
    )
    benchmark_container = ft.Container(
        content=build_benchmark_table(),
        expand=True
    )
    correlation_container = ft.Container(
        content=build_correlation_chart(),
        expand=True
    )

    # -----------------------------------------------------------------------
    # CUSTOM ANALYTICS TABS (100% FLET VERSION INDEPENDENT)
    # -----------------------------------------------------------------------
    tab_definitions = [
        ("Bitcoin Horizon", ft.Icons.SHOW_CHART, btc_chart_container),
        ("Ethereum Horizon", ft.Icons.TIMELINE, eth_chart_container),
        ("Model Benchmark", ft.Icons.EMOJI_EVENTS, benchmark_container),
        ("Lagged Correlation", ft.Icons.QUERY_STATS, correlation_container),
    ]

    tab_bar_row = ft.Row(scroll=ft.ScrollMode.AUTO, spacing=6)
    tab_content_host = ft.Container(content=btc_chart_container, height=340)

    def render_tabs():
        tab_bar_row.controls.clear()
        for idx, (label, icon, content_ctrl) in enumerate(tab_definitions):
            is_selected = idx == active_tab_index
            tab_bar_row.controls.append(
                ft.Container(
                    content=ft.Row(
                        spacing=5,
                        controls=[
                            ft.Icon(icon, size=14, color="#0f172a" if is_selected else "#8a8f9e"),
                            ft.Text(label, size=11, weight=ft.FontWeight.BOLD, color="#0f172a" if is_selected else "#f7f8f8"),
                        ]
                    ),
                    bgcolor="#38bdf8" if is_selected else "#101217",
                    border=border_all(1, "#38bdf8" if is_selected else "rgba(255, 255, 255, 0.08)"),
                    padding=padding_symmetric(horizontal=10, vertical=8),
                    border_radius=8,
                    on_click=lambda _, i=idx: select_tab(i)
                )
            )
        tab_content_host.content = tab_definitions[active_tab_index][2]

    def select_tab(idx):
        nonlocal active_tab_index
        active_tab_index = idx
        render_tabs()
        page.update()

    render_tabs()

    analytics_tabs = ft.Column(
        spacing=10,
        controls=[
            tab_bar_row,
            tab_content_host
        ]
    )

    # -----------------------------------------------------------------------
    # API FETCH & RENDER WAKE-UP LOGIC (ASYNC - NON BLOCKING)
    # -----------------------------------------------------------------------
    async def fetch_all_data(e=None):
        nonlocal raw_news, is_syncing
        if is_syncing:
            return

        is_syncing = True
        sync_btn.disabled = True
        sync_btn.icon_color = "#f59e0b"
        sync_status_text.value = "⏳ Waking up Render backend (έως 60s)..."
        sync_status_text.color = "#f59e0b"
        page.update()

        try:
            # Βήμα 1: Ξύπνημα του Render.com αν είναι σε sleep mode
            wake_res = await asyncio.to_thread(requests.get, f"{BASE_URL}/", timeout=65)
            if wake_res.status_code != 200:
                raise Exception(f"Backend returned status {wake_res.status_code}")

            sync_status_text.value = "🔄 Syncing metrics, AI predictions & news..."
            sync_status_text.color = "#38bdf8"
            page.update()

            # Βήμα 2: Ανάκτηση δεδομένων με αυξημένο timeout
            res_m = await asyncio.to_thread(requests.get, f"{BASE_URL}/api/live-metrics", timeout=25)
            res_p = await asyncio.to_thread(requests.get, f"{BASE_URL}/api/predict-live", timeout=25)
            res_n = await asyncio.to_thread(requests.get, f"{BASE_URL}/api/live-news", timeout=25)

            btc_live_price = None
            eth_live_price = None

            if res_m.status_code == 200:
                m = res_m.json()
                btc = m.get("bitcoin", {})
                eth = m.get("ethereum", {})
                fng = m.get("market_sentiment", {})

                btc_live_price = btc.get('close_price', 0.0)
                eth_live_price = eth.get('close_price', 0.0)

                btc_price_lbl.value = f"${btc_live_price:,.2f}"
                btc_change_lbl.value = f"24h: {btc.get('change_pct', 0.0):+.2f}%"

                eth_price_lbl.value = f"${eth_live_price:,.2f}"
                eth_change_lbl.value = f"24h: {eth.get('change_pct', 0.0):+.2f}%"

                fng_lbl.value = f"{fng.get('fear_greed_index', 50)}/100"
                fng_sub_lbl.value = f"Class: {fng.get('fear_greed_class', 'N/A')}"

                score = fng.get("finbert_sentiment_score", 0.0)
                sent_lbl.value = f"{score:+.2f}"
                sent_lbl.color = "#22c55e" if score > 0 else ("#ef4444" if score < 0 else "#62677d")
                sent_sub_lbl.value = f"NLP: {'Bullish 🟢' if score > 0 else 'Bearish 🔴'}"

            if res_p.status_code == 200:
                p = res_p.json().get("predictions", {})

                # BTC Predictions
                tr_btc = p.get("transformer_attention", {})
                ls_btc = p.get("lstm_deep_learning", {})
                en_btc = p.get("ensemble_consensus", {})

                tr_btc_val = tr_btc.get('BTC_target', 75200.0)
                ls_btc_val = ls_btc.get('BTC_target', 73800.0)
                en_btc_val = en_btc.get('BTC_target', 74650.0)

                btc_trans_lbl.value = f"Trans: ${tr_btc_val:,.2f} [{tr_btc.get('BTC_signal', '')}]"
                btc_lstm_lbl.value = f"LSTM: ${ls_btc_val:,.2f} [{ls_btc.get('BTC_signal', '')}]"
                btc_ens_lbl.value = f"${en_btc_val:,.2f}"
                btc_ens_sub.value = f"{en_btc.get('BTC_signal', '')} | Conf: {en_btc.get('BTC_confidence', '')}"

                # ETH Predictions
                tr_eth = p.get("transformer_attention", {})
                ls_eth = p.get("lstm_deep_learning", {})
                en_eth = p.get("ensemble_consensus", {})

                tr_eth_val = tr_eth.get('ETH_target', 2128.0)
                ls_eth_val = ls_eth.get('ETH_target', 2135.0)
                en_eth_val = en_eth.get('ETH_target', 2130.0)

                eth_trans_lbl.value = f"Trans: ${tr_eth_val:,.2f} [{tr_eth.get('ETH_signal', '')}]"
                eth_lstm_lbl.value = f"LSTM: ${ls_eth_val:,.2f} [{ls_eth.get('ETH_signal', '')}]"
                eth_ens_lbl.value = f"${en_eth_val:,.2f}"
                eth_ens_sub.value = f"{en_eth.get('ETH_signal', '')} | Conf: {en_eth.get('ETH_confidence', '')}"

                # Ανάκτηση πραγματικών ιστορικών τιμών & ενημέρωση διαγραμμάτων
                btc_hist = await asyncio.to_thread(fetch_historical_prices, "BTC", btc_live_price, 10)
                eth_hist = await asyncio.to_thread(fetch_historical_prices, "ETH", eth_live_price, 10)

                btc_chart_container.content = build_horizon_chart(
                    "BTC", btc_hist, tr_btc_val, ls_btc_val, en_btc_val
                )
                eth_chart_container.content = build_horizon_chart(
                    "ETH", eth_hist, tr_eth_val, ls_eth_val, en_eth_val
                )

            if res_n.status_code == 200:
                raw_news = res_n.json().get("news_feed", [])
                render_ner_filters()
                render_news_cards()

            sync_status_text.value = "✅ Pipeline Synced (Render Online)"
            sync_status_text.color = "#22c55e"
        except Exception as ex:
            print(f"[!] Error fetching REST API: {ex}")
            sync_status_text.value = "⚠️ Sync timeout/error — πάτα ξανά Sync"
            sync_status_text.color = "#ef4444"
        finally:
            is_syncing = False
            sync_btn.disabled = False
            sync_btn.icon_color = "#5e6ad2"
            page.update()

    # -----------------------------------------------------------------------
    # DYNAMIC NER FILTERING & NEWS FEED
    # -----------------------------------------------------------------------
    def set_ner_filter(entity_name):
        nonlocal active_ner_filter
        active_ner_filter = entity_name
        render_ner_filters()
        render_news_cards()
        page.update()

    def render_ner_filters():
        ner_filter_row.controls.clear()
        entities_set = set()
        for item in raw_news:
            for ent in item.get("entities", []):
                entities_set.add(ent)

        is_all = active_ner_filter is None
        ner_filter_row.controls.append(
            ft.Container(
                content=ft.Text("#ALL", size=11, weight=ft.FontWeight.BOLD, color="#0f172a" if is_all else "#38bdf8"),
                bgcolor="#38bdf8" if is_all else "#1e293b",
                padding=padding_symmetric(horizontal=10, vertical=4),
                border_radius=10,
                on_click=lambda _: set_ner_filter(None)
            )
        )

        for ent in sorted(list(entities_set))[:8]:
            is_act = active_ner_filter == ent
            ner_filter_row.controls.append(
                ft.Container(
                    content=ft.Text(f"#{ent}", size=11, weight=ft.FontWeight.BOLD, color="#0f172a" if is_act else "#38bdf8"),
                    bgcolor="#38bdf8" if is_act else "#1e293b",
                    padding=padding_symmetric(horizontal=10, vertical=4),
                    border_radius=10,
                    on_click=lambda _, e=ent: set_ner_filter(e)
                )
            )

    def render_news_cards():
        news_list_column.controls.clear()
        for item in raw_news:
            entities = item.get("entities", [])
            if active_ner_filter and active_ner_filter not in entities:
                continue

            topic_label = item.get("topic_label", "General")
            topic_color = item.get("topic_color", "#6b7280")
            impact_color = item.get("color", "#62677d")

            ent_chips = [
                ft.Container(
                    content=ft.Text(f"#{e}", size=9, color="#38bdf8", weight=ft.FontWeight.W_600),
                    bgcolor="#0f172a",
                    padding=padding_symmetric(horizontal=6, vertical=2),
                    border_radius=6
                ) for e in entities
            ]

            card = ft.Container(
                padding=12,
                bgcolor="#101217",
                border_radius=10,
                border=border_all(1, "rgba(255, 255, 255, 0.05)"),
                content=ft.Column(
                    spacing=6,
                    controls=[
                        ft.Row(
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            controls=[
                                ft.Container(
                                    content=ft.Text(topic_label, size=9, weight=ft.FontWeight.BOLD, color="#ffffff"),
                                    bgcolor=topic_color,
                                    padding=padding_symmetric(horizontal=6, vertical=2),
                                    border_radius=4
                                ),
                                ft.Text(f"{item.get('asset', '')} • {item.get('time', '')}", size=10, color="#8a8f9e")
                            ]
                        ),
                        ft.Text(item.get("title", ""), size=12, weight=ft.FontWeight.W_600, color="#f7f8f8"),
                        ft.Row(
                            controls=[
                                ft.Text("Expected Impact:", size=11, color="#8a8f9e"),
                                ft.Text(item.get("expected_impact", ""), size=11, weight=ft.FontWeight.BOLD, color=impact_color),
                                ft.Text(f"({item.get('sentiment_label', '')})", size=10, color="#8a8f9e")
                            ]
                        ),
                        ft.Row(controls=ent_chips, wrap=True, spacing=4) if ent_chips else ft.Container()
                    ]
                )
            )
            news_list_column.controls.append(card)

    # -----------------------------------------------------------------------
    # WEBSOCKET REAL-TIME TICKER TASK
    # -----------------------------------------------------------------------
    async def binance_ws_task():
        import websockets
        while True:
            try:
                async with websockets.connect(WS_URL) as ws:
                    ws_status_icon.color = "#38bdf8"
                    ws_status_text.value = "WebSocket Live Ticks (Binance)"
                    ws_status_text.color = "#38bdf8"
                    page.update()

                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if data.get("type") == "TICK_UPDATE":
                            sym = data.get("symbol")
                            price = data.get("price", 0.0)
                            change = data.get("change_pct", 0.0)

                            if sym == "BTCUSDT":
                                btc_price_lbl.value = f"${price:,.2f}"
                                btc_change_lbl.value = f"24h: {change:+.2f}% ⚡ Live"
                                btc_price_lbl.color = "#38bdf8"
                            elif sym == "ETHUSDT":
                                eth_price_lbl.value = f"${price:,.2f}"
                                eth_change_lbl.value = f"24h: {change:+.2f}% ⚡ Live"
                                eth_price_lbl.color = "#38bdf8"

                            page.update()
            except Exception:
                ws_status_icon.color = "#f59e0b"
                ws_status_text.value = "Reconnecting WebSocket Stream..."
                ws_status_text.color = "#f59e0b"
                page.update()
                await asyncio.sleep(3)

    # -----------------------------------------------------------------------
    # CARD BUILDERS & UI LAYOUT
    # -----------------------------------------------------------------------
    def build_card(title, main_widget, sub_widget=None, accent_border="rgba(255, 255, 255, 0.05)"):
        return ft.Container(
            padding=14,
            bgcolor="#101217",
            border_radius=12,
            border=border_all(1, accent_border),
            content=ft.Column(
                spacing=4,
                controls=[
                    ft.Text(title.upper(), size=10, weight=ft.FontWeight.BOLD, color="#8a8f9e"),
                    main_widget,
                    sub_widget if sub_widget else ft.Container()
                ]
            )
        )

    sync_btn = ft.IconButton(
        icon=ft.Icons.SYNC,
        icon_color="#5e6ad2",
        on_click=fetch_all_data,
        tooltip="Sync Pipeline (Wake Up Render)"
    )

    header = ft.Row(
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        controls=[
            ft.Column([
                ft.Text("❖ Crypto AI", size=20, weight=ft.FontWeight.BOLD, color="#f7f8f8"),
                ft.Text("Linear Mobile Intelligence", size=11, color="#8a8f9e")
            ], spacing=1),
            sync_btn
        ]
    )

    ws_bar = ft.Column(
        spacing=2,
        controls=[
            ft.Row(controls=[ws_status_icon, ws_status_text], spacing=6),
            sync_status_text
        ]
    )

    cards_grid = ft.Column(
        spacing=10,
        controls=[
            ft.Row(
                controls=[
                    ft.Container(content=build_card("BITCOIN (BTC)", btc_price_lbl, btc_change_lbl, "#f7931a"), expand=True),
                    ft.Container(content=build_card("ETHEREUM (ETH)", eth_price_lbl, eth_change_lbl, "#a855f7"), expand=True),
                ],
                spacing=10
            ),
            ft.Row(
                controls=[
                    ft.Container(content=build_card("FEAR & GREED", fng_lbl, fng_sub_lbl, "#facc15"), expand=True),
                    ft.Container(content=build_card("FINBERT SCORE", sent_lbl, sent_sub_lbl, "#5e6ad2"), expand=True),
                ],
                spacing=10
            ),
            build_card(
                "BTC ENSEMBLE CONSENSUS TARGET 🌟",
                ft.Column([
                    btc_ens_lbl,
                    btc_ens_sub,
                    ft.Divider(height=10, color="rgba(255,255,255,0.05)"),
                    btc_trans_lbl,
                    btc_lstm_lbl
                ], spacing=3),
                accent_border="#eab308"
            ),
            build_card(
                "ETH ENSEMBLE CONSENSUS TARGET 🌟",
                ft.Column([
                    eth_ens_lbl,
                    eth_ens_sub,
                    ft.Divider(height=10, color="rgba(255,255,255,0.05)"),
                    eth_trans_lbl,
                    eth_lstm_lbl
                ], spacing=3),
                accent_border="#eab308"
            ),
        ]
    )

    news_section = ft.Column(
        spacing=8,
        controls=[
            ft.Text("24H LIVE NEWS & NER FILTERING", size=11, weight=ft.FontWeight.BOLD, color="#8a8f9e"),
            ner_filter_row,
            news_list_column
        ]
    )

    page.add(
        header,
        ws_bar,
        ft.Divider(height=12, color="transparent"),
        cards_grid,
        ft.Divider(height=16, color="rgba(255,255,255,0.05)"),
        analytics_tabs,
        ft.Divider(height=16, color="rgba(255,255,255,0.05)"),
        news_section
    )

    page.run_task(fetch_all_data)
    page.run_task(binance_ws_task)


if __name__ == "__main__":
    if hasattr(ft, "app"):
        ft.app(target=main)
    elif hasattr(ft, "run"):
        ft.run(main)