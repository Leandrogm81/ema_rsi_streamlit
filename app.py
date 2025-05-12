# app.py
import streamlit as st
import pandas as pd
import pandas_ta as ta # Usaremos para EMA, RSI, ATR
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Para subplots
from datetime import datetime, timedelta, date
import quantstats as qs # Para métricas avançadas
import time # Para pequeno delay opcional

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Estratégia Fast-Trend Confirmed",
    page_icon="⚡", # Novo Ícone
    layout="wide"
)

# --- Inicialização do Session State ---
# Guarda valores entre execuções da página
default_watchlist = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = default_watchlist # Começa com alguns exemplos
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None # Guarda resultado da simulação individual
if 'last_ticker_simulated' not in st.session_state:
    st.session_state.last_ticker_simulated = "" # Qual ticker foi simulado por último
if 'current_signals' not in st.session_state:
    st.session_state.current_signals = {} # Guarda {ticker: sinal} da verificação atual
# Define valores iniciais para os novos parâmetros no state (se não existirem)
if 'ticker_input_value' not in st.session_state: st.session_state.ticker_input_value = "PETR4.SA"
if 'cfg_ema_fast' not in st.session_state: st.session_state.cfg_ema_fast = 10
if 'cfg_ema_slow' not in st.session_state: st.session_state.cfg_ema_slow = 50
if 'cfg_rsi_len' not in st.session_state: st.session_state.cfg_rsi_len = 14
if 'cfg_rsi_buy' not in st.session_state: st.session_state.cfg_rsi_buy = 55.0
if 'cfg_rsi_sell' not in st.session_state: st.session_state.cfg_rsi_sell = 45.0
if 'cfg_atr_len' not in st.session_state: st.session_state.cfg_atr_len = 14
if 'cfg_atr_mult' not in st.session_state: st.session_state.cfg_atr_mult = 2.0
if 'cfg_start' not in st.session_state: st.session_state.cfg_start = date.today() - timedelta(days=5*365)
if 'cfg_end' not in st.session_state: st.session_state.cfg_end = date.today()
if 'cfg_capital' not in st.session_state: st.session_state.cfg_capital = 1000.0
if 'min_cagr_input' not in st.session_state: st.session_state.min_cagr_input = 30.0
# Guarda resultado do scan
if 'scan_results_df' not in st.session_state: st.session_state.scan_results_df = pd.DataFrame()

# --- Lista de Tickers Comuns ---
COMMON_TICKERS = {
    "IBOV (Exemplos)": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "MGLU3.SA", "WEGE3.SA", "B3SA3.SA", "RENT3.SA", "PRIO3.SA"],
    "US Tech (Exemplos)": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CSCO"],
    "Cripto (Exemplos)": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"]
}
FLAT_TICKER_LIST = [""] + sorted(list(set(ticker for sublist in COMMON_TICKERS.values() for ticker in sublist)))

# --- Função de Métricas ---
def calculate_metrics(trades_df, equity_curve_df, initial_capital):
    """Calcula métricas de desempenho básicas e avançadas."""
    metrics = {"winRate": 0, "payoffRatio": 0, "avgGain": 0, "avgLoss": 0, "maxDrawdown": 0, "sharpeRatio": 0, "cagr": 0}
    if trades_df.empty or equity_curve_df.empty:
        return metrics

    # Cálculos Manuais
    wins = trades_df[trades_df['profit'] > 0]; losses = trades_df[trades_df['profit'] <= 0]
    total_trades = len(trades_df)
    metrics["winRate"] = round((len(wins) / total_trades * 100), 2) if total_trades > 0 else 0
    metrics["avgGain"] = round(wins['profit'].mean(), 2) if not wins.empty else 0
    metrics["avgLoss"] = round(abs(losses['profit'].mean()), 2) if not losses.empty else 0 # Média da perda (positiva)
    metrics["payoffRatio"] = round(metrics["avgGain"] / metrics["avgLoss"], 2) if metrics["avgLoss"] > 0 else 0

    # Cálculos com QuantStats
    try:
        equity_curve_df['Date'] = pd.to_datetime(equity_curve_df['date'])
        equity_curve_df = equity_curve_df.set_index('Date')
        daily_returns = equity_curve_df['equity'].pct_change().dropna()

        if not daily_returns.empty and daily_returns.std() != 0 and len(daily_returns) >= 3:
            metrics["maxDrawdown"] = round(qs.stats.max_drawdown(daily_returns) * 100, 2) # Em %
            metrics["sharpeRatio"] = round(qs.stats.sharpe(daily_returns), 2)
            metrics["cagr"] = round(qs.stats.cagr(daily_returns) * 100, 2) # CAGR em %
        else:
             st.caption("Dados insuficientes para métricas QuantStats.")

    except Exception as e:
        st.caption(f"Aviso: Erro métricas QuantStats: {e}")

    for k, v in metrics.items():
        if pd.isna(v): metrics[k] = 0
    return metrics

# --- Função do Backtest (Estratégia Fast-Trend Confirmed) ---
@st.cache_data(ttl=3600) # Cache de 1 hora
def run_strategy_backtest(ticker: str, start_date: date, end_date: date,
                          initial_capital: float = 1000.0,
                          ema_fast_len: int = 10, ema_slow_len: int = 50,
                          rsi_len: int = 14, rsi_buy_level: float = 55.0, rsi_sell_level: float = 45.0,
                          atr_len: int = 14, atr_multiplier: float = 2.0):
    """
    Executa o backtest da estratégia "Fast-Trend Confirmed".
    """
    st.caption(f"Iniciando backtest para {ticker}...")

    # --- 1. Buscar Dados Históricos ---
    fetch_start_date = start_date - timedelta(days=max(ema_slow_len, rsi_len, atr_len) + 50)
    start_str = fetch_start_date.strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        data = yf.download(ticker, start=start_str, end=end_str, interval="1d", auto_adjust=True, progress=False)
        if data.empty: raise ValueError("Nenhum dado retornado.")
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[(data.index >= pd.Timestamp(start_date)) & (data.index < pd.Timestamp(end_date + timedelta(days=1)))]
        if data.empty: raise ValueError(f"Nenhum dado no período {start_date} a {end_date}.")
    except Exception as e:
        st.error(f"Erro ao buscar dados para {ticker}: {e}", icon="📉")
        return None

    # --- 2. Calcular Indicadores ---
    try:
        ema_fast_col = f'EMA_{ema_fast_len}'; ema_slow_col = f'EMA_{ema_slow_len}'
        data.ta.ema(close='Close', length=ema_fast_len, append=True, col_names=(ema_fast_col,))
        data.ta.ema(close='Close', length=ema_slow_len, append=True, col_names=(ema_slow_col,))
        rsi_col = f'RSI_{rsi_len}'
        data.ta.rsi(close='Close', length=rsi_len, append=True, col_names=(rsi_col,))
        atr_col = f'ATR_{atr_len}'
        data.ta.atr(length=atr_len, append=True, col_names=(atr_col,))
        initial_len = len(data)
        data.dropna(inplace=True)
        if data.empty: raise ValueError(f"Dados insuficientes após indicadores. {initial_len} linhas.")
    except Exception as e:
        st.error(f"Erro ao calcular indicadores para {ticker}: {e}", icon="📊")
        return None

    # --- 3. Simulação ---
    cash = initial_capital; shares = 0.0; in_position = False
    entry_price = 0.0; entry_date = None; entry_atr = 0.0
    stop_loss_price = 0.0; trailing_stop_price = 0.0; highest_price_since_entry = 0.0
    trades = []; signals = []
    equity_curve = [{"date": data.index[0].date().strftime('%Y-%m-%d'), "equity": initial_capital}]

    for i in range(1, len(data)):
        current_date = data.index[i]; current_open = data['Open'].iloc[i]
        current_high = data['High'].iloc[i]; current_low = data['Low'].iloc[i]
        current_close = data['Close'].iloc[i]; prev_row = data.iloc[i-1]
        prev_prev_row = data.iloc[i-2] if i > 1 else None

        prev_ema_fast = prev_row[ema_fast_col]; prev_ema_slow = prev_row[ema_slow_col]
        prev_rsi = prev_row[rsi_col]; prev_atr = prev_row[atr_col]

        exit_signal = False; exit_reason = None; exit_price = 0.0
        if in_position:
            highest_price_since_entry = max(highest_price_since_entry, current_high)
            new_trailing_stop = highest_price_since_entry - (atr_multiplier * entry_atr)
            trailing_stop_price = max(trailing_stop_price, new_trailing_stop)

            if current_low <= stop_loss_price:
                exit_signal, exit_reason, exit_price = True, f"Stop Loss ({atr_multiplier}xATR)", stop_loss_price
            elif current_low <= trailing_stop_price:
                exit_signal, exit_reason, exit_price = True, f"Trailing Stop ({atr_multiplier}xATR)", trailing_stop_price
            elif prev_prev_row is not None:
                ema_crossed_down = prev_ema_fast < prev_ema_slow and prev_prev_row[ema_fast_col] >= prev_prev_row[ema_slow_col]
                if ema_crossed_down: exit_signal, exit_reason, exit_price = True, "EMA Cross Down", current_open
                elif prev_rsi < rsi_sell_level: exit_signal, exit_reason, exit_price = True, f"RSI < {rsi_sell_level}", current_open

            if exit_signal:
                if shares > 0:
                    cash += shares * exit_price
                    profit = (exit_price - entry_price) * shares
                    return_pct = ((exit_price / entry_price) - 1) * 100 if entry_price != 0 else 0
                    days_held = (current_date.date() - entry_date).days + 1
                    trades.append({"entryDate": entry_date.strftime('%Y-%m-%d'), "entryPrice": round(entry_price, 2),
                                   "exitDate": current_date.date().strftime('%Y-%m-%d'), "exitPrice": round(exit_price, 2),
                                   "shares": round(shares, 4), "profit": round(profit, 2), "returnPercent": round(return_pct, 2),
                                   "exitReason": exit_reason, "daysHeld": days_held})
                    signals.append({"date": current_date.date().strftime('%Y-%m-%d'), "type": "sell", "price": round(exit_price, 2)})
                shares, in_position, entry_price, entry_date, entry_atr, stop_loss_price, trailing_stop_price, highest_price_since_entry = \
                    0.0, False, 0.0, None, 0.0, 0.0, 0.0, 0.0

        if not in_position and prev_prev_row is not None:
            ema_crossed_up = prev_ema_fast > prev_ema_slow and prev_prev_row[ema_fast_col] <= prev_prev_row[ema_slow_col]
            rsi_confirm = prev_rsi > rsi_buy_level
            if ema_crossed_up and rsi_confirm:
                entry_price = current_open
                if entry_price > 0 and cash > 0 and prev_atr > 0:
                    entry_atr = prev_atr
                    stop_loss_price = entry_price - (atr_multiplier * entry_atr)
                    trailing_stop_price = stop_loss_price
                    highest_price_since_entry = entry_price
                    shares_to_buy = cash / entry_price # 100% capital
                    cash -= shares_to_buy * entry_price
                    shares, in_position, entry_date = shares_to_buy, True, current_date.date()
                    signals.append({"date": current_date.date().strftime('%Y-%m-%d'), "type": "buy", "price": round(entry_price, 2)})

        current_equity = cash + (shares * current_close)
        equity_curve.append({"date": current_date.date().strftime('%Y-%m-%d'), "equity": round(current_equity, 2)})

    # --- 4. Calcular Resultados Finais ---
    final_equity = equity_curve[-1]['equity'] if equity_curve else initial_capital
    total_profit = final_equity - initial_capital
    total_return_percent = (total_profit / initial_capital) * 100 if initial_capital > 0 else 0
    number_of_trades = len(trades)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(); equity_curve_df = pd.DataFrame(equity_curve)
    adv_metrics = calculate_metrics(trades_df, equity_curve_df, initial_capital)

    # --- 5. Preparar Dicionário de Retorno ---
    chart_data_df = data[['Open', 'High', 'Low', 'Close', ema_fast_col, ema_slow_col, rsi_col]].reset_index()
    chart_data_df['Date'] = pd.to_datetime(chart_data_df['Date']).dt.date
    metrics_start_date = data.index.min().date() if not data.empty else None
    metrics_end_date = data.index.max().date() if not data.empty else None

    results = {
        "ticker": ticker,
        "params": {"ema_f": ema_fast_len, "ema_s": ema_slow_len, "rsi": rsi_len, "rsi_buy": rsi_buy_level, "rsi_sell": rsi_sell_level, "atr_len": atr_len, "atr_mult": atr_multiplier, "capital": initial_capital},
        "metrics": {"initialCapital": round(initial_capital, 2), "finalEquity": round(final_equity, 2), "totalProfit": round(total_profit, 2),
                    "totalReturnPercent": round(total_return_percent, 2), "numberOfTrades": number_of_trades,
                    "startDate": metrics_start_date.strftime('%Y-%m-%d') if metrics_start_date else 'N/A',
                    "endDate": metrics_end_date.strftime('%Y-%m-%d') if metrics_end_date else 'N/A', **adv_metrics},
        "trades": trades_df, "signals": signals, "chartData": chart_data_df,
        "indicators": {"ema_f": ema_fast_col, "ema_s": ema_slow_col, "rsi": rsi_col}
    }
    st.caption(f"Backtest '{ticker}' concluído.")
    return results


# --- Função para Plotar o Gráfico (Estratégia Fast-Trend) ---
def plot_results(chart_data_df, signals, ticker, indicators_cols, show_indicators=False):
    """Plota o gráfico de velas, sinais e opcionalmente os indicadores EMA e RSI."""
    if chart_data_df.empty: st.warning("Não há dados para plotar."); return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Gráfico de Velas (Linha 1)
    fig.add_trace(go.Candlestick(x=chart_data_df['Date'], open=chart_data_df['Open'], high=chart_data_df['High'],
                               low=chart_data_df['Low'], close=chart_data_df['Close'], name='OHLC',
                               increasing_line_color='green', decreasing_line_color='red'), row=1, col=1)
    # Plotar EMAs
    ema_f_col = indicators_cols.get('ema_f'); ema_s_col = indicators_cols.get('ema_s')
    if ema_f_col and ema_s_col:
         fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[ema_f_col], name=ema_f_col.replace('_',' '), line=dict(color='blue', width=1)), row=1, col=1)
         fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[ema_s_col], name=ema_s_col.replace('_',' '), line=dict(color='orange', width=1)), row=1, col=1)
    # Sinais Compra/Venda
    buy_signals_df = pd.DataFrame([s for s in signals if s['type'] == 'buy'])
    if not buy_signals_df.empty: fig.add_trace(go.Scatter(x=buy_signals_df['date'], y=buy_signals_df['price'], mode='markers', name='Compra', marker=dict(color='green', size=10, symbol='triangle-up', line=dict(width=1, color='DarkSlateGrey'))), row=1, col=1)
    sell_signals_df = pd.DataFrame([s for s in signals if s['type'] == 'sell'])
    if not sell_signals_df.empty: fig.add_trace(go.Scatter(x=sell_signals_df['date'], y=sell_signals_df['price'], mode='markers', name='Venda', marker=dict(color='red', size=10, symbol='triangle-down', line=dict(width=1, color='DarkSlateGrey'))), row=1, col=1)

    # Plotar RSI (Linha 2)
    if show_indicators:
        rsi_col = indicators_cols.get('rsi')
        rsi_buy_level = st.session_state.cfg_rsi_buy; rsi_sell_level = st.session_state.cfg_rsi_sell
        if rsi_col:
            fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[rsi_col], name=rsi_col.replace('_',' '), line=dict(color='purple', width=1.5)), row=2, col=1)
            fig.add_hline(y=rsi_buy_level, line_dash="dot", line_color="rgba(0,150,0,0.6)", annotation_text=f"RSI Compra > {rsi_buy_level}", annotation_position="bottom right", row=2, col=1)
            fig.add_hline(y=rsi_sell_level, line_dash="dot", line_color="rgba(150,0,0,0.6)", annotation_text=f"RSI Venda < {rsi_sell_level}", annotation_position="top right", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, showgrid=True)
    else: fig.update_layout(yaxis2_visible=False)

    # Layout Geral
    fig.update_layout(title={'text': f'Backtest Fast-Trend: {ticker}', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      height=650, xaxis_rangeslider_visible=False, xaxis_showticklabels=True, xaxis2_showticklabels=True,
                      yaxis_title="Preço / EMA", row=1, col=1, yaxis_fixedrange=False, yaxis2_fixedrange=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5), margin=dict(l=50, r=50, t=80, b=50))
    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True, row=1, col=1)
    if show_indicators: fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True, row=2, col=1)
    return fig


# --- Função para Verificar Sinal Atual (Estratégia Fast-Trend) ---
@st.cache_data(ttl=900) # Cache de 15 minutos
def get_current_signal(ticker: str,
                       ema_fast_len: int = 10, ema_slow_len: int = 50,
                       rsi_len: int = 14, rsi_buy_level: float = 55.0, rsi_sell_level: float = 45.0):
    """Verifica o sinal da estratégia Fast-Trend (COMPRA/VENDA/NEUTRO) no último dia."""
    st.caption(f"Verificando sinal {ticker}...")
    try:
        data = yf.download(ticker, period=f"{ema_slow_len + 20}d", interval="1d", auto_adjust=True, progress=False)
        if data.empty or len(data) < ema_slow_len + 2: return "Dados Insuf."
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data.ta.ema(close='Close', length=ema_fast_len, append=True, col_names=('EMA_F',))
        data.ta.ema(close='Close', length=ema_slow_len, append=True, col_names=('EMA_S',))
        data.ta.rsi(close='Close', length=rsi_len, append=True, col_names=('RSI',))
        data.dropna(inplace=True)
        if len(data) < 3: return "Dados Insuf. pós Ind."

        last = data.iloc[-1]; prev = data.iloc[-2]; prev_prev = data.iloc[-3]

        # Condição de Compra (baseado no fechamento anterior)
        ema_crossed_up = prev['EMA_F'] > prev['EMA_S'] and prev_prev['EMA_F'] <= prev_prev['EMA_S']
        rsi_confirm = prev['RSI'] > rsi_buy_level
        if ema_crossed_up and rsi_confirm: return "COMPRA"

        # Condição de Saída Técnica (baseado no fechamento anterior)
        ema_crossed_down = prev['EMA_F'] < prev['EMA_S'] and prev_prev['EMA_F'] >= prev_prev['EMA_S']
        rsi_below_level = prev['RSI'] < rsi_sell_level
        if ema_crossed_down or rsi_below_level: return "VENDA" # Sinaliza possível fechamento

        return "NEUTRO"
    except Exception as e:
        st.caption(f"Erro {ticker}: {str(e)[:30]}...")
        return "Erro"

# --- Formatação de Moeda ---
def format_currency(value):
    try: return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except: return "R$ 0,00"

# ==============================================================================
# --- Interface Principal do Streamlit ---
# ==============================================================================

st.title("⚡ Estratégia Fast-Trend Confirmed")
st.caption("Simulador, Scanner e Monitor (EMA 10x50 + RSI 14 + ATR Stop)")
st.divider()

# --- Barra Lateral (Sidebar - ATUALIZADA com novos parâmetros) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/crossing.png", width=80)
    st.header("⚙️ Configurações Globais")
    st.caption("Afetam Simulador e Scanner.")

    with st.expander("Parâmetros EMA & RSI", expanded=True):
        # Usa chaves do session_state para pegar valores, mas key do widget é diferente
        # para evitar conflito de chave se o mesmo widget fosse usado fora da sidebar
        st.session_state.cfg_ema_fast = st.number_input("EMA Rápida (dias)", min_value=3, max_value=100, value=st.session_state.cfg_ema_fast, step=1, key="cfg_ema_fast_in")
        st.session_state.cfg_ema_slow = st.number_input("EMA Lenta (dias)", min_value=10, max_value=200, value=st.session_state.cfg_ema_slow, step=1, key="cfg_ema_slow_in")
        st.session_state.cfg_rsi_len = st.number_input("RSI Período (dias)", min_value=5, max_value=50, value=st.session_state.cfg_rsi_len, step=1, key="cfg_rsi_len_in")
        st.session_state.cfg_rsi_buy = st.number_input("RSI Nível Compra >", min_value=50.0, max_value=70.0, value=st.session_state.cfg_rsi_buy, step=0.5, format="%.1f", key="cfg_rsi_buy_in")
        st.session_state.cfg_rsi_sell = st.number_input("RSI Nível Venda <", min_value=30.0, max_value=50.0, value=st.session_state.cfg_rsi_sell, step=0.5, format="%.1f", key="cfg_rsi_sell_in")

    with st.expander("Parâmetros Stop Loss (ATR)", expanded=True):
        st.session_state.cfg_atr_len = st.number_input("ATR Período (dias)", min_value=5, max_value=50, value=st.session_state.cfg_atr_len, step=1, key="cfg_atr_len_in")
        st.session_state.cfg_atr_mult = st.number_input("ATR Multiplicador (Stop/Trail)", min_value=1.0, max_value=5.0, value=st.session_state.cfg_atr_mult, step=0.1, format="%.1f", key="cfg_atr_mult_in")

    with st.expander("Período do Backtest", expanded=True):
        today = datetime.now().date()
        # Usa valores do state ou defaults
        cfg_start = st.date_input("Data Inicial", value=st.session_state.cfg_start, max_value=today - timedelta(days=30), key="cfg_start_in")
        cfg_end = st.date_input("Data Final", value=st.session_state.cfg_end, min_value=cfg_start + timedelta(days=30), max_value=today, key="cfg_end_in")
        st.session_state.cfg_start = cfg_start # Atualiza state
        st.session_state.cfg_end = cfg_end     # Atualiza state
        st.caption(f"Duração: {(cfg_end - cfg_start).days} dias")

    with st.expander("Financeiro", expanded=True):
        cfg_capital = st.number_input("Capital Inicial (R$)", min_value=1.0, value=st.session_state.cfg_capital, step=100.0, format="%.2f", key="cfg_capital_in")
        st.session_state.cfg_capital = cfg_capital # Atualiza state

    st.divider()
    st.header("⭐ Lista de Monitoramento")

    if st.button("📡 Verificar Sinais Atuais", key="check_signals", use_container_width=True, help="Verifica o sinal da estratégia para hoje."):
        current_signals_temp = {}
        if st.session_state.watchlist:
            progress_bar = st.progress(0, text="Verificando sinais...")
            num_tickers = len(st.session_state.watchlist)
            for i, ticker_to_check in enumerate(st.session_state.watchlist):
                signal = get_current_signal(ticker_to_check, st.session_state.cfg_ema_fast, st.session_state.cfg_ema_slow, st.session_state.cfg_rsi_len, st.session_state.cfg_rsi_buy, st.session_state.cfg_rsi_sell)
                current_signals_temp[ticker_to_check] = signal
                progress_bar.progress((i + 1) / num_tickers, text=f"Verificando {ticker_to_check}...")
                time.sleep(0.05)
            st.session_state.current_signals = current_signals_temp
            progress_bar.empty(); st.toast("Verificação concluída!", icon="✅")
        else: st.toast("Watchlist vazia.", icon="ℹ️")

    if not st.session_state.watchlist: st.info("Nenhum ativo na lista.")
    else:
        st.caption("Ativos:")
        for i in range(len(st.session_state.watchlist) - 1, -1, -1):
            ticker_in_list = st.session_state.watchlist[i]
            signal_status = st.session_state.current_signals.get(ticker_in_list, "")
            signal_color = {"COMPRA": "green", "VENDA": "red", "NEUTRO": "gray", "Erro": "orange", "Dados Insuf.": "orange", "Dados Insuf. pós Ind.": "orange"}.get(signal_status, "gray")
            signal_icon = {"COMPRA": "🔼", "VENDA": "🔽", "NEUTRO": "⏸️"}.get(signal_status, "❔")
            signal_display = f"<span style='color:{signal_color}; font-weight:bold; font-size:small;'>{signal_icon} {signal_status}</span>" if signal_status else ""
            col1, col2, col3 = st.columns([0.55, 0.25, 0.2])
            with col1:
                if st.button(ticker_in_list, key=f"sim_{ticker_in_list}_{i}", type="secondary", help=f"Carregar {ticker_in_list}", use_container_width=True):
                     st.session_state.ticker_input_value = ticker_in_list; st.rerun()
            with col2: st.markdown(signal_display, unsafe_allow_html=True)
            with col3:
                if st.button("➖", key=f"remove_{ticker_in_list}_{i}", type="secondary", help="Remover", use_container_width=True):
                    removed = st.session_state.watchlist.pop(i); st.session_state.current_signals.pop(removed, None)
                    st.toast(f"{removed} removido.");
                    if st.session_state.get('last_ticker_simulated') == removed: st.session_state['backtest_results'] = None; st.session_state['last_ticker_simulated'] = ""
                    st.rerun()

    if st.session_state.watchlist:
        st.markdown("<a href='?clear_watchlist=true' target='_self' style='color: tomato; font-size: small;'>Limpar Lista Completa</a>", unsafe_allow_html=True)

# Lógica para limpar watchlist
if st.query_params.get("clear_watchlist") == "true":
    st.session_state.watchlist = []; st.session_state.current_signals = {}
    st.session_state['backtest_results'] = None; st.session_state['last_ticker_simulated'] = ""
    st.session_state['scan_results_df'] = pd.DataFrame(); st.toast("Lista e resultados limpos.")
    st.query_params.clear(); st.rerun()


# ==============================================================================
# --- Abas para Organizar a Interface ---
# ==============================================================================
tab1, tab2 = st.tabs(["📈 **Simulador Individual**", "🔍 **Scanner de Ativos**"])

# --- Aba 1: Simulador Individual ---
with tab1:
    st.header("Simulador de Backtest Individual")
    st.markdown("Teste a estratégia Fast-Trend em um ativo com os parâmetros da barra lateral.")
    sim_col1, sim_col2 = st.columns([0.6, 0.4])
    with sim_col1:
        ticker_input_sim = st.text_input("Código do Ativo:", value=st.session_state.ticker_input_value,
                                         placeholder="Digite ou selecione abaixo...", key="ticker_input_main_tab1", label_visibility="collapsed").upper()
        if ticker_input_sim != st.session_state.ticker_input_value:
             st.session_state.ticker_input_value = ticker_input_sim
             st.session_state.backtest_results = None; st.session_state.last_ticker_simulated = ""
    with sim_col2:
        selected_common_ticker_sim = st.selectbox("Selecionar ativo comum:", options=FLAT_TICKER_LIST,
                                                  index=FLAT_TICKER_LIST.index(st.session_state.ticker_input_value) if st.session_state.ticker_input_value in FLAT_TICKER_LIST else 0,
                                                  key="common_ticker_select_tab1", label_visibility="collapsed")
        if selected_common_ticker_sim and selected_common_ticker_sim != st.session_state.ticker_input_value:
            st.session_state.ticker_input_value = selected_common_ticker_sim
            st.session_state.backtest_results = None; st.session_state.last_ticker_simulated = ""
            st.rerun()

    simulate_button = st.button("Executar Simulação Individual", type="primary", use_container_width=True, key="sim_button_tab1")
    st.divider()

    # Execução Simulação Individual
    if simulate_button:
        current_ticker_input = st.session_state.ticker_input_value
        if current_ticker_input:
            sim_params = {"start_date": st.session_state.cfg_start, "end_date": st.session_state.cfg_end, "initial_capital": st.session_state.cfg_capital,
                          "ema_fast_len": st.session_state.cfg_ema_fast, "ema_slow_len": st.session_state.cfg_ema_slow, "rsi_len": st.session_state.cfg_rsi_len,
                          "rsi_buy_level": st.session_state.cfg_rsi_buy, "rsi_sell_level": st.session_state.cfg_rsi_sell, "atr_len": st.session_state.cfg_atr_len,
                          "atr_multiplier": st.session_state.cfg_atr_mult}
            if sim_params["start_date"] >= sim_params["end_date"]: st.error("Data Inicial >= Data Final.")
            else:
                with st.spinner(f"Executando backtest para {current_ticker_input}..."):
                    results = run_strategy_backtest(ticker=current_ticker_input, **sim_params)
                    st.session_state['backtest_results'] = results; st.session_state['last_ticker_simulated'] = current_ticker_input
        else: st.warning("Insira um código de ativo."); st.session_state['backtest_results'] = None

    # Exibição Resultados Simulação Individual
    if 'backtest_results' in st.session_state and st.session_state['backtest_results'] is not None and \
       st.session_state['backtest_results']['ticker'] == st.session_state.get('last_ticker_simulated'):
        results = st.session_state['backtest_results']
        st.header(f"Resultados para: {results['ticker']}")
        add_col, info_col = st.columns([0.3, 0.7])
        with add_col:
            ticker_to_add = results['ticker']
            if ticker_to_add not in st.session_state.watchlist:
                if st.button(f"⭐ Adicionar à Lista", key=f"add_watch_{ticker_to_add}"):
                    st.session_state.watchlist.append(ticker_to_add); st.toast(f"{ticker_to_add} adicionado!"); st.rerun()
            else: st.success(f"✔️ Na Watchlist")

        with st.container(border=True):
            st.subheader("📊 Métricas de Desempenho")
            metrics = results['metrics']
            m_col1, m_col2, m_col3 = st.columns(3); m_col4, m_col5, m_col6 = st.columns(3)
            m_col1.metric("CAGR (Retorno Anualizado)", f"{metrics['cagr']:.2f}%")
            m_col2.metric("Retorno Total Período", f"{metrics['totalReturnPercent']:.2f}%")
            m_col3.metric("Nº Trades", metrics['numberOfTrades'])
            m_col4.metric("Taxa de Acerto", f"{metrics['winRate']:.2f}%")
            m_col5.metric("Payoff Ratio", f"{metrics['payoffRatio']:.2f}")
            m_col6.metric("Max Drawdown", f"{metrics['maxDrawdown']:.2f}%")
            st.caption(f"Período: {metrics['startDate']} a {metrics['endDate']} | Capital: {format_currency(metrics['initialCapital'])} -> {format_currency(metrics['finalEquity'])} ({format_currency(metrics['totalProfit'])}) | Sharpe: {metrics['sharpeRatio']:.2f}")
            st.divider()

            st.subheader("📈 Gráfico e Operações")
            show_indicators_opt = st.toggle("Mostrar RSI no Gráfico", value=False, key="show_indicators_toggle_tab1")
            fig = plot_results(results['chartData'], results['signals'], results['ticker'], results['indicators'], show_indicators=show_indicators_opt)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📜 Ver Histórico de Operações Detalhado"):
                trades_df = results['trades']
                if not trades_df.empty:
                    trades_df_display = trades_df.copy()
                    trades_df_display['profit'] = trades_df_display['profit'].apply(format_currency)
                    trades_df_display['entryPrice'] = trades_df_display['entryPrice'].map('{:.2f}'.format)
                    trades_df_display['exitPrice'] = trades_df_display['exitPrice'].map('{:.2f}'.format)
                    trades_df_display['returnPercent'] = trades_df_display['returnPercent'].map('{:.2f}%'.format)
                    st.dataframe(trades_df_display[['entryDate', 'entryPrice', 'exitDate', 'exitPrice', 'daysHeld', 'profit', 'returnPercent', 'exitReason']], use_container_width=True)
                else: st.info("Nenhuma operação realizada.")

    elif simulate_button and st.session_state['backtest_results'] is None:
         st.warning("Não foi possível gerar resultados. Verifique ticker/período/erros.")


# --- Aba 2: Scanner de Ativos (Completa e Corrigida) ---
with tab2:
    st.header("Scanner de Ativos Lucrativos")
    st.markdown("Encontre ativos com **CAGR positivo** usando os parâmetros atuais.")
    st.warning("Scanner pode demorar. Listas maiores podem falhar no Streamlit Cloud.", icon="⚠️")

    scan_col1, scan_col2 = st.columns([0.6, 0.4])
    with scan_col1:
        scan_list_options = ["Use Minha Watchlist Atual"] + list(COMMON_TICKERS.keys())
        selected_scan_list_name = st.radio("Lista para escanear:", options=scan_list_options, key="scan_list_radio_tab2", horizontal=True)
    with scan_col2:
        min_cagr_scan = st.number_input("CAGR Mínimo Desejado (%)", min_value=0.0, value=st.session_state.min_cagr_input, step=5.0, format="%.1f", key="min_cagr_input_tab2")
        st.session_state.min_cagr_input = min_cagr_scan # Atualiza state

    list_is_empty = True; tickers_for_scan = []
    if selected_scan_list_name == "Use Minha Watchlist Atual":
        tickers_for_scan = st.session_state.watchlist; list_is_empty = not tickers_for_scan
        if list_is_empty: st.warning("Sua watchlist está vazia.")
    else: tickers_for_scan = COMMON_TICKERS[selected_scan_list_name]; list_is_empty = False
    st.caption(f"Serão escaneados {len(tickers_for_scan)} ativos da lista: '{selected_scan_list_name}'")

    scan_button = st.button("🔎 Escanear Ativos Agora", key="scan_button_tab2", type="primary", disabled=list_is_empty)
    st.divider()

    if scan_button and not list_is_empty:
        params = {"start_date": st.session_state.cfg_start, "end_date": st.session_state.cfg_end, "initial_capital": st.session_state.cfg_capital,
                  "ema_fast_len": st.session_state.cfg_ema_fast, "ema_slow_len": st.session_state.cfg_ema_slow, "rsi_len": st.session_state.cfg_rsi_len,
                  "rsi_buy_level": st.session_state.cfg_rsi_buy, "rsi_sell_level": st.session_state.cfg_rsi_sell, "atr_len": st.session_state.cfg_atr_len,
                  "atr_multiplier": st.session_state.cfg_atr_mult}
        min_cagr_threshold = st.session_state.min_cagr_input

        st.info(f"Iniciando escaneamento com CAGR > {min_cagr_threshold}%...")
        scan_progress = st.progress(0, text="Iniciando..."); profitable_list = []; skipped_count = 0; error_messages = []
        total_tickers = len(tickers_for_scan)

        for i, ticker in enumerate(tickers_for_scan):
            progress_percentage = (i + 1) / total_tickers
            scan_progress.progress(progress_percentage, text=f"Escaneando: {ticker} ({i+1}/{total_tickers})")
            try:
                results = run_strategy_backtest(ticker=ticker, **params)
                if results and results.get('metrics'):
                    cagr_value = results['metrics'].get('cagr', -999)
                    if cagr_value > min_cagr_threshold:
                        profitable_list.append({'Ticker': ticker, 'CAGR (%)': cagr_value, 'Retorno Total (%)': results['metrics']['totalReturnPercent'],
                                                'Nº Trades': results['metrics']['numberOfTrades'], 'Taxa Acerto (%)': results['metrics']['winRate'],
                                                'Max Drawdown (%)': results['metrics']['maxDrawdown']})
            except Exception as e:
                error_msg = f"Erro {ticker}: {str(e)[:100]}..."; error_messages.append(error_msg); skipped_count += 1
            # time.sleep(0.05)

        scan_progress.empty()
        st.subheader("Resultados do Escaneamento")
        if profitable_list:
            profitable_df = pd.DataFrame(profitable_list).sort_values(by='CAGR (%)', ascending=False).reset_index(drop=True)
            for col in ['CAGR (%)', 'Retorno Total (%)', 'Taxa Acerto (%)', 'Max Drawdown (%)']:
                 if col in profitable_df.columns: profitable_df[col] = profitable_df[col].map('{:.2f}%'.format)
            st.session_state['scan_results_df'] = profitable_df
            st.success(f"Scan concluído! {len(profitable_df)} ativos com CAGR > {min_cagr_threshold}%.")
            if skipped_count > 0: st.warning(f"{skipped_count} ativos pulados devido a erros.")
            st.dataframe(profitable_df, use_container_width=True, height=min( (len(profitable_df) + 1) * 35 + 3, 600) )
            profitable_tickers = profitable_df['Ticker'].tolist()
            missing_in_watchlist = [t for t in profitable_tickers if t not in st.session_state.watchlist]
            if missing_in_watchlist:
                if st.button(f"⭐ Adicionar {len(missing_in_watchlist)} resultados à Watchlist", key="add_scan_results"):
                     st.session_state.watchlist.extend(missing_in_watchlist); st.session_state.watchlist = sorted(list(set(st.session_state.watchlist)))
                     st.toast(f"{len(missing_in_watchlist)} adicionados!"); st.rerun()
            if error_messages:
                with st.expander("Ver detalhes dos erros"): [st.caption(msg) for msg in error_messages]
        else:
            st.session_state['scan_results_df'] = pd.DataFrame()
            st.info(f"Nenhum ativo com CAGR > {min_cagr_threshold}% na lista '{selected_scan_list_name}'.")
            if skipped_count > 0: st.warning(f"{skipped_count} ativos pulados.")
            if error_messages:
                 with st.expander("Ver detalhes dos erros"): [st.caption(msg) for msg in error_messages]

    elif not scan_button and not st.session_state.scan_results_df.empty:
         st.subheader("Resultado do Último Escaneamento Realizado:")
         st.dataframe(st.session_state.scan_results_df, use_container_width=True)
         st.caption("Clique em 'Escanear Ativos Agora' para atualizar.")
