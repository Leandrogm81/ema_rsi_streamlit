# app.py
import streamlit as st
import pandas as pd
import pandas_ta as ta # Usaremos para EMA, RSI, ATR
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import quantstats as qs
import time

# --- Configuração da Página Streamlit ---
st.set_page_config(page_title="Estratégia Fast-Trend Confirmed", page_icon="⚡", layout="wide") # Novo Ícone

# --- Inicialização do Session State (Mantém a estrutura) ---
default_watchlist = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]
if 'watchlist' not in st.session_state: st.session_state.watchlist = default_watchlist
if 'backtest_results' not in st.session_state: st.session_state.backtest_results = None
if 'last_ticker_simulated' not in st.session_state: st.session_state.last_ticker_simulated = ""
if 'current_signals' not in st.session_state: st.session_state.current_signals = {}
# Define valores iniciais para os novos parâmetros no state
if 'ticker_input_value' not in st.session_state: st.session_state.ticker_input_value = "PETR4.SA"
if 'cfg_ema_fast' not in st.session_state: st.session_state.cfg_ema_fast = 10
if 'cfg_ema_slow' not in st.session_state: st.session_state.cfg_ema_slow = 50
if 'cfg_rsi_len' not in st.session_state: st.session_state.cfg_rsi_len = 14
if 'cfg_rsi_buy' not in st.session_state: st.session_state.cfg_rsi_buy = 55.0
if 'cfg_rsi_sell' not in st.session_state: st.session_state.cfg_rsi_sell = 45.0
if 'cfg_atr_len' not in st.session_state: st.session_state.cfg_atr_len = 14
if 'cfg_atr_mult' not in st.session_state: st.session_state.cfg_atr_mult = 2.0
# Mantém os outros (start, end, capital)
if 'scan_results_df' not in st.session_state: st.session_state.scan_results_df = pd.DataFrame()

# --- Lista de Tickers Comuns (Mantém a estrutura) ---
COMMON_TICKERS = {
    "IBOV (Exemplos)": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "MGLU3.SA", "WEGE3.SA", "B3SA3.SA", "RENT3.SA", "PRIO3.SA"],
    "US Tech (Exemplos)": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CSCO"],
    "Cripto (Exemplos)": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"]
}
FLAT_TICKER_LIST = [""] + sorted(list(set(ticker for sublist in COMMON_TICKERS.values() for ticker in sublist)))

# --- Função de Métricas (Permanece a mesma) ---
def calculate_metrics(trades_df, equity_curve_df, initial_capital):
    # >>> COLE O CÓDIGO DA FUNÇÃO calculate_metrics DA VERSÃO ANTERIOR AQUI <<<
    pass

# --- Função do Backtest (ATUALIZADA PARA NOVA ESTRATÉGIA) ---
@st.cache_data(ttl=3600)
def run_strategy_backtest(ticker: str, start_date: date, end_date: date,
                          initial_capital: float = 1000.0,
                          # Novos Parâmetros da Estratégia
                          ema_fast_len: int = 10, ema_slow_len: int = 50,
                          rsi_len: int = 14, rsi_buy_level: float = 55.0, rsi_sell_level: float = 45.0,
                          atr_len: int = 14, atr_multiplier: float = 2.0):
    """
    Executa o backtest da estratégia "Fast-Trend Confirmed" (EMA Cross + RSI Filter + ATR Stop).
    """
    st.caption(f"Iniciando backtest para {ticker}...")

    # --- 1. Buscar Dados Históricos (Adaptação para período maior de EMA) ---
    fetch_start_date = start_date - timedelta(days=max(ema_slow_len, rsi_len, atr_len) + 50) # Garante dados suficientes para a EMA mais longa
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

    # --- 2. Calcular Novos Indicadores ---
    try:
        # EMAs
        ema_fast_col = f'EMA_{ema_fast_len}'
        ema_slow_col = f'EMA_{ema_slow_len}'
        data.ta.ema(close='Close', length=ema_fast_len, append=True, col_names=(ema_fast_col,))
        data.ta.ema(close='Close', length=ema_slow_len, append=True, col_names=(ema_slow_col,))

        # RSI
        rsi_col = f'RSI_{rsi_len}'
        data.ta.rsi(close='Close', length=rsi_len, append=True, col_names=(rsi_col,))

        # ATR (Usado para Stops)
        atr_col = f'ATR_{atr_len}'
        data.ta.atr(length=atr_len, append=True, col_names=(atr_col,)) # Usa High, Low, Close por padrão

        initial_len = len(data)
        data.dropna(inplace=True) # Remove NaNs iniciais
        if data.empty: raise ValueError(f"Dados insuficientes após indicadores. {initial_len} linhas iniciais.")
    except Exception as e:
        st.error(f"Erro ao calcular indicadores para {ticker}: {e}", icon="📊")
        return None

    # --- 3. Simulação / Lógica de Backtesting (Nova Estratégia) ---
    cash = initial_capital
    shares = 0.0
    in_position = False
    entry_price = 0.0
    entry_date = None
    entry_atr = 0.0 # Guarda o ATR no momento da entrada para o stop
    stop_loss_price = 0.0 # Nível do stop loss inicial
    trailing_stop_price = 0.0 # Nível do trailing stop
    highest_price_since_entry = 0.0 # Máximo preço desde a entrada para o trailing

    trades = []
    signals = []
    equity_curve = [{"date": data.index[0].date().strftime('%Y-%m-%d'), "equity": initial_capital}]

    # Loop dia a dia
    for i in range(1, len(data)):
        current_date = data.index[i]
        current_open = data['Open'].iloc[i]
        current_high = data['High'].iloc[i]
        current_low = data['Low'].iloc[i]
        current_close = data['Close'].iloc[i]
        prev_row = data.iloc[i-1] # Dados do dia anterior (para sinais e stops)
        prev_prev_row = data.iloc[i-2] if i > 1 else None # Dia anterior ao anterior (para cruzamentos)

        # Valores dos indicadores no dia anterior
        prev_ema_fast = prev_row[ema_fast_col]
        prev_ema_slow = prev_row[ema_slow_col]
        prev_rsi = prev_row[rsi_col]
        prev_atr = prev_row[atr_col] # ATR do dia anterior

        # --- Lógica de Saída (Verificada Primeiro) ---
        exit_signal = False; exit_reason = None
        if in_position:
            # Atualiza trailing stop se necessário
            highest_price_since_entry = max(highest_price_since_entry, current_high) # Usa a máxima do dia atual
            new_trailing_stop = highest_price_since_entry - (atr_multiplier * entry_atr) # Usa ATR da entrada
            trailing_stop_price = max(trailing_stop_price, new_trailing_stop) # Trailing só sobe

            # 1. Verifica Stops (Baseado na Mínima do Dia Atual vs Stops definidos)
            if current_low <= stop_loss_price:
                exit_signal, exit_reason = True, f"Stop Loss ({atr_multiplier}xATR)"
                exit_price = stop_loss_price # Sai no preço do stop
            elif current_low <= trailing_stop_price:
                exit_signal, exit_reason = True, f"Trailing Stop ({atr_multiplier}xATR)"
                exit_price = trailing_stop_price # Sai no preço do trailing stop

            # 2. Verifica Condições de Saída por Indicadores (Baseado nos dados do dia anterior)
            elif prev_prev_row is not None:
                # Cruzamento EMA descendente no dia anterior?
                ema_crossed_down = prev_ema_fast < prev_ema_slow and prev_prev_row[ema_fast_col] >= prev_prev_row[ema_slow_col]
                if ema_crossed_down:
                    exit_signal, exit_reason = True, "EMA Cross Down"
                    exit_price = current_open # Sai na abertura
                # RSI abaixo do nível no dia anterior?
                elif prev_rsi < rsi_sell_level:
                    exit_signal, exit_reason = True, f"RSI < {rsi_sell_level}"
                    exit_price = current_open # Sai na abertura

            # Executa a venda
            if exit_signal:
                if shares > 0: # Segurança
                    cash += shares * exit_price
                    profit = (exit_price - entry_price) * shares
                    return_pct = ((exit_price / entry_price) - 1) * 100 if entry_price != 0 else 0
                    days_held = (current_date.date() - entry_date).days + 1 # Calcula dias
                    trades.append({
                        "entryDate": entry_date.strftime('%Y-%m-%d'), "entryPrice": round(entry_price, 2),
                        "exitDate": current_date.date().strftime('%Y-%m-%d'), "exitPrice": round(exit_price, 2),
                        "shares": round(shares, 4), "profit": round(profit, 2),
                        "returnPercent": round(return_pct, 2), "exitReason": exit_reason, "daysHeld": days_held
                    })
                    signals.append({"date": current_date.date().strftime('%Y-%m-%d'), "type": "sell", "price": round(exit_price, 2)})
                shares, in_position, entry_price, entry_date, entry_atr, stop_loss_price, trailing_stop_price, highest_price_since_entry = \
                    0.0, False, 0.0, None, 0.0, 0.0, 0.0, 0.0 # Reseta tudo


        # --- Lógica de Entrada (Verificada se não saiu e não está em posição) ---
        if not in_position and prev_prev_row is not None:
            # Condições baseadas nos dados do dia anterior
            ema_crossed_up = prev_ema_fast > prev_ema_slow and prev_prev_row[ema_fast_col] <= prev_prev_row[ema_slow_col]
            rsi_confirm = prev_rsi > rsi_buy_level

            if ema_crossed_up and rsi_confirm:
                entry_price = current_open # Entra na abertura do dia atual
                if entry_price > 0 and cash > 0 and prev_atr > 0: # Precisa de ATR > 0 para stop
                    entry_atr = prev_atr # Guarda ATR do dia anterior para cálculo do stop
                    stop_loss_price = entry_price - (atr_multiplier * entry_atr)
                    trailing_stop_price = stop_loss_price # Trailing começa no stop inicial
                    highest_price_since_entry = entry_price # Preço inicial para trailing

                    # Calcula tamanho da posição (100% do capital por enquanto)
                    shares_to_buy = cash / entry_price
                    cash -= shares_to_buy * entry_price
                    shares, in_position, entry_date = shares_to_buy, True, current_date.date()
                    signals.append({"date": current_date.date().strftime('%Y-%m-%d'), "type": "buy", "price": round(entry_price, 2)})


        # --- Atualiza Curva de Equity ---
        current_equity = cash + (shares * current_close) # Usa fechamento do dia atual
        equity_curve.append({"date": current_date.date().strftime('%Y-%m-%d'), "equity": round(current_equity, 2)})

    # --- 4. Calcular Resultados Finais ---
    final_equity = equity_curve[-1]['equity'] if equity_curve else initial_capital
    total_profit = final_equity - initial_capital
    total_return_percent = (total_profit / initial_capital) * 100 if initial_capital > 0 else 0
    number_of_trades = len(trades)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_curve_df = pd.DataFrame(equity_curve)

    adv_metrics = calculate_metrics(trades_df, equity_curve_df, initial_capital)

    # --- 5. Preparar Dicionário de Retorno ---
    chart_data_df = data[['Open', 'High', 'Low', 'Close', ema_fast_col, ema_slow_col, rsi_col]].reset_index() # Seleciona indicadores relevantes
    chart_data_df['Date'] = pd.to_datetime(chart_data_df['Date']).dt.date

    metrics_start_date = data.index.min().date() if not data.empty else None
    metrics_end_date = data.index.max().date() if not data.empty else None

    results = {
        "ticker": ticker,
        "params": {"ema_f": ema_fast_len, "ema_s": ema_slow_len, "rsi": rsi_len, "rsi_buy": rsi_buy_level, "rsi_sell": rsi_sell_level, "atr_len": atr_len, "atr_mult": atr_multiplier, "capital": initial_capital},
        "metrics": {
            "initialCapital": round(initial_capital, 2), "finalEquity": round(final_equity, 2),
            "totalProfit": round(total_profit, 2), "totalReturnPercent": round(total_return_percent, 2),
            "numberOfTrades": number_of_trades,
            "startDate": metrics_start_date.strftime('%Y-%m-%d') if metrics_start_date else 'N/A',
            "endDate": metrics_end_date.strftime('%Y-%m-%d') if metrics_end_date else 'N/A',
            **adv_metrics
        },
        "trades": trades_df,
        "signals": signals,
        "chartData": chart_data_df, # Agora inclui EMAs e RSI
        "indicators": {"ema_f": ema_fast_col, "ema_s": ema_slow_col, "rsi": rsi_col} # Nomes das colunas relevantes
    }
    st.caption(f"Backtest '{ticker}' concluído.")
    return results


# --- Função para Plotar o Gráfico (ATUALIZADA para novos indicadores) ---
def plot_results(chart_data_df, signals, ticker, indicators_cols, show_indicators=False):
    """Plota o gráfico de velas, sinais e opcionalmente os indicadores EMA e RSI."""
    if chart_data_df.empty:
        st.warning("Não há dados para plotar o gráfico.")
        return go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # --- Gráfico de Velas (Linha 1) ---
    fig.add_trace(go.Candlestick(x=chart_data_df['Date'], open=chart_data_df['Open'], high=chart_data_df['High'],
                               low=chart_data_df['Low'], close=chart_data_df['Close'], name='OHLC',
                               increasing_line_color='green', decreasing_line_color='red'),
                  row=1, col=1)

    # Plotar EMAs junto com o preço
    ema_f_col = indicators_cols.get('ema_f')
    ema_s_col = indicators_cols.get('ema_s')
    if ema_f_col and ema_s_col:
         fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[ema_f_col], name=ema_f_col.replace('_',' '), line=dict(color='blue', width=1)), row=1, col=1)
         fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[ema_s_col], name=ema_s_col.replace('_',' '), line=dict(color='orange', width=1)), row=1, col=1)

    # --- Sinais de Compra/Venda (Linha 1) ---
    buy_signals_df = pd.DataFrame([s for s in signals if s['type'] == 'buy'])
    if not buy_signals_df.empty:
        fig.add_trace(go.Scatter(x=buy_signals_df['date'], y=buy_signals_df['price'], mode='markers', name='Compra',
                                 marker=dict(color='green', size=10, symbol='triangle-up', line=dict(width=1, color='DarkSlateGrey'))),
                      row=1, col=1)
    sell_signals_df = pd.DataFrame([s for s in signals if s['type'] == 'sell'])
    if not sell_signals_df.empty:
        fig.add_trace(go.Scatter(x=sell_signals_df['date'], y=sell_signals_df['price'], mode='markers', name='Venda',
                                  marker=dict(color='red', size=10, symbol='triangle-down', line=dict(width=1, color='DarkSlateGrey'))),
                      row=1, col=1)

    # --- Plotar Indicadores (RSI na Linha 2) se selecionado ---
    if show_indicators:
        rsi_col = indicators_cols.get('rsi')
        rsi_buy_level = st.session_state.cfg_rsi_buy # Pega do state
        rsi_sell_level = st.session_state.cfg_rsi_sell # Pega do state

        if rsi_col:
            fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[rsi_col], name=rsi_col.replace('_',' '), line=dict(color='purple', width=1.5)), row=2, col=1)
            # Linhas de referência para níveis de compra/venda do RSI
            fig.add_hline(y=rsi_buy_level, line_dash="dot", line_color="rgba(0,150,0,0.6)", annotation_text=f"RSI Compra > {rsi_buy_level}", row=2, col=1)
            fig.add_hline(y=rsi_sell_level, line_dash="dot", line_color="rgba(150,0,0,0.6)", annotation_text=f"RSI Venda < {rsi_sell_level}", row=2, col=1)
            # Linha de 50 opcional
            # fig.add_hline(y=50, line_dash="dash", line_color="rgba(100,100,100,0.4)", row=2, col=1)

        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, showgrid=True) # Eixo fixo para RSI
    else:
         fig.update_layout(yaxis2_visible=False)


    # --- Layout Geral ---
    fig.update_layout(
        title={'text': f'Backtest Fast-Trend: {ticker}', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        height=650, # Aumenta um pouco a altura
        xaxis_rangeslider_visible=False,
        xaxis_showticklabels=True, xaxis2_showticklabels=True,
        yaxis_title="Preço / EMA", row=1, col=1, # Atualiza título eixo Y1
        yaxis_fixedrange=False, yaxis2_fixedrange=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True, row=1, col=1)
    if show_indicators: fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True, row=2, col=1)

    return fig


# --- Função para Verificar Sinal Atual (ATUALIZADA para nova estratégia) ---
@st.cache_data(ttl=900)
def get_current_signal(ticker: str,
                       ema_fast_len: int = 10, ema_slow_len: int = 50,
                       rsi_len: int = 14, rsi_buy_level: float = 55.0, rsi_sell_level: float = 45.0):
    """Verifica o sinal da estratégia Fast-Trend (COMPRA/VENDA/NEUTRO) no último dia."""
    st.caption(f"Verificando sinal {ticker}...")
    try:
        # Pega dados suficientes para EMA lenta + alguns dias
        data = yf.download(ticker, period=f"{ema_slow_len + 20}d", interval="1d", auto_adjust=True, progress=False)
        if data.empty or len(data) < ema_slow_len + 2: return "Dados Insuf."

        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Calcula indicadores necessários
        data.ta.ema(close='Close', length=ema_fast_len, append=True, col_names=('EMA_F',))
        data.ta.ema(close='Close', length=ema_slow_len, append=True, col_names=('EMA_S',))
        data.ta.rsi(close='Close', length=rsi_len, append=True, col_names=('RSI',))
        data.dropna(inplace=True)
        if len(data) < 3: return "Dados Insuf. pós Ind." # Precisa de 3 pontos para verificar cruzamento

        # Pega as 3 últimas linhas válidas
        last = data.iloc[-1]; prev = data.iloc[-2]; prev_prev = data.iloc[-3]

        # Verifica Condição de Compra (baseado no fechamento anterior)
        ema_crossed_up = prev['EMA_F'] > prev['EMA_S'] and prev_prev['EMA_F'] <= prev_prev['EMA_S']
        rsi_confirm = prev['RSI'] > rsi_buy_level
        if ema_crossed_up and rsi_confirm:
            return "COMPRA"

        # Verifica Condição de Saída Técnica (baseado no fechamento anterior)
        ema_crossed_down = prev['EMA_F'] < prev['EMA_S'] and prev_prev['EMA_F'] >= prev_prev['EMA_S']
        rsi_below_level = prev['RSI'] < rsi_sell_level
        # Retorna 'VENDA' se qualquer condição de saída for atendida (para alertar sobre possível fechamento)
        if ema_crossed_down or rsi_below_level:
            return "VENDA"

        return "NEUTRO"

    except Exception as e:
        st.caption(f"Erro {ticker}: {str(e)[:30]}...")
        return "Erro"


# --- Formatação de Moeda (sem mudanças) ---
def format_currency(value):
    try: return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except: return "R$ 0,00"

# ==============================================================================
# --- Interface Principal do Streamlit ---
# ==============================================================================

st.title("⚡ Estratégia Fast-Trend Confirmed") # Novo Título
st.caption("Simulador, Scanner e Monitor (EMA 10x50 + RSI 14 + ATR Stop)")
st.divider()

# --- Barra Lateral (Sidebar - ATUALIZADA com novos parâmetros) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/crossing.png", width=80) # Novo ícone
    st.header("⚙️ Configurações Globais")
    st.caption("Afetam Simulador e Scanner.")

    with st.expander("Parâmetros EMA & RSI", expanded=True):
        st.session_state.cfg_ema_fast = st.number_input("EMA Rápida (dias)", min_value=3, max_value=100, value=10, step=1, key="cfg_ema_fast_in")
        st.session_state.cfg_ema_slow = st.number_input("EMA Lenta (dias)", min_value=10, max_value=200, value=50, step=1, key="cfg_ema_slow_in")
        st.session_state.cfg_rsi_len = st.number_input("RSI Período (dias)", min_value=5, max_value=50, value=14, step=1, key="cfg_rsi_len_in")
        st.session_state.cfg_rsi_buy = st.number_input("RSI Nível Compra >", min_value=50.0, max_value=70.0, value=55.0, step=0.5, format="%.1f", key="cfg_rsi_buy_in")
        st.session_state.cfg_rsi_sell = st.number_input("RSI Nível Venda <", min_value=30.0, max_value=50.0, value=45.0, step=0.5, format="%.1f", key="cfg_rsi_sell_in")

    with st.expander("Parâmetros Stop Loss (ATR)", expanded=True):
        st.session_state.cfg_atr_len = st.number_input("ATR Período (dias)", min_value=5, max_value=50, value=14, step=1, key="cfg_atr_len_in")
        st.session_state.cfg_atr_mult = st.number_input("ATR Multiplicador (Stop/Trail)", min_value=1.0, max_value=5.0, value=2.0, step=0.1, format="%.1f", key="cfg_atr_mult_in")

    with st.expander("Período do Backtest", expanded=True):
        today = datetime.now().date()
        default_start = today - timedelta(days=5*365) # Padrão 5 anos
        cfg_start = st.date_input("Data Inicial", value=st.session_state.get('cfg_start', default_start), max_value=today - timedelta(days=30), key="cfg_start_in")
        cfg_end = st.date_input("Data Final", value=st.session_state.get('cfg_end', today), min_value=cfg_start + timedelta(days=30), max_value=today, key="cfg_end_in")
        st.session_state.cfg_start = cfg_start # Atualiza state
        st.session_state.cfg_end = cfg_end     # Atualiza state
        st.caption(f"Duração: {(cfg_end - cfg_start).days} dias")

    with st.expander("Financeiro", expanded=True):
        cfg_capital = st.number_input("Capital Inicial (R$)", min_value=1.0, value=st.session_state.get('cfg_capital', 1000.0), step=100.0, format="%.2f", key="cfg_capital_in")
        st.session_state.cfg_capital = cfg_capital # Atualiza state

    st.divider()
    st.header("⭐ Lista de Monitoramento")
    # --- Watchlist (Código da versão anterior - adaptado para novos parâmetros de sinal) ---
    if st.button("📡 Verificar Sinais Atuais", key="check_signals", use_container_width=True, help="Verifica o sinal da estratégia para hoje."):
        current_signals_temp = {}
        if st.session_state.watchlist:
            progress_bar = st.progress(0, text="Verificando sinais...")
            num_tickers = len(st.session_state.watchlist)
            for i, ticker_to_check in enumerate(st.session_state.watchlist):
                # Usa parâmetros atuais da sidebar
                signal = get_current_signal(
                    ticker_to_check,
                    st.session_state.cfg_ema_fast, st.session_state.cfg_ema_slow,
                    st.session_state.cfg_rsi_len, st.session_state.cfg_rsi_buy, st.session_state.cfg_rsi_sell
                )
                current_signals_temp[ticker_to_check] = signal
                progress_bar.progress((i + 1) / num_tickers, text=f"Verificando {ticker_to_check}...")
                time.sleep(0.05)
            st.session_state.current_signals = current_signals_temp
            progress_bar.empty(); st.toast("Verificação concluída!", icon="✅")
        else: st.toast("Watchlist vazia.", icon="ℹ️")

    if not st.session_state.watchlist: st.info("Nenhum ativo na lista.")
    else:
        st.caption("Ativos:")
        # >>> COLE O CÓDIGO DE EXIBIÇÃO DA WATCHLIST (LOOP FOR) DA VERSÃO ANTERIOR AQUI <<<
        # (A lógica interna de exibição é a mesma, só precisa garantir que chame get_current_signal com os params certos)
        pass

    if st.session_state.watchlist:
        st.markdown("<a href='?clear_watchlist=true' ... >Limpar Lista Completa</a>", unsafe_allow_html=True) # Manter link


# --- Lógica para limpar watchlist (sem mudanças) ---
if st.query_params.get("clear_watchlist") == "true":
    # >>> COLE O CÓDIGO DE LIMPEZA DA WATCHLIST DA VERSÃO ANTERIOR AQUI <<<
    pass


# ==============================================================================
# --- Abas para Organizar a Interface ---
# ==============================================================================
tab1, tab2 = st.tabs(["📈 **Simulador Individual**", "🔍 **Scanner de Ativos**"])

# --- Aba 1: Simulador Individual (Adaptada para novos parâmetros) ---
with tab1:
    st.header("Simulador de Backtest Individual")
    st.markdown("Teste a estratégia Fast-Trend em um ativo com os parâmetros da barra lateral.")
    # --- Input Ticker (Código da versão anterior) ---
    # >>> COLE O CÓDIGO DE INPUT DO TICKER (COLUNAS + SELECTBOX + TEXT_INPUT) DA ABA 1 DA VERSÃO ANTERIOR AQUI <<<
    pass

    simulate_button = st.button("Executar Simulação Individual", type="primary", use_container_width=True, key="sim_button_tab1")
    st.divider()

    # --- Execução da Simulação Individual (Adaptada) ---
    if simulate_button:
        current_ticker_input = st.session_state.ticker_input_value
        if current_ticker_input:
            # Pega todos os parâmetros atuais da sidebar/state
            sim_params = {
                "start_date": st.session_state.cfg_start, "end_date": st.session_state.cfg_end,
                "initial_capital": st.session_state.cfg_capital,
                "ema_fast_len": st.session_state.cfg_ema_fast, "ema_slow_len": st.session_state.cfg_ema_slow,
                "rsi_len": st.session_state.cfg_rsi_len, "rsi_buy_level": st.session_state.cfg_rsi_buy, "rsi_sell_level": st.session_state.cfg_rsi_sell,
                "atr_len": st.session_state.cfg_atr_len, "atr_multiplier": st.session_state.cfg_atr_mult
            }
            if sim_params["start_date"] >= sim_params["end_date"]: st.error("Data Inicial >= Data Final.")
            else:
                with st.spinner(f"Executando backtest para {current_ticker_input}..."):
                    results = run_strategy_backtest(ticker=current_ticker_input, **sim_params)
                    st.session_state['backtest_results'] = results
                    st.session_state['last_ticker_simulated'] = current_ticker_input
        else: st.warning("Insira um código de ativo."); st.session_state['backtest_results'] = None

    # --- Exibição dos Resultados da Simulação Individual (Adaptada) ---
    if 'backtest_results' in st.session_state and st.session_state['backtest_results'] is not None and \
       st.session_state['backtest_results']['ticker'] == st.session_state.get('last_ticker_simulated'):
        results = st.session_state['backtest_results']
        st.header(f"Resultados para: {results['ticker']}")
        # --- Botão Add Watchlist (Código da versão anterior) ---
        # >>> COLE O CÓDIGO DO BOTÃO "ADICIONAR À LISTA" DA VERSÃO ANTERIOR AQUI <<<
        pass

        with st.container(border=True):
             # --- Métricas (Código da versão anterior - já inclui CAGR etc.) ---
             # >>> COLE O CÓDIGO DE EXIBIÇÃO DAS MÉTRICAS DA VERSÃO ANTERIOR AQUI <<<
            pass
            st.divider()
            # --- Gráfico e Tabela (Código da versão anterior) ---
            st.subheader("📈 Gráfico e Operações")
            show_indicators_opt = st.toggle("Mostrar RSI no Gráfico", value=False, key="show_indicators_toggle_tab1") # Label ajustado
            fig = plot_results(results['chartData'], results['signals'], results['ticker'], results['indicators'], show_indicators=show_indicators_opt)
            st.plotly_chart(fig, use_container_width=True)
            # >>> COLE O CÓDIGO DO EXPANDER DA TABELA DE TRADES DA VERSÃO ANTERIOR AQUI <<<
            pass

    elif simulate_button and st.session_state['backtest_results'] is None:
         st.warning("Não foi possível gerar resultados. Verifique ticker/período/erros.")


# --- Aba 2: Scanner de Ativos (Adaptada para novos parâmetros) ---
with tab2:
    st.header("Scanner de Ativos Lucrativos")
    st.markdown("Encontre ativos com **CAGR positivo** usando os parâmetros atuais.")
    st.warning("Scanner pode demorar. Listas maiores podem falhar no Streamlit Cloud.", icon="⚠️")
    # --- Seleção Lista e CAGR Mínimo (Código da versão anterior) ---
    # >>> COLE O CÓDIGO DE SELEÇÃO DE LISTA E CAGR MÍNIMO DA ABA 2 DA VERSÃO ANTERIOR AQUI <<<
    pass

    scan_button = st.button("🔎 Escanear Ativos Agora", key="scan_button_tab2", type="primary", disabled=list_is_empty)
    st.divider()

    if scan_button and not list_is_empty:
        # Pega todos os parâmetros atuais para passar ao backtest no loop
        params = {
            "start_date": st.session_state.cfg_start, "end_date": st.session_state.cfg_end,
            "initial_capital": st.session_state.cfg_capital,
            "ema_fast_len": st.session_state.cfg_ema_fast, "ema_slow_len": st.session_state.cfg_ema_slow,
            "rsi_len": st.session_state.cfg_rsi_len, "rsi_buy_level": st.session_state.cfg_rsi_buy, "rsi_sell_level": st.session_state.cfg_rsi_sell,
            "atr_len": st.session_state.cfg_atr_len, "atr_multiplier": st.session_state.cfg_atr_mult
        }
        min_cagr_scan = st.session_state.min_cagr_input # Pega valor do input

        st.info(f"Iniciando escaneamento com CAGR > {min_cagr_scan}%...")
        # --- Loop do Scanner (Código da versão anterior - já usa CAGR) ---
        # >>> COLE O CÓDIGO DO LOOP DO SCANNER DA ABA 2 DA VERSÃO ANTERIOR AQUI <<<
        # (Ele já pega o CAGR dos results e filtra por 'min_cagr_threshold')
        pass

    # --- Exibição Resultados Último Scan (Código da versão anterior) ---
    elif not scan_button and not st.session_state.scan_results_df.empty:
         # >>> COLE O CÓDIGO DE EXIBIÇÃO DOS RESULTADOS DO SCAN ANTERIOR AQUI <<<
        pass
