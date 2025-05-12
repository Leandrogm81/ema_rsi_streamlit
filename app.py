# app.py
import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Para subplots
from datetime import datetime, timedelta, date
import quantstats as qs # Para métricas avançadas

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Estratégia EMA RSI Diário",
    page_icon="📈",
    layout="wide"
)

# --- Inicialização do Session State ---
default_watchlist = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = default_watchlist # Começa com alguns exemplos
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'last_ticker_simulated' not in st.session_state:
    st.session_state.last_ticker_simulated = ""
if 'current_signals' not in st.session_state:
    st.session_state.current_signals = {} # Guarda {ticker: sinal}
if 'ticker_input_value' not in st.session_state:
        st.session_state.ticker_input_value = "PETR4.SA"

# --- Lista de Tickers Comuns (Exemplo) ---
# Poderia vir de um arquivo ou API no futuro
COMMON_TICKERS = {
    "Brasil - IBOV (Exemplos)": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "MGLU3.SA", "WEGE3.SA"],
    "EUA - Tech (Exemplos)": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "Cripto (Exemplos)": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"]
}
# Flatten a lista para o selectbox
FLAT_TICKER_LIST = [""] + [ticker for sublist in COMMON_TICKERS.values() for ticker in sublist]


# --- Função de Métricas (Usando Quantstats e Manual) ---
def calculate_metrics(trades_df, equity_curve_df, initial_capital):
    if trades_df.empty or equity_curve_df.empty:
        return {
            "winRate": 0, "payoffRatio": 0, "avgGain": 0, "avgLoss": 0,
            "maxDrawdown": 0, "sharpeRatio": 0
        }

    # Cálculos Manuais
    wins = trades_df[trades_df['profit'] > 0]
    losses = trades_df[trades_df['profit'] <= 0]
    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
    avg_gain = wins['profit'].mean() if not wins.empty else 0
    avg_loss = abs(losses['profit'].mean()) if not losses.empty else 0 # Média da perda (positiva)
    payoff_ratio = avg_gain / avg_loss if avg_loss > 0 else 0

    # Cálculos com QuantStats (requer retornos diários)
    try:
        equity_curve_df['Date'] = pd.to_datetime(equity_curve_df['date'])
        equity_curve_df = equity_curve_df.set_index('Date')
        daily_returns = equity_curve_df['equity'].pct_change().dropna()

        if daily_returns.empty or daily_returns.std() == 0 or len(daily_returns) < 3:
             max_drawdown = 0
             sharpe = 0
             st.warning("Não há dados suficientes para calcular Max Drawdown ou Sharpe Ratio via QuantStats.")
        else:
            max_drawdown = qs.stats.max_drawdown(daily_returns) * 100 # Em percentual
            sharpe = qs.stats.sharpe(daily_returns) # Usa Risk-Free Rate = 0% por padrão

    except Exception as e:
        st.error(f"Erro ao calcular métricas com QuantStats: {e}")
        max_drawdown = 0
        sharpe = 0

    return {
        "winRate": round(win_rate * 100, 2),
        "payoffRatio": round(payoff_ratio, 2),
        "avgGain": round(avg_gain, 2),
        "avgLoss": round(avg_loss, 2),
        "maxDrawdown": round(max_drawdown, 2), # Já em %
        "sharpeRatio": round(sharpe, 2) if pd.notna(sharpe) else 0 # Tratar NaN no Sharpe
    }

# --- Função do Backtest (Atualizada para retornar indicadores e usar métricas) ---
@st.cache_data(ttl=3600)
def run_strategy_backtest(ticker: str, start_date: date, end_date: date,
                          initial_capital: float = 1000.0, rsi_len: int = 2,
                          ema_len: int = 2, exit_days: int = 4):
    st.write(f"Buscando dados para {ticker} de {start_date} a {end_date}...")
    fetch_start_date = start_date - timedelta(days=50)
    start_str = fetch_start_date.strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        data = yf.download(ticker, start=start_str, end=end_str, interval="1d", auto_adjust=True, progress=False)
        if data.empty: st.error(f"Não foi possível obter dados para {ticker}."); return None
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[(data.index >= pd.Timestamp(start_date)) & (data.index < pd.Timestamp(end_date + timedelta(days=1)))]
    except Exception as e: st.error(f"Erro ao buscar dados: {e}"); return None

    if data.empty: st.error(f"Não há dados para {ticker} no período de {start_date} a {end_date}."); return None

    st.write(f"Calculando indicadores (RSI({rsi_len}), EMA({ema_len}))...")
    try:
        data.ta.rsi(close='Close', length=rsi_len, append=True, col_names=(f'RSI_{rsi_len}_C',))
        data.ta.rsi(close='Open', length=rsi_len, append=True, col_names=(f'RSI_{rsi_len}_O',))
        rsi_close_col = f'RSI_{rsi_len}_C'
        rsi_open_col = f'RSI_{rsi_len}_O'
        data.ta.ema(close=rsi_close_col, length=ema_len, append=True, col_names=(f'EMA_{ema_len}_RSI_C',))
        data.ta.ema(close=rsi_open_col, length=ema_len, append=True, col_names=(f'EMA_{ema_len}_RSI_O',))
        ema_rsi_c_col = f'EMA_{ema_len}_RSI_C'
        ema_rsi_o_col = f'EMA_{ema_len}_RSI_O'
        data.dropna(inplace=True)
        if data.empty: st.error(f"Dados insuficientes após cálculo dos indicadores."); return None
    except Exception as e: st.error(f"Erro ao calcular indicadores: {e}"); return None

    st.write(f"Executando simulação...")
    # --- Simulação ---
    cash = initial_capital
    shares = 0.0
    in_position = False
    entry_price = 0.0
    entry_date = None
    days_in_trade = 0
    trades = []
    signals = []
    equity_curve = [{"date": data.index[0].date().strftime('%Y-%m-%d'), "equity": initial_capital}] # Ponto inicial

    for i in range(1, len(data)):
        current_date = data.index[i]
        current_open_price = data['Open'].iloc[i]
        prev_row = data.iloc[i-1]
        prev_ema_rsi_c = prev_row[ema_rsi_c_col]
        prev_ema_rsi_o = prev_row[ema_rsi_o_col]
        prev_prev_ema_rsi_c = data[ema_rsi_c_col].iloc[i-2] if i > 1 else np.nan
        prev_prev_ema_rsi_o = data[ema_rsi_o_col].iloc[i-2] if i > 1 else np.nan

        exit_signal = False
        exit_reason = None
        if in_position:
            days_in_trade += 1
            is_crossunder = prev_ema_rsi_c < prev_ema_rsi_o and prev_prev_ema_rsi_c >= prev_prev_ema_rsi_o
            is_ema_falling = prev_ema_rsi_c < prev_prev_ema_rsi_c
            if is_crossunder and is_ema_falling: exit_signal, exit_reason = True, "Saída Técnica"
            elif days_in_trade == exit_days: exit_signal, exit_reason = True, f"Saída por Tempo ({exit_days}d)"

            if exit_signal:
                exit_price = current_open_price
                if shares > 0:
                    cash += shares * exit_price
                    profit = (exit_price - entry_price) * shares
                    return_pct = ((exit_price / entry_price) - 1) * 100 if entry_price != 0 else 0
                    trades.append({
                        "entryDate": entry_date.strftime('%Y-%m-%d'), "entryPrice": round(entry_price, 2),
                        "exitDate": current_date.date().strftime('%Y-%m-%d'), "exitPrice": round(exit_price, 2),
                        "shares": round(shares, 4), "profit": round(profit, 2),
                        "returnPercent": round(return_pct, 2), "exitReason": exit_reason, "daysHeld": days_in_trade
                    })
                    signals.append({"date": current_date.date().strftime('%Y-%m-%d'), "type": "sell", "price": round(exit_price, 2)})
                shares, in_position, entry_price, entry_date, days_in_trade = 0.0, False, 0.0, None, 0

        if not in_position and i > 1:
            is_crossover = prev_ema_rsi_c > prev_ema_rsi_o and prev_prev_ema_rsi_c <= prev_prev_ema_rsi_o
            is_ema_rising = prev_ema_rsi_c > prev_prev_ema_rsi_c
            if is_crossover and is_ema_rising:
                entry_price = current_open_price
                if entry_price > 0 and cash > 0:
                    shares_to_buy = cash / entry_price
                    cash -= shares_to_buy * entry_price
                    shares, in_position, entry_date, days_in_trade = shares_to_buy, True, current_date.date(), 0
                    signals.append({"date": current_date.date().strftime('%Y-%m-%d'), "type": "buy", "price": round(entry_price, 2)})

        current_equity = cash + (shares * data['Close'].iloc[i])
        equity_curve.append({"date": current_date.date().strftime('%Y-%m-%d'), "equity": round(current_equity, 2)})

    # --- Resultados Finais ---
    final_equity = equity_curve[-1]['equity'] if equity_curve else initial_capital
    total_profit = final_equity - initial_capital
    total_return_percent = (total_profit / initial_capital) * 100 if initial_capital > 0 else 0
    number_of_trades = len(trades)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_curve_df = pd.DataFrame(equity_curve)

    # Calcular Métricas Adicionais
    adv_metrics = calculate_metrics(trades_df, equity_curve_df, initial_capital)

    # Preparar dados para gráfico (incluindo indicadores)
    chart_data_df = data[['Open', 'High', 'Low', 'Close', rsi_close_col, rsi_open_col, ema_rsi_c_col, ema_rsi_o_col]].reset_index()
    chart_data_df['Date'] = pd.to_datetime(chart_data_df['Date']).dt.date

    metrics_start_date = data.index.min().date() if not data.empty else None
    metrics_end_date = data.index.max().date() if not data.empty else None

    results = {
        "ticker": ticker,
        "params": {"rsi": rsi_len, "ema": ema_len, "exit": exit_days, "capital": initial_capital},
        "metrics": {
            "initialCapital": round(initial_capital, 2), "finalEquity": round(final_equity, 2),
            "totalProfit": round(total_profit, 2), "totalReturnPercent": round(total_return_percent, 2),
            "numberOfTrades": number_of_trades,
            "startDate": metrics_start_date.strftime('%Y-%m-%d') if metrics_start_date else 'N/A',
            "endDate": metrics_end_date.strftime('%Y-%m-%d') if metrics_end_date else 'N/A',
            **adv_metrics # Adiciona as métricas avançadas
        },
        "trades": trades_df,
        "signals": signals,
        "chartData": chart_data_df, # Agora inclui indicadores
        "indicators": {"rsi_c": rsi_close_col, "rsi_o": rsi_open_col, "ema_c": ema_rsi_c_col, "ema_o": ema_rsi_o_col} # Nomes das colunas
    }
    return results

# --- Função para Plotar o Gráfico com Subplots ---
def plot_results(chart_data_df, signals, ticker, indicators_cols, show_indicators=False):
    if chart_data_df.empty: st.warning("Não há dados para plotar."); return go.Figure()

    # Cria figura com 2 subplots (preços em cima, indicadores embaixo)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3]) # Ajuste row_heights

    # Gráfico de Velas
    fig.add_trace(go.Candlestick(x=chart_data_df['Date'], open=chart_data_df['Open'], high=chart_data_df['High'],
                               low=chart_data_df['Low'], close=chart_data_df['Close'], name='OHLC'), row=1, col=1)

    # Sinais de Compra/Venda (no gráfico de preços)
    buy_signals_df = pd.DataFrame([s for s in signals if s['type'] == 'buy'])
    if not buy_signals_df.empty:
        fig.add_trace(go.Scatter(x=buy_signals_df['date'], y=buy_signals_df['price'], mode='markers', name='Compra',
                                 marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
    sell_signals_df = pd.DataFrame([s for s in signals if s['type'] == 'sell'])
    if not sell_signals_df.empty:
        fig.add_trace(go.Scatter(x=sell_signals_df['date'], y=sell_signals_df['price'], mode='markers', name='Venda',
                                  marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

    # Plotar Indicadores (no subplot inferior) se selecionado
    if show_indicators:
        rsi_c_col = indicators_cols['rsi_c']
        rsi_o_col = indicators_cols['rsi_o']
        ema_c_col = indicators_cols['ema_c']
        ema_o_col = indicators_cols['ema_o']

        fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[ema_c_col], name='EMA(RSI C)', line=dict(color='blue', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[ema_o_col], name='EMA(RSI O)', line=dict(color='orange', width=1.5)), row=2, col=1)
        # Opcional: plotar RSIs originais (pode poluir)
        # fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[rsi_c_col], name='RSI Close', line=dict(color='lightblue', width=1, dash='dot')), row=2, col=1)
        # fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[rsi_o_col], name='RSI Open', line=dict(color='lightsalmon', width=1, dash='dot')), row=2, col=1)

        fig.update_yaxes(title_text="Indicadores", row=2, col=1) # Título para eixo Y do subplot 2
        # Adicionar linhas de referência (ex: sobrecompra/venda no RSI - ajustar eixos se plotar RSI)
        # fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        # fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # Layout Geral
    fig.update_layout(
        title=f'Backtest: {ticker}',
        height=600, # Aumentar altura para acomodar subplots
        xaxis_rangeslider_visible=False, # Esconder slider do eixo X principal
        xaxis_showticklabels=True, xaxis2_showticklabels=True, # Mostrar labels do eixo X em ambos
        yaxis_title="Preço", # Eixo Y do Gráfico 1
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

    return fig

# --- Função para Verificar Sinal Atual ---
@st.cache_data(ttl=900) # Cache de 15 minutos para sinais atuais
def get_current_signal(ticker: str, rsi_len: int = 2, ema_len: int = 2):
    st.write(f"Verificando sinal atual para {ticker}...") # Feedback sutil
    try:
        # Pega só os últimos ~10 dias (suficiente p/ RSI(2), EMA(2))
        data = yf.download(ticker, period="15d", interval="1d", auto_adjust=True, progress=False)
        if data.empty or len(data) < rsi_len + ema_len: return "Dados Insuficientes"

        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Calcula indicadores
        data.ta.rsi(close='Close', length=rsi_len, append=True, col_names=('RSI_C',))
        data.ta.rsi(close='Open', length=rsi_len, append=True, col_names=('RSI_O',))
        data.ta.ema(close='RSI_C', length=ema_len, append=True, col_names=('EMA_RSI_C',))
        data.ta.ema(close='RSI_O', length=ema_len, append=True, col_names=('EMA_RSI_O',))
        data.dropna(inplace=True)
        if len(data) < 2: return "Dados Insuficientes pós Ind."

        # Pega as duas últimas linhas válidas
        last = data.iloc[-1]
        prev = data.iloc[-2]

        # Verifica Condição de Compra
        is_crossover = last['EMA_RSI_C'] > last['EMA_RSI_O'] and prev['EMA_RSI_C'] <= prev['EMA_RSI_O']
        is_ema_rising = last['EMA_RSI_C'] > prev['EMA_RSI_C']
        if is_crossover and is_ema_rising:
            return "COMPRA"

        # Verifica Condição de Saída Técnica
        is_crossunder = last['EMA_RSI_C'] < last['EMA_RSI_O'] and prev['EMA_RSI_C'] >= prev['EMA_RSI_O']
        is_ema_falling = last['EMA_RSI_C'] < prev['EMA_RSI_C']
        if is_crossunder and is_ema_falling:
            return "VENDA" # Sinal de saída técnica

        return "NEUTRO"

    except Exception as e:
        st.error(f"Erro ao verificar sinal para {ticker}: {e}", icon="⚠️")
        return "Erro"


# --- Formatação de Moeda ---
def format_currency(value):
    try: return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except: return "R$ 0,00"

# ==============================================================================
# --- Interface Principal do Streamlit ---
# ==============================================================================

st.title("📈 Simulador de Estratégia EMA RSI Diário")
st.markdown("Teste a estratégia de cruzamento de EMA sobre RSI diário com parâmetros configuráveis e acompanhe sua lista.")
st.divider()

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Configurações")
    st.subheader("Parâmetros da Estratégia")
    cfg_rsi_len = st.number_input("Período RSI", min_value=2, max_value=50, value=2, step=1, key="cfg_rsi")
    cfg_ema_len = st.number_input("Período EMA", min_value=2, max_value=50, value=2, step=1, key="cfg_ema")
    cfg_exit_days = st.number_input("Dias Saída Tempo", min_value=1, max_value=30, value=4, step=1, key="cfg_exit")

    st.subheader("Período do Backtest")
    today = datetime.now().date()
    one_year_ago = today - timedelta(days=365)
    cfg_start_date = st.date_input("Data Inicial", value=one_year_ago, max_value=today - timedelta(days=1), key="cfg_start")
    cfg_end_date = st.date_input("Data Final", value=today, min_value=cfg_start_date + timedelta(days=1), max_value=today, key="cfg_end")

    st.subheader("Financeiro")
    cfg_initial_capital = st.number_input("Capital Inicial", min_value=1.0, value=1000.0, step=100.0, format="%.2f", key="cfg_capital")

    st.divider()
    st.header("⭐ Lista de Monitoramento")

    # Botão para Verificar Sinais Atuais
    if st.button("Verificar Sinais Atuais da Lista", key="check_signals", use_container_width=True):
        current_signals_temp = {}
        if not st.session_state.watchlist:
            st.toast("Lista de monitoramento vazia.", icon="ℹ️")
        else:
            progress_bar = st.progress(0, text="Verificando sinais...")
            num_tickers = len(st.session_state.watchlist)
            for i, ticker_to_check in enumerate(st.session_state.watchlist):
                signal = get_current_signal(ticker_to_check, cfg_rsi_len, cfg_ema_len)
                current_signals_temp[ticker_to_check] = signal
                progress_bar.progress((i + 1) / num_tickers, text=f"Verificando {ticker_to_check}...")
            st.session_state.current_signals = current_signals_temp # Atualiza o estado
            progress_bar.empty() # Limpa a barra de progresso
            st.toast("Verificação de sinais concluída!", icon="✅")


    # Exibe a lista e botões de remover + sinal atual (se verificado)
    if not st.session_state.watchlist:
        st.info("Nenhum ativo na lista.")
    else:
        st.caption("Ativos:")
        # Iterar para exibir com sinal atual
        for i in range(len(st.session_state.watchlist) - 1, -1, -1):
            ticker_in_list = st.session_state.watchlist[i]
            signal_status = st.session_state.current_signals.get(ticker_in_list, "") # Pega sinal se existir

            # Formatação do status
            signal_color = {"COMPRA": "green", "VENDA": "red", "NEUTRO": "gray", "Erro": "orange", "Dados Insuficientes": "orange", "Dados Insuficientes pós Ind.": "orange"}.get(signal_status, "gray")
            signal_icon = {"COMPRA": "🔼", "VENDA": "🔽", "NEUTRO": "⏸️"}.get(signal_status, "❔")
            signal_display = f"<span style='color:{signal_color}; font-weight:bold;'>{signal_icon} {signal_status}</span>" if signal_status else ""

            col1_watch, col2_watch, col3_watch = st.columns([0.5, 0.3, 0.2]) # Nome, Sinal, Remover
            with col1_watch:
                 # Clicar no nome atualiza o input principal
                if st.button(ticker_in_list, key=f"sim_{ticker_in_list}_{i}", type="secondary", help=f"Simular {ticker_in_list}"):
                     st.session_state.ticker_input_value = ticker_in_list
                     st.rerun()
            with col2_watch:
                st.markdown(signal_display, unsafe_allow_html=True) # Exibe o status do sinal
            with col3_watch:
                if st.button("➖", key=f"remove_{ticker_in_list}_{i}", type="secondary", help="Remover da lista", use_container_width=True):
                    removed_ticker = st.session_state.watchlist.pop(i)
                    # Remove sinal atual se existir
                    st.session_state.current_signals.pop(removed_ticker, None)
                    st.toast(f"{removed_ticker} removido.")
                    if st.session_state.get('last_ticker_simulated') == removed_ticker:
                        st.session_state['backtest_results'] = None; st.session_state['last_ticker_simulated'] = ""
                    st.rerun()

    st.markdown("[Limpar Lista](?clear_watchlist=true)", unsafe_allow_html=True)

# Lógica para limpar watchlist
if st.query_params.get("clear_watchlist") == "true":
    st.session_state.watchlist = []; st.session_state.current_signals = {}; st.session_state['backtest_results'] = None; st.session_state['last_ticker_simulated'] = ""
    st.toast("Lista de monitoramento limpa."); st.query_params.clear(); st.rerun()


# --- Área Principal ---
st.subheader("Entrada de Ativo")
col_sel, col_input = st.columns([0.4, 0.6]) # Coluna para Selectbox, Coluna para Input

# Selectbox para tickers comuns
with col_sel:
    selected_common_ticker = st.selectbox(
        "Ou selecione um ativo comum:",
        options=FLAT_TICKER_LIST,
        index=0, # Começa com a opção vazia
        key="common_ticker_select",
        label_visibility="collapsed" # Esconde o label padrão
    )
    # Se o usuário selecionou algo no selectbox, atualiza o input principal
    if selected_common_ticker and selected_common_ticker != st.session_state.ticker_input_value:
        st.session_state.ticker_input_value = selected_common_ticker
        st.rerun() # Força o rerun para atualizar o st.text_input

# Input principal (controlado pelo session_state)
with col_input:
    ticker_input = st.text_input(
        "Digite o Código do Ativo (ex: PETR4.SA):",
        value=st.session_state.ticker_input_value,
        placeholder="Digite ou selecione ao lado...",
        key="ticker_input_main"
    ).upper()
    # Atualiza o state se o usuário digitar algo manualmente
    if ticker_input != st.session_state.ticker_input_value:
         st.session_state.ticker_input_value = ticker_input

# Botão de Simulação
simulate_button = st.button("Simular Estratégia", type="primary", use_container_width=True)
st.divider()

# --- Execução e Exibição ---
if simulate_button:
    current_ticker_input = st.session_state.ticker_input_value
    if current_ticker_input:
        if cfg_start_date >= cfg_end_date: st.error("Erro: Data Inicial >= Data Final.")
        else:
            with st.spinner(f"Executando backtest para {current_ticker_input}..."):
                results = run_strategy_backtest(ticker=current_ticker_input, start_date=cfg_start_date, end_date=cfg_end_date,
                                                  initial_capital=cfg_initial_capital, rsi_len=cfg_rsi_len,
                                                  ema_len=cfg_ema_len, exit_days=cfg_exit_days)
                st.session_state['backtest_results'] = results
                st.session_state['last_ticker_simulated'] = current_ticker_input
                # Limpa sinais atuais ao simular novo backtest
                st.session_state.current_signals = {}
    else: st.warning("Insira um código de ativo."); st.session_state['backtest_results'] = None


# Exibe resultados se existirem e corresponderem ao último ticker
if 'backtest_results' in st.session_state and st.session_state['backtest_results'] is not None and \
   st.session_state['backtest_results']['ticker'] == st.session_state.get('last_ticker_simulated'):

    results = st.session_state['backtest_results']
    st.header(f"Resultados para: {results['ticker']}")

    # Botão Adicionar/Info Watchlist
    ticker_to_add = results['ticker']
    if ticker_to_add not in st.session_state.watchlist:
        if st.button(f"⭐ Adicionar {ticker_to_add} à Lista", help="Salva este ativo na lista da barra lateral."):
            st.session_state.watchlist.append(ticker_to_add); st.toast(f"{ticker_to_add} adicionado!"); st.rerun()
    else: st.info(f"{ticker_to_add} já está na Lista de Monitoramento.")

    with st.container(border=True):
        st.subheader("📊 Métricas de Desempenho")
        metrics = results['metrics']
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Rentabilidade Total", f"{metrics['totalReturnPercent']:.2f}%")
        m_col2.metric("Lucro Total", format_currency(metrics['totalProfit']))
        m_col3.metric("Nº Trades", metrics['numberOfTrades'])

        m_col4, m_col5, m_col6 = st.columns(3)
        m_col4.metric("Taxa de Acerto", f"{metrics['winRate']:.2f}%")
        m_col5.metric("Payoff Ratio", f"{metrics['payoffRatio']:.2f}")
        m_col6.metric("Max Drawdown", f"{metrics['maxDrawdown']:.2f}%")
        # Poderia adicionar Sharpe, Avg Gain/Loss se quisesse mais colunas

        st.caption(f"Período: {metrics['startDate']} a {metrics['endDate']} | Capital Inicial: {format_currency(metrics['initialCapital'])} | Capital Final: {format_currency(metrics['finalEquity'])}")

        st.divider()
        # Controles do Gráfico
        show_indicators_opt = st.toggle("Mostrar Indicadores no Gráfico", value=False, key="show_indicators_toggle")

        # Gráfico
        st.subheader("📈 Gráfico e Operações")
        fig = plot_results(results['chartData'], results['signals'], results['ticker'], results['indicators'], show_indicators=show_indicators_opt)
        st.plotly_chart(fig, use_container_width=True)

        # Tabela de Trades
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
     st.warning("Não foi possível gerar resultados. Verifique ticker, período e mensagens de erro.")
