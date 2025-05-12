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
import time # Para pequeno delay opcional

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Ferramenta EMA RSI Diário",
    page_icon="🎯",
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
if 'ticker_input_value' not in st.session_state:
    st.session_state.ticker_input_value = "PETR4.SA" # Valor inicial do input de ticker
if 'scan_results_df' not in st.session_state:
    st.session_state.scan_results_df = pd.DataFrame() # Guarda resultado do scan

# --- Lista de Tickers Comuns ---
# Listas estáticas de exemplo
COMMON_TICKERS = {
    "IBOV (Exemplos)": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "MGLU3.SA", "WEGE3.SA", "B3SA3.SA", "RENT3.SA", "PRIO3.SA"],
    "US Tech (Exemplos)": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CSCO"],
    "Cripto (Exemplos)": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"]
}
# Cria uma lista única e ordenada para o selectbox, incluindo uma opção vazia no início
FLAT_TICKER_LIST = [""] + sorted(list(set(ticker for sublist in COMMON_TICKERS.values() for ticker in sublist)))


# --- Função de Métricas (Atualizada para CAGR) ---
def calculate_metrics(trades_df, equity_curve_df, initial_capital):
    """Calcula métricas de desempenho básicas e avançadas."""
    metrics = {"winRate": 0, "payoffRatio": 0, "avgGain": 0, "avgLoss": 0, "maxDrawdown": 0, "sharpeRatio": 0, "cagr": 0}
    if trades_df.empty or equity_curve_df.empty:
        return metrics

    # Cálculos Manuais
    wins = trades_df[trades_df['profit'] > 0]
    losses = trades_df[trades_df['profit'] <= 0]
    total_trades = len(trades_df)
    metrics["winRate"] = round((len(wins) / total_trades * 100), 2) if total_trades > 0 else 0
    metrics["avgGain"] = round(wins['profit'].mean(), 2) if not wins.empty else 0
    metrics["avgLoss"] = round(abs(losses['profit'].mean()), 2) if not losses.empty else 0 # Média da perda (positiva)
    metrics["payoffRatio"] = round(metrics["avgGain"] / metrics["avgLoss"], 2) if metrics["avgLoss"] > 0 else 0

    # Cálculos com QuantStats
    try:
        # Prepara a curva de equity para o quantstats
        equity_curve_df['Date'] = pd.to_datetime(equity_curve_df['date'])
        equity_curve_df = equity_curve_df.set_index('Date')
        daily_returns = equity_curve_df['equity'].pct_change().dropna() # Calcula retornos diários

        if not daily_returns.empty and daily_returns.std() != 0 and len(daily_returns) >= 3: # Precisa de variância e dados suficientes
            metrics["maxDrawdown"] = round(qs.stats.max_drawdown(daily_returns) * 100, 2) # Em %
            metrics["sharpeRatio"] = round(qs.stats.sharpe(daily_returns), 2) # Sharpe anualizado (padrão)
            metrics["cagr"] = round(qs.stats.cagr(daily_returns) * 100, 2) # CAGR em %
        else:
            # Não calcula se não houver dados suficientes
             st.caption("Dados insuficientes para calcular Max Drawdown, Sharpe ou CAGR via QuantStats.")

    except Exception as e:
        st.caption(f"Aviso: Erro no cálculo de métricas QuantStats: {e}") # Aviso em vez de erro fatal

    # Garantir que NaN não seja retornado
    for k, v in metrics.items():
        if pd.isna(v):
            metrics[k] = 0 # Substitui NaN por 0
    return metrics

# --- Função do Backtest (Completa e Corrigida) ---
@st.cache_data(ttl=3600) # Cache de 1 hora
def run_strategy_backtest(ticker: str, start_date: date, end_date: date,
                          initial_capital: float = 1000.0, rsi_len: int = 2,
                          ema_len: int = 2, exit_days: int = 4):
    """
    Executa o backtest da estratégia EMA RSI Diário e retorna resultados incluindo métricas e CAGR.
    """
    st.caption(f"Iniciando backtest para {ticker}...") # Feedback sutil

    # --- 1. Buscar Dados Históricos ---
    fetch_start_date = start_date - timedelta(days=50) # Para aquecimento dos indicadores
    start_str = fetch_start_date.strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d') # +1 dia para incluir end_date

    try:
        data = yf.download(ticker, start=start_str, end=end_str, interval="1d", auto_adjust=True, progress=False)
        if data.empty: raise ValueError("Nenhum dado retornado pelo yfinance.")
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] # Padroniza nomes
        # Filtra para o período exato solicitado APÓS buscar mais para indicadores
        data = data[(data.index >= pd.Timestamp(start_date)) & (data.index < pd.Timestamp(end_date + timedelta(days=1)))]
        if data.empty: raise ValueError(f"Nenhum dado encontrado no período de {start_date} a {end_date}.")
    except Exception as e:
        st.error(f"Erro ao buscar dados para {ticker}: {e}", icon="📉")
        return None # Retorna None em caso de erro na busca

    # --- 2. Calcular Indicadores ---
    try:
        data.ta.rsi(close='Close', length=rsi_len, append=True, col_names=(f'RSI_{rsi_len}_C',))
        data.ta.rsi(close='Open', length=rsi_len, append=True, col_names=(f'RSI_{rsi_len}_O',))
        rsi_close_col = f'RSI_{rsi_len}_C'; rsi_open_col = f'RSI_{rsi_len}_O'
        data.ta.ema(close=rsi_close_col, length=ema_len, append=True, col_names=(f'EMA_{ema_len}_RSI_C',))
        data.ta.ema(close=rsi_open_col, length=ema_len, append=True, col_names=(f'EMA_{ema_len}_RSI_O',))
        ema_rsi_c_col = f'EMA_{ema_len}_RSI_C'; ema_rsi_o_col = f'EMA_{ema_len}_RSI_O'

        # Remove linhas com NaN gerados pelos indicadores (geralmente no início)
        initial_len = len(data)
        data.dropna(inplace=True)
        if data.empty: raise ValueError(f"Dados insuficientes após cálculo de indicadores (período RSI/EMA longo demais?). {initial_len} linhas iniciais.")
    except Exception as e:
        st.error(f"Erro ao calcular indicadores para {ticker}: {e}", icon="📊")
        return None

    # --- 3. Simulação / Lógica de Backtesting ---
    cash = initial_capital
    equity = initial_capital
    shares = 0.0
    in_position = False
    entry_price = 0.0
    entry_date = None
    days_in_trade = 0
    trades = []
    signals = []
    # Ponto inicial da curva de equity (data do primeiro dia com indicadores válidos)
    equity_curve = [{"date": data.index[0].date().strftime('%Y-%m-%d'), "equity": initial_capital}]

    # Loop dia a dia (começando do segundo dia com dados válidos)
    for i in range(1, len(data)):
        current_date = data.index[i]
        current_open_price = data['Open'].iloc[i]
        prev_row = data.iloc[i-1] # Dados do dia anterior (para gerar sinal)

        # Indicadores do dia anterior
        prev_ema_rsi_c = prev_row[ema_rsi_c_col]; prev_ema_rsi_o = prev_row[ema_rsi_o_col]
        # Indicadores do dia anterior ao anterior (para verificar cruzamentos e tendência da EMA)
        prev_prev_ema_rsi_c = data[ema_rsi_c_col].iloc[i-2] if i > 1 else np.nan
        prev_prev_ema_rsi_o = data[ema_rsi_o_col].iloc[i-2] if i > 1 else np.nan

        # --- Lógica de Saída (Verificada Primeiro) ---
        exit_signal = False; exit_reason = None
        if in_position:
            days_in_trade += 1
            # Condição 1: Saída Técnica (Crossunder + EMA caindo)
            is_crossunder = prev_ema_rsi_c < prev_ema_rsi_o and prev_prev_ema_rsi_c >= prev_prev_ema_rsi_o # Crossunder no dia anterior
            is_ema_falling = prev_ema_rsi_c < prev_prev_ema_rsi_c # EMA caiu no dia anterior
            if is_crossunder and is_ema_falling:
                exit_signal, exit_reason = True, "Saída Técnica"
            # Condição 2: Saída por Tempo
            elif days_in_trade == exit_days:
                exit_signal, exit_reason = True, f"Saída por Tempo ({exit_days}d)"

            # Executa a venda se o sinal for verdadeiro
            if exit_signal:
                exit_price = current_open_price # Sai na abertura do dia atual
                if shares > 0 and exit_price > 0: # Segurança
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
                # Reseta estado da posição
                shares, in_position, entry_price, entry_date, days_in_trade = 0.0, False, 0.0, None, 0

        # --- Lógica de Entrada (Verificada se não saiu e não está em posição) ---
        if not in_position and i > 1: # Precisa de i > 1 para ter prev_prev
            # Condição: Crossover + EMA subindo
            is_crossover = prev_ema_rsi_c > prev_ema_rsi_o and prev_prev_ema_rsi_c <= prev_prev_ema_rsi_o # Crossover no dia anterior
            is_ema_rising = prev_ema_rsi_c > prev_prev_ema_rsi_c # EMA subiu no dia anterior

            if is_crossover and is_ema_rising:
                entry_price = current_open_price # Entra na abertura do dia atual
                if entry_price > 0 and cash > 0: # Segurança
                    shares_to_buy = cash / entry_price
                    cash -= shares_to_buy * entry_price # Atualiza caixa
                    # Define estado da posição
                    shares, in_position, entry_date, days_in_trade = shares_to_buy, True, current_date.date(), 0
                    signals.append({"date": current_date.date().strftime('%Y-%m-%d'), "type": "buy", "price": round(entry_price, 2)})

        # Atualiza valor do Portfólio Diário (Equity) para a curva
        current_equity = cash + (shares * data['Close'].iloc[i]) # Usa o fechamento do dia atual
        equity_curve.append({"date": current_date.date().strftime('%Y-%m-%d'), "equity": round(current_equity, 2)})

    # --- 4. Calcular Resultados Finais ---
    final_equity = equity_curve[-1]['equity'] if equity_curve else initial_capital
    total_profit = final_equity - initial_capital
    total_return_percent = (total_profit / initial_capital) * 100 if initial_capital > 0 else 0
    number_of_trades = len(trades)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_curve_df = pd.DataFrame(equity_curve)

    # Calcular Métricas Adicionais (incluindo CAGR)
    adv_metrics = calculate_metrics(trades_df, equity_curve_df, initial_capital)

    # Preparar dados para gráfico (incluindo indicadores)
    chart_data_df = data[['Open', 'High', 'Low', 'Close', rsi_close_col, rsi_open_col, ema_rsi_c_col, ema_rsi_o_col]].reset_index()
    chart_data_df['Date'] = pd.to_datetime(chart_data_df['Date']).dt.date # Garante que é date

    # Datas de início/fim efetivas do backtest (após remover NaNs)
    metrics_start_date = data.index.min().date() if not data.empty else None
    metrics_end_date = data.index.max().date() if not data.empty else None

    # --- 5. Retornar Dicionário Completo de Resultados ---
    results = {
        "ticker": ticker,
        "params": {"rsi": rsi_len, "ema": ema_len, "exit": exit_days, "capital": initial_capital},
        "metrics": {
            "initialCapital": round(initial_capital, 2), "finalEquity": round(final_equity, 2),
            "totalProfit": round(total_profit, 2), "totalReturnPercent": round(total_return_percent, 2),
            "numberOfTrades": number_of_trades,
            "startDate": metrics_start_date.strftime('%Y-%m-%d') if metrics_start_date else 'N/A',
            "endDate": metrics_end_date.strftime('%Y-%m-%d') if metrics_end_date else 'N/A',
            **adv_metrics # Adiciona winRate, payoffRatio, avgGain, avgLoss, maxDrawdown, sharpeRatio, cagr
        },
        "trades": trades_df, # DataFrame de trades
        "signals": signals, # Lista de dicionários de sinais
        "chartData": chart_data_df, # DataFrame com OHLC e indicadores
        "indicators": {"rsi_c": rsi_close_col, "rsi_o": rsi_open_col, "ema_c": ema_rsi_c_col, "ema_o": ema_rsi_o_col} # Nomes das colunas
    }
    st.caption(f"Backtest concluído para {ticker}.") # Feedback sutil
    return results


# --- Função para Plotar o Gráfico com Subplots (Completa) ---
def plot_results(chart_data_df, signals, ticker, indicators_cols, show_indicators=False):
    """Plota o gráfico de velas, sinais e opcionalmente os indicadores."""
    if chart_data_df.empty:
        st.warning("Não há dados para plotar o gráfico.")
        return go.Figure() # Retorna figura vazia

    # Cria figura com 2 subplots (preços em cima, indicadores embaixo)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, # Espaço entre gráficos
                        row_heights=[0.7, 0.3]) # Proporção de altura

    # --- Gráfico de Velas (Linha 1) ---
    fig.add_trace(go.Candlestick(x=chart_data_df['Date'], open=chart_data_df['Open'], high=chart_data_df['High'],
                               low=chart_data_df['Low'], close=chart_data_df['Close'], name='OHLC',
                               increasing_line_color='green', decreasing_line_color='red'), # Cores das velas
                  row=1, col=1)

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

    # --- Plotar Indicadores (Linha 2) se selecionado ---
    if show_indicators:
        rsi_c_col = indicators_cols['rsi_c']; rsi_o_col = indicators_cols['rsi_o']
        ema_c_col = indicators_cols['ema_c']; ema_o_col = indicators_cols['ema_o']

        # Plotar EMAs sobre RSI
        fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[ema_c_col], name='EMA(RSI C)', line=dict(color='blue', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[ema_o_col], name='EMA(RSI O)', line=dict(color='orange', width=1.5)), row=2, col=1)

        # Opcional: Plotar RSIs originais (pode poluir, descomente se quiser)
        # fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[rsi_c_col], name='RSI Close', line=dict(color='lightblue', width=1, dash='dot')), row=2, col=1)
        # fig.add_trace(go.Scatter(x=chart_data_df['Date'], y=chart_data_df[rsi_o_col], name='RSI Open', line=dict(color='lightsalmon', width=1, dash='dot')), row=2, col=1)

        fig.update_yaxes(title_text="Indicadores", row=2, col=1, showgrid=True) # Título e grid para eixo Y do subplot 2
        # Adicionar linhas de referência (ex: sobrecompra/venda - ajustar se necessário)
        # fig.add_hline(y=80, line_dash="dot", line_color="rgba(255,0,0,0.5)", row=2, col=1)
        # fig.add_hline(y=20, line_dash="dot", line_color="rgba(0,255,0,0.5)", row=2, col=1)
    else:
        # Se indicadores não são mostrados, remove o espaço reservado para o segundo eixo Y
         fig.update_layout(yaxis2_visible=False)


    # --- Layout Geral do Gráfico ---
    fig.update_layout(
        title={'text': f'Backtest: {ticker}', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}, # Centraliza título
        height=600, # Altura total
        xaxis_rangeslider_visible=False, # Esconder slider do eixo X principal
        xaxis_showticklabels=True, xaxis2_showticklabels=True, # Mostrar labels do eixo X em ambos (se o 2º existir)
        yaxis_title="Preço", # Eixo Y do Gráfico 1
        yaxis_fixedrange=False, # Permite zoom no eixo Y de preço
        yaxis2_fixedrange=False, # Permite zoom no eixo Y de indicadores (se existir)
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5), # Legenda horizontal acima
        margin=dict(l=50, r=50, t=80, b=50) # Margens
    )
    # Melhora aparência dos eixos
    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True, row=1, col=1)
    if show_indicators:
        fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey', mirror=True, row=2, col=1)

    return fig

# --- Função para Verificar Sinal Atual (Completa) ---
@st.cache_data(ttl=900) # Cache de 15 minutos
def get_current_signal(ticker: str, rsi_len: int = 2, ema_len: int = 2):
    """Verifica o sinal da estratégia (COMPRA/VENDA/NEUTRO) no último dia disponível."""
    st.caption(f"Verificando sinal para {ticker}...") # Feedback sutil
    try:
        # Pega um pouco mais de dados (ex: 20 dias) para garantir cálculo
        data = yf.download(ticker, period="20d", interval="1d", auto_adjust=True, progress=False)
        if data.empty or len(data) < rsi_len + ema_len: return "Dados Insuficientes"

        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Calcula indicadores necessários
        data.ta.rsi(close='Close', length=rsi_len, append=True, col_names=('RSI_C',))
        data.ta.rsi(close='Open', length=rsi_len, append=True, col_names=('RSI_O',))
        data.ta.ema(close='RSI_C', length=ema_len, append=True, col_names=('EMA_RSI_C',))
        data.ta.ema(close='RSI_O', length=ema_len, append=True, col_names=('EMA_RSI_O',))
        data.dropna(inplace=True) # Remove NaNs no início
        if len(data) < 2: return "Dados Insuficientes pós Ind." # Precisa de pelo menos 2 pontos para comparar

        # Pega as duas últimas linhas válidas
        last = data.iloc[-1]
        prev = data.iloc[-2]

        # Verifica Condição de Compra no último dia (baseado nos dados do penúltimo)
        is_crossover = prev['EMA_RSI_C'] > prev['EMA_RSI_O'] and data['EMA_RSI_C'].iloc[-3] <= data['EMA_RSI_O'].iloc[-3] if len(data) > 2 else False # Verifica cruzamento no penúltimo
        is_ema_rising = prev['EMA_RSI_C'] > data['EMA_RSI_C'].iloc[-3] if len(data) > 2 else False # Verifica se EMA subiu no penúltimo
        if is_crossover and is_ema_rising:
            return "COMPRA" # Sinal de compra GERADO no fechamento anterior, válido para HOJE

        # Verifica Condição de Saída Técnica no último dia
        is_crossunder = prev['EMA_RSI_C'] < prev['EMA_RSI_O'] and data['EMA_RSI_C'].iloc[-3] >= data['EMA_RSI_O'].iloc[-3] if len(data) > 2 else False
        is_ema_falling = prev['EMA_RSI_C'] < data['EMA_RSI_C'].iloc[-3] if len(data) > 2 else False
        if is_crossunder and is_ema_falling:
            return "VENDA" # Sinal de venda GERADO no fechamento anterior, válido para HOJE

        # Nenhuma condição de entrada ou saída técnica ativa
        return "NEUTRO"

    except Exception as e:
        st.caption(f"Erro ao verificar {ticker}: {str(e)[:50]}...", ) # Mostra erro sutilmente
        return "Erro"


# --- Formatação de Moeda ---
def format_currency(value):
    """Formata um valor numérico como moeda brasileira (R$)."""
    try:
        return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        # Retorna 0 ou o próprio valor se não for numérico? Decidi por 0.
        return "R$ 0,00"

# ==============================================================================
# --- Interface Principal do Streamlit ---
# ==============================================================================

st.title("🎯 Ferramenta EMA RSI Diário")
st.caption("Simulador, Scanner e Monitor de Sinais para a estratégia EMA(2)/RSI(2)")
st.divider()

# --- Barra Lateral (Sidebar - Completa) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=80) # Ícone temático
    st.header("⚙️ Configurações Globais")
    st.caption("Estes parâmetros afetam o Simulador e o Scanner.")

    # Expander para Parâmetros da Estratégia
    with st.expander("Parâmetros da Estratégia", expanded=True):
        cfg_rsi_len = st.number_input("Período RSI", min_value=2, max_value=50, value=2, step=1, key="cfg_rsi")
        cfg_ema_len = st.number_input("Período EMA (sobre RSI)", min_value=2, max_value=50, value=2, step=1, key="cfg_ema")
        cfg_exit_days = st.number_input("Dias Saída por Tempo", min_value=1, max_value=30, value=4, step=1, key="cfg_exit")

    # Expander para Período do Backtest
    with st.expander("Período do Backtest", expanded=True):
        today = datetime.now().date()
        one_year_ago = today - timedelta(days=365)
        cfg_start_date = st.date_input("Data Inicial", value=one_year_ago, max_value=today - timedelta(days=1), key="cfg_start")
        cfg_end_date = st.date_input("Data Final", value=today, min_value=cfg_start_date + timedelta(days=1), max_value=today, key="cfg_end")
        st.caption(f"Duração: {(cfg_end_date - cfg_start_date).days} dias")

    # Expander para Financeiro
    with st.expander("Financeiro", expanded=True):
        cfg_initial_capital = st.number_input("Capital Inicial (R$)", min_value=1.0, value=1000.0, step=100.0, format="%.2f", key="cfg_capital")

    st.divider()

    # --- Watchlist ---
    st.header("⭐ Lista de Monitoramento")
    st.caption("Acompanhe seus ativos e verifique sinais.")

    # Botão para Verificar Sinais Atuais da Lista
    if st.button("📡 Verificar Sinais Atuais", key="check_signals", use_container_width=True, help="Verifica o sinal COMPRA/VENDA/NEUTRO para hoje nos ativos da lista."):
        current_signals_temp = {}
        if not st.session_state.watchlist:
            st.toast("Lista de monitoramento vazia.", icon="ℹ️")
        else:
            progress_bar = st.progress(0, text="Verificando sinais...")
            num_tickers = len(st.session_state.watchlist)
            for i, ticker_to_check in enumerate(st.session_state.watchlist):
                # Usa parâmetros atuais da sidebar para checar sinal
                signal = get_current_signal(ticker_to_check, st.session_state.cfg_rsi, st.session_state.cfg_ema)
                current_signals_temp[ticker_to_check] = signal
                progress_bar.progress((i + 1) / num_tickers, text=f"Verificando {ticker_to_check}...")
                time.sleep(0.05) # Pequeno delay para não sobrecarregar API
            st.session_state.current_signals = current_signals_temp # Atualiza o estado
            progress_bar.empty()
            st.toast("Verificação de sinais concluída!", icon="✅")

    # Exibe a lista e botões de remover + sinal atual
    if not st.session_state.watchlist:
        st.info("Nenhum ativo na lista. Adicione após simular um ativo.")
    else:
        st.caption("Ativos:")
        for i in range(len(st.session_state.watchlist) - 1, -1, -1): # Loop reverso seguro para remoção
            ticker_in_list = st.session_state.watchlist[i]
            signal_status = st.session_state.current_signals.get(ticker_in_list, "") # Pega sinal atual se existir

            signal_color = {"COMPRA": "green", "VENDA": "red", "NEUTRO": "gray", "Erro": "orange", "Dados Insuficientes": "orange", "Dados Insuficientes pós Ind.": "orange"}.get(signal_status, "gray")
            signal_icon = {"COMPRA": "🔼", "VENDA": "🔽", "NEUTRO": "⏸️"}.get(signal_status, "❔")
            signal_display = f"<span style='color:{signal_color}; font-weight:bold; font-size:small;'>{signal_icon} {signal_status}</span>" if signal_status else ""

            col1_watch, col2_watch, col3_watch = st.columns([0.55, 0.25, 0.2]) # Nome, Sinal, Remover
            with col1_watch:
                 # Clicar no nome preenche o input da aba Simulador
                if st.button(ticker_in_list, key=f"sim_{ticker_in_list}_{i}", type="secondary", help=f"Carregar {ticker_in_list} no Simulador", use_container_width=True):
                     st.session_state.ticker_input_value = ticker_in_list
                     st.rerun() # Força atualização do input na aba 1
            with col2_watch:
                st.markdown(signal_display, unsafe_allow_html=True)
            with col3_watch:
                # Botão Remover (-)
                if st.button("➖", key=f"remove_{ticker_in_list}_{i}", type="secondary", help="Remover da lista", use_container_width=True):
                    removed_ticker = st.session_state.watchlist.pop(i)
                    st.session_state.current_signals.pop(removed_ticker, None) # Remove sinal também
                    st.toast(f"{removed_ticker} removido.")
                    if st.session_state.get('last_ticker_simulated') == removed_ticker:
                        st.session_state['backtest_results'] = None; st.session_state['last_ticker_simulated'] = ""
                    st.rerun()

    # Link para Limpar Lista
    if st.session_state.watchlist: # Só mostra se a lista não estiver vazia
        st.markdown("<a href='?clear_watchlist=true' target='_self' style='color: tomato; font-size: small;'>Limpar Lista Completa</a>", unsafe_allow_html=True)


# Lógica para limpar watchlist (Completa)
if st.query_params.get("clear_watchlist") == "true":
    st.session_state.watchlist = []
    st.session_state.current_signals = {} # Limpa sinais atuais
    st.session_state['backtest_results'] = None # Limpa resultados da simulação
    st.session_state['last_ticker_simulated'] = ""
    st.session_state['scan_results_df'] = pd.DataFrame() # Limpa resultados do scan
    st.toast("Lista de monitoramento e resultados limpos.")
    st.query_params.clear() # Limpa o parâmetro da URL
    st.rerun() # Atualiza a interface


# ==============================================================================
# --- Abas para Organizar a Interface ---
# ==============================================================================
tab1, tab2 = st.tabs(["📈 **Simulador Individual**", "🔍 **Scanner de Ativos**"])

# --- Aba 1: Simulador Individual (Completa) ---
with tab1:
    st.header("Simulador de Backtest Individual")
    st.markdown("Teste a estratégia em um único ativo com os parâmetros definidos na barra lateral.")

    sim_col1, sim_col2 = st.columns([0.6, 0.4])
    with sim_col1:
        # Input principal (controlado pelo session_state)
        ticker_input_sim = st.text_input(
            "Código do Ativo (ex: PETR4.SA, AAPL, BTC-USD):",
            value=st.session_state.ticker_input_value, # Usa valor do estado
            placeholder="Digite ou selecione abaixo...",
            key="ticker_input_main_tab1",
            label_visibility="collapsed"
        ).upper()
        # Atualiza o state se o usuário digitar algo manualmente
        if ticker_input_sim != st.session_state.ticker_input_value:
             st.session_state.ticker_input_value = ticker_input_sim
             st.session_state.backtest_results = None # Limpa resultado antigo se ticker mudar
             st.session_state.last_ticker_simulated = ""

    with sim_col2:
        # Selectbox para tickers comuns (afeta o input acima)
        selected_common_ticker_sim = st.selectbox(
            "Selecionar ativo comum:",
            options=FLAT_TICKER_LIST,
            index=FLAT_TICKER_LIST.index(st.session_state.ticker_input_value) if st.session_state.ticker_input_value in FLAT_TICKER_LIST else 0, # Define index atual
            key="common_ticker_select_tab1",
            label_visibility="collapsed"
        )
        # Se o usuário selecionou algo diferente no selectbox, atualiza o state
        if selected_common_ticker_sim and selected_common_ticker_sim != st.session_state.ticker_input_value:
            st.session_state.ticker_input_value = selected_common_ticker_sim
            st.session_state.backtest_results = None # Limpa resultado antigo
            st.session_state.last_ticker_simulated = ""
            st.rerun() # Força o rerun para atualizar o text_input

    # Botão de Simulação
    simulate_button = st.button("Executar Simulação Individual", type="primary", use_container_width=True, key="sim_button_tab1")
    st.divider()

    # --- Execução da Simulação Individual ---
    if simulate_button:
        current_ticker_input = st.session_state.ticker_input_value # Pega valor atual
        if current_ticker_input:
            # Pega parâmetros da sidebar (usando as chaves dos widgets)
            sim_params = {
                "start_date": st.session_state.cfg_start, "end_date": st.session_state.cfg_end,
                "initial_capital": st.session_state.cfg_capital, "rsi_len": st.session_state.cfg_rsi,
                "ema_len": st.session_state.cfg_ema, "exit_days": st.session_state.cfg_exit
            }
            if sim_params["start_date"] >= sim_params["end_date"]:
                st.error("Erro: Data Inicial deve ser anterior à Data Final.")
            else:
                with st.spinner(f"Executando backtest para {current_ticker_input}..."):
                    results = run_strategy_backtest(ticker=current_ticker_input, **sim_params)
                    st.session_state['backtest_results'] = results # Salva resultado na sessão
                    st.session_state['last_ticker_simulated'] = current_ticker_input # Guarda qual ticker foi simulado
        else:
            st.warning("Insira um código de ativo para simular.")
            st.session_state['backtest_results'] = None


    # --- Exibição dos Resultados da Simulação Individual (Completa) ---
    # Verifica se há resultados E se pertencem ao último ticker simulado
    if 'backtest_results' in st.session_state and st.session_state['backtest_results'] is not None and \
       st.session_state['backtest_results']['ticker'] == st.session_state.get('last_ticker_simulated'):

        results = st.session_state['backtest_results']
        st.header(f"Resultados para: {results['ticker']}")

        # Botão Adicionar/Info Watchlist
        ticker_to_add = results['ticker']
        add_col, info_col = st.columns([0.3, 0.7])
        with add_col:
            if ticker_to_add not in st.session_state.watchlist:
                if st.button(f"⭐ Adicionar à Lista", key=f"add_watch_{ticker_to_add}", help="Salva este ativo na lista da barra lateral."):
                    st.session_state.watchlist.append(ticker_to_add); st.toast(f"{ticker_to_add} adicionado!"); st.rerun()
            else:
                st.success(f"✔️ Na Watchlist")

        with st.container(border=True): # Agrupa resultados com borda
            st.subheader("📊 Métricas de Desempenho")
            metrics = results['metrics']
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("CAGR (Retorno Anualizado)", f"{metrics['cagr']:.2f}%") # CAGR adicionado
            m_col2.metric("Retorno Total Período", f"{metrics['totalReturnPercent']:.2f}%")
            m_col3.metric("Nº Trades", metrics['numberOfTrades'])

            m_col4, m_col5, m_col6 = st.columns(3)
            m_col4.metric("Taxa de Acerto", f"{metrics['winRate']:.2f}%")
            m_col5.metric("Payoff Ratio", f"{metrics['payoffRatio']:.2f}")
            m_col6.metric("Max Drawdown", f"{metrics['maxDrawdown']:.2f}%")

            st.caption(f"Período: {metrics['startDate']} a {metrics['endDate']} | Capital: {format_currency(metrics['initialCapital'])} -> {format_currency(metrics['finalEquity'])} ({format_currency(metrics['totalProfit'])}) | Sharpe: {metrics['sharpeRatio']:.2f}")
            st.divider()

            # Controles do Gráfico
            st.subheader("📈 Gráfico e Operações")
            show_indicators_opt = st.toggle("Mostrar Indicadores no Gráfico", value=False, key="show_indicators_toggle_tab1")

            # Gráfico
            fig = plot_results(results['chartData'], results['signals'], results['ticker'], results['indicators'], show_indicators=show_indicators_opt)
            st.plotly_chart(fig, use_container_width=True)

            # Tabela de Trades
            with st.expander("📜 Ver Histórico de Operações Detalhado"):
                trades_df = results['trades']
                if not trades_df.empty:
                    trades_df_display = trades_df.copy()
                    # Formatar colunas
                    trades_df_display['profit'] = trades_df_display['profit'].apply(format_currency)
                    trades_df_display['entryPrice'] = trades_df_display['entryPrice'].map('{:.2f}'.format)
                    trades_df_display['exitPrice'] = trades_df_display['exitPrice'].map('{:.2f}'.format)
                    trades_df_display['returnPercent'] = trades_df_display['returnPercent'].map('{:.2f}%'.format)
                    st.dataframe(trades_df_display[['entryDate', 'entryPrice', 'exitDate', 'exitPrice', 'daysHeld', 'profit', 'returnPercent', 'exitReason']], use_container_width=True)
                else:
                    st.info("Nenhuma operação realizada neste período.")

    # Mensagem se a simulação foi tentada mas não gerou resultado (ex: erro)
    elif simulate_button and st.session_state['backtest_results'] is None:
         st.warning("Não foi possível gerar resultados para a simulação individual. Verifique o ticker, período e mensagens de erro.")


# --- Aba 2: Scanner de Ativos (Completa) ---
with tab2:
    st.header("Scanner de Ativos Lucrativos")
    st.markdown("Encontre ativos que tiveram **retorno anualizado (CAGR) positivo** com os parâmetros de estratégia definidos na barra lateral, dentro do período de backtest selecionado.")
    st.warning("Escanear listas longas pode demorar e exceder limites no Streamlit Cloud. Comece com listas menores.", icon="⚠️")

    scan_col1, scan_col2 = st.columns([0.6, 0.4])

    with scan_col1:
        scan_list_options = ["Use Minha Watchlist Atual"] + list(COMMON_TICKERS.keys())
        selected_scan_list_name = st.radio(
            "Selecione a lista de ativos para escanear:",
            options=scan_list_options,
            key="scan_list_radio_tab2",
            horizontal=True,
        )
    with scan_col2:
        min_cagr_threshold = st.number_input(
            "Rentabilidade Anual Mínima Desejada (%)",
            min_value=0.0, value=30.0, step=5.0, format="%.1f", key="min_cagr_input"
        )

    if selected_scan_list_name == "Use Minha Watchlist Atual":
        tickers_for_scan = st.session_state.watchlist
        list_is_empty = not tickers_for_scan
        if list_is_empty: st.warning("Sua watchlist está vazia. Adicione ativos ou selecione outra lista.")
    else:
        tickers_for_scan = COMMON_TICKERS[selected_scan_list_name]
        list_is_empty = False

    st.caption(f"Serão escaneados {len(tickers_for_scan)} ativos da lista: '{selected_scan_list_name}'")

    scan_button = st.button("🔎 Escanear Ativos Agora", key="scan_button_tab2", type="primary", disabled=list_is_empty)
    st.divider()

    if scan_button and not list_is_empty:
        # Pega parâmetros da sidebar usando as chaves dos widgets
        params = {
            "start_date": st.session_state.cfg_start, "end_date": st.session_state.cfg_end,
            "initial_capital": st.session_state.cfg_capital, "rsi_len": st.session_state.cfg_rsi,
            "ema_len": st.session_state.cfg_ema, "exit_days": st.session_state.cfg_exit
        }

        st.info(f"Iniciando escaneamento com CAGR > {min_cagr_threshold}%...")
        scan_progress = st.progress(0, text="Iniciando...")
        profitable_list = []
        skipped_count = 0
        error_messages = []
        total_tickers = len(tickers_for_scan)

        # Loop principal do Scanner
        for i, ticker in enumerate(tickers_for_scan):
            progress_percentage = (i + 1) / total_tickers
            scan_progress.progress(progress_percentage, text=f"Escaneando: {ticker} ({i+1}/{total_tickers})")
            try:
                # Chama o backtest (aproveita cache se parâmetros não mudaram)
                results = run_strategy_backtest(ticker=ticker, **params)

                if results and results.get('metrics'):
                    cagr_value = results['metrics'].get('cagr', -999) # Pega CAGR, default negativo se não existir
                    if cagr_value > min_cagr_threshold:
                        profitable_list.append({
                            'Ticker': ticker,
                            'CAGR (%)': cagr_value,
                            'Retorno Total (%)': results['metrics']['totalReturnPercent'],
                            'Nº Trades': results['metrics']['numberOfTrades'],
                            'Taxa Acerto (%)': results['metrics']['winRate'],
                            'Max Drawdown (%)': results['metrics']['maxDrawdown'],
                            # 'Payoff Ratio': results['metrics']['payoffRatio'] # Opcional
                        })
            except Exception as e:
                # Captura erro GERAL no processamento do ticker
                error_msg = f"Erro ao processar {ticker}: {str(e)[:100]}..."
                # st.caption(error_msg) # Pode poluir muito, mostra no final
                error_messages.append(error_msg)
                skipped_count += 1
            # time.sleep(0.05) # Delay opcional

        scan_progress.empty() # Limpa barra de progresso

        # --- Exibe Resultados do Scan ---
        st.subheader("Resultados do Escaneamento")
        if profitable_list:
            profitable_df = pd.DataFrame(profitable_list).sort_values(by='CAGR (%)', ascending=False).reset_index(drop=True)
            # Formata colunas percentuais
            for col in ['CAGR (%)', 'Retorno Total (%)', 'Taxa Acerto (%)', 'Max Drawdown (%)']:
                 if col in profitable_df.columns:
                    profitable_df[col] = profitable_df[col].map('{:.2f}%'.format)

            st.session_state['scan_results_df'] = profitable_df # Salva no estado
            st.success(f"Encontrados {len(profitable_df)} ativos com CAGR > {min_cagr_threshold}%.")
            if skipped_count > 0: st.warning(f"{skipped_count} ativos foram pulados devido a erros.")

            # Exibe a tabela
            st.dataframe(profitable_df, use_container_width=True, height=min( (len(profitable_df) + 1) * 35 + 3, 600) ) # Altura dinâmica até 600px

            # Adiciona opção de adicionar todos os resultados à watchlist
            profitable_tickers = profitable_df['Ticker'].tolist()
            missing_in_watchlist = [t for t in profitable_tickers if t not in st.session_state.watchlist]
            if missing_in_watchlist:
                if st.button(f"⭐ Adicionar {len(missing_in_watchlist)} resultados à Watchlist"):
                     st.session_state.watchlist.extend(missing_in_watchlist)
                     # Remove duplicados caso existam por alguma razão
                     st.session_state.watchlist = sorted(list(set(st.session_state.watchlist)))
                     st.toast(f"{len(missing_in_watchlist)} ativos adicionados à watchlist!")
                     st.rerun()

            if error_messages: # Mostra erros se houveram
                with st.expander("Ver detalhes dos erros durante o scan"):
                    for msg in error_messages: st.caption(msg)
        else:
            st.session_state['scan_results_df'] = pd.DataFrame() # Salva vazio
            st.info(f"Nenhum ativo encontrado com CAGR > {min_cagr_threshold}% na lista '{selected_scan_list_name}'.")
            if skipped_count > 0: st.warning(f"{skipped_count} ativos foram pulados devido a erros.")
            if error_messages:
                 with st.expander("Ver detalhes dos erros durante o scan"):
                    for msg in error_messages: st.caption(msg)


    # Exibir resultados do último scan se existirem e o botão não foi pressionado agora
    elif not scan_button and not st.session_state.scan_results_df.empty:
         st.subheader("Resultado do Último Escaneamento Realizado:")
         st.dataframe(st.session_state.scan_results_df, use_container_width=True)
         st.caption("Clique em 'Escanear Ativos Agora' para atualizar os resultados com os parâmetros atuais.")
