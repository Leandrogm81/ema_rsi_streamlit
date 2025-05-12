# app.py
import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Estratégia EMA RSI Diário",
    page_icon="📈",
    layout="wide"
)

# --- Inicialização do Session State ---
# Usado para guardar a watchlist e os resultados entre reruns
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = [] # Lista para os tickers monitorados
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'last_ticker_simulated' not in st.session_state:
    st.session_state.last_ticker_simulated = ""


# --- Função do Backtest (Adaptada para receber mais parâmetros) ---
@st.cache_data(ttl=3600) # Cache por 1 hora
def run_strategy_backtest(ticker: str, start_date: datetime, end_date: datetime,
                          initial_capital: float = 1000.0, rsi_len: int = 2,
                          ema_len: int = 2, exit_days: int = 4):
    """
    Executa o backtest da estratégia EMA RSI Diário com parâmetros configuráveis.
    Retorna um dicionário com resultados ou None em caso de erro.
    """
    st.write(f"Buscando dados para {ticker} de {start_date.date()} a {end_date.date()}...")

    # --- 1. Buscar Dados Históricos ---
    # Buscar um pouco mais de dados para aquecimento dos indicadores
    fetch_start_date = start_date - timedelta(days=50) # Pegar 50 dias antes para cálculo inicial

    try:
        data = yf.download(ticker, start=fetch_start_date, end=end_date + timedelta(days=1), # +1 dia para incluir end_date
                           interval="1d", auto_adjust=True, progress=False)
        if data.empty:
            st.error(f"Não foi possível obter dados para {ticker} no período solicitado.")
            return None
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Filtrar dados para o período exato solicitado APÓS buscar mais para indicadores
        data = data[data.index >= pd.Timestamp(start_date)] # Filtra a partir da data de início real

    except Exception as e:
        st.error(f"Erro ao buscar dados para {ticker}: {str(e)}")
        return None

    if data.empty:
        st.error(f"Não há dados para {ticker} no período de {start_date.date()} a {end_date.date()}.")
        return None

    st.write(f"Calculando indicadores (RSI({rsi_len}), EMA({ema_len}))...")
    # --- 2. Calcular Indicadores ---
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
        if data.empty:
             st.error(f"Não há dados suficientes para {ticker} após cálculo dos indicadores.")
             return None
    except Exception as e:
        st.error(f"Erro ao calcular indicadores para {ticker}: {str(e)}")
        return None

    st.write(f"Executando simulação...")
    # --- 3. Simulação / Lógica de Backtesting ---
    # (Lógica interna da simulação permanece a mesma, usando os parâmetros rsi_len, ema_len, exit_days)
    cash = initial_capital
    equity = initial_capital
    shares = 0.0
    in_position = False
    entry_price = 0.0
    entry_date = None
    days_in_trade = 0
    trades = []
    signals = []
    equity_curve = []

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
            if is_crossunder and is_ema_falling:
                exit_signal = True
                exit_reason = "Saída Técnica (Crossunder)"
            elif days_in_trade == exit_days:
                exit_signal = True
                exit_reason = f"Saída por Tempo ({exit_days} dias)"

            if exit_signal:
                exit_price = current_open_price
                if shares > 0:
                    cash += shares * exit_price
                    profit = (exit_price - entry_price) * shares
                    return_pct = ((exit_price / entry_price) - 1) * 100 if entry_price != 0 else 0
                    trades.append({
                        "entryDate": entry_date.strftime('%Y-%m-%d'), "entryPrice": round(entry_price, 2),
                        "exitDate": current_date.strftime('%Y-%m-%d'), "exitPrice": round(exit_price, 2),
                        "shares": round(shares, 4), "profit": round(profit, 2),
                        "returnPercent": round(return_pct, 2), "exitReason": exit_reason, "daysHeld": days_in_trade
                    })
                    signals.append({"date": current_date.strftime('%Y-%m-%d'), "type": "sell", "price": round(exit_price, 2)})
                shares = 0.0
                in_position = False
                entry_price = 0.0
                entry_date = None
                days_in_trade = 0

        if not in_position and i > 1:
            is_crossover = prev_ema_rsi_c > prev_ema_rsi_o and prev_prev_ema_rsi_c <= prev_prev_ema_rsi_o
            is_ema_rising = prev_ema_rsi_c > prev_prev_ema_rsi_c
            if is_crossover and is_ema_rising:
                entry_price = current_open_price
                if entry_price > 0 and cash > 0:
                    shares_to_buy = cash / entry_price
                    cash -= shares_to_buy * entry_price
                    shares = shares_to_buy
                    in_position = True
                    entry_date = current_date
                    days_in_trade = 0
                    signals.append({"date": current_date.strftime('%Y-%m-%d'), "type": "buy", "price": round(entry_price, 2)})

        current_equity = cash + (shares * data['Close'].iloc[i])
        equity_curve.append({"date": current_date.strftime('%Y-%m-%d'), "equity": round(current_equity, 2)})

    # --- 4. Calcular Resultados Finais ---
    final_equity = equity_curve[-1]['equity'] if equity_curve else initial_capital
    total_profit = final_equity - initial_capital
    total_return_percent = (total_profit / initial_capital) * 100 if initial_capital > 0 else 0
    number_of_trades = len(trades)
    chart_data_df = data[['Open', 'High', 'Low', 'Close']].reset_index() # Manter como DataFrame

    # --- 5. Retornar Resultados ---
    results = {
        "ticker": ticker,
        "params": {"rsi": rsi_len, "ema": ema_len, "exit": exit_days, "capital": initial_capital},
        "metrics": {
            "initialCapital": round(initial_capital, 2), "finalEquity": round(final_equity, 2),
            "totalProfit": round(total_profit, 2), "totalReturnPercent": round(total_return_percent, 2),
            "numberOfTrades": number_of_trades,
            "startDate": data.index.min().strftime('%Y-%m-%d') if not data.empty else 'N/A',
            "endDate": data.index.max().strftime('%Y-%m-%d') if not data.empty else 'N/A',
        },
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
        "signals": signals,
        "chartData": chart_data_df
    }
    return results

# --- Função para Plotar o Gráfico ---
def plot_results(chart_data_df, signals, ticker):
    if chart_data_df.empty:
        st.warning("Não há dados para plotar o gráfico.")
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=chart_data_df['Date'], open=chart_data_df['Open'], high=chart_data_df['High'],
                               low=chart_data_df['Low'], close=chart_data_df['Close'], name='OHLC'))
    buy_signals_df = pd.DataFrame([s for s in signals if s['type'] == 'buy'])
    if not buy_signals_df.empty:
        fig.add_trace(go.Scatter(x=buy_signals_df['date'], y=buy_signals_df['price'], mode='markers', name='Compra',
                                 marker=dict(color='green', size=10, symbol='triangle-up')))
    sell_signals_df = pd.DataFrame([s for s in signals if s['type'] == 'sell'])
    if not sell_signals_df.empty:
        fig.add_trace(go.Scatter(x=sell_signals_df['date'], y=sell_signals_df['price'], mode='markers', name='Venda',
                                  marker=dict(color='red', size=10, symbol='triangle-down')))
    fig.update_layout(
        title=f'Backtest: {ticker}', xaxis_title='Data', yaxis_title='Preço',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Formatação de Moeda ---
def format_currency(value):
    try:
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "R$ 0,00"

# ==============================================================================
# --- Interface Principal do Streamlit ---
# ==============================================================================

st.title("📈 Simulador de Estratégia EMA RSI Diário")
st.markdown("Teste a estratégia de cruzamento de EMA sobre RSI diário com parâmetros configuráveis.")
st.divider()

# --- Barra Lateral (Sidebar) para Configurações e Watchlist ---
with st.sidebar:
    st.header("⚙️ Configurações")

    # Parâmetros da Estratégia
    st.subheader("Parâmetros da Estratégia")
    cfg_rsi_len = st.number_input("Período RSI", min_value=2, max_value=50, value=2, step=1)
    cfg_ema_len = st.number_input("Período EMA sobre RSI", min_value=2, max_value=50, value=2, step=1)
    cfg_exit_days = st.number_input("Dias para Saída por Tempo", min_value=1, max_value=30, value=4, step=1)

    # Período do Backtest
    st.subheader("Período do Backtest")
    today = datetime.now().date()
    one_year_ago = today - timedelta(days=365)
    cfg_start_date = st.date_input("Data Inicial", value=one_year_ago, max_value=today - timedelta(days=1))
    cfg_end_date = st.date_input("Data Final", value=today, min_value=cfg_start_date + timedelta(days=1), max_value=today)

    # Capital Inicial
    st.subheader("Financeiro")
    cfg_initial_capital = st.number_input("Capital Inicial", min_value=1.0, value=1000.0, step=100.0, format="%.2f")

    st.divider()

    # --- Watchlist ---
    st.header("⭐ Lista de Monitoramento")
    st.caption("Ativos salvos nesta sessão.")

    # Exibe a lista e botões de remover
    if not st.session_state.watchlist:
        st.info("Nenhum ativo na lista.")
    else:
        # Usar um loop reverso para evitar problemas de índice ao remover
        for i in range(len(st.session_state.watchlist) - 1, -1, -1):
            ticker_in_list = st.session_state.watchlist[i]
            col1_watch, col2_watch = st.columns([0.7, 0.3]) # Coluna p/ nome, coluna p/ botão
            with col1_watch:
                st.write(ticker_in_list)
            with col2_watch:
                # Botão Remover com chave única para cada item
                if st.button("Remover", key=f"remove_{ticker_in_list}_{i}", type="secondary", use_container_width=True):
                    removed_ticker = st.session_state.watchlist.pop(i)
                    st.toast(f"{removed_ticker} removido da lista.")
                    st.rerun() # Força re-execução para atualizar a lista visualmente

    st.markdown("[Limpar Lista](?clear_watchlist=true)", unsafe_allow_html=True) # Link para limpar (ver lógica abaixo)

# Lógica para limpar a watchlist via query param
if st.query_params.get("clear_watchlist") == "true":
    st.session_state.watchlist = []
    st.toast("Lista de monitoramento limpa.")
    st.query_params.clear() # Limpa o query param para evitar re-limpeza
    st.rerun()


# --- Área Principal ---
col_input, col_button = st.columns([0.8, 0.2])

with col_input:
    ticker_input = st.text_input("Código do Ativo (ex: PETR4.SA, AAPL, BTC-USD):",
                                 value=st.session_state.get('last_ticker_simulated', "PETR4.SA"), # Lembra último ticker
                                 placeholder="Digite o ticker..."
                                 ).upper()
with col_button:
    st.markdown(" ", unsafe_allow_html=True) # Espaço para alinhar botão
    simulate_button = st.button("Simular Estratégia", type="primary", use_container_width=True)


# --- Execução da Simulação ---
if simulate_button:
    if ticker_input:
        # Validar datas
        if cfg_start_date >= cfg_end_date:
            st.error("Erro: A Data Inicial deve ser anterior à Data Final.")
        else:
            with st.spinner(f"Executando backtest para {ticker_input}... Aguarde!"):
                # Chama a função de backtest com os parâmetros da sidebar
                results = run_strategy_backtest(
                    ticker=ticker_input,
                    start_date=cfg_start_date,
                    end_date=cfg_end_date,
                    initial_capital=cfg_initial_capital,
                    rsi_len=cfg_rsi_len,
                    ema_len=cfg_ema_len,
                    exit_days=cfg_exit_days
                )
                st.session_state['backtest_results'] = results # Salva na sessão
                st.session_state['last_ticker_simulated'] = ticker_input # Lembra o ticker
    else:
        st.warning("Por favor, insira um código de ativo.")
        st.session_state['backtest_results'] = None


# --- Exibição dos Resultados ---
# Verifica se há resultados na sessão E se pertencem ao último ticker simulado
if st.session_state['backtest_results'] is not None and st.session_state['backtest_results']['ticker'] == st.session_state['last_ticker_simulated']:
    results = st.session_state['backtest_results']

    st.success(f"Backtest concluído para {results['ticker']}")

    # --- Botão Adicionar à Lista ---
    ticker_to_add = results['ticker']
    if ticker_to_add not in st.session_state.watchlist:
        if st.button(f"⭐ Adicionar {ticker_to_add} à Lista de Monitoramento"):
            st.session_state.watchlist.append(ticker_to_add)
            st.toast(f"{ticker_to_add} adicionado à lista!")
            st.rerun() # Atualiza a sidebar
    else:
        st.info(f"{ticker_to_add} já está na Lista de Monitoramento.")


    # Container para os resultados
    with st.container(border=True): # Adiciona borda ao redor dos resultados
        # Seção de Métricas
        st.subheader("📊 Métricas Principais")
        metrics = results['metrics']
        cols_metrics = st.columns(3)
        cols_metrics[0].metric("Rentabilidade Total", f"{metrics['totalReturnPercent']:.2f}%",
                               delta_color=("green" if metrics['totalReturnPercent'] >= 0 else "red"))
        cols_metrics[1].metric("Lucro/Prejuízo Total", format_currency(metrics['totalProfit']),
                               delta_color=("green" if metrics['totalProfit'] >= 0 else "red"))
        cols_metrics[2].metric("Nº de Trades", metrics['numberOfTrades'])

        cols_metrics2 = st.columns(3)
        cols_metrics2[0].metric("Capital Inicial", format_currency(metrics['initialCapital']))
        cols_metrics2[1].metric("Capital Final", format_currency(metrics['finalEquity']))
        cols_metrics2[2].metric("Período Analisado", f"{metrics['startDate']} a {metrics['endDate']}")


        # Seção do Gráfico com Expander
        with st.expander("📈 Ver Gráfico de Velas com Sinais", expanded=True):
            fig = plot_results(results['chartData'], results['signals'], results['ticker'])
            st.plotly_chart(fig, use_container_width=True)

        # Seção da Tabela de Trades com Expander
        with st.expander("📜 Ver Histórico de Operações", expanded=False):
            trades_df = results['trades']
            if not trades_df.empty:
                trades_df_display = trades_df.copy()
                # Formatação condicional pode ser adicionada aqui se necessário
                trades_df_display['profit'] = trades_df_display['profit'].apply(format_currency)
                trades_df_display['entryPrice'] = trades_df_display['entryPrice'].map('{:.2f}'.format)
                trades_df_display['exitPrice'] = trades_df_display['exitPrice'].map('{:.2f}'.format)
                trades_df_display['returnPercent'] = trades_df_display['returnPercent'].map('{:.2f}%'.format)
                st.dataframe(trades_df_display[[
                    'entryDate', 'entryPrice', 'exitDate', 'exitPrice',
                    'daysHeld', 'profit', 'returnPercent', 'exitReason'
                ]], use_container_width=True)
            else:
                st.info("Nenhuma operação realizada no período.")

# Mensagem se nenhum resultado estiver pronto para exibição
elif simulate_button and st.session_state['backtest_results'] is None:
     st.warning("Não foi possível gerar resultados. Verifique as mensagens de erro acima.")
