"""
PulseTrade AI — Backend Server
================================
LOCAL USE:
  pip install flask flask-cors yfinance pandas groq requests gunicorn
  python server.py
  Open http://localhost:5000

RAILWAY HOSTING:
  Set environment variables on Railway:
  GROQ_API_KEY  = your groq key
  NEWS_API_KEY  = your newsapi key
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import math
from groq import Groq
import requests as req
import os

# ─────────────────────────────────────────────
#  API KEYS — reads from environment variables
#  For local use: set these in your terminal OR
#  paste keys directly here for testing only
# ─────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "paste_your_groq_key_here")
NEWS_API_KEY  = os.environ.get("NEWS_API_KEY",  "paste_your_newsapi_key_here")

# ─────────────────────────────────────────────
#  AI MODEL — update this one line if Groq
#  ever changes their model names
# ─────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────
#  HELPER — safely convert to float (NaN → None)
# ─────────────────────────────────────────────
def safe(val, decimals=2):
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, decimals)
    except Exception:
        return None

app = Flask(__name__, static_folder=".")

# Allow all origins — needed for Render hosting
CORS(app, origins="*", supports_credentials=True)

# Only create Groq client if key is available
client = None
if GROQ_API_KEY and GROQ_API_KEY != "paste_your_groq_key_here":
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("Groq client ready")
    except Exception as e:
        print(f"Groq client error: {e}")


# ─────────────────────────────────────────────
#  SERVE FRONTEND
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "groq": client is not None})

@app.route("/manifest.json")
def manifest():
    return send_from_directory(".", "manifest.json")

@app.route("/sw.js")
def service_worker():
    response = send_from_directory(".", "sw.js")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Service-Worker-Allowed"] = "/"
    return response

@app.route("/icon-192.png")
def icon192():
    return send_from_directory(".", "icon-192.png")

@app.route("/icon-512.png")
def icon512():
    return send_from_directory(".", "icon-512.png")


# ─────────────────────────────────────────────
#  STOCK DATA ENDPOINT
# ─────────────────────────────────────────────
@app.route("/api/stock/<ticker>")
def get_stock(ticker):
    try:
        period = request.args.get("period", "6mo")
        stock  = yf.Ticker(ticker.upper())
        df     = stock.history(period=period)

        if df.empty:
            return jsonify({"error": f"No data found for {ticker}"}), 404

        # Calculate indicators
        df["MA20"]      = df["Close"].rolling(20).mean()
        df["MA50"]      = df["Close"].rolling(50).mean()
        delta           = df["Close"].diff()
        gain            = delta.clip(lower=0).rolling(14).mean()
        loss            = (-delta.clip(upper=0)).rolling(14).mean()
        df["RSI"]       = 100 - (100 / (1 + gain / loss))
        df["Change"]    = df["Close"].pct_change() * 100
        df["Vol20"]     = df["Change"].rolling(20).std()

        # MACD
        exp12           = df["Close"].ewm(span=12).mean()
        exp26           = df["Close"].ewm(span=26).mean()
        df["MACD"]      = exp12 - exp26
        df["Signal"]    = df["MACD"].ewm(span=9).mean()

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        # Build chart data (last 180 points max)
        chart_df = df.tail(180).copy()
        chart_df = chart_df.where(pd.notnull(chart_df), None)

        chart_data = []
        for idx, row in chart_df.iterrows():
            chart_data.append({
                "date":        idx.strftime("%Y-%m-%d"),
                "open":        safe(row["Open"]),
                "high":        safe(row["High"]),
                "low":         safe(row["Low"]),
                "close":       safe(row["Close"]),
                "volume":      int(row["Volume"]) if row["Volume"] is not None and not math.isnan(float(row["Volume"])) else None,
                "ma20":        safe(row["MA20"]),
                "ma50":        safe(row["MA50"]),
                "rsi":         safe(row["RSI"]),
                "macd":        safe(row["MACD"], 4),
                "macd_signal": safe(row["Signal"], 4),
            })

        # Summary stats
        price       = float(latest["Close"])
        prev_price  = float(prev["Close"])
        change_pct  = ((price - prev_price) / prev_price) * 100
        high_period = float(df["High"].max())
        low_period  = float(df["Low"].min())

        info = {}
        try:
            info = stock.info or {}
        except Exception:
            pass

        return jsonify({
            "ticker":      ticker.upper(),
            "name":        info.get("longName", ticker.upper()),
            "price":       safe(latest["Close"]),
            "change":      safe(change_pct),
            "high":        safe(high_period),
            "low":         safe(low_period),
            "ma20":        safe(latest["MA20"]),
            "ma50":        safe(latest["MA50"]),
            "rsi":         safe(latest["RSI"]),
            "volatility":  safe(latest["Vol20"]),
            "volume":      int(latest["Volume"]) if not math.isnan(float(latest["Volume"])) else None,
            "market_cap":  info.get("marketCap"),
            "sector":      info.get("sector", ""),
            "chart":       chart_data,
            "period":      period,
            "days":        len(df),
            "change_7d":   safe(((float(latest["Close"]) - float(df["Close"].iloc[-7]))  / float(df["Close"].iloc[-7]))  * 100) if len(df) >= 7  else None,
            "change_30d":  safe(((float(latest["Close"]) - float(df["Close"].iloc[-30])) / float(df["Close"].iloc[-30])) * 100) if len(df) >= 30 else None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
#  AI ANALYSIS ENDPOINT
# ─────────────────────────────────────────────
@app.route("/api/ai/<ticker>")
def get_ai(ticker):
    try:
        if not client:
            return jsonify({"error": "AI service not configured. Add GROQ_API_KEY in Render environment variables."}), 503

        period = request.args.get("period", "6mo")
        stock  = yf.Ticker(ticker.upper())
        df     = stock.history(period=period)

        if df.empty:
            return jsonify({"error": "No data"}), 404

        df["MA20"]   = df["Close"].rolling(20).mean()
        df["MA50"]   = df["Close"].rolling(50).mean()
        delta        = df["Close"].diff()
        gain         = delta.clip(lower=0).rolling(14).mean()
        loss         = (-delta.clip(upper=0)).rolling(14).mean()
        df["RSI"]    = 100 - (100 / (1 + gain / loss))
        df["Change"] = df["Close"].pct_change() * 100
        df["Vol20"]  = df["Change"].rolling(20).std()

        latest     = df.iloc[-1]
        price      = float(latest["Close"])
        rsi        = float(latest["RSI"])
        ma20       = float(latest["MA20"])
        ma50       = float(latest["MA50"])
        volatility = float(latest["Vol20"])
        change_7d  = ((price - float(df["Close"].iloc[-7]))  / float(df["Close"].iloc[-7]))  * 100 if len(df) >= 7  else 0
        change_30d = ((price - float(df["Close"].iloc[-30])) / float(df["Close"].iloc[-30])) * 100 if len(df) >= 30 else 0

        prompt = f"""You are an expert stock market analyst. Analyse this data and respond ONLY in the exact JSON format shown. No extra text, no markdown, no backticks.

Stock: {ticker.upper()}
Price: ${price:.2f}
7-day change: {change_7d:+.2f}%
30-day change: {change_30d:+.2f}%
RSI: {rsi:.1f}
MA20: ${ma20:.2f}
MA50: ${ma50:.2f}
Volatility: {volatility:.2f}%
Period high: ${df["High"].max():.2f}
Period low: ${df["Low"].min():.2f}

Respond with ONLY this JSON:
{{
  "recommendation": "STRONG BUY or BUY or HOLD or SELL or STRONG SELL",
  "confidence": <number 1-99>,
  "rise_chance": <number 1-99>,
  "drop_chance": <number that makes rise_chance + drop_chance = 100>,
  "risk": "LOW or MEDIUM or HIGH",
  "entry_price": <suggested entry price number>,
  "target_price": <suggested target price number>,
  "stop_loss": <suggested stop loss price number>,
  "summary": "<2-3 sentences explaining what the stock has been doing in plain English>",
  "prediction": "<1-2 sentences on what might happen short term>",
  "reason": "<1 sentence explaining the recommendation>",
  "sentiment": "BULLISH or NEUTRAL or BEARISH"
}}"""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
            timeout=30,
        )

        text = response.choices[0].message.content.strip()
        # Clean up any markdown if model adds it
        text = text.replace("```json", "").replace("```", "").strip()

        import json
        data = json.loads(text)
        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
#  WATCHLIST SEARCH ENDPOINT
# ─────────────────────────────────────────────
@app.route("/api/quote/<ticker>")
def get_quote(ticker):
    try:
        stock  = yf.Ticker(ticker.upper())
        df     = stock.history(period="5d")
        if df.empty:
            return jsonify({"error": "Not found"}), 404

        price      = float(df["Close"].iloc[-1])
        prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else price
        change     = ((price - prev_price) / prev_price) * 100

        info = {}
        try:
            info = stock.info or {}
        except Exception:
            pass

        return jsonify({
            "ticker": ticker.upper(),
            "name":   info.get("longName", ticker.upper()),
            "price":  round(price, 2),
            "change": round(change, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
#  MARKET MOVERS ENDPOINT
# ─────────────────────────────────────────────
@app.route("/api/movers")
def get_movers():
    tickers = ["AAPL", "TSLA", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BTC-USD", "ETH-USD", "AMD"]
    results = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            df    = stock.history(period="5d")
            if df.empty or len(df) < 2:
                continue
            price      = float(df["Close"].iloc[-1])
            prev_price = float(df["Close"].iloc[-2])
            change     = ((price - prev_price) / prev_price) * 100
            results.append({
                "ticker": t,
                "price":  round(price, 2),
                "change": round(change, 2),
            })
        except Exception:
            continue
    return jsonify(results)


# ─────────────────────────────────────────────
#  NEWS ENDPOINT — General market news
# ─────────────────────────────────────────────
@app.route("/api/news")
def get_news():
    try:
        category = request.args.get("category", "general")

        queries = {
            "general":  "stock market finance investing",
            "stocks":   "stocks equities shares earnings",
            "crypto":   "bitcoin cryptocurrency crypto blockchain",
            "economy":  "economy federal reserve interest rates inflation GDP",
            "earnings": "earnings report quarterly results revenue profit",
        }
        query = queries.get(category, queries["general"])

        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}"
            f"&language=en"
            f"&sortBy=publishedAt"
            f"&pageSize=20"
            f"&apiKey={NEWS_API_KEY}"
        )
        response = req.get(url, timeout=10)
        data     = response.json()

        if data.get("status") != "ok":
            return jsonify({"error": data.get("message", "News fetch failed")}), 500

        articles = []
        for a in data.get("articles", []):
            if not a.get("title") or a.get("title") == "[Removed]":
                continue
            articles.append({
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "url":         a.get("url", ""),
                "source":      a.get("source", {}).get("name", ""),
                "published":   a.get("publishedAt", ""),
                "image":       a.get("urlToImage", ""),
            })

        # Add AI sentiment tags via Groq
        articles = tag_sentiment(articles)
        return jsonify(articles)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
#  NEWS ENDPOINT — Stock specific news
# ─────────────────────────────────────────────
@app.route("/api/news/<ticker>")
def get_stock_news(ticker):
    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={ticker}+stock"
            f"&language=en"
            f"&sortBy=publishedAt"
            f"&pageSize=10"
            f"&apiKey={NEWS_API_KEY}"
        )
        response = req.get(url, timeout=10)
        data     = response.json()

        if data.get("status") != "ok":
            return jsonify({"error": data.get("message", "News fetch failed")}), 500

        articles = []
        for a in data.get("articles", []):
            if not a.get("title") or a.get("title") == "[Removed]":
                continue
            articles.append({
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "url":         a.get("url", ""),
                "source":      a.get("source", {}).get("name", ""),
                "published":   a.get("publishedAt", ""),
                "image":       a.get("urlToImage", ""),
            })

        articles = tag_sentiment(articles)
        return jsonify(articles)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
#  SENTIMENT TAGGING
# ─────────────────────────────────────────────
def tag_sentiment(articles):
    """Add bullish/bearish/neutral sentiment to each article using keywords."""
    bullish_words = ["surge", "soar", "rally", "gain", "jump", "rise", "record", "growth",
                     "profit", "beat", "upgrade", "buy", "bullish", "high", "strong", "positive"]
    bearish_words = ["crash", "fall", "drop", "decline", "plunge", "loss", "miss", "downgrade",
                     "sell", "bearish", "low", "weak", "warning", "risk", "recession", "cut"]

    for a in articles:
        text  = (a.get("title", "") + " " + a.get("description", "")).lower()
        bull  = sum(1 for w in bullish_words if w in text)
        bear  = sum(1 for w in bearish_words if w in text)
        if bull > bear:
            a["sentiment"]   = "BULLISH"
            a["impact"]      = min(10, bull * 2)
        elif bear > bull:
            a["sentiment"]   = "BEARISH"
            a["impact"]      = min(10, bear * 2)
        else:
            a["sentiment"]   = "NEUTRAL"
            a["impact"]      = 3

        # Format time
        pub = a.get("published", "")
        if pub:
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                a["published"] = dt.strftime("%b %d, %Y · %H:%M UTC")
            except Exception:
                pass

    return articles


# ─────────────────────────────────────────────
#  AI RECOMMENDATIONS ENDPOINT
#  Returns top picks for short term, long term
#  and best profit potential
# ─────────────────────────────────────────────
@app.route("/api/recommendations")
def get_recommendations():
    try:
        # Stocks to scan across all categories
        candidates = {
            "stocks":  ["AAPL","MSFT","NVDA","TSLA","AMZN","GOOGL","META","AMD","NFLX","JPM","V","MA","DIS","PYPL","UBER"],
            "crypto":  ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"],
            "etf":     ["SPY","QQQ","VTI","ARKK","VGT"],
        }
        all_tickers = candidates["stocks"] + candidates["crypto"] + candidates["etf"]

        # Fetch quick data for all tickers
        scored = []
        for ticker in all_tickers:
            try:
                stock = yf.Ticker(ticker)
                df    = stock.history(period="3mo")
                if df.empty or len(df) < 30:
                    continue

                df["MA20"]   = df["Close"].rolling(20).mean()
                df["MA50"]   = df["Close"].rolling(50).mean()
                delta        = df["Close"].diff()
                gain         = delta.clip(lower=0).rolling(14).mean()
                loss         = (-delta.clip(upper=0)).rolling(14).mean()
                df["RSI"]    = 100 - (100 / (1 + gain / loss))
                df["Change"] = df["Close"].pct_change() * 100
                df["Vol20"]  = df["Change"].rolling(20).std()

                latest     = df.iloc[-1]
                price      = safe(latest["Close"])
                ma20       = safe(latest["MA20"])
                ma50       = safe(latest["MA50"])
                rsi        = safe(latest["RSI"])
                vol        = safe(latest["Vol20"])
                change_1m  = safe(((float(latest["Close"]) - float(df["Close"].iloc[-22])) / float(df["Close"].iloc[-22])) * 100) if len(df) >= 22 else 0
                change_3m  = safe(((float(latest["Close"]) - float(df["Close"].iloc[0]))  / float(df["Close"].iloc[0]))  * 100)

                if not all([price, ma20, ma50, rsi, vol]):
                    continue

                # Scoring logic
                long_score  = 0
                short_score = 0
                profit_score= 0

                # Long term — stable uptrend, low volatility, price above MAs
                if price > ma20 > ma50:    long_score += 30
                if rsi > 40 and rsi < 65:  long_score += 20
                if vol < 2.5:              long_score += 20
                if change_3m > 5:          long_score += 20
                if change_3m > 15:         long_score += 10

                # Short term — momentum, RSI building, recent breakout
                if price > ma20:           short_score += 25
                if rsi > 50 and rsi < 70:  short_score += 25
                if change_1m > 3:          short_score += 25
                if change_1m > 8:          short_score += 15
                if vol > 1.5:              short_score += 10

                # Best profit — high momentum + higher volatility = bigger moves
                if change_1m > 5:          profit_score += 25
                if change_3m > 10:         profit_score += 25
                if rsi > 55:               profit_score += 20
                if vol > 2:                profit_score += 20
                if price > ma20 > ma50:    profit_score += 10

                # Determine category
                if ticker in candidates["crypto"]:
                    category = "Crypto"
                elif ticker in candidates["etf"]:
                    category = "ETF"
                else:
                    category = "Stock"

                scored.append({
                    "ticker":       ticker,
                    "price":        price,
                    "change_1m":    change_1m,
                    "change_3m":    change_3m,
                    "rsi":          rsi,
                    "volatility":   vol,
                    "long_score":   long_score,
                    "short_score":  short_score,
                    "profit_score": profit_score,
                    "category":     category,
                })
            except Exception:
                continue

        if not scored:
            return jsonify({"error": "Could not fetch market data"}), 500

        # Sort and pick top 5 for each category
        long_term  = sorted(scored, key=lambda x: x["long_score"],   reverse=True)[:5]
        short_term = sorted(scored, key=lambda x: x["short_score"],  reverse=True)[:5]
        top_profit = sorted(scored, key=lambda x: x["profit_score"], reverse=True)[:5]

        # Get AI summary for the top picks
        def ai_blurb(picks, mode):
            try:
                tickers_info = "\n".join([
                    f"{p['ticker']}: price=${p['price']}, 1M={p['change_1m']:+.1f}%, 3M={p['change_3m']:+.1f}%, RSI={p['rsi']:.0f}"
                    for p in picks
                ])
                prompt = f"""You are a stock analyst. Based on this data, write a single short sentence (max 15 words) explaining WHY each stock is recommended for {mode} investing. Be specific.

{tickers_info}

Respond ONLY in this exact JSON format, no extra text:
{{
  "{picks[0]['ticker']}": "reason here",
  "{picks[1]['ticker']}": "reason here",
  "{picks[2]['ticker']}": "reason here",
  "{picks[3]['ticker']}": "reason here",
  "{picks[4]['ticker']}": "reason here"
}}"""
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300,
                    timeout=30,
                )
                import json
                text = response.choices[0].message.content.strip()
                text = text.replace("```json","").replace("```","").strip()
                return json.loads(text)
            except Exception:
                return {}

        long_blurbs  = ai_blurb(long_term,  "long term")
        short_blurbs = ai_blurb(short_term, "short term")
        profit_blurbs= ai_blurb(top_profit, "maximum profit")

        # Add blurbs to results
        for p in long_term:
            p["reason"] = long_blurbs.get(p["ticker"], "Strong fundamentals and steady trend.")
        for p in short_term:
            p["reason"] = short_blurbs.get(p["ticker"], "Strong short-term momentum detected.")
        for p in top_profit:
            p["reason"] = profit_blurbs.get(p["ticker"], "High momentum and volatility for bigger moves.")

        return jsonify({
            "long_term":   long_term,
            "short_term":  short_term,
            "top_profit":  top_profit,
            "scanned":     len(scored),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ─────────────────────────────────────────────
#  FEAR & GREED ENDPOINT
# ─────────────────────────────────────────────
@app.route("/api/feargreed")
def get_fear_greed():
    try:
        # Scan major indices to calculate fear/greed
        tickers = ["SPY","QQQ","VIX=F","GLD","BTC-USD"]
        data    = {}
        for t in tickers:
            try:
                df = yf.Ticker(t).history(period="1mo")
                if not df.empty:
                    data[t] = df
            except Exception:
                continue

        score     = 50  # Start neutral
        signals   = []

        # SPY momentum
        if "SPY" in data and len(data["SPY"]) >= 20:
            spy    = data["SPY"]
            change = ((float(spy["Close"].iloc[-1]) - float(spy["Close"].iloc[-20])) / float(spy["Close"].iloc[-20])) * 100
            if change > 5:
                score += 15
                signals.append({"label":"Market Momentum","value":"Strong Uptrend","bull":True})
            elif change > 2:
                score += 8
                signals.append({"label":"Market Momentum","value":"Mild Uptrend","bull":True})
            elif change < -5:
                score -= 15
                signals.append({"label":"Market Momentum","value":"Strong Downtrend","bull":False})
            elif change < -2:
                score -= 8
                signals.append({"label":"Market Momentum","value":"Mild Downtrend","bull":False})
            else:
                signals.append({"label":"Market Momentum","value":"Sideways","bull":None})

        # QQQ tech strength
        if "QQQ" in data and len(data["QQQ"]) >= 10:
            qqq    = data["QQQ"]
            change = ((float(qqq["Close"].iloc[-1]) - float(qqq["Close"].iloc[-10])) / float(qqq["Close"].iloc[-10])) * 100
            if change > 3:
                score += 10
                signals.append({"label":"Tech Strength","value":"Bullish","bull":True})
            elif change < -3:
                score -= 10
                signals.append({"label":"Tech Strength","value":"Bearish","bull":False})
            else:
                signals.append({"label":"Tech Strength","value":"Neutral","bull":None})

        # Bitcoin as risk appetite indicator
        if "BTC-USD" in data and len(data["BTC-USD"]) >= 7:
            btc    = data["BTC-USD"]
            change = ((float(btc["Close"].iloc[-1]) - float(btc["Close"].iloc[-7])) / float(btc["Close"].iloc[-7])) * 100
            if change > 5:
                score += 10
                signals.append({"label":"Crypto Sentiment","value":"Risk-On","bull":True})
            elif change < -5:
                score -= 10
                signals.append({"label":"Crypto Sentiment","value":"Risk-Off","bull":False})
            else:
                signals.append({"label":"Crypto Sentiment","value":"Neutral","bull":None})

        # Gold as safe haven indicator
        if "GLD" in data and len(data["GLD"]) >= 10:
            gld    = data["GLD"]
            change = ((float(gld["Close"].iloc[-1]) - float(gld["Close"].iloc[-10])) / float(gld["Close"].iloc[-10])) * 100
            if change > 2:
                score -= 8  # Gold rising = fear
                signals.append({"label":"Safe Haven Demand","value":"High (Fearful)","bull":False})
            elif change < -1:
                score += 8  # Gold falling = confidence
                signals.append({"label":"Safe Haven Demand","value":"Low (Confident)","bull":True})
            else:
                signals.append({"label":"Safe Haven Demand","value":"Neutral","bull":None})

        # Clamp score
        score = max(0, min(100, score))

        # Label
        if score >= 80:   label = "Extreme Greed"
        elif score >= 60: label = "Greed"
        elif score >= 45: label = "Neutral"
        elif score >= 25: label = "Fear"
        else:             label = "Extreme Fear"

        # What it means for investors
        advice = {
            "Extreme Greed": "Market is very overheated. Consider taking some profits. Be cautious entering new positions.",
            "Greed":         "Market is bullish but be selective. Good time to review your positions.",
            "Neutral":       "Market is balanced. Good time to research and plan your next move.",
            "Fear":          "Market is nervous. Historically a good time to look for buying opportunities in quality stocks.",
            "Extreme Fear":  "Market is panicking. Strong buying opportunities may exist for patient long-term investors.",
        }

        return jsonify({
            "score":   round(score),
            "label":   label,
            "advice":  advice[label],
            "signals": signals,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
#  PORTFOLIO PRICE ENDPOINT
# ─────────────────────────────────────────────
@app.route("/api/portfolio/prices", methods=["POST"])
def get_portfolio_prices():
    try:
        holdings = request.json.get("holdings", [])
        results  = []
        for h in holdings:
            ticker = h.get("ticker","").upper()
            try:
                df    = yf.Ticker(ticker).history(period="5d")
                if df.empty:
                    continue
                price      = safe(df["Close"].iloc[-1])
                prev_price = safe(df["Close"].iloc[-2]) if len(df) > 1 else price
                change     = safe(((price - prev_price) / prev_price) * 100) if prev_price else 0
                results.append({
                    "ticker":    ticker,
                    "price":     price,
                    "change":    change,
                    "shares":    h.get("shares", 0),
                    "buy_price": h.get("buy_price", 0),
                })
            except Exception:
                continue
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
#  AI CHAT ENDPOINT
# ─────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def ai_chat():
    try:
        if not client:
            return jsonify({"error": "AI service not configured. Add GROQ_API_KEY in Render environment variables."}), 503

        messages  = request.json.get("messages", [])
        user_msg  = request.json.get("message", "")

        # Fetch quick market context
        context = ""
        try:
            spy_df  = yf.Ticker("SPY").history(period="5d")
            btc_df  = yf.Ticker("BTC-USD").history(period="5d")
            spy_chg = safe(((float(spy_df["Close"].iloc[-1]) - float(spy_df["Close"].iloc[-2])) / float(spy_df["Close"].iloc[-2])) * 100) if len(spy_df) > 1 else 0
            btc_chg = safe(((float(btc_df["Close"].iloc[-1]) - float(btc_df["Close"].iloc[-2])) / float(btc_df["Close"].iloc[-2])) * 100) if len(btc_df) > 1 else 0
            context = f"Today's market: S&P500 (SPY) is {spy_chg:+.2f}% today. Bitcoin is {btc_chg:+.2f}% today."
        except Exception:
            context = "Live market data temporarily unavailable."

        system = f"""You are a friendly, helpful stock market assistant called PulseAI inside the PulseTrade app.
You help everyday people — including beginners — understand the stock market and make informed decisions.

{context}

Rules:
- Always be clear, friendly and use simple language
- Never give direct buy/sell orders — give balanced information instead
- Always remind users this is educational, not financial advice
- Keep responses concise — 3 to 5 sentences max unless they ask for more detail
- Use emojis occasionally to keep it friendly
- If asked about a specific stock, give a balanced view of pros and cons"""

        chat_messages = [{"role":"system","content":system}]
        for m in messages[-10:]:  # Last 10 messages for context
            chat_messages.append({"role": m["role"], "content": m["content"]})
        chat_messages.append({"role":"user","content":user_msg})

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=chat_messages,
            temperature=0.5,
            max_tokens=400,
            timeout=30,
        )

        return jsonify({"reply": response.choices[0].message.content.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n  PulseTrade AI starting on port {port}")
    print(f"  Groq AI: {'ready' if client else 'not configured'}")
    print(f"  NewsAPI: {'ready' if NEWS_API_KEY != 'paste_your_newsapi_key_here' else 'not configured'}\n")
    app.run(debug=debug, host="0.0.0.0", port=port)
