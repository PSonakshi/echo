# Implementation Plan: Crypto Narrative Pulse Tracker

## Overview

This implementation builds on the existing Pathway RAG template. Phase 1 (Core) must be completed first, followed by Phase 2 (Demo-Ready) for a working demo. Phases 3-4 are stretch goals.

**Base Template:** Pathway RAG template (already cloned)

- `app.py` - Main application entry point
- `app.yaml` - YAML configuration for RAG components
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies

## Tasks

- [x] 1. Adapt Project Structure for Crypto Pulse Tracker
  - Extend requirements.txt with additional dependencies (vaderSentiment, python-telegram-bot, streamlit, requests)
  - Create .env.example with required environment variables (TELEGRAM_TOKEN, TRACKED_COIN, etc.)
  - Create docker-compose.yml to orchestrate pipeline, telegram bot, dashboard, and simulator services
  - Update app.yaml to configure crypto-specific RAG settings
  - _Requirements: 13.1, 13.4, 13.5_

- [x] 2. Core Data Models and Schema
  - [x] 2.1 Create schemas.py with MessageSchema and PriceSchema
    - Define MessageSchema with fields (message_id, text, author_id, author_followers, timestamp, tags, engagement_count, source_platform)
    - Define PriceSchema with fields (coin_symbol, price_usd, timestamp, volume_24h)
    - Use Pathway's pw.Schema pattern from template
    - _Requirements: 1.1, 4.1_
  - [ ]* 2.2 Write property test for MessageSchema round-trip
    - **Property 1: Message Schema Validation Round-Trip**
    - **Validates: Requirements 1.1, 1.2**
  - [x] 2.3 Implement schema validation with error handling
    - Validate required fields
    - Return structured error messages for invalid payloads
    - _Requirements: 1.3_

- [x] 3. Hype Simulator (Primary Data Source)
  - [x] 3.1 Create simulator/hype_simulator.py
    - Define phase configuration (seed 10%, growth 40%, peak 30%, decline 20%)
    - Implement message generation with phase-appropriate sentiment and phrases
    - Include influencer accounts with high follower counts
    - _Requirements: 10.1, 10.3, 10.4_
  - [x] 3.2 Add price data generation to simulator
    - Generate correlated price data following hype cycle phases
    - _Requirements: 10.5_
  - [x] 3.3 Implement HTTP webhook sender in simulator
    - Send messages to pipeline webhook endpoint using requests library
    - Configurable duration (3-5 mins) and message count (200)
    - _Requirements: 10.2_
  - [ ]* 3.4 Write property test for simulator phase sentiment

    - **Property 22: Hype Simulator Phase Sentiment**
    - **Validates: Requirements 10.3**

- [x] 4. Sentiment Analysis Pipeline
  - [x] 4.1 Create transforms/sentiment.py with SentimentAnalyzer class
    - Use VADER sentiment analyzer (vaderSentiment library)
    - Add crypto-specific lexicon (moon, pump, dump, rug, bullish, bearish, hodl, fomo, fud)
    - Return score in [-1, 1] range
    - _Requirements: 3.1_
  - [ ]* 4.2 Write property test for sentiment score range
    - **Property 5: Sentiment Score Range**
    - **Validates: Requirements 3.1**
  - [x] 4.3 Integrate sentiment into Pathway pipeline
    - Use pw.apply() to add sentiment scores to message stream
    - Use pw.temporal.sliding() with 5-min duration, 1-min hop for velocity
    - Use pw.reducers.avg() for velocity aggregation
    - _Requirements: 3.2, 3.3, 3.4, 3.5_

- [x] 5. Basic Pulse Score Calculator
  - [x] 5.1 Create transforms/pulse_score.py with PulseScoreCalculator class
    - Implement scoring formula: sentiment (0-4) + phrase (0-3) + influencer (0-3) - divergence (0-1)
    - Clamp output to [1, 10] range using max(1.0, min(10.0, score))
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9_
  - [ ]* 5.2 Write property test for pulse score calculation
    - **Property 15: Pulse Score Calculation**
    - **Validates: Requirements 7.1-7.9**

- [x] 6. Checkpoint - Core Pipeline Working
  - Ensure simulator can send messages to pipeline
  - Verify sentiment analysis produces valid scores
  - Verify pulse score calculation works
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. HTTP Webhook Connector for Messages
  - [x] 7.1 Create connectors/webhook.py with Pathway HTTP connector
    - Use pw.io.http.rest_connector() following template patterns
    - Configure for MessageSchema ingestion
    - _Requirements: 1.5_
  - [x] 7.2 Add tag filtering to pipeline
    - Filter messages by configurable tag patterns (e.g., #Solana, $MEME)
    - Use pw.Table.filter() with tag matching logic
    - _Requirements: 1.4_

- [x] 8. Price Data Integration
  - [x] 8.1 Create connectors/price_fetcher.py
    - Fetch from CoinGecko API or use simulated data from simulator
    - Implement caching with TTL for rate limit handling (50 calls/min)
    - _Requirements: 4.1, 4.7_
  - [x] 8.2 Add price delta calculation to pipeline
    - Use pw.temporal.sliding() with standard 5-min window
    - Calculate percentage change: (end - start) / start * 100
    - _Requirements: 4.2_
  - [x] 8.3 Create transforms/divergence.py for divergence detection
    - Detect bearish divergence: sentiment > 0.5 AND price_delta < -2%
    - Detect bullish divergence: sentiment < -0.5 AND price_delta > 2%
    - _Requirements: 4.4, 4.5_
  - [ ]* 8.4 Write property test for divergence detection
    - **Property 8: Divergence Detection**
    - **Validates: Requirements 4.4, 4.5**

- [x] 9. Document Store and RAG Integration
  - [x] 9.1 Extend app.yaml for crypto message indexing
    - Configure DocumentStore to index messages with metadata (sentiment, influence_score, source_platform)
    - Use existing SentenceTransformerEmbedder from template
    - _Requirements: 8.1, 8.2_
  - [x] 9.2 Create rag/live_metrics.py for state management
    - Create shared state dict for pulse_score, trending_phrases, influencer_consensus, divergence_status
    - Use pw.io.subscribe() to update state from pipeline outputs
    - _Requirements: 8.4_
  - [x] 9.3 Create rag/crypto_rag.py extending template's question answerer
    - Build context-enriched prompt with live metrics
    - Retrieve top 15 relevant messages
    - Use Ollama (from template config) for LLM responses
    - _Requirements: 8.3, 8.4_

- [ ] 10. Telegram Bot
  - [ ] 10.1 Create bot/telegram_bot.py with TelegramAlertBot class
    - Use python-telegram-bot library
    - Send alerts for score >= 7 (🚀 Strong Buy Signal) and score <= 3 (❄️ Cooling Off)
    - _Requirements: 9.1, 9.2, 9.5_
  - [ ] 10.2 Add divergence warning alerts
    - Send ⚠️ warning when bearish/bullish divergence detected
    - _Requirements: 9.3_
  - [ ] 10.3 Implement /query command handler
    - Handle user queries via RAG system from task 9.3
    - Return formatted response with pulse score, phrases, and risk assessment
    - _Requirements: 9.4_

- [ ] 11. Checkpoint - Demo-Ready
  - Verify end-to-end flow: simulator → pipeline → alerts
  - Test Telegram bot alerts and queries
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Streamlit Dashboard
  - [ ] 12.1 Create dashboard/streamlit_app.py with basic layout
    - Display real-time pulse score with large number indicator
    - _Requirements: 11.1_
  - [ ] 12.2 Add pulse score line chart
    - Use st.line_chart() to plot score history over time
    - Auto-refresh every 5 seconds
    - _Requirements: 11.1_
  - [ ] 12.3 Add trending phrases display
    - Show top 5 phrases as a list or simple word cloud
    - Update every 30 seconds
    - _Requirements: 11.2_
  - [ ] 12.4 Add price and correlation status panel
    - Display current price and divergence status with color coding
    - _Requirements: 11.4_

- [ ] 13. Phrase Clustering (Phase 3)
  - [ ] 13.1 Create transforms/phrase_clusterer.py
    - Extract bigrams and trigrams using simple tokenization
    - Filter stopwords and short words (< 3 chars)
    - _Requirements: 5.1_
  - [ ] 13.2 Add trending phrase tracking to pipeline
    - Use pw.temporal.sliding() with 10-min window
    - Use pw.groupby() and pw.reducers.count() for frequency
    - Mark phrases with frequency >= 5 as trending
    - _Requirements: 5.2, 5.3, 5.4_

- [ ] 14. Influencer Tracking (Phase 3)
  - [ ] 14.1 Create transforms/influence.py with influence score calculation
    - Formula: (followers × 0.6) + (engagement × 0.4)
    - Use pw.apply() to add influence_score to message stream
    - _Requirements: 6.1_
  - [ ] 14.2 Add influencer classification to pipeline
    - Use pw.Table.filter() to classify authors with score > 10000 as influencers
    - _Requirements: 6.2_
  - [ ] 14.3 Add influencer consensus tracking
    - Use pw.temporal.tumbling() with 10-min windows
    - Track bullish (sentiment > 0.3) vs bearish (sentiment < -0.3) counts
    - Calculate ratio: bullish_count / (bullish_count + bearish_count)
    - _Requirements: 6.3, 6.4_

- [ ] 15. Performance Monitoring (Phase 3)
  - [ ] 15.1 Add latency tracking
    - Measure end-to-end latency from ingestion to alert
    - Log warning if latency > 5 seconds
    - _Requirements: 12.1, 12.4_
  - [ ] 15.2 Add throughput metrics
    - Track messages per second
    - Display in dashboard
    - _Requirements: 12.2, 12.3_

- [ ] 16. Checkpoint - Impressive Features Complete
  - Verify phrase clustering and influencer tracking work
  - Test full pulse score with all components
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. Discord Connector (Stretch Goal)
  - [ ] 17.1 Create connectors/discord_connector.py
    - Use pw.io.python.read() pattern from Pathway docs
    - Transform Discord webhook messages to MessageSchema
    - _Requirements: 2.1, 2.2, 2.3_
  - [ ] 17.2 Add retry logic with exponential backoff
    - Retry 3 times on webhook failure
    - Use exponential backoff (1s, 2s, 4s delays)
    - _Requirements: 2.4_

- [ ] 18. Final Integration and Polish
  - [ ] 18.1 Complete Docker deployment
    - Verify all services start correctly
    - Test fault tolerance (container restart)
    - _Requirements: 13.2, 13.3, 13.4_
  - [ ] 18.2 Add influencer leaderboard to dashboard
    - Show top contributors and their sentiment
    - _Requirements: 11.3_
  - [ ] 18.3 Add RAG relevance score logging
    - Log retrieval scores for monitoring
    - _Requirements: 12.5_

- [ ] 19. Final Checkpoint - All Tests Pass
  - Run all unit tests
  - Run all property tests
  - Verify demo flow works end-to-end
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based tests
- Phase 1 (Tasks 1-6): Core functionality - MUST complete
- Phase 2 (Tasks 7-12): Demo-ready features - HIGH priority
- Phase 3 (Tasks 13-16): Impressive features - MEDIUM priority
- Phase 4 (Tasks 17-19): Stretch goals - LOW priority
- **Use existing template patterns** - app.yaml syntax, pw.xpacks.llm imports, etc.
- **LLM**: Use Ollama (already configured in app.yaml) - no API costs
- **Embeddings**: Use SentenceTransformerEmbedder from template (no API key needed)
- **In case of syntax conflicts**: Template syntax takes precedence
- Checkpoints ensure incremental validation
