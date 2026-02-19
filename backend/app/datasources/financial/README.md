# Financial Datasources

Implemented adapters:

1. Tushare financial adapter.
2. Eastmoney financial adapter.

The service layer uses fallback order:

1. Tushare (token required)
2. Eastmoney
3. Deterministic mock fallback
