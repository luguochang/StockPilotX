# Datasources Module

This module provides a dedicated integration layer for all external market
data providers used by StockPilotX.

## Design Goals

1. Keep adapter logic isolated from business services.
2. Normalize data from heterogeneous upstream APIs.
3. Provide a stable factory API for service wiring.
4. Keep source-specific failures local and observable.

## Round-AC Scope

Round-AC only scaffolds the module and factory entry points. Existing runtime
implementations still reuse `backend.app.data.sources` internals to avoid
breaking behavior while migration is in progress.

## Planned Submodules

1. `base`: shared adapter protocols, HTTP client, utility helpers.
2. `quote`: market quote sources.
3. `financial`: financial statement and factor sources.
4. `news`: market news and event intelligence sources.
5. `research`: research report sources.
6. `macro`: macroeconomic indicator sources.
7. `fund`: fund data sources.

