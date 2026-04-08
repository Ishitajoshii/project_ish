# CircuitRL UI

Vite + React frontend for the CircuitRL benchmark console.

## Commands

```bash
npm install
npm run dev
npm run build
npm run lint
```

## Development

The dev server runs on `http://127.0.0.1:5173` and proxies `/api/*` to the FastAPI backend on `http://127.0.0.1:8000`.

The frontend code uses:

- `src/circuit_rl_app.tsx` for root state and playback control
- `src/components/` for modular UI panels
- `src/api_client.ts` for backend requests
- `src/ui_types.ts` for shared payload types
- `src/ui_helpers.ts` for formatting and chart helpers
