<#
.SYNOPSIS
    Automates the full Fantacalcio Analysis pipeline:
    setup → migrations → scraping → ML training → frontend
.DESCRIPTION
    Runs every step in order with minimal prompts.
    Use -SkipX flags to skip individual phases.
.PARAMETER SkipBuild
    Skip Docker image build.
.PARAMETER SkipScraper
    Skip match stats scraping.
.PARAMETER SkipStats
    Skip per-season stats scraping.
.PARAMETER SkipFantavoto
    Skip fantavoto CSV generation.
.PARAMETER SkipML
    Skip ML pipeline training.
.PARAMETER SkipMapping
    Skip Fantacalcio↔FotMob ID mapping.
.PARAMETER SkipMantra
    Skip MANTRA scoring computation.
.PARAMETER SkipFrontend
    Skip Angular frontend startup.
.PARAMETER League
    League name(s) to scrape (default: "Serie A").
.PARAMETER Seasons
    Season(s) for match stats (default: "2026-2027").
.PARAMETER StatsSeasons
    Season(s) for per-season stats (default: "2023-2024,2024-2025,2025-2026,2026-2027").
#>

param(
    [switch]$SkipBuild,
    [switch]$SkipScraper,
    [switch]$SkipStats,
    [switch]$SkipFantavoto,
    [switch]$SkipML,
    [switch]$SkipMapping,
    [switch]$SkipMantra,
    [switch]$SkipFrontend,
    [string]$League = "Serie A",
    [string]$Seasons = "2026-2027",
    [string]$StatsSeasons = "2023-2024,2024-2025,2025-2026,2026-2027"
)

$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot

function Log($Msg) {
    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] ⚡ $Msg" -ForegroundColor Cyan
}

function CheckLastExitCode {
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        Write-Host "❌ Command failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ── 0. Prerequisites ──────────────────────────────────────────────────────
Log "STEP 0/12 — Verifica prerequisiti"

if (-not (Test-Path "$ROOT\.env")) {
    Log "⚠️  .env non trovato. Lo creo da .env.example..."
    Copy-Item "$ROOT\.env.example" "$ROOT\.env"
    Write-Host "   ✏️  Ricordati di impostare POSTGRES_PASSWORD in .env!" -ForegroundColor Yellow
}

# Check Docker
try { docker ps | Out-Null } catch {
    Write-Host "❌ Docker non in esecuzione. Avvialo e riprova." -ForegroundColor Red
    exit 1
}

# Check frontend deps
if (-not (Test-Path "$ROOT\frontend\node_modules")) {
    Log "⏳ Installazione dipendenze frontend..."
    Push-Location "$ROOT\frontend"
    npm install | Out-Null
    Pop-Location
}

# ── 1. Build ─────────────────────────────────────────────────────────────
if (-not $SkipBuild) {
    Log "STEP 1/12 — Build immagini Docker"
    docker compose build
    CheckLastExitCode
} else { Log "⏭️  STEP 1 — Build saltato" }

# ── 2. Start services ────────────────────────────────────────────────────
Log "STEP 2/12 — Avvio servizi (db, redis, api)"
docker compose up -d db redis api
CheckLastExitCode

# Wait for healthy
Start-Sleep -Seconds 5
$retries = 0
do {
    $status = docker compose ps --format json | ConvertFrom-Json
    $dbOk = ($status | Where-Object Service -eq db).Status -like "*healthy*"
    $redisOk = ($status | Where-Object Service -eq redis).Status -like "*healthy*"
    if ($dbOk -and $redisOk) { break }
    $retries++
    Start-Sleep -Seconds 3
} while ($retries -lt 20)

if ($retries -ge 20) { Write-Host "❌ Timeout attesa servizi" -ForegroundColor Red; exit 1 }
Log "✅ db e redis healthy"

# ── 3. Init DB + Migrations ──────────────────────────────────────────────
Log "STEP 3/12 — Inizializzazione DB (init.sql + migrations)"
if (Test-Path "$ROOT\db\init.sql") {
    Get-Content "$ROOT\db\init.sql" | docker compose exec -T db psql -U fbref -d fbref
    CheckLastExitCode
}
Get-ChildItem "$ROOT\db\migrations\*.sql" | Sort-Object Name | ForEach-Object {
    Log "   Eseguo $($_.Name)..."
    Get-Content $_.FullName | docker compose exec -T db psql -U fbref -d fbref
    CheckLastExitCode
}
Log "✅ DB inizializzato e migration completate"

# ── 4. Scrape match stats ────────────────────────────────────────────────
if (-not $SkipScraper) {
    Log "STEP 4/12 — Scraping match stats ($League $Seasons)"
    docker compose --profile scraper run --rm scraper --leagues "$League" --seasons "$Seasons"
    CheckLastExitCode
    Log "✅ Match stats scraping completato"
} else { Log "⏭️  STEP 4 — Match stats saltato" }

# ── 5. Scrape per-season stats ───────────────────────────────────────────
if (-not $SkipStats) {
    Log "STEP 5/12 — Scraping per-season stats ($League $StatsSeasons)"
    docker compose --profile scraper run --rm scraper --leagues "$League" --seasons "$StatsSeasons" --stats
    CheckLastExitCode
    Log "✅ Per-season stats scraping completato"
} else { Log "⏭️  STEP 5 — Stats saltato" }

# ── 6. Import quotations (if available) ─────────────────────────────────
Log "STEP 6/12 — (Opzionale) Import quotazioni"
$quotazioniDir = "$ROOT\quotazioni"
if (Test-Path $quotazioniDir) {
    $xlsxFiles = Get-ChildItem "$quotazioniDir\*.xlsx" -ErrorAction SilentlyContinue
    if ($xlsxFiles) {
        docker compose run --rm -v "${quotazioniDir}:/app/quotazioni:ro" api python -m ml.data.import_quotations --quotazioni-dir /app/quotazioni --source listone_fantagazzetta
        CheckLastExitCode
        Log "✅ Quotazioni importate"
    } else { Log "⏭️  Nessun file .xlsx in ./quotazioni, salto..." }
} else { Log "⏭️  Cartella ./quotazioni non trovata, salto..." }

# ── 7. Generate fantavoto CSV ────────────────────────────────────────────
if (-not $SkipFantavoto) {
    Log "STEP 7/12 — Generazione fantavoto CSV"
    $votiDir = "$ROOT\voti"
    if (Test-Path $votiDir) {
        # Ensure ml image is built
        docker compose --profile ml build --no-cache ml 2>&1 | Out-Null
        docker compose --profile ml run --rm --entrypoint python -v "${votiDir}:/app/voti:ro" ml -m ml.data.voti_loader --voti-dir /app/voti --out /app/artifacts/fantavoto.csv
        CheckLastExitCode
        Log "✅ Fantavoto CSV generato"
    } else { Write-Host "⚠️  Cartella ./voti non trovata" -ForegroundColor Yellow }
} else { Log "⏭️  STEP 7 — Fantavoto saltato" }

# ── 8. ML Pipeline ──────────────────────────────────────────────────────
if (-not $SkipML) {
    Log "STEP 8/12 — Pipeline ML ($League con fantavoto)"
    docker compose --profile ml run --rm ml --league "$League" --fantavoto-csv /app/artifacts/fantavoto.csv --predict-next
    CheckLastExitCode
    Log "✅ Pipeline ML completata"
} else { Log "⏭️  STEP 8 — ML saltato" }

# ── 9. ID Mapping ─────────────────────────────────────────────────────────
if (-not $SkipMapping) {
    Log "STEP 9/12 — ID Mapping Fantacalcio ↔ FotMob"
    $pgPassword = (Select-String '(?<=POSTGRES_PASSWORD=).*' "$ROOT\.env").Matches.Value
    docker compose exec -e ML_DATABASE_URL="postgresql+psycopg2://fbref:${pgPassword}@db:5432/fbref" api python -m ml.data.run_id_mapping
    CheckLastExitCode
    Log "✅ ID Mapping completato"

    # Backfill manual resolutions (idempotent, safe to run every time)
    Log "   Backfill risoluzioni manuali in manual_resolutions..."
    docker compose exec -e ML_DATABASE_URL="postgresql+psycopg2://fbref:${pgPassword}@db:5432/fbref" api python scripts/backfill_manual_resolutions.py
    Log "✅ Backfill risoluzioni completato"
} else { Log "⏭️  STEP 9 — ID Mapping + backfill saltato" }

# ── 10. MANTRA Computation ───────────────────────────────────────────────
if (-not $SkipMantra) {
    Log "STEP 10/12 — Computazione MANTRA scoring"
    # Wait for API to be ready
    $apiRetries = 0
    do {
        $apiOk = curl.exe -s -o "$env:TEMP\mantra_health.json" -w "%{http_code}" "http://localhost:8000/api/v1/health"
        if ($apiOk -eq 200) { break }
        $apiRetries++
        Start-Sleep -Seconds 3
    } while ($apiRetries -lt 10)
    if ($apiRetries -ge 10) { Write-Host "❌ API non disponibile" -ForegroundColor Red; exit 1 }
    
    $mantraResult = curl.exe -s -X POST "http://localhost:8000/api/v1/mantra/run" -H "Content-Type: application/json" -d '{}' | ConvertFrom-Json
    if ($mantraResult.status -eq "ok") {
        Log "✅ MANTRA completato: $($mantraResult.n_players) giocatori"
    } else {
        Write-Host "❌ MANTRA fallito: $mantraResult" -ForegroundColor Red
    }
} else { Log "⏭️  STEP 10 — MANTRA saltato" }

# ── 11. Frontend ─────────────────────────────────────────────────────────
if (-not $SkipFrontend) {
    Log "STEP 11/12 — Avvio frontend Angular"
    Push-Location "$ROOT\frontend"
    Start-Process -NoNewWindow -FilePath "npx" -ArgumentList "ng serve --host 0.0.0.0 --port 4200"
    Pop-Location
    Log "✅ Frontend avviato su http://localhost:4200"
} else { Log "⏭️  STEP 11 — Frontend saltato" }

# ── 12. Verify ────────────────────────────────────────────────────────────
Log "STEP 12/12 — Verifica finale"
Start-Sleep -Seconds 3
$health = curl.exe -s -o "$env:TEMP\final_health.json" -w "%{http_code}" "http://localhost:8000/api/v1/health"
if ($health -eq 200) {
    Log "✅ API risponde correttamente"
} else {
    Write-Host "⚠️  API non raggiungibile (HTTP $health). Verifica con 'docker compose logs api'" -ForegroundColor Yellow
}

# ── Done ─────────────────────────────────────────────────────────────────
Write-Host @"

╔════════════════════════════════════════════════════╗
║         🎯 Pipeline completata!                   ║
╠════════════════════════════════════════════════════╣
║  API:    http://localhost:8000/api/v1/health       ║
║  Swagger: http://localhost:8000/docs               ║
║  MANTRA:  http://localhost:8000/api/v1/mantra/players║
║  Frontend: http://localhost:4200                   ║
║  DB:     docker compose exec db psql -U fbref fbref║
╚════════════════════════════════════════════════════╝

Comandi rapidi:
  docker compose ps                        → stato servizi
  docker compose logs -f api               → log API
  docker compose down                      → ferma tutto
"@
