const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');
const path = require('path');

// ── CLI / env configuration ────────────────────────────────────────────────
// Supports both legacy batch mode (1→38) and incremental single-matchday mode
// used by the voti-refresh workflow:
//
//   node scraper.js --anno=2025-26 --giornata=12
//   node scraper.js --anno=2025-26 --start=10 --end=12 --output=stdout
//
// Env fallbacks: ANNO, START_GIORNATA, END_GIORNATA, OUTPUT (file|stdout)

function parseArgs(argv) {
    const args = {
        anno: process.env.ANNO || '2025-26',
        start: parseInt(process.env.START_GIORNATA || '1', 10),
        end: parseInt(process.env.END_GIORNATA || '38', 10),
        output: process.env.OUTPUT || 'file', // 'file' | 'stdout'
        outfile: null,
    };
    for (const raw of argv.slice(2)) {
        if (raw.startsWith('--anno=')) args.anno = raw.slice(7);
        else if (raw.startsWith('--giornata=')) {
            const g = parseInt(raw.slice(11), 10);
            args.start = g;
            args.end = g;
        } else if (raw.startsWith('--start=')) args.start = parseInt(raw.slice(8), 10);
        else if (raw.startsWith('--end=')) args.end = parseInt(raw.slice(6), 10);
        else if (raw.startsWith('--output=')) args.output = raw.slice(9);
        else if (raw.startsWith('--outfile=')) args.outfile = raw.slice(10);
        else if (raw === '--help' || raw === '-h') {
            console.error(`Usage: node scraper.js [options]
  --anno=YYYY-YY       Season label (default: 2025-26 or $ANNO)
  --giornata=N         Scrape a single matchday (sets start=end=N)
  --start=N --end=N    Inclusive matchday range (default 1-38)
  --output=file|stdout Where to write JSON (default: file)
  --outfile=PATH       Explicit output path (default: voti_fantacalcio-{anno}.json)`);
            process.exit(0);
        }
    }
    if (args.start > args.end) {
        console.error(`Invalid range: start (${args.start}) > end (${args.end})`);
        process.exit(1);
    }
    return args;
}

const ARGS = parseArgs(process.argv);
const ANNO = ARGS.anno;
const BASE_URL = `https://www.fantacalcio.it/voti-fantacalcio-serie-a/${ANNO}`;
const START_GIORNATA = ARGS.start;
const END_GIORNATA = ARGS.end;
const REQUEST_DELAY_MS = 1000;
const OUTPUT_FILE = ARGS.outfile || `voti_fantacalcio-${ANNO}.json`;

// Mappatura dei ruoli (escludiamo 'all')
const ROLE_MAP = {
    'p': 'Portiere',
    'd': 'Difensore',
    'c': 'Centrocampista',
    'a': 'Attaccante'
};

/**
 * Estrae i dati di una singola giornata
 */
async function scrapeGiornata(giornata) {
    const url = `${BASE_URL}/${giornata}`;
    // Log to stderr so stdout stays clean when --output=stdout
    console.error(`Scraping giornata ${giornata}...`);

    try {
        const response = await axios.get(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        });
        const html = response.data;
        const $ = cheerio.load(html);

        const teamsContainer = $('ul.teams').first();
        if (!teamsContainer.length) {
            console.error(`Nessun elemento ul.teams trovato per la giornata ${giornata}`);
            return null;
        }

        const squadre = [];

        teamsContainer.find('li.team-table').each((index, teamElement) => {
            const team = $(teamElement);

            // Estrai il nome della squadra da team-info
            const teamInfo = team.find('.team-info');
            const teamNameLink = teamInfo.find('a.team-name');
            const squadraNome = teamNameLink.length ? teamNameLink.text().trim() : 'Sconosciuto';

            // Estrazione match score
            const matchScore = team.find('header .match-score');
            const spans = matchScore.find('span');
            const spansArray = spans.toArray().map(el => $(el).text().trim());
            const parts = spansArray.filter(s => s !== '' && s !== '-');
            let squadraCasa, punteggioCasa, punteggioOspite, squadraOspite;
            if (parts.length === 4) {
                [squadraCasa, punteggioCasa, punteggioOspite, squadraOspite] = parts;
            } else {
                const matchText = matchScore.text().trim();
                const matchRegex = /(.+?)\s+(\d+)\s*[-–]\s*(\d+)\s+(.+)/;
                const match = matchText.match(matchRegex);
                if (match) {
                    [, squadraCasa, punteggioCasa, punteggioOspite, squadraOspite] = match;
                } else {
                    squadraCasa = 'Sconosciuto';
                    squadraOspite = 'Sconosciuto';
                    punteggioCasa = '?';
                    punteggioOspite = '?';
                }
            }

            const matchDate = team.find('header .match-date').text().trim();

            // Estrazione giocatori (esclusi allenatori)
            const giocatori = [];
            const table = team.find('table.grades-table');
            table.find('tbody tr').each((rowIndex, rowElement) => {
                const row = $(rowElement);
                const cols = row.find('td');
                if (cols.length < 3) return;

                const playerCell = cols.eq(0);
                const roleSpan = playerCell.find('span.role');
                const ruoloRaw = roleSpan.attr('data-value');
                const ruolo = ROLE_MAP[ruoloRaw] || ruoloRaw || 'Sconosciuto';

                // Salta allenatori
                if (ruoloRaw === 'all' || ruolo === 'Allenatore') {
                    return;
                }

                const nameLink = playerCell.find('a.player-name');
                let nome = nameLink.length ? nameLink.text().trim() : playerCell.find('span.player-name').text().trim();
                if (!nome) nome = playerCell.text().trim();

                const iconSub = playerCell.find('img[src*="in.webp"]').length ? 'subentrato' : '';
                const iconOut = playerCell.find('img[src*="out.webp"]').length ? 'sostituito' : '';
                const stato = iconSub || iconOut || '';

                // Voti - estrai da data-value, non dal testo
                const voteCell = cols.eq(1);
                const pills = voteCell.find('.pill');
                const voti = [];
                pills.each((i, pill) => {
                    const grade = $(pill).find('span.player-grade').attr('data-value') || null;
                    const fantaGrade = $(pill).find('span.player-fanta-grade').attr('data-value') || null;
                    voti.push({
                        voto: grade,
                        fantavoto: fantaGrade
                    });
                });
                while (voti.length < 3) {
                    voti.push({ voto: null, fantavoto: null });
                }

                // Bonus e malus separati
                const bonusCell = cols.eq(2);
                const bonusItems = bonusCell.find('.player-bonus');
                const bonus = {};
                const malus = {};
                bonusItems.each((i, item) => {
                    const el = $(item);
                    const type = el.hasClass('bonus') ? 'bonus' : 'malus';
                    const value = parseInt(el.attr('data-value') || '0', 10);
                    const title = el.attr('title') || '';
                    let key = title.toLowerCase().replace(/ /g, '_');
                    if (!key) key = `item_${i}`;

                    if (type === 'bonus') {
                        bonus[key] = value;
                    } else {
                        malus[key] = value;
                    }
                });

                giocatori.push({
                    squadra: squadraNome,
                    nome,
                    ruolo,
                    stato,
                    voti: {
                        fantacalcio: voti[0] || null,
                        statistico: voti[1] || null,
                        italia: voti[2] || null
                    },
                    bonus: bonus,
                    malus: malus
                });
            });

            squadre.push({
                squadraCasa,
                squadraOspite,
                punteggioCasa,
                punteggioOspite,
                data: matchDate,
                giocatori
            });
        });

        return {
            giornata,
            url,
            squadre
        };

    } catch (error) {
        console.error(`Errore durante lo scraping della giornata ${giornata}:`, error.message);
        return null;
    }
}

/**
 * Esegue lo scraping del range richiesto e scrive su file o stdout.
 */
async function scrapeTutteGiornate() {
    console.error(`Avvio scraping ${ANNO} da giornata ${START_GIORNATA} a ${END_GIORNATA}...`);

    const risultati = [];

    for (let g = START_GIORNATA; g <= END_GIORNATA; g++) {
        const data = await scrapeGiornata(g);
        if (data) {
            risultati.push(data);
        }
        if (g < END_GIORNATA) {
            console.error(`Attendo ${REQUEST_DELAY_MS}ms prima della prossima richiesta...`);
            await new Promise(resolve => setTimeout(resolve, REQUEST_DELAY_MS));
        }
    }

    const payload = JSON.stringify(risultati, null, 2);

    if (ARGS.output === 'stdout') {
        process.stdout.write(payload);
        console.error(`✅ Scraping completato! ${risultati.length} giornata(e) scritte su stdout`);
    } else {
        const outputPath = path.resolve(OUTPUT_FILE);
        fs.writeFileSync(outputPath, payload, 'utf-8');
        console.error(`✅ Scraping completato! Dati salvati in ${outputPath}`);
    }
}

scrapeTutteGiornate().catch((err) => {
    console.error(err);
    process.exit(1);
});
