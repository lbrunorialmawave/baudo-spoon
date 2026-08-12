const INTERVALO_MS = 4 * 60 * 1000; // 4 minuti
const URL = 'https://baudo-spoon.onrender.com/api/v1/health';

async function chiamaServizio() {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] Chiamata in corso...`);

  try {
    const response = await fetch(URL, {
      method: 'GET',
      headers: { accept: 'application/json' }
    });

    const body = await response.text(); 
    console.log(`[${timestamp}] ✅ Risposta: ${response.status} ${response.statusText}`);
    console.log(`[${timestamp}] 📦 Body: ${body}`);
  } catch (error) {
    console.error(`[${timestamp}] ❌ Errore: ${error.message}`);
  }
}
chiamaServizio();
const idInterval = setInterval(chiamaServizio, INTERVALO_MS);
process.on('SIGINT', () => {
  console.log('\n🛑 Interruzione ricevuta. Arresto dello script...');
  clearInterval(idInterval);
  process.exit(0);
});

console.log(`🚀 Script avviato. Chiamate ogni ${INTERVALO_MS / 1000} secondi. Premi Ctrl+C per fermare.`);