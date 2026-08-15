/**
 * WS-10 — Frontend auction config contract (vitest runtime).
 *
 * Garantisce che i nuovi parametri limited-cohort (applyReliabilityWeight,
 * riskAversion) siano preservati fedelmente lungo tutta la catena:
 *
 *   backend schema → AuctionConfig (TS) → form (AuctionComponent)
 *     → setupAuctionConfig getter → backend schema (richiesta startAuction)
 *
 * Le verifiche statiche (regex sul sorgente TS) restano in
 * `ml/tests/test_frontend_auction_contract.py` (WS7): qui si testa il
 * comportamento runtime del componente, non la forma del codice.
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { of, throwError } from 'rxjs';
import { AuctionComponent } from './auction.component';
import { AuctionService } from '../../core/services/auction.service';
import { QuotationService } from '../../core/services/quotation.service';
import {
  AuctionConfig,
  InitializeAuctionRequest,
  InitializeAuctionResponse,
} from '../../core/models/auction.models';

/** Costruisce una AuctionConfig minima valida per i test di idratazione. */
function makeConfig(overrides: Partial<AuctionConfig> = {}): AuctionConfig {
  return {
    numParticipants: 8,
    roleQuotas: { P: 3, D: 8, C: 8, A: 6 },
    ruleset: 'CLASSIC',
    marketDriftConfig: {
      alpha: 0.3,
      spilloverAdjacentTier: 0.1,
      spilloverCrossRole: 0.05,
      minIndex: 0.5,
      maxIndex: 2.0,
      tierThresholds: [0.3, 0.7],
    },
    alternativesConfig: { lowCostPercentile: 0.3 },
    useInflationBaseline: false,
    referenceBudget: 300,
    budgetInitial: 500,
    ...overrides,
  };
}

/** Mock minimale di AuctionService: init, summary, varRanking e getSeasons coprono il setup. */
function makeAuctionServiceSpy() {
  return {
    init: vi.fn((_req: InitializeAuctionRequest) =>
      of<InitializeAuctionResponse>({ sessionId: 'sess-test' }),
    ),
    // summary() viene invocato da refreshSummary() nel callback `next` di init(),
    // che a sua volta chiama refreshVarRanking() → varRanking().
    summary: vi.fn(() => of({
      participants: [],
      assignments: [],
      priceIndex: {},
    } as never)),
    varRanking: vi.fn(() => of({
      sessionId: 'sess-test',
      items: [],
      usingLivePrices: false,
    } as never)),
    getSeasons: vi.fn(() => of<number[]>([])),
  };
}

/** Mock di QuotationService: deve restituire un Observable nel costruttore. */
function makeQuotationServiceSpy() {
  return {
    getSeasons: vi.fn(() => of<number[]>([])),
  };
}

describe('AuctionComponent — WS-10 frontend config contract', () => {
  let auctionService: ReturnType<typeof makeAuctionServiceSpy>;

  beforeEach(async () => {
    auctionService = makeAuctionServiceSpy();

    await TestBed.configureTestingModule({
      imports: [AuctionComponent],
      providers: [
        provideZonelessChangeDetection(),
        { provide: AuctionService, useValue: auctionService },
        { provide: QuotationService, useValue: makeQuotationServiceSpy() },
      ],
    }).compileComponents();
  });

  // ── 12.1 Default ─────────────────────────────────────────────────────
  describe('12.1 — default values', () => {
    it('applyReliabilityWeight default = true (ADR 0001)', () => {
      const fixture = TestBed.createComponent(AuctionComponent);
      const cmp = fixture.componentInstance;

      expect(cmp.applyReliabilityWeight).toBe(true);
      expect(cmp.setupAuctionConfig.applyReliabilityWeight).toBe(true);
    });

    it('riskAversion default = 0 (risk-neutral)', () => {
      const fixture = TestBed.createComponent(AuctionComponent);
      const cmp = fixture.componentInstance;

      expect(cmp.riskAversion).toBe(0);
      expect(cmp.setupAuctionConfig.riskAversion).toBe(0);
    });
  });

  // ── 12.2 Explicit false must be preserved ───────────────────────────
  describe('12.2 — explicit false preservation', () => {
    it('setupAuctionConfig returns false when form is set to false', () => {
      const fixture = TestBed.createComponent(AuctionComponent);
      const cmp = fixture.componentInstance;

      cmp.applyReliabilityWeight = false;

      expect(cmp.setupAuctionConfig.applyReliabilityWeight).toBe(false);
    });

    it('applyPreset({applyReliabilityWeight: false}) does NOT flip to true', () => {
      const fixture = TestBed.createComponent(AuctionComponent);
      const cmp = fixture.componentInstance;

      // Sanity: starts true
      expect(cmp.applyReliabilityWeight).toBe(true);

      // Hydrate a config that explicitly asks for false
      const cfg = makeConfig({ applyReliabilityWeight: false });
      // applyPreset is the method used by onPresetChange; passing the bare
      // config object requires the surrounding preset envelope, so we use
      // a minimal AuctionPreset-shaped object via the public method.
      cmp.applyPreset({ config: cfg } as never);

      expect(cmp.applyReliabilityWeight).toBe(false);
      expect(cmp.setupAuctionConfig.applyReliabilityWeight).toBe(false);
    });

    it('startAuction request body carries applyReliabilityWeight=false', () => {
      const fixture = TestBed.createComponent(AuctionComponent);
      const cmp = fixture.componentInstance;

      cmp.applyReliabilityWeight = false;
      cmp.startAuction();

      expect(auctionService.init).toHaveBeenCalledTimes(1);
      const sent = auctionService.init.mock.calls[0][0] as InitializeAuctionRequest;
      expect(sent.config.applyReliabilityWeight).toBe(false);
    });
  });

  // ── 12.3 riskAversion range pass-through ─────────────────────────────
  describe('12.3 — riskAversion values', () => {
    it.each([0, 0.5, 5])(
      'round-trips valid riskAversion=%s through setupAuctionConfig',
      (value) => {
        const fixture = TestBed.createComponent(AuctionComponent);
        const cmp = fixture.componentInstance;

        cmp.riskAversion = value;

        expect(cmp.setupAuctionConfig.riskAversion).toBe(value);
      },
    );

    it.each([-1, 6, Number.NaN])(
      'documents pass-through for out-of-range riskAversion=%s (backend is the gate)',
      (value) => {
        const fixture = TestBed.createComponent(AuctionComponent);
        const cmp = fixture.componentInstance;

        cmp.riskAversion = value;

        // The frontend does NOT clamp — backend Pydantic schema is the
        // single source of truth for range validation (fail-closed at
        // 422). This test pins the contract so any frontend-side guard
        // added later becomes a deliberate, reviewed change.
        expect(cmp.setupAuctionConfig.riskAversion).toBe(value);
      },
    );

    it('startAuction request body carries riskAversion=0.5', () => {
      const fixture = TestBed.createComponent(AuctionComponent);
      const cmp = fixture.componentInstance;

      cmp.riskAversion = 0.5;
      cmp.startAuction();

      const sent = auctionService.init.mock.calls[0][0] as InitializeAuctionRequest;
      expect(sent.config.riskAversion).toBe(0.5);
    });
  });

  // ── 12.4 Round trip ─────────────────────────────────────────────────
  describe('12.4 — backend schema ↔ frontend config round trip', () => {
    it('hydrates a config and emits an equivalent setupAuctionConfig', () => {
      const fixture = TestBed.createComponent(AuctionComponent);
      const cmp = fixture.componentInstance;

      const incoming: AuctionConfig = makeConfig({
        applyReliabilityWeight: false,
        riskAversion: 2.5,
        hybridBlend: 0.3,
      });

      cmp.applyPreset({ config: incoming } as never);

      const out = cmp.setupAuctionConfig;
      expect(out.applyReliabilityWeight).toBe(false);
      expect(out.riskAversion).toBe(2.5);
      expect(out.hybridBlend).toBe(0.3);
    });

    it('omitted applyReliabilityWeight in input does not override default', () => {
      const fixture = TestBed.createComponent(AuctionComponent);
      const cmp = fixture.componentInstance;

      const incoming: AuctionConfig = makeConfig({
        applyReliabilityWeight: undefined,
        riskAversion: undefined,
      });

      cmp.applyPreset({ config: incoming } as never);

      // Defaults preserved: backend will receive true / 0 because the
      // frontend never silently flips the user-facing toggle.
      expect(cmp.applyReliabilityWeight).toBe(true);
      expect(cmp.riskAversion).toBe(0);
    });
  });

  // ── 12.5 startAuction propagates both fields ─────────────────────────
  describe('12.5 — startAuction HTTP body contract', () => {
    it('POST body always contains applyReliabilityWeight and riskAversion', () => {
      const fixture = TestBed.createComponent(AuctionComponent);
      const cmp = fixture.componentInstance;

      cmp.startAuction();

      expect(auctionService.init).toHaveBeenCalledTimes(1);
      const sent = auctionService.init.mock.calls[0][0] as InitializeAuctionRequest;
      expect(sent.config).toHaveProperty('applyReliabilityWeight');
      expect(sent.config).toHaveProperty('riskAversion');
      expect(typeof sent.config.applyReliabilityWeight).toBe('boolean');
      expect(typeof sent.config.riskAversion).toBe('number');
    });

    it('does not call init when startAuction is invoked but seasons load fails (still attempts)', () => {
      // Reconfigure with a failing getSeasons mock to confirm the
      // constructor fallback path is exercised; init() is still called
      // because the form is submittable even without seasons.
      const failingQuotation = {
        getSeasons: vi.fn(() => throwError(() => new Error('boom'))),
      };
      auctionService.getSeasons = vi.fn(() => of<number[]>([]));

      TestBed.resetTestingModule();
      TestBed.configureTestingModule({
        imports: [AuctionComponent],
        providers: [
          provideZonelessChangeDetection(),
          { provide: AuctionService, useValue: auctionService },
          { provide: QuotationService, useValue: failingQuotation },
        ],
      });

      const fixture = TestBed.createComponent(AuctionComponent);
      const cmp = fixture.componentInstance;

      cmp.startAuction();

      // Body still contains both fields regardless of season-load state
      const sent = auctionService.init.mock.calls[0][0] as InitializeAuctionRequest;
      expect(sent.config.applyReliabilityWeight).toBe(true);
      expect(sent.config.riskAversion).toBe(0);
    });
  });
});
