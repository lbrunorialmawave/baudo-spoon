import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/**
 * Esempio concreto da mostrare nella legenda di un campo.
 * - `label`: valore/identificatore sintetico (es. "0.3")
 * - `value`: descrizione/significato pratico (es. "consigliato")
 */
export interface FieldLegendExample {
  readonly label: string;
  readonly value: string;
}

/**
 * Componente presentazionale riusabile per mostrare la legenda di un campo
 * form: una descrizione estesa seguita da una lista di esempi concreti d'uso.
 *
 * Caratteristiche:
 * - Standalone, OnPush, signal inputs.
 * - Accessibilità WCAG 2.2 AA: `aria-describedby` del campo target deve
 *   puntare a `fieldId`; questo <p> ha ruolo implicito di testo descrittivo.
 * - Visibilità costante (no tooltip): la legenda è sempre leggibile in pagina.
 *
 * Utilizzo tipico:
 * ```html
 * <label class="field-label" for="alpha">Fattore EWMA (alpha)</label>
 * <input class="field-input" id="alpha" type="number" min="0" max="1"
 *        step="0.05" [(ngModel)]="alpha"
 *        [attr.aria-describedby]="'legend-alpha'" />
 * <app-field-legend
 *   fieldId="legend-alpha"
 *   description="Peso attribuito all'ultima osservazione nell'aggiornamento
 *                ricorsivo dell'indice di prezzo EWMA."
 *   [examples]="alphaExamples" />
 * ```
 */
@Component({
  selector: 'app-field-legend',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (description() || (examples()?.length ?? 0) > 0) {
      <p class="field-legend" [id]="fieldId()">
        @if (description()) {
          <span class="field-legend-text">{{ description() }}</span>
        }
        @if (examples()?.length) {
          <span class="field-legend-examples">
            <span class="field-legend-tag">Esempi</span>
            @for (ex of examples()!; track ex.label) {
              <span class="field-legend-example">
                <code>{{ ex.label }}</code>
                <span class="field-legend-arrow" aria-hidden="true">→</span>
                <span>{{ ex.value }}</span>
              </span>
            }
          </span>
        }
      </p>
    }
  `,
  styles: [`
    :host { display: block; }

    .field-legend {
      font-size: 11px;
      line-height: 1.5;
      color: var(--color-text-secondary);
      margin: 4px 0 0;
      padding: 6px 8px;
      background: color-mix(in srgb, var(--color-accent) 5%, transparent);
      border-left: 2px solid var(--color-accent);
      border-radius: 0 4px 4px 0;
    }

    .field-legend-text {
      display: block;
    }

    .field-legend-examples {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px 10px;
      margin-top: 4px;
    }

    .field-legend-tag {
      font-weight: 700;
      color: var(--color-text-primary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 9px;
    }

    .field-legend-example {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 11px;
    }

    .field-legend-example code {
      background: var(--color-bg);
      padding: 1px 5px;
      border-radius: 3px;
      font-size: 10px;
      font-family: var(--font-mono, ui-monospace, monospace);
      color: var(--color-text-primary);
      border: 1px solid var(--color-border);
    }

    .field-legend-arrow {
      color: var(--color-text-secondary);
      opacity: 0.7;
    }
  `],
})
export class FieldLegendComponent {
  /** ID univoco usato per `aria-describedby` dal campo target. Richiesto. */
  public readonly fieldId = input.required<string>();
  /** Descrizione estesa: cosa fa il campo e come viene utilizzato. */
  public readonly description = input<string>('');
  /** Lista di esempi concreti d'uso. */
  public readonly examples = input<readonly FieldLegendExample[] | null>(null);
}
