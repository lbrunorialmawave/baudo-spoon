import { Component, afterNextRender, inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { HealthCheckService } from './core/services/health-check.service';
import { HealthGateComponent } from './shared/components/health-gate/health-gate.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, HealthGateComponent],
  template: `
    @if (health.ready()) {
      <router-outlet />
    } @else {
      <app-health-gate />
    }
  `,
  styleUrl: './app.scss',
})
export class App {
  protected readonly health = inject(HealthCheckService);

  constructor() {
    afterNextRender(() => this.health.verify());
  }
}
